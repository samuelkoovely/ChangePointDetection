from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.sparse.csgraph import connected_components

from TemporalNetwork import ContTempNetwork


PRIMARY_SCHOOL_PATH = "./data/primaryschoolnet"
OUTPUT_BASE = Path("./gridsearch_results/primaryschool_day1_limit")

DAY1_STOP_INDEX = 1556
WINDOWS_MINUTES = [2, 30, 60]
WINDOWS_SECONDS = [60 * minutes for minutes in WINDOWS_MINUTES]


@dataclass(frozen=True)
class WindowLimitPlan:
    """
    Full-scan window positions for one window length.
    """

    window: float
    k_samples: np.ndarray
    t_samples: np.ndarray


def load_primary_school_day1_network(
    network_path: str | Path = PRIMARY_SCHOOL_PATH,
    day1_stop_index: int = DAY1_STOP_INDEX,
) -> tuple[ContTempNetwork, float]:
    """
    Load the primary-school network and keep only day 1.

    The cut is defined by the reference time `full_net.times[day1_stop_index]`.
    The resulting day-1 network keeps the full node set from the original
    dataset, even if a few nodes only appear later.
    """

    resolved_path = str(network_path)
    full_net = ContTempNetwork.load(
        resolved_path,
        matrices_list=[],
        attributes_list=["node_to_label_dict", "events_table"],
    )
    full_net._compute_time_grid()

    if day1_stop_index >= len(full_net.times):
        raise IndexError(
            f"day1_stop_index={day1_stop_index} is out of bounds for "
            f"{len(full_net.times)} time points."
        )

    cutoff_time = float(full_net.times[day1_stop_index])
    events_day1 = full_net.events_table.loc[
        full_net.events_table.ending_times <= cutoff_time
    ].copy()

    day1_net = ContTempNetwork(
        events_table=events_day1,
        relabel_nodes=False,
        node_to_label_dict=full_net.node_to_label_dict,
    )

    # Preserve the full node set so all window statistics live in the same
    # state space as the complete primary-school network.
    day1_net.node_array = full_net.node_array.copy()
    day1_net.num_nodes = full_net.num_nodes
    day1_net._compute_time_grid()

    return day1_net, cutoff_time


def prepare_full_window_scan(
    net: ContTempNetwork,
    windows_seconds: Sequence[float] = WINDOWS_SECONDS,
) -> dict[float, WindowLimitPlan]:
    """
    Enumerate every valid window start for each requested window length.
    """

    window_plans: dict[float, WindowLimitPlan] = {}

    for window in tuple(float(window) for window in windows_seconds):
        k_samples = np.flatnonzero(net.times < net.times[-1] - window).astype(int)
        t_samples = np.asarray(net.times[k_samples], dtype=float)
        window_plans[window] = WindowLimitPlan(
            window=window,
            k_samples=k_samples,
            t_samples=t_samples,
        )

    return window_plans


def compute_component_log_sum(
    net: ContTempNetwork,
    start_time: float,
    window_seconds: float,
) -> float:
    """
    Compute sum_C (|C| / N) log |C| on the aggregated window graph.
    """

    adjacency = net.compute_static_adjacency_matrix(
        start_time=float(start_time),
        end_time=float(start_time) + float(window_seconds),
    ).tocsr()

    n_components, labels = connected_components(
        adjacency,
        directed=False,
        return_labels=True,
    )
    component_sizes = np.bincount(labels, minlength=n_components).astype(float)
    weights = component_sizes / float(net.num_nodes)

    return float(np.sum(weights * np.log(component_sizes)))


def compute_window_limit_curve(
    net: ContTempNetwork,
    plan: WindowLimitPlan,
) -> dict:
    """
    Compute the connected-component limit statistic for all starts of one window.
    """

    values = np.empty(len(plan.t_samples), dtype=float)

    for idx, start_time in enumerate(plan.t_samples):
        values[idx] = compute_component_log_sum(
            net=net,
            start_time=float(start_time),
            window_seconds=float(plan.window),
        )

        if (idx + 1) % 250 == 0 or idx + 1 == len(plan.t_samples):
            print(
                f"window={int(plan.window)}s: "
                f"{idx + 1}/{len(plan.t_samples)} samples"
            )

    time_limit_array = (
        np.column_stack((plan.t_samples, values))
        if len(plan.t_samples) > 0
        else np.empty((0, 2), dtype=float)
    )

    return {
        "window": float(plan.window),
        "window_seconds": float(plan.window),
        "window_minutes": float(plan.window) / 60.0,
        "k_samples": np.asarray(plan.k_samples, dtype=int),
        "t_samples": np.asarray(plan.t_samples, dtype=float),
        "component_log_sums": values,
        # Compatibility with downstream plotting helpers that already expect a
        # generic 1D signal array plus explicit sampled times.
        "signal": values,
        "signal_array": values,
        # Shape (n_selected_times, 2) with columns [t_sample, component_log_sum].
        "time_component_log_sums": time_limit_array,
        "statistic": "sum((size / N) * log(size)) over connected components",
    }


def save_window_result(result: dict, output_base: Path) -> None:
    """
    Save one window payload.
    """

    outdir = output_base / "window_limit_selected" / str(int(result["window_seconds"]))
    outdir.mkdir(parents=True, exist_ok=True)

    with open(outdir / "window_limit.pkl", "wb") as handle:
        pickle.dump(result, handle)


def save_metadata(
    output_base: Path,
    network_path: str,
    net: ContTempNetwork,
    window_plans: dict[float, WindowLimitPlan],
    day1_stop_time: float,
    elapsed_seconds: float | None = None,
) -> None:
    """
    Save run metadata alongside the computed limit curves.
    """

    windows_seconds = np.asarray(sorted(window_plans), dtype=float)
    metadata = {
        "network_path": network_path,
        "day_stop_index_reference": DAY1_STOP_INDEX,
        "day_stop_time_reference": float(day1_stop_time),
        "day1_start_time": float(net.times[0]),
        "day1_last_time": float(net.times[-1]),
        "num_nodes": int(net.num_nodes),
        "num_events": int(net.num_events),
        "num_times": int(len(net.times)),
        "windows_seconds": windows_seconds,
        "windows_minutes": windows_seconds / 60.0,
        "window_sample_counts": {
            float(window): int(len(plan.t_samples))
            for window, plan in window_plans.items()
        },
        "statistic": "sum((size / N) * log(size)) over connected components",
        "time_component_log_sums_columns": ["t_sample", "component_log_sum"],
        "output_layout": "window_limit_selected/<window_seconds>/window_limit.pkl",
        "elapsed_seconds": elapsed_seconds,
    }

    output_base.mkdir(parents=True, exist_ok=True)
    with open(output_base / "metadata.pkl", "wb") as handle:
        pickle.dump(metadata, handle)


def main() -> None:
    network_path = PRIMARY_SCHOOL_PATH
    day1_net, day1_stop_time = load_primary_school_day1_network(network_path)
    window_plans = prepare_full_window_scan(day1_net, windows_seconds=WINDOWS_SECONDS)

    save_metadata(
        output_base=OUTPUT_BASE,
        network_path=network_path,
        net=day1_net,
        window_plans=window_plans,
        day1_stop_time=day1_stop_time,
    )

    t_start = time.time()

    for window_seconds in WINDOWS_SECONDS:
        plan = window_plans[float(window_seconds)]
        print(
            f"starting window={int(window_seconds)}s with {len(plan.t_samples)} samples"
        )
        result = compute_window_limit_curve(day1_net, plan)
        result["day_stop_index"] = DAY1_STOP_INDEX
        result["day_stop_time_reference"] = float(day1_stop_time)
        save_window_result(result=result, output_base=OUTPUT_BASE)

    elapsed = time.time() - t_start
    save_metadata(
        output_base=OUTPUT_BASE,
        network_path=network_path,
        net=day1_net,
        window_plans=window_plans,
        day1_stop_time=day1_stop_time,
        elapsed_seconds=elapsed,
    )

    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(
        f"Total runtime: {elapsed:.2f} s "
        f"({int(hours):02d}:{int(minutes):02d}:{seconds:05.2f})"
    )


if __name__ == "__main__":
    main()
