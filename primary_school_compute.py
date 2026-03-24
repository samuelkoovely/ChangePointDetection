from __future__ import annotations

import os
import pickle
import time
from pathlib import Path
from typing import Sequence

import numpy as np
from joblib import Parallel, delayed

from TemporalNetwork import ContTempNetwork
from signal_generation import (
    PreparedSignalSample,
    WindowSamplingPlan,
    compute_signals_for_lambdas_prepared,
)


PRIMARY_SCHOOL_PATH = (
    "/home/b/skoove/Desktop/entropy/paper_data/socio_pat_primary_school/"
    "primaryschoolnet"
)
FALLBACK_PRIMARY_SCHOOL_PATH = "./data/primaryschoolnet"
OUTPUT_BASE = Path("./gridsearch_results/primaryschool_day1")

DAY1_STOP_INDEX = 1556
LAMBDAS = np.logspace(-5, 0, 10)
WINDOWS_MINUTES = [2, 30, 60]
WINDOWS_SECONDS = [60 * minutes for minutes in WINDOWS_MINUTES]

# One sample means parallelism has to happen over lambda blocks. Each worker gets
# its own prepared copy and processes a subset of lambdas sequentially.
BACKEND = "loky"
N_JOBS = min(4, len(LAMBDAS), os.cpu_count() or 1)


def resolve_network_path(network_path: str | Path = PRIMARY_SCHOOL_PATH) -> str:
    """
    Resolve the primary-school network path, falling back to the local repo copy.
    """
    requested = Path(network_path)
    if Path(f"{requested}.pickle").exists():
        return str(requested)

    fallback = Path(FALLBACK_PRIMARY_SCHOOL_PATH)
    if Path(f"{fallback}.pickle").exists():
        return str(fallback)

    return str(requested)


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
    resolved_path = resolve_network_path(network_path)
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

    # Preserve the full node set so day-1 signals live in the same state space
    # as the full primary-school network.
    day1_net.node_array = full_net.node_array.copy()
    day1_net.num_nodes = full_net.num_nodes
    day1_net._compute_time_grid()

    return day1_net, cutoff_time


def prepare_full_window_sample(
    net: ContTempNetwork,
    windows_seconds: Sequence[float] = WINDOWS_SECONDS,
) -> PreparedSignalSample:
    """
    Prepare a full-scan sample context for the selected windows.

    Unlike the generic helper in `signal_generation.py`, this computes all valid
    window start indices exactly instead of sampling them.
    """
    windows = tuple(float(window) for window in windows_seconds)
    p0 = np.ones(net.num_nodes, dtype=float) / net.num_nodes

    net.compute_laplacian_matrices(
        t_start=net.times[0],
        t_stop=net.times[-1],
        random_walk=False,
    )

    window_plans: dict[float, WindowSamplingPlan] = {}
    for window in windows:
        considered_times = net.times[net.times < net.times[-1] - window]
        M = len(considered_times)

        if M <= 0:
            k_samples = np.array([], dtype=int)
            t_samples = np.array([], dtype=float)
            window_ends = np.array([], dtype=np.int64)
        else:
            k_samples = np.arange(M, dtype=int)
            t_samples = np.asarray(net.times[k_samples], dtype=float)
            window_ends = np.asarray(
                net._get_window_ends_vectorized(window, n_windows=M),
                dtype=np.int64,
            )

        window_plans[window] = WindowSamplingPlan(
            window=window,
            k_samples=k_samples,
            t_samples=t_samples,
            window_ends=window_ends,
        )

    return PreparedSignalSample(
        net=net,
        windows=windows,
        sample_fraction=1.0,
        p0=p0,
        window_plans=window_plans,
    )


def lambda_blocks(lambdas: Sequence[float], n_jobs: int) -> list[list[float]]:
    """
    Split lambdas into contiguous blocks for worker-local sequential processing.
    """
    blocks = np.array_split(np.asarray(lambdas, dtype=float), n_jobs)
    return [block.tolist() for block in blocks if len(block) > 0]


def save_lambda_block_results(
    signals_by_lambda: dict[float, dict[float, dict]],
    output_base: Path,
    day1_stop_time: float,
) -> None:
    """
    Save all signal results in the primary-school day-1 layout.
    """
    for lamda, window_results in signals_by_lambda.items():
        lamda_key = f"{float(lamda):.11f}"
        for window_seconds, signal_result in window_results.items():
            payload = {
                "lamda": lamda_key,
                "lamda_float": float(lamda),
                "window": float(window_seconds),
                "window_seconds": float(window_seconds),
                "window_minutes": float(window_seconds) / 60.0,
                "k_samples": np.asarray(signal_result["k_samples"], dtype=int),
                "t_samples": np.asarray(signal_result["t_samples"], dtype=float),
                # Keep both layouts: direct array for new code, lambda-keyed dict
                # for older primary-school plotting scripts.
                "signal_array": np.asarray(signal_result["signal"], dtype=float),
                "signal": {lamda_key: np.asarray(signal_result["signal"], dtype=float)},
                "day_stop_index": DAY1_STOP_INDEX,
                "day_stop_time_reference": float(day1_stop_time),
            }

            outdir = output_base / "window_S_selected" / str(int(window_seconds))
            outdir.mkdir(parents=True, exist_ok=True)
            outfile = outdir / f"window_S{lamda_key}"
            with open(outfile, "wb") as handle:
                pickle.dump(payload, handle)


def worker(
    prepared: PreparedSignalSample,
    lamda_block: Sequence[float],
    output_base: Path,
    day1_stop_time: float,
) -> list[float]:
    """
    Process one lambda block for the prepared primary-school day-1 sample.
    """
    try:
        signals_by_lambda = compute_signals_for_lambdas_prepared(
            prepared=prepared,
            lambdas=lamda_block,
            use_linear_approx=False,
            lin_t_s=10,
            window_backend="segment_tree",
        )
        save_lambda_block_results(
            signals_by_lambda=signals_by_lambda,
            output_base=output_base,
            day1_stop_time=day1_stop_time,
        )
    except Exception as exc:
        for lamda in lamda_block:
            lamda_key = f"{float(lamda):.11f}"
            for window_seconds in prepared.windows:
                outdir = output_base / "window_S_selected" / str(int(window_seconds))
                outdir.mkdir(parents=True, exist_ok=True)
                outfile = outdir / f"window_S{lamda_key}"
                error_payload = {
                    "lamda": lamda_key,
                    "lamda_float": float(lamda),
                    "window": float(window_seconds),
                    "window_seconds": float(window_seconds),
                    "window_minutes": float(window_seconds) / 60.0,
                    "signal": 10,
                    "error": f"{type(exc).__name__}: {exc}",
                    "day_stop_index": DAY1_STOP_INDEX,
                    "day_stop_time_reference": float(day1_stop_time),
                }
                with open(outfile, "wb") as handle:
                    pickle.dump(error_payload, handle)
        print(f"error for lambda block {lamda_block}: {type(exc).__name__}: {exc}")
        return []

    print("finished lambda block:", [f"{lamda:.11f}" for lamda in lamda_block])
    return [float(lamda) for lamda in lamda_block]


def save_metadata(
    output_base: Path,
    network_path: str,
    prepared: PreparedSignalSample,
    day1_stop_time: float,
    elapsed_seconds: float | None = None,
) -> None:
    """
    Save run metadata alongside the computed signals.
    """
    metadata = {
        "network_path": network_path,
        "day_stop_index_reference": DAY1_STOP_INDEX,
        "day_stop_time_reference": float(day1_stop_time),
        "day1_start_time": float(prepared.net.times[0]),
        "day1_last_time": float(prepared.net.times[-1]),
        "num_nodes": int(prepared.net.num_nodes),
        "num_events": int(prepared.net.num_events),
        "num_times": int(len(prepared.net.times)),
        "lambdas": np.asarray(LAMBDAS, dtype=float),
        "windows_seconds": np.asarray(WINDOWS_SECONDS, dtype=float),
        "windows_minutes": np.asarray(WINDOWS_MINUTES, dtype=float),
        "backend": BACKEND,
        "n_jobs": int(N_JOBS),
        "window_backend": "segment_tree",
        "elapsed_seconds": elapsed_seconds,
    }
    output_base.mkdir(parents=True, exist_ok=True)
    with open(output_base / "metadata.pkl", "wb") as handle:
        pickle.dump(metadata, handle)


def main() -> None:
    network_path = resolve_network_path(PRIMARY_SCHOOL_PATH)
    day1_net, day1_stop_time = load_primary_school_day1_network(network_path)
    prepared = prepare_full_window_sample(day1_net, windows_seconds=WINDOWS_SECONDS)

    save_metadata(
        output_base=OUTPUT_BASE,
        network_path=network_path,
        prepared=prepared,
        day1_stop_time=day1_stop_time,
    )

    blocks = lambda_blocks(LAMBDAS, N_JOBS)
    t_start = time.time()

    if len(blocks) == 1:
        worker(
            prepared=prepared,
            lamda_block=blocks[0],
            output_base=OUTPUT_BASE,
            day1_stop_time=day1_stop_time,
        )
    else:
        Parallel(n_jobs=len(blocks), backend=BACKEND)(
            delayed(worker)(
                prepared=prepared,
                lamda_block=block,
                output_base=OUTPUT_BASE,
                day1_stop_time=day1_stop_time,
            )
            for block in blocks
        )

    elapsed = time.time() - t_start
    save_metadata(
        output_base=OUTPUT_BASE,
        network_path=network_path,
        prepared=prepared,
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
