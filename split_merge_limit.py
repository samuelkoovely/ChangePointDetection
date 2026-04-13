from __future__ import annotations

import argparse
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.sparse.csgraph import connected_components


NETWORK_PATH = Path("./data/split_merge.pkl")
SIGNAL_RESULTS_CANDIDATES = (
    Path("./gridsearch_results/motifs_f/split_merge"),
    Path("./gridsearch_results/motifs/split_merge"),
    Path("./gridsearch_results/motifs_run1/split_merge"),
)
OUTPUT_BASE = Path("./gridsearch_results/split_merge_limit")
DEFAULT_WINDOWS = [1.0, 3.0, 5.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute connected-component upper-bound curves for the split-merge "
            "motif over one or more local window lengths."
        )
    )
    parser.add_argument(
        "--network-path",
        type=Path,
        default=NETWORK_PATH,
        help="Path to the split-merge temporal network pickle.",
    )
    parser.add_argument(
        "--signal-results-dir",
        type=Path,
        default=None,
        help=(
            "Optional forward entropy results directory. When omitted, the script "
            "searches the standard motif result locations."
        ),
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=OUTPUT_BASE,
        help="Directory where window limit curves will be written.",
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Explicit window lengths to process in seconds. When omitted, the "
            "script falls back to signal metadata or the built-in defaults."
        ),
    )
    return parser.parse_args()


@dataclass(frozen=True)
class WindowLimitPlan:
    """
    Full-scan window positions for one window length.
    """

    window: float
    k_samples: np.ndarray
    t_samples: np.ndarray


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def window_key(window: float) -> str:
    """
    Format a window value for directory names.
    """

    return f"{float(window):g}"


def resolve_signal_results_dir(
    requested_dir: Path | None = None,
    allow_missing: bool = False,
) -> Path | None:
    """
    Find the forward split-merge entropy results if they already exist.
    """

    if requested_dir is not None:
        metadata_path = requested_dir / "metadata.pkl"
        signal_dir = requested_dir / "window_S"
        if metadata_path.exists() and signal_dir.exists():
            return requested_dir
        if allow_missing:
            return None
        raise FileNotFoundError(
            f"Explicit signal results directory {requested_dir} is missing "
            "metadata.pkl or window_S."
        )

    for candidate in SIGNAL_RESULTS_CANDIDATES:
        metadata_path = candidate / "metadata.pkl"
        signal_dir = candidate / "window_S"
        if metadata_path.exists() and signal_dir.exists():
            return candidate

    if allow_missing:
        return None

    raise FileNotFoundError(
        "Could not find forward split-merge entropy results. Checked: "
        + ", ".join(str(path) for path in SIGNAL_RESULTS_CANDIDATES)
    )


def load_signal_metadata(signal_results_dir: Path | None) -> dict | None:
    """
    Load entropy metadata if a forward split-merge run already exists.
    """

    if signal_results_dir is None:
        return None

    metadata_path = signal_results_dir / "metadata.pkl"
    if not metadata_path.exists():
        return None

    return load_pickle(metadata_path)


def get_windows(
    signal_metadata: dict | None,
    requested_windows: Sequence[float] | None = None,
) -> list[float]:
    """
    Return the window lengths to process.
    """

    if requested_windows is not None:
        return list(dict.fromkeys(float(window) for window in requested_windows))

    if signal_metadata is None:
        return list(DEFAULT_WINDOWS)

    if "windows" in signal_metadata:
        return [float(window) for window in signal_metadata["windows"]]

    return list(DEFAULT_WINDOWS)


def load_split_merge_network(network_path: str | Path = NETWORK_PATH) -> Any:
    """
    Load the split-merge temporal network.
    """

    return load_pickle(Path(network_path))


def prepare_full_window_scan(
    net: Any,
    windows: Sequence[float],
) -> dict[float, WindowLimitPlan]:
    """
    Enumerate every valid window start for each requested window length.
    """

    window_plans: dict[float, WindowLimitPlan] = {}

    for window in tuple(float(window) for window in windows):
        k_samples = np.flatnonzero(net.times < net.times[-1] - window).astype(int)
        t_samples = np.asarray(net.times[k_samples], dtype=float)
        window_plans[window] = WindowLimitPlan(
            window=window,
            k_samples=k_samples,
            t_samples=t_samples,
        )

    return window_plans


def compute_component_log_sum(
    net: Any,
    start_time: float,
    window: float,
) -> float:
    """
    Compute sum_C (|C| / N) log |C| on the aggregated window graph.
    """

    adjacency = net.compute_static_adjacency_matrix(
        start_time=float(start_time),
        end_time=float(start_time) + float(window),
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
    net: Any,
    plan: WindowLimitPlan,
) -> dict[str, Any]:
    """
    Compute the connected-component limit statistic for all starts of one window.
    """

    values = np.empty(len(plan.t_samples), dtype=float)

    for idx, start_time in enumerate(plan.t_samples):
        values[idx] = compute_component_log_sum(
            net=net,
            start_time=float(start_time),
            window=float(plan.window),
        )

        if (idx + 1) % 250 == 0 or idx + 1 == len(plan.t_samples):
            print(
                f"window={float(plan.window):g}: "
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
        "k_samples": np.asarray(plan.k_samples, dtype=int),
        "t_samples": np.asarray(plan.t_samples, dtype=float),
        "component_log_sums": values,
        # Compatibility with plotting helpers that expect a generic 1D signal.
        "signal": values,
        "signal_array": values,
        # Shape (n_selected_times, 2) with columns [t_sample, component_log_sum].
        "time_component_log_sums": time_limit_array,
        "statistic": "sum((size / N) * log(size)) over connected components",
    }


def save_window_result(result: dict[str, Any], output_base: Path) -> None:
    """
    Save one window payload.
    """

    outdir = output_base / "window_limit_selected" / window_key(result["window_seconds"])
    outdir.mkdir(parents=True, exist_ok=True)

    with (outdir / "window_limit.pkl").open("wb") as handle:
        pickle.dump(result, handle)


def save_metadata(
    output_base: Path,
    network_path: str | Path,
    net: Any,
    window_plans: dict[float, WindowLimitPlan],
    signal_results_dir: Path | None,
    signal_metadata: dict | None,
    elapsed_seconds: float | None = None,
) -> None:
    """
    Save run metadata alongside the computed limit curves.
    """

    windows = np.asarray(sorted(window_plans), dtype=float)
    metadata = {
        "network_path": str(network_path),
        "source_signal_results_dir": (
            str(signal_results_dir) if signal_results_dir is not None else None
        ),
        "source_signal_metadata_path": (
            str(signal_results_dir / "metadata.pkl")
            if signal_results_dir is not None
            else None
        ),
        "num_nodes": int(net.num_nodes),
        "num_events": int(net.num_events),
        "num_times": int(len(net.times)),
        "windows": windows,
        "window_sample_counts": {
            float(window): int(len(plan.t_samples))
            for window, plan in window_plans.items()
        },
        "signal_lambdas": (
            np.asarray(signal_metadata["lambdas"], dtype=float)
            if signal_metadata is not None and "lambdas" in signal_metadata
            else None
        ),
        "statistic": "sum((size / N) * log(size)) over connected components",
        "time_component_log_sums_columns": ["t_sample", "component_log_sum"],
        "output_layout": "window_limit_selected/<window>/window_limit.pkl",
        "elapsed_seconds": elapsed_seconds,
    }

    output_base.mkdir(parents=True, exist_ok=True)
    with (output_base / "metadata.pkl").open("wb") as handle:
        pickle.dump(metadata, handle)


def main() -> None:
    args = parse_args()
    signal_results_dir = resolve_signal_results_dir(
        requested_dir=args.signal_results_dir,
        allow_missing=True,
    )
    signal_metadata = load_signal_metadata(signal_results_dir)
    windows = get_windows(
        signal_metadata,
        requested_windows=args.windows,
    )

    net = load_split_merge_network(args.network_path)
    window_plans = prepare_full_window_scan(net, windows=windows)

    save_metadata(
        output_base=args.output_base,
        network_path=args.network_path,
        net=net,
        window_plans=window_plans,
        signal_results_dir=signal_results_dir,
        signal_metadata=signal_metadata,
    )

    t_start = time.time()

    for window in windows:
        plan = window_plans[float(window)]
        print(f"starting window={float(window):g} with {len(plan.t_samples)} samples")
        result = compute_window_limit_curve(net, plan)
        save_window_result(result=result, output_base=args.output_base)

    elapsed = time.time() - t_start
    save_metadata(
        output_base=args.output_base,
        network_path=args.network_path,
        net=net,
        window_plans=window_plans,
        signal_results_dir=signal_results_dir,
        signal_metadata=signal_metadata,
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
