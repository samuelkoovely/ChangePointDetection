from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from joblib import Parallel, delayed

from signal_generation import (
    PreparedSignalSample,
    WindowSamplingPlan,
    build_window_sampling_plan,
    compute_signals_for_lambdas_prepared,
    prepare_signal_sample,
    save_signal_result,
)
from ct_examples_common import (
    DEFAULT_DATA_DIR,
    DEFAULT_ENTROPY_WINDOWS,
    SPECS,
    get_specs,
    load_pickle,
    sample_path,
)


DEFAULT_LAMBDAS = np.logspace(-5, 0, 10)
DEFAULT_OUTPUT_BASE = Path("gridsearch_results/ct_examples")
WINDOW_BACKEND = "segment_tree"
BACKEND = "loky"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute sparse block-activity entropy signals for many "
            "(lambda, window) pairs using lambda-block parallelism."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[spec.key for spec in SPECS],
        choices=[spec.key for spec in SPECS],
        help="Subset of sparse examples to process.",
    )
    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=DEFAULT_LAMBDAS.tolist(),
        help="Lambda values to evaluate.",
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        type=float,
        default=DEFAULT_ENTROPY_WINDOWS,
        help="Window lengths to evaluate.",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=1.0,
        help=(
            "Fraction of valid window centers to sample. Use 1.0 to compute "
            "the full entropy curves."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the generated sparse sample pickles.",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=DEFAULT_OUTPUT_BASE,
        help="Base directory where per-dataset signal grids will be written.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=min(4, len(DEFAULT_LAMBDAS), os.cpu_count() or 1),
        help="Number of parallel workers across lambda blocks.",
    )
    parser.add_argument(
        "--reverse-time",
        action="store_true",
        help="Compute backward entropy curves instead of forward ones.",
    )
    return parser.parse_args()


def normalize_windows(windows: Sequence[float]) -> tuple[float, ...]:
    return tuple(dict.fromkeys(float(window) for window in windows))


def prepare_full_window_sample(
    net: Any,
    windows: Sequence[float],
    reverse_time: bool = False,
) -> PreparedSignalSample:
    windows_tuple = normalize_windows(windows)
    p0 = np.ones(net.num_nodes, dtype=float) / net.num_nodes

    net.compute_laplacian_matrices(
        t_start=net.times[0],
        t_stop=net.times[-1],
        random_walk=False,
    )

    window_plans: dict[float, WindowSamplingPlan] = {}
    for window in windows_tuple:
        window_plans[float(window)] = build_window_sampling_plan(
            net=net,
            window=float(window),
            reverse_time=reverse_time,
            full_scan=True,
        )

    return PreparedSignalSample(
        net=net,
        windows=windows_tuple,
        sample_fraction=1.0,
        reverse_time=bool(reverse_time),
        p0=p0,
        window_plans=window_plans,
    )


def prepare_sparse_sample(
    net: Any,
    windows: Sequence[float],
    sample_fraction: float,
    reverse_time: bool = False,
) -> PreparedSignalSample:
    if float(sample_fraction) >= 1.0:
        return prepare_full_window_sample(
            net=net,
            windows=windows,
            reverse_time=reverse_time,
        )

    return prepare_signal_sample(
        net=net,
        windows=windows,
        sample_fraction=float(sample_fraction),
        p0=np.ones(net.num_nodes, dtype=float) / net.num_nodes,
        reverse_time=reverse_time,
    )


def lambda_blocks(lambdas: Sequence[float], n_jobs: int) -> list[list[float]]:
    effective_jobs = max(1, int(n_jobs))
    blocks = np.array_split(np.asarray(lambdas, dtype=float), effective_jobs)
    return [block.tolist() for block in blocks if len(block) > 0]


def save_lambda_block_results(
    signals_by_lambda: dict[float, dict[float, dict[str, Any]]],
    dataset_output_dir: Path,
) -> None:
    signal_dir = dataset_output_dir / "signals"
    for window_results in signals_by_lambda.values():
        for signal_result in window_results.values():
            save_signal_result(signal_result, signal_dir)


def save_metadata(
    spec_key: str,
    sample_input_path: Path,
    sample: dict[str, Any],
    prepared: PreparedSignalSample,
    lambdas: Sequence[float],
    output_dir: Path,
    sample_fraction: float,
    n_jobs: int,
    reverse_time: bool,
    elapsed_seconds: float | None = None,
) -> None:
    metadata = {
        "dataset": spec_key,
        "sample_path": str(sample_input_path),
        "num_nodes": int(prepared.net.num_nodes),
        "num_events": int(prepared.net.num_events),
        "num_times": int(len(prepared.net.times)),
        "lambdas": np.asarray(lambdas, dtype=float),
        "windows": np.asarray(prepared.windows, dtype=float),
        "sample_fraction": float(sample_fraction),
        "n_jobs": int(n_jobs),
        "reverse_time": bool(reverse_time),
        "direction": "backward" if reverse_time else "forward",
        "window_backend": WINDOW_BACKEND,
        "breakpoints": np.asarray(sample.get("bkps", []), dtype=float),
        "sample_metadata": sample.get("metadata"),
        "elapsed_seconds": elapsed_seconds,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metadata.pkl", "wb") as handle:
        pickle.dump(metadata, handle)


def worker(
    prepared: PreparedSignalSample,
    lamda_block: Sequence[float],
    dataset_output_dir: Path,
) -> list[float]:
    signals_by_lambda = compute_signals_for_lambdas_prepared(
        prepared=prepared,
        lambdas=lamda_block,
        use_linear_approx=False,
        lin_t_s=10,
        window_backend=WINDOW_BACKEND,
    )
    save_lambda_block_results(
        signals_by_lambda=signals_by_lambda,
        dataset_output_dir=dataset_output_dir,
    )
    finished = [float(lamda) for lamda in lamda_block]
    print("finished lambda block:", [f"{lamda:.11f}" for lamda in finished])
    return finished


def run_lambda_blocks(
    prepared: PreparedSignalSample,
    blocks: Sequence[Sequence[float]],
    dataset_output_dir: Path,
) -> None:
    if len(blocks) == 1:
        worker(
            prepared=prepared,
            lamda_block=blocks[0],
            dataset_output_dir=dataset_output_dir,
        )
        return

    try:
        Parallel(n_jobs=len(blocks), backend=BACKEND)(
            delayed(worker)(
                prepared=prepared,
                lamda_block=block,
                dataset_output_dir=dataset_output_dir,
            )
            for block in blocks
        )
    except PermissionError as exc:
        print(
            f"Parallel backend {BACKEND!r} unavailable ({exc}). "
            "Falling back to sequential lambda blocks."
        )
        for block in blocks:
            worker(
                prepared=prepared,
                lamda_block=block,
                dataset_output_dir=dataset_output_dir,
            )


def compute_signal_grid_for_dataset(
    spec_key: str,
    data_dir: Path,
    lambdas: Sequence[float],
    windows: Sequence[float],
    sample_fraction: float,
    output_base: Path,
    n_jobs: int,
    reverse_time: bool = False,
) -> None:
    spec = get_specs([spec_key])[0]
    sample_input_path = sample_path(spec, data_dir)
    if not sample_input_path.exists():
        raise FileNotFoundError(
            f"Missing sparse sample {sample_input_path}. "
            "Run generate_ct_examples.py first."
        )

    sample = load_pickle(sample_input_path)
    prepared = prepare_sparse_sample(
        net=sample["tnet"],
        windows=windows,
        sample_fraction=sample_fraction,
        reverse_time=reverse_time,
    )
    dataset_output_dir = output_base / spec.key

    save_metadata(
        spec_key=spec.key,
        sample_input_path=sample_input_path,
        sample=sample,
        prepared=prepared,
        lambdas=lambdas,
        output_dir=dataset_output_dir,
        sample_fraction=sample_fraction,
        n_jobs=n_jobs,
        reverse_time=reverse_time,
    )

    blocks = lambda_blocks(lambdas, n_jobs=n_jobs)
    t_start = time.time()
    run_lambda_blocks(
        prepared=prepared,
        blocks=blocks,
        dataset_output_dir=dataset_output_dir,
    )

    elapsed = time.time() - t_start
    save_metadata(
        spec_key=spec.key,
        sample_input_path=sample_input_path,
        sample=sample,
        prepared=prepared,
        lambdas=lambdas,
        output_dir=dataset_output_dir,
        sample_fraction=sample_fraction,
        n_jobs=n_jobs,
        reverse_time=reverse_time,
        elapsed_seconds=elapsed,
    )

    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(
        f"{spec.key}: {elapsed:.2f} s "
        f"({int(hours):02d}:{int(minutes):02d}:{seconds:05.2f})"
    )


def main() -> None:
    args = parse_args()
    lambdas = [float(lamda) for lamda in args.lambdas]
    windows = [float(window) for window in args.windows]

    for spec in get_specs(args.datasets):
        compute_signal_grid_for_dataset(
            spec_key=spec.key,
            data_dir=args.data_dir,
            lambdas=lambdas,
            windows=windows,
            sample_fraction=float(args.sample_fraction),
            output_base=args.output_base,
            n_jobs=int(args.n_jobs),
            reverse_time=bool(args.reverse_time),
        )


if __name__ == "__main__":
    main()
