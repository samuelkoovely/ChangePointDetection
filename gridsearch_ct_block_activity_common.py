from __future__ import annotations

import argparse
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from gridsearch_score import CPSample, extract_true_change_points, grid_search
from signal_generation import SUPPORTED_WINDOW_BACKENDS, get_sampled_window_indices_and_times
from sparse_block_activity_common import DEFAULT_ENTROPY_WINDOWS, load_pickle, normalize_windows


DEFAULT_LAMBDAS = np.logspace(-5, 0, 10)
DEFAULT_MARGIN = 5.0
DEFAULT_SELECTION_METRIC = "hausdorff"
DEFAULT_KERNEL = "linear"
DEFAULT_WINDOW_BACKEND = "segment_tree"
DEFAULT_VERBOSE = 10


@dataclass(frozen=True)
class CTGridSearchSpec:
    dataset_key: str
    dataset_path: Path
    output_dir: Path
    sample_fraction: float


def parse_args(spec: CTGridSearchSpec) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"Grid-search lambda and window for the {spec.dataset_key} "
            "continuous-time sparse block-activity dataset."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=spec.dataset_path,
        help="Input dataset pickle.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=spec.output_dir,
        help=(
            "Directory where gridsearch_results.pkl and, by default, the "
            "per-sample entropy signals will be written."
        ),
    )
    parser.add_argument(
        "--signals-dir",
        type=Path,
        default=None,
        help=(
            "Optional override for the per-sample signal output directory. "
            "Defaults to OUTPUT_DIR/signals."
        ),
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
        "--margin",
        type=float,
        default=DEFAULT_MARGIN,
        help="Tolerance margin used by the evaluation metrics.",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=spec.sample_fraction,
        help=(
            "Fraction of valid window centers to sample when generating "
            "entropy signals."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=min(5, os.cpu_count() or 1),
        help="Number of parallel workers across samples.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="loky",
        help="Joblib backend used for sample-parallel grid search.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=DEFAULT_VERBOSE,
        help="Joblib verbosity passed to grid_search.",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default=DEFAULT_KERNEL,
        help="Kernel passed to ruptures.KernelCPD.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["f1", "hausdorff"],
        default=DEFAULT_SELECTION_METRIC,
        help="Criterion used to select the best (lambda, window) pair.",
    )
    parser.add_argument(
        "--window-backend",
        choices=sorted(SUPPORTED_WINDOW_BACKENDS),
        default=DEFAULT_WINDOW_BACKEND,
        help="Backend used for local-entropy window products.",
    )
    parser.add_argument(
        "--use-linear-approx",
        action="store_true",
        help="Use the linear approximation for transition matrices.",
    )
    parser.add_argument(
        "--lin-t-s",
        type=int,
        default=10,
        help="Linear-approximation time step used when enabled.",
    )
    parser.set_defaults(save_signals=True)
    parser.add_argument(
        "--no-save-signals",
        dest="save_signals",
        action="store_false",
        help="Skip writing per-sample entropy signal pickles.",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> list[dict[str, Any]]:
    dataset = load_pickle(path)
    if not isinstance(dataset, list):
        raise TypeError(
            f"Expected {path} to contain a list of dataset entries, got {type(dataset)!r}."
        )
    return dataset


def build_training_samples(dataset: Sequence[dict[str, Any]]) -> list[CPSample]:
    training_samples: list[CPSample] = []
    for sample_index, entry in enumerate(dataset):
        true_change_points, n_bkps = extract_true_change_points(entry)
        training_samples.append(
            CPSample(
                data=entry["tnet"],
                true_change_points=true_change_points,
                n_bkps=n_bkps,
                name=f"sample_{sample_index}",
            )
        )
    return training_samples


def warn_if_sampling_too_sparse(
    dataset: Sequence[dict[str, Any]],
    *,
    windows: Sequence[float],
    sample_fraction: float,
) -> None:
    max_n_bkps = max(int(entry.get("n_bkps", 1)) for entry in dataset)
    required_points = max_n_bkps + 1
    warnings: list[str] = []

    for window in normalize_windows(windows):
        sampled_counts = [
            len(
                get_sampled_window_indices_and_times(
                    entry["tnet"],
                    window=float(window),
                    sample_fraction=float(sample_fraction),
                )[0]
            )
            for entry in dataset
        ]
        min_sampled_count = min(sampled_counts)
        if min_sampled_count > max_n_bkps:
            continue

        full_scan_counts = [
            len(
                get_sampled_window_indices_and_times(
                    entry["tnet"],
                    window=float(window),
                    sample_fraction=1.0,
                )[0]
            )
            for entry in dataset
        ]
        min_full_scan_count = min(full_scan_counts)
        if min_full_scan_count <= max_n_bkps:
            warnings.append(
                f"window={float(window):g}: even a full scan only yields "
                f"{min_full_scan_count} valid centers."
            )
            continue

        suggested_fraction = required_points / float(min_full_scan_count)
        warnings.append(
            f"window={float(window):g}: sample_fraction={float(sample_fraction):g} "
            f"yields as few as {min_sampled_count} sampled centers; "
            f"increase to at least {suggested_fraction:.3g} to make "
            f"{max_n_bkps} breakpoint(s) feasible."
        )

    if warnings:
        print("Warning: sparse sampling may be too aggressive for change-point detection.")
        for warning in warnings:
            print("  -", warning)


def augment_and_save_summary(
    summary: dict[str, Any],
    *,
    spec: CTGridSearchSpec,
    args: argparse.Namespace,
    dataset: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    enriched_summary = dict(summary)
    enriched_summary["dataset"] = spec.dataset_key
    enriched_summary["data_path"] = str(args.data_path)
    enriched_summary["output_dir"] = str(args.output_dir)
    enriched_summary["signals_dir"] = (
        str(args.signals_dir) if args.signals_dir is not None else None
    )
    enriched_summary["num_dataset_entries"] = len(dataset)
    enriched_summary["dataset_breakpoints"] = [
        [float(change_point) for change_point in entry.get("bkps", [])]
        for entry in dataset
    ]
    enriched_summary["cli_args"] = {
        "lambdas": np.asarray(args.lambdas, dtype=float),
        "windows": np.asarray(normalize_windows(args.windows), dtype=float),
        "margin": float(args.margin),
        "sample_fraction": float(args.sample_fraction),
        "n_jobs": int(args.n_jobs),
        "backend": str(args.backend),
        "verbose": int(args.verbose),
        "kernel": str(args.kernel),
        "selection_metric": str(args.selection_metric),
        "window_backend": str(args.window_backend),
        "use_linear_approx": bool(args.use_linear_approx),
        "lin_t_s": int(args.lin_t_s),
        "save_signals": bool(args.save_signals),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "gridsearch_results.pkl", "wb") as handle:
        pickle.dump(enriched_summary, handle)
    return enriched_summary


def print_summary(summary: dict[str, Any], *, output_dir: Path, signals_dir: Path | None) -> None:
    print("Score array shape:", summary["score_array"].shape)
    print("Selection metric:", summary["selection_metric"])
    print("Best lamda:", summary["best_lamda"])
    print("Best window:", summary["best_window"])
    print("Best selected score:", summary["best_score"])
    print("Best mean F1:", summary["best_f1"])
    print("Best mean Hausdorff:", summary["best_hausdorff"])
    print("Total runtime:", summary["elapsed_seconds"])
    print("Signal generation phase runtime:", summary["signal_generation_phase_seconds"])
    print("Detection and metrics phase runtime:", summary["detection_metrics_phase_seconds"])
    print("Signals saved:", summary["save_signals"])
    print("Signals output directory:", str(signals_dir) if signals_dir is not None else None)
    print("Summary output directory:", output_dir)


def main(spec: CTGridSearchSpec) -> None:
    args = parse_args(spec)
    dataset = load_dataset(args.data_path)
    training_samples = build_training_samples(dataset)
    warn_if_sampling_too_sparse(
        dataset,
        windows=args.windows,
        sample_fraction=float(args.sample_fraction),
    )
    signals_dir = None
    if args.save_signals:
        signals_dir = args.signals_dir or (args.output_dir / "signals")

    summary = grid_search(
        samples=training_samples,
        lambdas=np.asarray(args.lambdas, dtype=float),
        windows=normalize_windows(args.windows),
        margin=float(args.margin),
        n_jobs=int(args.n_jobs),
        backend=str(args.backend),
        verbose=int(args.verbose),
        outdir=args.output_dir,
        sample_fraction=float(args.sample_fraction),
        kernel=str(args.kernel),
        use_linear_approx=bool(args.use_linear_approx),
        lin_t_s=int(args.lin_t_s),
        save_signals=bool(args.save_signals),
        signals_outdir=signals_dir,
        selection_metric=str(args.selection_metric),
        window_backend=str(args.window_backend),
    )
    enriched_summary = augment_and_save_summary(
        summary,
        spec=spec,
        args=args,
        dataset=dataset,
    )

    print("Number of samples:", len(training_samples))
    print("Dataset path:", args.data_path)
    print_summary(
        enriched_summary,
        output_dir=args.output_dir,
        signals_dir=signals_dir,
    )
