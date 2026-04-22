from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path
from typing import Any, Sequence

from joblib import Parallel, delayed

from gridsearch_score import (
    CPSample,
    compute_and_store_signals_for_sample,
    evaluate_precomputed_lambda_signals,
    extract_true_change_points,
    reorganize_sample_signal_results_by_lambda,
)


DEFAULT_RESULTS_FILENAME = "test_set_results.pkl"
DEFAULT_TEST_SIGNALS_DIRNAME = "test_signals"
DEFAULT_N_JOBS = 1
DEFAULT_BACKEND = "loky"
DEFAULT_VERBOSE = 0


def _load_pickle(path: str | Path) -> Any:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _write_pickle(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(obj, handle)


def _normalize_point(point: float | int) -> float | int:
    value = float(point)
    if value.is_integer():
        return int(value)
    return value


def _normalize_points(points: Sequence[float | int]) -> list[float | int]:
    return [_normalize_point(point) for point in points]


def load_ct_samples(dataset_path: str | Path) -> list[CPSample]:
    dataset_path = Path(dataset_path)
    dataset = _load_pickle(dataset_path)
    if not isinstance(dataset, Sequence) or len(dataset) == 0:
        raise ValueError(f"Dataset at {dataset_path} is empty or not sequence-like.")

    samples: list[CPSample] = []
    for sample_index, entry in enumerate(dataset):
        if "tnet" not in entry:
            raise KeyError(
                f"Dataset entry {sample_index} in {dataset_path} is missing the 'tnet' key."
            )
        true_change_points, n_bkps = extract_true_change_points(entry)
        samples.append(
            CPSample(
                data=entry["tnet"],
                true_change_points=true_change_points,
                n_bkps=n_bkps,
                name=f"sample_{sample_index}",
            )
        )

    return samples


def _require_value(summary: dict[str, Any], key: str) -> Any:
    value = summary.get(key)
    if value is None:
        raise ValueError(
            f"Training summary at runtime is missing a usable value for '{key}'."
        )
    return value


def _build_per_sample_results(
    samples: Sequence[CPSample],
    predicted_change_points: Sequence[Sequence[float | int]],
    f1_scores: Sequence[float],
    hausdorff_scores: Sequence[float],
) -> list[dict[str, Any]]:
    per_sample_results = []
    for sample, pred_cps, f1_value, hausdorff_value in zip(
        samples,
        predicted_change_points,
        f1_scores,
        hausdorff_scores,
    ):
        per_sample_results.append(
            {
                "sample_name": sample.name,
                "n_bkps": int(sample.n_bkps),
                "true_change_points": _normalize_points(sample.true_change_points),
                "predicted_change_points": _normalize_points(pred_cps),
                "f1": float(f1_value),
                "hausdorff": float(hausdorff_value),
            }
        )
    return per_sample_results


def build_ct_test_evaluation_parser(
    *,
    default_training_results_path: str | Path,
    default_test_dataset_path: str | Path,
    default_outdir: str | Path,
    description: str,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--training-results-path",
        type=Path,
        default=Path(default_training_results_path),
        help="Path to the training grid-search summary pickle.",
    )
    parser.add_argument(
        "--test-dataset-path",
        type=Path,
        default=Path(default_test_dataset_path),
        help="Path to the test dataset pickle.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(default_outdir),
        help="Directory where evaluation outputs will be written.",
    )
    parser.set_defaults(save_signals=True)
    parser.add_argument(
        "--no-save-signals",
        dest="save_signals",
        action="store_false",
        help="Skip writing per-sample test-set signal pickles.",
    )
    parser.add_argument(
        "--results-filename",
        type=str,
        default=DEFAULT_RESULTS_FILENAME,
        help="Filename used for the saved evaluation summary pickle.",
    )
    parser.add_argument(
        "--test-signals-dirname",
        type=str,
        default=DEFAULT_TEST_SIGNALS_DIRNAME,
        help="Subdirectory name used for saved test-set signals.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=DEFAULT_N_JOBS,
        help=(
            "Number of parallel workers across test samples during signal generation. "
            "Use 1 to disable sample-level parallelism."
        ),
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=DEFAULT_BACKEND,
        help="Joblib backend used for sample-parallel signal generation.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=DEFAULT_VERBOSE,
        help="Joblib verbosity used during sample-parallel signal generation.",
    )
    parser.add_argument(
        "--recommend-max-jobs",
        action="store_true",
        help=(
            "Print a conservative default worker suggestion based on the current machine "
            "and exit."
        ),
    )
    return parser


def recommended_parallel_jobs() -> int:
    return min(5, os.cpu_count() or 1)


def run_ct_entropy_test_evaluation(
    *,
    training_results_path: str | Path,
    test_dataset_path: str | Path,
    outdir: str | Path,
    save_signals: bool = True,
    results_filename: str = DEFAULT_RESULTS_FILENAME,
    test_signals_dirname: str = DEFAULT_TEST_SIGNALS_DIRNAME,
    n_jobs: int = DEFAULT_N_JOBS,
    backend: str = DEFAULT_BACKEND,
    verbose: int = DEFAULT_VERBOSE,
) -> dict[str, Any]:
    training_results_path = Path(training_results_path)
    test_dataset_path = Path(test_dataset_path)
    outdir = Path(outdir)
    results_path = outdir / results_filename
    signals_outdir = outdir / test_signals_dirname if save_signals else None

    training_summary = _load_pickle(training_results_path)
    samples = load_ct_samples(test_dataset_path)

    best_lamda = float(_require_value(training_summary, "best_lamda"))
    best_window = float(_require_value(training_summary, "best_window"))
    windows = [best_window]
    lambdas = [best_lamda]
    n_jobs = int(n_jobs)
    if n_jobs <= 0:
        raise ValueError("n_jobs must be a positive integer.")

    t0 = time.time()
    t_signal_phase_start = time.time()
    sample_signal_results = Parallel(
        n_jobs=n_jobs,
        backend=str(backend),
        verbose=int(verbose),
    )(
        delayed(compute_and_store_signals_for_sample)(
            sample=sample,
            sample_idx=sample_idx,
            lambdas=lambdas,
            windows=windows,
            sample_fraction=float(training_summary.get("sample_fraction", 1.0)),
            p0=None,
            use_linear_approx=bool(training_summary.get("use_linear_approx", False)),
            lin_t_s=int(training_summary.get("lin_t_s", 10)),
            save_signals=save_signals,
            signals_outdir=signals_outdir,
            window_backend=training_summary.get("window_backend", "auto"),
            reverse_time=False,
        )
        for sample_idx, sample in enumerate(samples)
    )
    lambda_signal_results = reorganize_sample_signal_results_by_lambda(
        sample_signal_results=sample_signal_results,
        samples=samples,
        lambdas=lambdas,
        windows=windows,
    )
    signal_generation_phase_seconds = time.time() - t_signal_phase_start

    t_detection_phase_start = time.time()
    evaluation = evaluate_precomputed_lambda_signals(
        samples=samples,
        lamda=best_lamda,
        windows=windows,
        margin=float(training_summary["margin"]),
        signals_by_window=lambda_signal_results[0]["signals_by_window"],
        kernel=training_summary["kernel"],
    )
    detection_metrics_phase_seconds = time.time() - t_detection_phase_start
    elapsed_seconds = time.time() - t0

    window_key = float(best_window)
    per_sample_results = _build_per_sample_results(
        samples=samples,
        predicted_change_points=evaluation["predicted_change_points"][window_key],
        f1_scores=evaluation["per_window_f1_scores"][window_key],
        hausdorff_scores=evaluation["per_window_hausdorff"][window_key],
    )

    summary = {
        "method": "entropy",
        "training_results_path": str(training_results_path),
        "test_dataset_path": str(test_dataset_path),
        "results_path": str(results_path),
        "signals_outdir": None if signals_outdir is None else str(signals_outdir),
        "num_test_samples": len(samples),
        "sample_names": [sample.name for sample in samples],
        "selection_metric": training_summary.get("selection_metric"),
        "margin": float(training_summary["margin"]),
        "kernel": training_summary["kernel"],
        "train_best_params": {
            "lamda": float(best_lamda),
            "window": float(best_window),
            "sample_fraction": float(training_summary.get("sample_fraction", 1.0)),
            "use_linear_approx": bool(training_summary.get("use_linear_approx", False)),
            "lin_t_s": int(training_summary.get("lin_t_s", 10)),
            "window_backend": training_summary.get("window_backend", "auto"),
        },
        "signal_generation_parallel_axis": "sample",
        "signal_generation_n_jobs": int(n_jobs),
        "signal_generation_backend": str(backend),
        "train_best_score": float(training_summary["best_score"]),
        "train_best_f1": float(training_summary["best_f1"]),
        "train_best_hausdorff": float(training_summary["best_hausdorff"]),
        "test_mean_f1": float(evaluation["mean_f1_per_window"][window_key]),
        "test_mean_hausdorff": float(evaluation["mean_hausdorff_per_window"][window_key]),
        "signal_generation_phase_seconds": float(signal_generation_phase_seconds),
        "detection_metrics_phase_seconds": float(detection_metrics_phase_seconds),
        "elapsed_seconds": float(elapsed_seconds),
        "per_sample_results": per_sample_results,
        "evaluation": evaluation,
    }
    _write_pickle(summary, results_path)
    return summary


def print_test_summary(summary: dict[str, Any]) -> None:
    print("Method:", summary["method"])
    print("Training results:", summary["training_results_path"])
    print("Test dataset:", summary["test_dataset_path"])
    print("Saved test results:", summary["results_path"])
    print("Saved test signals:", summary["signals_outdir"])
    print(
        "Signal generation parallelism:",
        {
            "axis": summary.get("signal_generation_parallel_axis"),
            "n_jobs": summary.get("signal_generation_n_jobs"),
            "backend": summary.get("signal_generation_backend"),
        },
    )
    print("Selection metric on training set:", summary["selection_metric"])
    print("Selected training parameters:", summary["train_best_params"])
    print("Number of test samples:", summary["num_test_samples"])
    print("Test mean F1:", summary["test_mean_f1"])
    print("Test mean Hausdorff:", summary["test_mean_hausdorff"])
    print("Signal generation phase runtime:", summary["signal_generation_phase_seconds"])
    print(
        "Detection and metrics phase runtime:",
        summary["detection_metrics_phase_seconds"],
    )
    print("Total runtime:", summary["elapsed_seconds"])
