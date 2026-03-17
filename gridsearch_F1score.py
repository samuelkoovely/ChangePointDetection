"""
Parallel grid-search over (lambda, window) using the F1-score for change-point detection.

This version uses the reusable signal-generation helpers from `signal_generation.py`.
The expensive lambda-dependent preprocessing is performed once per lambda worker,
then reused across all candidate window lengths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import math
import pickle
import time

import numpy as np
import ruptures as rpt
from joblib import Parallel, delayed

from evaluation_metrics import f1_score, hausdorff_distance
from signal_generation import compute_signals_for_lambda_timed, save_signal_result


# -----------------------------------------------------------------------------
# Data structure
# -----------------------------------------------------------------------------

@dataclass
class CPSample:
    """
    One training example for the grid-search.

    Attributes
    ----------
    data:
        Input object used to generate the signal. In the current intended use,
        this is typically a temporal network object.
    true_change_points:
        Ground-truth change-point times for this sample.
    n_bkps:
        Number of breakpoints to predict with ruptures.
    name:
        Optional identifier, useful for debugging / storing outputs.
    """
    data: Any
    true_change_points: Sequence[float]
    n_bkps: int = 1
    name: str | None = None


# -----------------------------------------------------------------------------
# Detection helper
# -----------------------------------------------------------------------------

def detect_change_points_from_signal(
    signal: np.ndarray,
    selected_times: Sequence[float],
    n_bkps: int,
    kernel: str = "linear",
) -> list[float]:
    """
    Run ruptures on a precomputed signal and convert breakpoint indices to times.
    """
    signal = np.asarray(signal)
    selected_times = np.asarray(selected_times, dtype=float)

    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    if signal.ndim != 2:
        raise ValueError(
            f"signal must be a 1D or 2D array-like object, got shape {signal.shape}."
        )

    if signal.shape[0] != len(selected_times):
        raise ValueError(
            "signal and selected_times must have the same number of time points. "
            f"Got {signal.shape[0]} and {len(selected_times)}."
        )

    if signal.shape[0] == 0:
        return []

    algo = rpt.KernelCPD(kernel=kernel).fit(signal)
    breakpoint_indices = np.asarray(algo.predict(n_bkps=n_bkps), dtype=int)

    if breakpoint_indices.size == 0:
        return []

    # ruptures typically includes the terminal endpoint n_samples, which is not
    # a change point. Remove it before converting indices to times.
    breakpoint_indices = breakpoint_indices[breakpoint_indices < len(selected_times)]

    if breakpoint_indices.size == 0:
        return []

    return selected_times[breakpoint_indices].tolist()


def merge_wall_clock_intervals(intervals: Sequence[tuple[float, float]]) -> float:
    """
    Merge wall-clock intervals and return the covered duration in seconds.

    This is used to build summary timings that remain comparable to the overall
    elapsed runtime when lambda workers execute in parallel.
    """
    normalized = sorted(
        (float(start), float(stop))
        for start, stop in intervals
        if float(stop) > float(start)
    )

    if not normalized:
        return 0.0

    total = 0.0
    current_start, current_stop = normalized[0]
    for start, stop in normalized[1:]:
        if start <= current_stop:
            current_stop = max(current_stop, stop)
            continue

        total += current_stop - current_start
        current_start, current_stop = start, stop

    total += current_stop - current_start
    return total


# -----------------------------------------------------------------------------
# Lambda worker
# -----------------------------------------------------------------------------

def evaluate_lambda(
    samples: Sequence[CPSample],
    lamda: float,
    windows: Sequence[float],
    margin: float,
    sample_fraction: float = 0.1,
    kernel: str = "linear",
    p0: np.ndarray | None = None,
    save_signals: bool = False,
    signals_outdir: str | Path | None = None,
    signal_dir_order: str = "lambda_window",
) -> dict:
    """
    Evaluate all candidate windows for one lambda value.

    The lambda-dependent preprocessing is performed once per sample and then
    reused across all window values.

    If requested, the generated entropy signals are also saved to disk in an
    ordered folder layout, either `outdir / lambda / window` or
    `outdir / window / lambda`.
    """
    windows = [float(w) for w in windows]

    score_array = np.empty(len(windows), dtype=float)
    score_array[:] = np.nan

    signal_generation_time_total = 0.0
    change_point_detection_time_total = 0.0
    metrics_time_total = 0.0
    signal_generation_preprocessing_time_total = 0.0

    signal_generation_time_per_window = {float(window): 0.0 for window in windows}
    signal_generation_window_only_time_per_window = {float(window): 0.0 for window in windows}
    change_point_detection_time_per_window = {float(window): 0.0 for window in windows}
    metrics_time_per_window = {float(window): 0.0 for window in windows}

    signal_generation_intervals: list[tuple[float, float]] = []
    change_point_detection_intervals: list[tuple[float, float]] = []
    metrics_intervals: list[tuple[float, float]] = []

    per_window_f1_scores: dict[float, np.ndarray] = {}
    per_window_hausdorff: dict[float, np.ndarray] = {}
    predicted_change_points: dict[float, list[list[float]]] = {}
    sample_names = [sample.name for sample in samples]

    f1_accumulator = {float(window): [] for window in windows}
    hausdorff_accumulator = {float(window): [] for window in windows}
    preds_accumulator = {float(window): [] for window in windows}

    if signal_dir_order not in {"lambda_window", "window_lambda"}:
        raise ValueError(
            "signal_dir_order must be either 'lambda_window' or 'window_lambda'."
        )

    for sample_idx, sample in enumerate(samples):
        signal_wall_start = time.time()
        signal_bundle = compute_signals_for_lambda_timed(
            net=sample.data,
            lamda=lamda,
            windows=windows,
            sample_fraction=sample_fraction,
            p0=p0,
        )
        signal_wall_stop = time.time()
        if signal_wall_stop > signal_wall_start:
            signal_generation_intervals.append((signal_wall_start, signal_wall_stop))

        signals_by_window = signal_bundle["signals_by_window"]
        signal_timing = signal_bundle["timing"]
        signal_generation_time_total += float(signal_timing["total_seconds"])
        signal_generation_preprocessing_time_total += float(signal_timing["preprocessing_seconds"])
        for window in windows:
            window_key = float(window)
            signal_generation_time_per_window[window_key] += float(
                signal_timing["attributed_per_window_seconds"][window_key]
            )
            signal_generation_window_only_time_per_window[window_key] += float(
                signal_timing["window_signal_seconds"][window_key]
            )

        for window in windows:
            window_key = float(window)
            signal_result = signals_by_window[window_key]

            if save_signals and signals_outdir is not None:
                lamda_dir = f"lambda_{lamda:.11f}"
                window_dir = f"window_{window:g}"
                sample_dir = sample.name if sample.name is not None else f"sample_{sample_idx}"

                if signal_dir_order == "lambda_window":
                    signal_outdir = Path(signals_outdir) / sample_dir / lamda_dir / window_dir
                else:
                    signal_outdir = Path(signals_outdir) / sample_dir / window_dir / lamda_dir

                save_signal_result(signal_result, signal_outdir)

            detection_wall_start = time.time()
            t_detection_start = time.perf_counter()
            pred_cps = detect_change_points_from_signal(
                signal=signal_result["signal"],
                selected_times=signal_result["t_samples"],
                n_bkps=sample.n_bkps,
                kernel=kernel,
            )
            detection_elapsed = time.perf_counter() - t_detection_start
            detection_wall_stop = time.time()
            if detection_wall_stop > detection_wall_start:
                change_point_detection_intervals.append((detection_wall_start, detection_wall_stop))
            change_point_detection_time_total += detection_elapsed
            change_point_detection_time_per_window[window_key] += detection_elapsed

            metrics_wall_start = time.time()
            t_metrics_start = time.perf_counter()
            f1 = f1_score(sample.true_change_points, pred_cps, margin)
            haus = hausdorff_distance(sample.true_change_points, pred_cps)
            metrics_elapsed = time.perf_counter() - t_metrics_start
            metrics_wall_stop = time.time()
            if metrics_wall_stop > metrics_wall_start:
                metrics_intervals.append((metrics_wall_start, metrics_wall_stop))
            metrics_time_total += metrics_elapsed
            metrics_time_per_window[window_key] += metrics_elapsed

            f1_accumulator[window_key].append(f1)
            hausdorff_accumulator[window_key].append(haus)
            preds_accumulator[window_key].append(pred_cps)

    for j, window in enumerate(windows):
        window_f1_scores = np.asarray(f1_accumulator[float(window)], dtype=float)
        window_hausdorff_scores = np.asarray(hausdorff_accumulator[float(window)], dtype=float)
        per_window_f1_scores[float(window)] = window_f1_scores
        per_window_hausdorff[float(window)] = window_hausdorff_scores
        predicted_change_points[float(window)] = preds_accumulator[float(window)]
        score_array[j] = float(np.mean(window_f1_scores)) if len(window_f1_scores) > 0 else math.nan

    return {
        "lamda": float(lamda),
        "windows": np.asarray(windows, dtype=float),
        "score_array": score_array,
        "per_window_f1_scores": per_window_f1_scores,
        "per_window_hausdorff": per_window_hausdorff,
        "mean_f1_per_window": {
            float(window): (float(np.mean(per_window_f1_scores[float(window)])) if len(per_window_f1_scores[float(window)]) > 0 else math.nan)
            for window in windows
        },
        "mean_hausdorff_per_window": {
            float(window): (float(np.mean(per_window_hausdorff[float(window)])) if len(per_window_hausdorff[float(window)]) > 0 else math.nan)
            for window in windows
        },
        "timing": {
            "signal_generation_total_seconds": signal_generation_time_total,
            "signal_generation_preprocessing_total_seconds": signal_generation_preprocessing_time_total,
            "change_point_detection_total_seconds": change_point_detection_time_total,
            "metrics_total_seconds": metrics_time_total,
            "signal_generation_per_window_seconds": signal_generation_time_per_window,
            "signal_generation_window_only_per_window_seconds": signal_generation_window_only_time_per_window,
            "change_point_detection_per_window_seconds": change_point_detection_time_per_window,
            "metrics_per_window_seconds": metrics_time_per_window,
            "signal_generation_intervals": signal_generation_intervals,
            "change_point_detection_intervals": change_point_detection_intervals,
            "metrics_intervals": metrics_intervals,
        },
        "predicted_change_points": predicted_change_points,
        "sample_names": sample_names,
    }


# -----------------------------------------------------------------------------
# Grid-search core
# -----------------------------------------------------------------------------

def grid_search_f1(
    samples: Sequence[CPSample],
    lambdas: Sequence[float],
    windows: Sequence[float],
    margin: float,
    n_jobs: int = 1,
    backend: str = "loky",
    verbose: int = 10,
    outdir: str | Path | None = None,
    sample_fraction: float = 0.1,
    kernel: str = "linear",
    p0: np.ndarray | None = None,
    save_signals: bool = False,
    signals_outdir: str | Path | None = None,
    signal_dir_order: str = "lambda_window",
) -> dict:
    """
    Run a parallel grid-search over all (lambda, window) pairs.

    Parallelization is done over lambda values so that the expensive
    lambda-dependent preprocessing can be reused across all window values.

    Optionally, the generated entropy signals can be saved to disk during the
    grid-search in a structured layout, either grouped first by lambda or first
    by window.
    """
    lambdas = np.asarray(lambdas, dtype=float)
    windows = np.asarray(windows, dtype=float)

    t0 = time.time()
    lambda_results = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
        delayed(evaluate_lambda)(
            samples=samples,
            lamda=float(lamda),
            windows=windows,
            margin=margin,
            sample_fraction=sample_fraction,
            kernel=kernel,
            p0=p0,
            save_signals=save_signals,
            signals_outdir=signals_outdir,
            signal_dir_order=signal_dir_order,
        )
        for lamda in lambdas
    )
    elapsed = time.time() - t0

    score_array = np.empty((len(lambdas), len(windows)), dtype=float)
    score_array[:] = np.nan
    hausdorff_array = np.empty((len(lambdas), len(windows)), dtype=float)
    hausdorff_array[:] = np.nan

    signal_generation_time_array = np.empty((len(lambdas), len(windows)), dtype=float)
    signal_generation_time_array[:] = np.nan
    change_point_detection_time_array = np.empty((len(lambdas), len(windows)), dtype=float)
    change_point_detection_time_array[:] = np.nan
    metrics_time_array = np.empty((len(lambdas), len(windows)), dtype=float)
    metrics_time_array[:] = np.nan

    signal_generation_time_per_lambda = np.empty(len(lambdas), dtype=float)
    signal_generation_time_per_lambda[:] = np.nan
    change_point_detection_time_per_lambda = np.empty(len(lambdas), dtype=float)
    change_point_detection_time_per_lambda[:] = np.nan
    metrics_time_per_lambda = np.empty(len(lambdas), dtype=float)
    metrics_time_per_lambda[:] = np.nan

    signal_generation_intervals: list[tuple[float, float]] = []
    change_point_detection_intervals: list[tuple[float, float]] = []
    metrics_intervals: list[tuple[float, float]] = []

    results_by_lambda: dict[float, dict] = {}
    for i, res in enumerate(lambda_results):
        score_array[i, :] = res["score_array"]
        hausdorff_array[i, :] = np.array(
            [res["mean_hausdorff_per_window"][float(window)] for window in windows],
            dtype=float,
        )
        signal_generation_time_array[i, :] = np.array(
            [res["timing"]["signal_generation_per_window_seconds"][float(window)] for window in windows],
            dtype=float,
        )
        change_point_detection_time_array[i, :] = np.array(
            [res["timing"]["change_point_detection_per_window_seconds"][float(window)] for window in windows],
            dtype=float,
        )
        metrics_time_array[i, :] = np.array(
            [res["timing"]["metrics_per_window_seconds"][float(window)] for window in windows],
            dtype=float,
        )
        signal_generation_time_per_lambda[i] = float(res["timing"]["signal_generation_total_seconds"])
        change_point_detection_time_per_lambda[i] = float(res["timing"]["change_point_detection_total_seconds"])
        metrics_time_per_lambda[i] = float(res["timing"]["metrics_total_seconds"])
        signal_generation_intervals.extend(res["timing"]["signal_generation_intervals"])
        change_point_detection_intervals.extend(res["timing"]["change_point_detection_intervals"])
        metrics_intervals.extend(res["timing"]["metrics_intervals"])
        results_by_lambda[float(res["lamda"])] = res

    total_signal_generation_seconds = merge_wall_clock_intervals(signal_generation_intervals)
    total_change_point_detection_seconds = merge_wall_clock_intervals(change_point_detection_intervals)
    total_metrics_seconds = merge_wall_clock_intervals(metrics_intervals)
    timed_wall_clock_seconds = merge_wall_clock_intervals(
        signal_generation_intervals + change_point_detection_intervals + metrics_intervals
    )

    if np.all(np.isnan(score_array)):
        best_index = None
        best_lamda = None
        best_window = None
        best_score = math.nan
    else:
        best_flat = int(np.nanargmax(score_array))
        best_index = np.unravel_index(best_flat, score_array.shape)
        best_lamda = float(lambdas[best_index[0]])
        best_window = float(windows[best_index[1]])
        best_score = float(score_array[best_index])

    summary = {
        "lambdas": lambdas,
        "windows": windows,
        "margin": float(margin),
        "sample_fraction": float(sample_fraction),
        "kernel": kernel,
        "save_signals": save_signals,
        "signals_outdir": str(signals_outdir) if signals_outdir is not None else None,
        "signal_dir_order": signal_dir_order,
        "score_array": score_array,
        "f1_array": score_array,
        "hausdorff_array": hausdorff_array,
        "signal_generation_time_array": signal_generation_time_array,
        "change_point_detection_time_array": change_point_detection_time_array,
        "metrics_time_array": metrics_time_array,
        "signal_generation_time_per_lambda": signal_generation_time_per_lambda,
        "change_point_detection_time_per_lambda": change_point_detection_time_per_lambda,
        "metrics_time_per_lambda": metrics_time_per_lambda,
        "total_signal_generation_seconds": total_signal_generation_seconds,
        "total_change_point_detection_seconds": total_change_point_detection_seconds,
        "total_metrics_seconds": total_metrics_seconds,
        "cumulative_signal_generation_seconds": float(np.nansum(signal_generation_time_per_lambda)),
        "cumulative_change_point_detection_seconds": float(np.nansum(change_point_detection_time_per_lambda)),
        "cumulative_metrics_seconds": float(np.nansum(metrics_time_per_lambda)),
        "timed_wall_clock_seconds": timed_wall_clock_seconds,
        "lambda_results": lambda_results,
        "results_by_lambda": results_by_lambda,
        "best_index": best_index,
        "best_lamda": best_lamda,
        "best_window": best_window,
        "best_score": best_score,
        "best_f1": best_score,
        "elapsed_seconds": elapsed,
    }

    if outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        with open(outdir / "gridsearch_f1_results.pkl", "wb") as f:
            pickle.dump(summary, f)

    return summary


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    with open("block2activities.pkl", "rb") as f:
        dataset = pickle.load(f)

    lambdas = np.logspace(-5, 0, 2)
    windows = [1, 5]
    margin = 5.0
    n_jobs = 4

    training_samples = [
        CPSample(
            data=entry["tnet"],
            true_change_points=[float(entry["bkp"])],
            n_bkps=1,
            name=f"sample_{i}",
        )
        for i, entry in enumerate(dataset)
    ]

    # Generated entropy signals can optionally be saved while running the grid-search.
    # With `signal_dir_order="lambda_window"`, the layout is:
    #     signals_outdir / sample_name / lambda_xxx / window_x
    # With `signal_dir_order="window_lambda"`, the layout is:
    #     signals_outdir / sample_name / window_x / lambda_xxx
    summary = grid_search_f1(
        samples=training_samples,
        lambdas=lambdas,
        windows=windows,
        margin=margin,
        n_jobs=n_jobs,
        outdir="./gridsearch_results/block2activities",
        sample_fraction=0.01,
        kernel="linear",
        save_signals=True,
        signals_outdir="./gridsearch_results/signals",
        signal_dir_order="lambda_window",
    )

    print("Number of samples:", len(training_samples))
    print("Score array shape:", summary["score_array"].shape)
    print("Best lamda:", summary["best_lamda"])
    print("Best window:", summary["best_window"])
    print("Best mean F1:", summary["best_score"])
    print("Hausdorff at best F1 params:", summary["hausdorff_array"][summary["best_index"]] if summary["best_index"] is not None else math.nan)
    print("Total runtime:", summary["elapsed_seconds"])
    print("Total signal generation time:", summary["total_signal_generation_seconds"])
    print("Total change-point detection time:", summary["total_change_point_detection_seconds"])
    print("Total metrics computation time:", summary["total_metrics_seconds"])
    print("Signals saved:", summary["save_signals"])
    print("Signals output directory:", summary["signals_outdir"])
    print("Signals directory order:", summary["signal_dir_order"])
    if summary["best_index"] is not None:
        print("Signal generation time at best params:", summary["signal_generation_time_array"][summary["best_index"]])
        print("Change-point detection time at best params:", summary["change_point_detection_time_array"][summary["best_index"]])
        print("Metrics computation time at best params:", summary["metrics_time_array"][summary["best_index"]])
