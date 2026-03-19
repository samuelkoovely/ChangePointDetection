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
from signal_generation import compute_signals_for_lambda, save_signal_result


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


# -----------------------------------------------------------------------------
# Signal generation phase
# -----------------------------------------------------------------------------

def compute_and_store_signals_for_lambda(
    samples: Sequence[CPSample],
    lamda: float,
    windows: Sequence[float],
    sample_fraction: float = 0.1,
    p0: np.ndarray | None = None,
    save_signals: bool = False,
    signals_outdir: str | Path | None = None,
    signal_dir_order: str = "lambda_window",
) -> dict:
    """
    Compute and store all signals associated with one lambda value.

    If requested, the generated entropy signals are also saved to disk in an
    ordered folder layout, either `outdir / lambda / window` or
    `outdir / window / lambda`.
    """
    windows = [float(w) for w in windows]
    signals_by_window = {float(window): [] for window in windows}
    sample_names = [sample.name for sample in samples]

    if signal_dir_order not in {"lambda_window", "window_lambda"}:
        raise ValueError(
            "signal_dir_order must be either 'lambda_window' or 'window_lambda'."
        )

    for sample_idx, sample in enumerate(samples):
        sample_signals = compute_signals_for_lambda(
            net=sample.data,
            lamda=lamda,
            windows=windows,
            sample_fraction=sample_fraction,
            p0=p0,
        )
        for window in windows:
            window_key = float(window)
            signal_result = sample_signals[window_key]
            signals_by_window[window_key].append(signal_result)

            if save_signals and signals_outdir is not None:
                lamda_dir = f"lambda_{lamda:.11f}"
                window_dir = f"window_{window:g}"
                sample_dir = sample.name if sample.name is not None else f"sample_{sample_idx}"

                if signal_dir_order == "lambda_window":
                    signal_outdir = Path(signals_outdir) / sample_dir / lamda_dir / window_dir
                else:
                    signal_outdir = Path(signals_outdir) / sample_dir / window_dir / lamda_dir

                save_signal_result(signal_result, signal_outdir)

    return {
        "lamda": float(lamda),
        "windows": np.asarray(windows, dtype=float),
        "signals_by_window": signals_by_window,
        "sample_names": sample_names,
    }


# -----------------------------------------------------------------------------
# Detection and metrics phase
# -----------------------------------------------------------------------------

def evaluate_precomputed_lambda_signals(
    samples: Sequence[CPSample],
    lamda: float,
    windows: Sequence[float],
    margin: float,
    signals_by_window: dict[float, list[dict[str, Any]]],
    kernel: str = "linear",
) -> dict:
    """
    Evaluate all candidate windows for one lambda using precomputed signals.
    """
    windows = [float(w) for w in windows]

    score_array = np.empty(len(windows), dtype=float)
    score_array[:] = np.nan

    per_window_f1_scores: dict[float, np.ndarray] = {}
    per_window_hausdorff: dict[float, np.ndarray] = {}
    predicted_change_points: dict[float, list[list[float]]] = {}
    sample_names = [sample.name for sample in samples]

    for j, window in enumerate(windows):
        window_key = float(window)
        window_signals = signals_by_window[window_key]
        if len(window_signals) != len(samples):
            raise ValueError(
                f"Expected {len(samples)} precomputed signals for window={window_key}, "
                f"got {len(window_signals)}."
            )

        window_f1_scores: list[float] = []
        window_hausdorff_scores: list[float] = []
        window_predicted_change_points: list[list[float]] = []

        for sample, signal_result in zip(samples, window_signals):
            pred_cps = detect_change_points_from_signal(
                signal=signal_result["signal"],
                selected_times=signal_result["t_samples"],
                n_bkps=sample.n_bkps,
                kernel=kernel,
            )
            window_predicted_change_points.append(pred_cps)
            window_f1_scores.append(f1_score(sample.true_change_points, pred_cps, margin))
            window_hausdorff_scores.append(hausdorff_distance(sample.true_change_points, pred_cps))

        window_f1_scores_array = np.asarray(window_f1_scores, dtype=float)
        window_hausdorff_scores_array = np.asarray(window_hausdorff_scores, dtype=float)
        per_window_f1_scores[window_key] = window_f1_scores_array
        per_window_hausdorff[window_key] = window_hausdorff_scores_array
        predicted_change_points[window_key] = window_predicted_change_points
        score_array[j] = (
            float(np.mean(window_f1_scores_array))
            if len(window_f1_scores_array) > 0 else math.nan
        )

    return {
        "lamda": float(lamda),
        "windows": np.asarray(windows, dtype=float),
        "score_array": score_array,
        "per_window_f1_scores": per_window_f1_scores,
        "per_window_hausdorff": per_window_hausdorff,
        "mean_f1_per_window": {
            float(window): (
                float(np.mean(per_window_f1_scores[float(window)]))
                if len(per_window_f1_scores[float(window)]) > 0 else math.nan
            )
            for window in windows
        },
        "mean_hausdorff_per_window": {
            float(window): (
                float(np.mean(per_window_hausdorff[float(window)]))
                if len(per_window_hausdorff[float(window)]) > 0 else math.nan
            )
            for window in windows
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
    selection_metric: str = "f1",
) -> dict:
    """
    Run a parallel grid-search over all (lambda, window) pairs.

    Parallelization is done over lambda values so that the expensive
    lambda-dependent preprocessing can be reused across all window values.

    Optionally, the generated entropy signals can be saved to disk during the
    grid-search in a structured layout, either grouped first by lambda or first
    by window.

    The best parameter pair can be selected either by maximizing the F1 score
    (`selection_metric="f1"`) or by minimizing the Hausdorff distance
    (`selection_metric="hausdorff"`).
    """
    lambdas = np.asarray(lambdas, dtype=float)
    windows = np.asarray(windows, dtype=float)

    if selection_metric not in {"f1", "hausdorff"}:
        raise ValueError(
            "selection_metric must be either 'f1' or 'hausdorff'."
        )

    t0 = time.time()
    t_signal_phase_start = time.time()
    lambda_signal_results = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
        delayed(compute_and_store_signals_for_lambda)(
            samples=samples,
            lamda=float(lamda),
            windows=windows,
            sample_fraction=sample_fraction,
            p0=p0,
            save_signals=save_signals,
            signals_outdir=signals_outdir,
            signal_dir_order=signal_dir_order,
        )
        for lamda in lambdas
    )
    signal_generation_phase_seconds = time.time() - t_signal_phase_start

    t_detection_phase_start = time.time()
    lambda_results = [
        evaluate_precomputed_lambda_signals(
            samples=samples,
            lamda=float(signal_bundle["lamda"]),
            windows=windows,
            margin=margin,
            signals_by_window=signal_bundle["signals_by_window"],
            kernel=kernel,
        )
        for signal_bundle in lambda_signal_results
    ]
    detection_metrics_phase_seconds = time.time() - t_detection_phase_start
    elapsed = time.time() - t0

    score_array = np.empty((len(lambdas), len(windows)), dtype=float)
    score_array[:] = np.nan
    hausdorff_array = np.empty((len(lambdas), len(windows)), dtype=float)
    hausdorff_array[:] = np.nan

    results_by_lambda: dict[float, dict] = {}
    for i, res in enumerate(lambda_results):
        score_array[i, :] = res["score_array"]
        hausdorff_array[i, :] = np.array(
            [res["mean_hausdorff_per_window"][float(window)] for window in windows],
            dtype=float,
        )
        results_by_lambda[float(res["lamda"])] = res

    num_samples = len(samples)
    num_lambda_jobs = len(lambdas)
    num_parameter_pairs = len(lambdas) * len(windows)

    if selection_metric == "f1":
        selection_array = score_array
        if np.all(np.isnan(selection_array)):
            best_index = None
            best_lamda = None
            best_window = None
            best_score = math.nan
            best_f1 = math.nan
            best_hausdorff = math.nan
        else:
            best_flat = int(np.nanargmax(selection_array))
            best_index = np.unravel_index(best_flat, selection_array.shape)
            best_lamda = float(lambdas[best_index[0]])
            best_window = float(windows[best_index[1]])
            best_score = float(selection_array[best_index])
            best_f1 = float(score_array[best_index])
            best_hausdorff = float(hausdorff_array[best_index])
    else:
        selection_array = hausdorff_array.copy()
        if np.all(np.isnan(selection_array)):
            best_index = None
            best_lamda = None
            best_window = None
            best_score = math.nan
            best_f1 = math.nan
            best_hausdorff = math.nan
        else:
            selection_array[np.isnan(selection_array)] = math.inf
            if np.all(np.isinf(selection_array)):
                best_index = None
                best_lamda = None
                best_window = None
                best_score = math.nan
                best_f1 = math.nan
                best_hausdorff = math.nan
            else:
                best_flat = int(np.argmin(selection_array))
                best_index = np.unravel_index(best_flat, selection_array.shape)
                best_lamda = float(lambdas[best_index[0]])
                best_window = float(windows[best_index[1]])
                best_score = float(hausdorff_array[best_index])
                best_f1 = float(score_array[best_index])
                best_hausdorff = float(hausdorff_array[best_index])

    summary = {
        "lambdas": lambdas,
        "windows": windows,
        "margin": float(margin),
        "sample_fraction": float(sample_fraction),
        "kernel": kernel,
        "num_samples": num_samples,
        "num_lambda_jobs": num_lambda_jobs,
        "num_parameter_pairs": num_parameter_pairs,
        "save_signals": save_signals,
        "signals_outdir": str(signals_outdir) if signals_outdir is not None else None,
        "signal_dir_order": signal_dir_order,
        "selection_metric": selection_metric,
        "selection_array": score_array if selection_metric == "f1" else hausdorff_array,
        "score_array": score_array,
        "f1_array": score_array,
        "hausdorff_array": hausdorff_array,
        "signal_generation_phase_seconds": signal_generation_phase_seconds,
        "detection_metrics_phase_seconds": detection_metrics_phase_seconds,
        "lambda_results": lambda_results,
        "results_by_lambda": results_by_lambda,
        "best_index": best_index,
        "best_lamda": best_lamda,
        "best_window": best_window,
        "best_score": best_score,
        "best_f1": best_f1,
        "best_hausdorff": best_hausdorff,
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
        selection_metric="f1",
    )

    print("Number of samples:", len(training_samples))
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
    print("Signals output directory:", summary["signals_outdir"])
    print("Signals directory order:", summary["signal_dir_order"])
