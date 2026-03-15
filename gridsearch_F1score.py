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
from signal_generation import compute_signals_for_lambda


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
) -> dict:
    """
    Evaluate all candidate windows for one lambda value.

    The lambda-dependent preprocessing is performed once per sample and then
    reused across all window values.
    """
    windows = [float(w) for w in windows]

    score_array = np.empty(len(windows), dtype=float)
    score_array[:] = np.nan

    per_window_scores: dict[float, np.ndarray] = {}
    per_window_f1_scores: dict[float, np.ndarray] = {}
    per_window_hausdorff: dict[float, np.ndarray] = {}
    predicted_change_points: dict[float, list[list[float]]] = {}
    sample_names = [sample.name for sample in samples]

    f1_accumulator = {float(window): [] for window in windows}
    hausdorff_accumulator = {float(window): [] for window in windows}
    preds_accumulator = {float(window): [] for window in windows}

    for sample in samples:
        signals_by_window = compute_signals_for_lambda(
            net=sample.data,
            lamda=lamda,
            windows=windows,
            sample_fraction=sample_fraction,
            p0=p0,
        )

        for window in windows:
            signal_result = signals_by_window[float(window)]
            pred_cps = detect_change_points_from_signal(
                signal=signal_result["signal"],
                selected_times=signal_result["t_samples"],
                n_bkps=sample.n_bkps,
                kernel=kernel,
            )
            f1 = f1_score(sample.true_change_points, pred_cps, margin)
            haus = hausdorff_distance(sample.true_change_points, pred_cps)

            f1_accumulator[float(window)].append(f1)
            hausdorff_accumulator[float(window)].append(haus)
            preds_accumulator[float(window)].append(pred_cps)

    for j, window in enumerate(windows):
        window_f1_scores = np.asarray(f1_accumulator[float(window)], dtype=float)
        window_hausdorff_scores = np.asarray(hausdorff_accumulator[float(window)], dtype=float)
        per_window_scores[float(window)] = window_f1_scores
        per_window_f1_scores[float(window)] = window_f1_scores
        per_window_hausdorff[float(window)] = window_hausdorff_scores
        predicted_change_points[float(window)] = preds_accumulator[float(window)]
        score_array[j] = float(np.mean(window_f1_scores)) if len(window_f1_scores) > 0 else math.nan

    return {
        "lamda": float(lamda),
        "windows": np.asarray(windows, dtype=float),
        "score_array": score_array,
        "per_window_scores": per_window_scores,
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
) -> dict:
    """
    Run a parallel grid-search over all (lambda, window) pairs.

    Parallelization is done over lambda values so that the expensive
    lambda-dependent preprocessing can be reused across all window values.
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
        )
        for lamda in lambdas
    )
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
        "score_array": score_array,
        "f1_array": score_array,
        "hausdorff_array": hausdorff_array,
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

    summary = grid_search_f1(
        samples=training_samples,
        lambdas=lambdas,
        windows=windows,
        margin=margin,
        n_jobs=n_jobs,
        outdir="./gridsearch_results",
        sample_fraction=0.01,
        kernel="linear",
    )

    print("Number of samples:", len(training_samples))
    print("Score array shape:", summary["score_array"].shape)
    print("Best lamda:", summary["best_lamda"])
    print("Best window:", summary["best_window"])
    print("Best mean F1:", summary["best_score"])
    print("Hausdorff at best F1 params:", summary["hausdorff_array"][summary["best_index"]] if summary["best_index"] is not None else math.nan)