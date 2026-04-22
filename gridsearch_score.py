"""
Parallel grid-search over (lambda, window) for change-point detection.

This version reuses the signal-generation helpers from `signal_generation.py`.
Each worker owns one sample, prepares its reusable state once, then computes
all candidate lambdas and window lengths for that sample.

Unlike `gridsearch_score_snapshots.py`, the detection step maps breakpoint
indices back to sampled times before evaluation.
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
from signal_generation import (
    compute_signals_for_lambdas_prepared,
    prepare_signal_sample,
    save_signal_result,
)


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

    if n_bkps <= 0 or signal.shape[0] <= n_bkps:
        return []

    try:
        algo = rpt.KernelCPD(kernel=kernel).fit(signal)
        breakpoint_indices = np.asarray(algo.predict(n_bkps=n_bkps), dtype=int)
    except Exception:
        return []

    if breakpoint_indices.size == 0:
        return []

    # ruptures typically includes the terminal endpoint n_samples, which is not
    # a change point. Remove it before converting indices to times.
    breakpoint_indices = breakpoint_indices[
        (breakpoint_indices >= 0) & (breakpoint_indices < len(selected_times))
    ]

    if breakpoint_indices.size == 0:
        return []

    return selected_times[breakpoint_indices].tolist()


def extract_true_change_points(entry: dict[str, Any]) -> tuple[list[float], int]:
    """
    Read the canonical dataset label format into a sorted list of ground-truth
    change points and its count.
    """
    if "bkps" not in entry:
        raise KeyError("Dataset entry must contain the 'bkps' key.")
    if "n_bkps" not in entry:
        raise KeyError("Dataset entry must contain the 'n_bkps' key.")

    true_change_points = sorted(float(change_point) for change_point in entry["bkps"])
    n_bkps = int(entry["n_bkps"])
    if n_bkps != len(true_change_points):
        raise ValueError(
            "Dataset entry has inconsistent breakpoint metadata: "
            f"n_bkps={n_bkps}, len(bkps)={len(true_change_points)}."
        )

    return true_change_points, n_bkps


# -----------------------------------------------------------------------------
# Signal generation phase
# -----------------------------------------------------------------------------

def compute_and_store_signals_for_sample(
    sample: CPSample,
    sample_idx: int,
    lambdas: Sequence[float],
    windows: Sequence[float],
    sample_fraction: float = 0.1,
    p0: np.ndarray | None = None,
    use_linear_approx: bool = False,
    lin_t_s: int = 10,
    save_signals: bool = False,
    signals_outdir: str | Path | None = None,
    window_backend: str = "auto",
    reverse_time: bool = False,
) -> dict:
    """
    Compute and optionally store all signals associated with one sample.

    The worker owns one sample, prepares its reusable state once, then computes
    all lambda/window combinations locally.
    """
    lambdas = [float(lamda) for lamda in lambdas]
    windows = [float(w) for w in windows]

    prepared = prepare_signal_sample(
        net=sample.data,
        windows=windows,
        sample_fraction=sample_fraction,
        p0=p0,
        reverse_time=reverse_time,
    )
    signals_by_lambda = compute_signals_for_lambdas_prepared(
        prepared=prepared,
        lambdas=lambdas,
        use_linear_approx=use_linear_approx,
        lin_t_s=lin_t_s,
        window_backend=window_backend,
    )

    sample_name = sample.name if sample.name is not None else f"sample_{sample_idx}"
    if save_signals and signals_outdir is not None:
        signal_outdir = Path(signals_outdir) / sample_name
        for lamda in lambdas:
            for window in windows:
                signal_result = signals_by_lambda[lamda][window]
                save_signal_result(signal_result, signal_outdir)

    return {
        "sample_idx": int(sample_idx),
        "sample_name": sample_name,
        "signals_by_lambda": signals_by_lambda,
        "reverse_time": bool(reverse_time),
    }


def reorganize_sample_signal_results_by_lambda(
    sample_signal_results: Sequence[dict],
    samples: Sequence[CPSample],
    lambdas: Sequence[float],
    windows: Sequence[float],
) -> list[dict]:
    """
    Transpose sample-owned signal bundles into lambda-owned bundles.
    """
    ordered_results = sorted(sample_signal_results, key=lambda item: item["sample_idx"])
    lambdas = [float(lamda) for lamda in lambdas]
    windows = [float(window) for window in windows]
    sample_names = [
        sample.name if sample.name is not None else f"sample_{idx}"
        for idx, sample in enumerate(samples)
    ]

    lambda_signal_results: list[dict] = []
    for lamda in lambdas:
        signals_by_window = {window: [] for window in windows}
        for sample_result in ordered_results:
            sample_lambda_results = sample_result["signals_by_lambda"][lamda]
            for window in windows:
                signals_by_window[window].append(sample_lambda_results[window])

        lambda_signal_results.append(
            {
                "lamda": float(lamda),
                "windows": np.asarray(windows, dtype=float),
                "signals_by_window": signals_by_window,
                "sample_names": sample_names,
            }
        )

    return lambda_signal_results


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

def grid_search(
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
    use_linear_approx: bool = False,
    lin_t_s: int = 10,
    save_signals: bool = False,
    signals_outdir: str | Path | None = None,
    selection_metric: str = "f1",
    window_backend: str = "auto",
) -> dict:
    """
    Run a parallel grid-search over all (lambda, window) pairs.

    Parallelization is done over samples so that each worker owns its network,
    prepares sample-level state once, then reuses it across all lambdas and
    window lengths.

    Optionally, the generated entropy signals can be saved to disk during the
    grid-search, with one folder per sample and filenames that encode both the
    lambda and window.

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
    sample_signal_results = Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)(
        delayed(compute_and_store_signals_for_sample)(
            sample=sample,
            sample_idx=sample_idx,
            lambdas=lambdas,
            windows=windows,
            sample_fraction=sample_fraction,
            p0=p0,
            use_linear_approx=use_linear_approx,
            lin_t_s=lin_t_s,
            save_signals=save_signals,
            signals_outdir=signals_outdir,
            window_backend=window_backend,
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
    num_sample_jobs = len(samples)
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
        "use_linear_approx": bool(use_linear_approx),
        "lin_t_s": int(lin_t_s),
        "window_backend": window_backend,
        "parallel_axis": "sample",
        "num_samples": num_samples,
        "num_lambda_jobs": num_lambda_jobs,
        "num_sample_jobs": num_sample_jobs,
        "num_parameter_pairs": num_parameter_pairs,
        "save_signals": save_signals,
        "signals_outdir": str(signals_outdir) if signals_outdir is not None else None,
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
        with open(outdir / "gridsearch_results.pkl", "wb") as f:
            pickle.dump(summary, f)

    return summary


grid_search_f1 = grid_search


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    with open("data/block2activities_train.pkl", "rb") as f:
        dataset = pickle.load(f)

    lambdas = np.logspace(-5, 0, 10)
    windows = [2.0]
    margin = 5.0
    n_jobs = 6

    training_samples = []
    for i, entry in enumerate(dataset):
        true_change_points, n_bkps = extract_true_change_points(entry)
        training_samples.append(
            CPSample(
                data=entry["tnet"],
                true_change_points=true_change_points,
                n_bkps=n_bkps,
                name=f"sample_{i}",
            )
        )

    # Generated entropy signals can optionally be saved while running the
    # grid-search under:
    #     signals_outdir / sample_name / signal_lamda_xxx_window_y.pkl
    summary = grid_search(
        samples=training_samples,
        lambdas=lambdas,
        windows=windows,
        margin=margin,
        n_jobs=n_jobs,
        outdir="./gridsearch_results/block2activities",
        sample_fraction=1.0,
        kernel="linear",
        save_signals=True,
        signals_outdir="./gridsearch_results/block2activities/signals",
        selection_metric="hausdorff",
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
