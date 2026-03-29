"""
Grid-search training pipeline for snapshot change-point detection using
averaged Frobenius distance.

This mirrors the structure of the snapshot Laplacian grid-search:
- precompute signals for each candidate window length
- run ruptures on each signal
- score predictions with F1 and Hausdorff distance
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

from benchmark_methods import avg_frobenius_distance
from evaluation_metrics import f1_score, hausdorff_distance


@dataclass
class CPSample:
    """
    One training example for the grid-search.
    """

    data: Any
    true_change_points: Sequence[float]
    n_bkps: int = 1
    name: str | None = None


def get_signal_result_filename(
    window_length: int,
    suffix: str = ".pkl",
) -> str:
    """
    Return the canonical filename for one saved Frobenius-distance signal.
    """

    return f"signal_window_length_{int(window_length)}{suffix}"


def save_signal_result(result: dict[str, Any], outdir: str | Path) -> Path:
    """
    Save one signal result dictionary to disk inside a sample folder.
    """

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    outfile = outdir / get_signal_result_filename(
        window_length=int(result["window_length"]),
    )
    with open(outfile, "wb") as f:
        pickle.dump(result, f)

    return outfile


def detect_change_points_from_signal(
    signal: np.ndarray,
    snapshot_indices: Sequence[float],
    n_bkps: int,
    kernel: str = "linear",
) -> list[float]:
    """
    Run ruptures on a precomputed signal and convert breakpoint positions back
    to snapshot indices.
    """

    signal = np.asarray(signal)
    snapshot_indices = np.asarray(snapshot_indices, dtype=float)

    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)

    if signal.ndim != 2:
        raise ValueError(
            f"signal must be a 1D or 2D array-like object, got shape {signal.shape}."
        )

    if signal.shape[0] != len(snapshot_indices):
        raise ValueError(
            "signal and snapshot_indices must have the same number of points. "
            f"Got {signal.shape[0]} and {len(snapshot_indices)}."
        )

    if signal.shape[0] == 0 or signal.shape[0] <= n_bkps:
        return []

    try:
        algo = rpt.KernelCPD(kernel=kernel).fit(signal)
        breakpoint_indices = np.asarray(algo.predict(n_bkps=n_bkps), dtype=int)
    except Exception:
        return []

    if breakpoint_indices.size == 0:
        return []

    breakpoint_indices = breakpoint_indices[
        (breakpoint_indices >= 0) & (breakpoint_indices < len(snapshot_indices))
    ]
    if breakpoint_indices.size == 0:
        return []

    return snapshot_indices[breakpoint_indices].tolist()


def compute_and_store_signals_for_sample(
    sample: CPSample,
    sample_idx: int,
    window_lengths: Sequence[int],
    save_signals: bool = False,
    signals_outdir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Compute and optionally store all Frobenius-distance signals for one sample.
    """

    window_lengths = [int(window_length) for window_length in window_lengths]
    signals_by_window: dict[int, dict[str, Any]] = {}

    sample_name = sample.name if sample.name is not None else f"sample_{sample_idx}"
    for window_length in window_lengths:
        signal, snapshot_indices = avg_frobenius_distance(
            sample.data,
            window_length=window_length,
        )
        signal_result = {
            "window_length": window_length,
            "signal": np.asarray(signal, dtype=float),
            "snapshot_indices": np.asarray(snapshot_indices, dtype=int),
        }
        signals_by_window[window_length] = signal_result

        if save_signals and signals_outdir is not None:
            signal_outdir = Path(signals_outdir) / sample_name
            save_signal_result(signal_result, signal_outdir)

    return {
        "sample_idx": int(sample_idx),
        "sample_name": sample_name,
        "signals_by_window": signals_by_window,
    }


def reorganize_sample_signal_results_by_window(
    sample_signal_results: Sequence[dict[str, Any]],
    window_lengths: Sequence[int],
) -> dict[int, list[dict[str, Any]]]:
    """
    Transpose sample-owned signal bundles into window-owned bundles.
    """

    ordered_results = sorted(sample_signal_results, key=lambda item: item["sample_idx"])
    window_lengths = [int(window_length) for window_length in window_lengths]

    signals_by_window: dict[int, list[dict[str, Any]]] = {
        window_length: [] for window_length in window_lengths
    }
    for window_length in window_lengths:
        for sample_result in ordered_results:
            signals_by_window[window_length].append(
                sample_result["signals_by_window"][window_length]
            )

    return signals_by_window


def evaluate_precomputed_window_signals(
    samples: Sequence[CPSample],
    window_lengths: Sequence[int],
    margin: float,
    signals_by_window: dict[int, list[dict[str, Any]]],
    kernel: str = "linear",
) -> dict[str, Any]:
    """
    Evaluate all candidate window lengths using precomputed signals.
    """

    window_lengths = [int(window_length) for window_length in window_lengths]

    score_array = np.empty(len(window_lengths), dtype=float)
    score_array[:] = np.nan

    per_window_f1_scores: dict[int, np.ndarray] = {}
    per_window_hausdorff: dict[int, np.ndarray] = {}
    predicted_change_points: dict[int, list[list[float]]] = {}
    sample_names = [sample.name for sample in samples]

    for j, window_length in enumerate(window_lengths):
        window_signals = signals_by_window[window_length]
        if len(window_signals) != len(samples):
            raise ValueError(
                f"Expected {len(samples)} precomputed signals for window_length="
                f"{window_length}, got {len(window_signals)}."
            )

        window_f1_scores: list[float] = []
        window_hausdorff_scores: list[float] = []
        window_predicted_change_points: list[list[float]] = []

        for sample, signal_result in zip(samples, window_signals):
            pred_cps = detect_change_points_from_signal(
                signal=signal_result["signal"],
                snapshot_indices=signal_result["snapshot_indices"],
                n_bkps=sample.n_bkps,
                kernel=kernel,
            )
            window_predicted_change_points.append(pred_cps)
            window_f1_scores.append(
                f1_score(sample.true_change_points, pred_cps, margin)
            )
            window_hausdorff_scores.append(
                hausdorff_distance(sample.true_change_points, pred_cps)
            )

        window_f1_scores_array = np.asarray(window_f1_scores, dtype=float)
        window_hausdorff_scores_array = np.asarray(window_hausdorff_scores, dtype=float)
        per_window_f1_scores[window_length] = window_f1_scores_array
        per_window_hausdorff[window_length] = window_hausdorff_scores_array
        predicted_change_points[window_length] = window_predicted_change_points
        score_array[j] = (
            float(np.mean(window_f1_scores_array))
            if len(window_f1_scores_array) > 0
            else math.nan
        )

    return {
        "window_lengths": np.asarray(window_lengths, dtype=int),
        "score_array": score_array,
        "per_window_f1_scores": per_window_f1_scores,
        "per_window_hausdorff": per_window_hausdorff,
        "mean_f1_per_window_length": {
            int(window_length): (
                float(np.mean(per_window_f1_scores[int(window_length)]))
                if len(per_window_f1_scores[int(window_length)]) > 0
                else math.nan
            )
            for window_length in window_lengths
        },
        "mean_hausdorff_per_window_length": {
            int(window_length): (
                float(np.mean(per_window_hausdorff[int(window_length)]))
                if len(per_window_hausdorff[int(window_length)]) > 0
                else math.nan
            )
            for window_length in window_lengths
        },
        "predicted_change_points": predicted_change_points,
        "sample_names": sample_names,
    }


def grid_search_frobenius_distance(
    samples: Sequence[CPSample],
    window_lengths: Sequence[int],
    margin: float,
    n_jobs: int = 1,
    backend: str = "loky",
    verbose: int = 10,
    outdir: str | Path | None = None,
    kernel: str = "linear",
    save_signals: bool = False,
    signals_outdir: str | Path | None = None,
    selection_metric: str = "f1",
) -> dict[str, Any]:
    """
    Run a parallel grid-search over all candidate window lengths.
    """

    window_lengths = np.asarray(window_lengths, dtype=int)

    if len(window_lengths) == 0:
        raise ValueError("window_lengths must contain at least one value.")
    if np.any(window_lengths <= 0):
        raise ValueError("All window_lengths must be strictly positive.")
    if selection_metric not in {"f1", "hausdorff"}:
        raise ValueError("selection_metric must be either 'f1' or 'hausdorff'.")

    t0 = time.time()
    t_signal_phase_start = time.time()
    signal_tasks = [
        delayed(compute_and_store_signals_for_sample)(
            sample=sample,
            sample_idx=sample_idx,
            window_lengths=window_lengths,
            save_signals=save_signals,
            signals_outdir=signals_outdir,
        )
        for sample_idx, sample in enumerate(samples)
    ]

    backend_used = backend
    try:
        sample_signal_results = Parallel(
            n_jobs=n_jobs,
            backend=backend,
            verbose=verbose,
        )(signal_tasks)
    except PermissionError:
        if backend != "loky":
            raise
        backend_used = "threading"
        sample_signal_results = Parallel(
            n_jobs=n_jobs,
            backend=backend_used,
            verbose=verbose,
        )(signal_tasks)

    signals_by_window = reorganize_sample_signal_results_by_window(
        sample_signal_results=sample_signal_results,
        window_lengths=window_lengths,
    )
    signal_generation_phase_seconds = time.time() - t_signal_phase_start

    t_detection_phase_start = time.time()
    evaluation = evaluate_precomputed_window_signals(
        samples=samples,
        window_lengths=window_lengths,
        margin=margin,
        signals_by_window=signals_by_window,
        kernel=kernel,
    )
    detection_metrics_phase_seconds = time.time() - t_detection_phase_start
    elapsed = time.time() - t0

    score_array = evaluation["score_array"]
    hausdorff_array = np.array(
        [
            evaluation["mean_hausdorff_per_window_length"][int(window_length)]
            for window_length in window_lengths
        ],
        dtype=float,
    )

    if selection_metric == "f1":
        selection_array = score_array
        if np.all(np.isnan(selection_array)):
            best_index = None
            best_window_length = None
            best_score = math.nan
            best_f1 = math.nan
            best_hausdorff = math.nan
        else:
            best_index = int(np.nanargmax(selection_array))
            best_window_length = int(window_lengths[best_index])
            best_score = float(selection_array[best_index])
            best_f1 = float(score_array[best_index])
            best_hausdorff = float(hausdorff_array[best_index])
    else:
        selection_array = hausdorff_array.copy()
        if np.all(np.isnan(selection_array)):
            best_index = None
            best_window_length = None
            best_score = math.nan
            best_f1 = math.nan
            best_hausdorff = math.nan
        else:
            selection_array[np.isnan(selection_array)] = math.inf
            if np.all(np.isinf(selection_array)):
                best_index = None
                best_window_length = None
                best_score = math.nan
                best_f1 = math.nan
                best_hausdorff = math.nan
            else:
                best_index = int(np.argmin(selection_array))
                best_window_length = int(window_lengths[best_index])
                best_score = float(hausdorff_array[best_index])
                best_f1 = float(score_array[best_index])
                best_hausdorff = float(hausdorff_array[best_index])

    summary = {
        "window_lengths": window_lengths,
        "margin": float(margin),
        "kernel": kernel,
        "backend_requested": backend,
        "backend_used": backend_used,
        "num_samples": len(samples),
        "num_parameter_pairs": len(window_lengths),
        "save_signals": save_signals,
        "signals_outdir": str(signals_outdir) if signals_outdir is not None else None,
        "selection_metric": selection_metric,
        "selection_array": score_array if selection_metric == "f1" else hausdorff_array,
        "score_array": score_array,
        "f1_array": score_array,
        "hausdorff_array": hausdorff_array,
        "signal_generation_phase_seconds": signal_generation_phase_seconds,
        "detection_metrics_phase_seconds": detection_metrics_phase_seconds,
        "window_results": evaluation,
        "best_index": best_index,
        "best_window_length": best_window_length,
        "best_window": best_window_length,
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


if __name__ == "__main__":
    with open("data/block1activity_snapshots.pkl", "rb") as f:
        dataset = pickle.load(f)

    first_net = dataset[0]["tnet"]
    num_snapshots = max(1, len(first_net.times) - 1)
    max_window_length = max(1, min(8, num_snapshots - 1))

    window_lengths = list(range(1, max_window_length + 1))
    margin = 1.0
    n_jobs = 6

    training_samples = [
        CPSample(
            data=entry["tnet"],
            true_change_points=[float(change_point) for change_point in entry["bkps"]],
            n_bkps=int(entry["n_bkps"]),
            name=f"sample_{i}",
        )
        for i, entry in enumerate(dataset)
    ]

    demo_window_length = min(5, max_window_length)
    demo_signal, demo_indices = avg_frobenius_distance(
        first_net,
        window_length=demo_window_length,
    )
    print("Demo signal length:", len(demo_signal))
    print("Demo index range:", demo_indices[:3], "...", demo_indices[-3:])

    summary = grid_search_frobenius_distance(
        samples=training_samples,
        window_lengths=window_lengths,
        margin=margin,
        n_jobs=n_jobs,
        outdir="./gridsearch_results/block1activity_snapshots_frobenius",
        kernel="linear",
        save_signals=True,
        signals_outdir="./gridsearch_results/block1activity_snapshots_frobenius/signals",
        selection_metric="hausdorff",
    )

    print("Number of samples:", len(training_samples))
    print("Score array shape:", summary["score_array"].shape)
    print("Backend used:", summary["backend_used"])
    print("Selection metric:", summary["selection_metric"])
    print("Best window_length:", summary["best_window_length"])
    print("Best selected score:", summary["best_score"])
    print("Best mean F1:", summary["best_f1"])
    print("Best mean Hausdorff:", summary["best_hausdorff"])
    print("Total runtime:", summary["elapsed_seconds"])
    print(
        "Signal generation phase runtime:",
        summary["signal_generation_phase_seconds"],
    )
    print(
        "Detection and metrics phase runtime:",
        summary["detection_metrics_phase_seconds"],
    )
    print("Signals saved:", summary["save_signals"])
    print("Signals output directory:", summary["signals_outdir"])
