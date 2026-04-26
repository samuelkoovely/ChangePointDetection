"""
Grid-search training pipeline for snapshot change-point detection using an
upstream-style LAD signal.

Compared with the existing local Laplacian grid-search, this variant uses the
experimental LAD implementation from `benchmark_methods_lad.py`, which follows
the original LAD repository more closely:
- top Laplacian singular values instead of smallest eigenvalues
- optional first-difference score
- optional short/long two-window scoring
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import math
import pickle
import time

import numpy as np
from joblib import Parallel, delayed

from benchmark_methods_lad import (
    compute_laplacian_signatures as benchmark_compute_laplacian_signatures,
    compute_similarity_signal_from_signatures as benchmark_compute_similarity_signal_from_signatures,
    laplacian_spectrum_similarity as benchmark_laplacian_spectrum_similarity,
)
from evaluation_metrics import f1_score, hausdorff_distance


@dataclass
class CPSample:
    """
    One training example for the LAD grid-search.
    """

    data: Any
    true_change_points: Sequence[float]
    n_bkps: int = 1
    name: str | None = None


def _is_temporal_network_like(data: Any) -> bool:
    return hasattr(data, "compute_laplacian_matrices") and hasattr(data, "times")


def _clone_data_for_signal_computation(data: Any) -> Any:
    """
    Avoid sharing mutable temporal-network caches across parallel workers.

    The LAD signal path may populate cached Laplacians on the network object.
    When joblib falls back to the threading backend, parallel `n_eigen` tasks
    would otherwise mutate the same `TemporalNetwork` instances concurrently.
    """

    if _is_temporal_network_like(data):
        return copy.deepcopy(data)
    return data


def resolve_second_window_length(
    window_length: int,
    second_window_scale: float | None,
) -> int | None:
    """
    Convert a short-window length into a long-window length.

    When `second_window_scale` is `None`, the LAD signal uses a single window.
    """

    window_length = int(window_length)
    if window_length <= 0:
        raise ValueError("window_length must be strictly positive.")

    if second_window_scale is None:
        return None

    second_window_scale = float(second_window_scale)
    if second_window_scale <= 1.0:
        raise ValueError("second_window_scale must be greater than 1.0.")

    second_window_length = int(round(second_window_scale * window_length))
    return max(window_length + 1, second_window_length)


def compute_laplacian_signatures(
    data: Any,
    n_eigen: int,
    normalize: bool = False,
    top: bool = True,
) -> np.ndarray:
    """
    Compute upstream-style LAD Laplacian signatures for every snapshot.
    """

    return benchmark_compute_laplacian_signatures(
        data=data,
        n_eigen=n_eigen,
        normalize=normalize,
        top=top,
    )


def compute_similarity_signal_from_signatures(
    laplacian_signatures: np.ndarray,
    window_length: int,
    n_eigen: int,
    difference: bool = True,
    second_window_scale: float | None = 2.0,
) -> dict[str, Any]:
    """
    Compute the upstream-style LAD signal from precomputed signatures.
    """

    second_window_length = resolve_second_window_length(
        window_length=window_length,
        second_window_scale=second_window_scale,
    )
    result = benchmark_compute_similarity_signal_from_signatures(
        laplacian_signatures=laplacian_signatures,
        window_length=window_length,
        n_eigen=n_eigen,
        difference=difference,
        second_window_length=second_window_length,
    )
    result["second_window_scale"] = second_window_scale
    return result


def laplacian_spectrum_similarity(
    data: Any,
    window_length: int,
    normalize: bool = False,
    n_eigen: int = 6,
    top: bool = True,
    difference: bool = True,
    second_window_scale: float | None = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute an upstream-style LAD anomaly signal for one temporal network.
    """

    second_window_length = resolve_second_window_length(
        window_length=window_length,
        second_window_scale=second_window_scale,
    )
    return benchmark_laplacian_spectrum_similarity(
        data=data,
        window_length=window_length,
        normalize=normalize,
        n_eigen=n_eigen,
        top=top,
        difference=difference,
        second_window_length=second_window_length,
    )


def get_signal_result_filename(
    n_eigen: int,
    window_length: int,
    top: bool,
    difference: bool,
    second_window_length: int | None,
    suffix: str = ".pkl",
) -> str:
    """
    Return the canonical filename for one saved LAD signal.
    """

    parts = [
        f"signal_n_eigen_{int(n_eigen)}",
        f"window_length_{int(window_length)}",
        f"top_{int(bool(top))}",
        f"difference_{int(bool(difference))}",
    ]
    if second_window_length is not None:
        parts.append(f"second_window_length_{int(second_window_length)}")
    else:
        parts.append("single_window")
    return "_".join(parts) + suffix


def save_signal_result(result: dict[str, Any], outdir: str | Path) -> Path:
    """
    Save one signal result dictionary to disk inside a sample folder.
    """

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    outfile = outdir / get_signal_result_filename(
        n_eigen=int(result["n_eigen"]),
        window_length=int(result["window_length"]),
        top=bool(result["top"]),
        difference=bool(result["difference"]),
        second_window_length=(
            None
            if result.get("second_window_length") is None
            else int(result["second_window_length"])
        ),
    )
    with open(outfile, "wb") as handle:
        pickle.dump(result, handle)

    return outfile


def compute_and_store_signals_for_n_eigen(
    samples: Sequence[CPSample],
    n_eigen: int,
    window_lengths: Sequence[int],
    normalize: bool = False,
    save_signals: bool = False,
    signals_outdir: str | Path | None = None,
    top: bool = True,
    difference: bool = True,
    second_window_scale: float | None = 2.0,
) -> dict[str, Any]:
    """
    Compute and optionally store all LAD signals associated with one `n_eigen`.
    """

    n_eigen = int(n_eigen)
    window_lengths = [int(window_length) for window_length in window_lengths]
    signals_by_window = {int(window_length): [] for window_length in window_lengths}
    sample_names = [sample.name for sample in samples]

    for sample_idx, sample in enumerate(samples):
        sample_data = _clone_data_for_signal_computation(sample.data)
        laplacian_signatures = compute_laplacian_signatures(
            data=sample_data,
            n_eigen=n_eigen,
            normalize=normalize,
            top=top,
        )

        for window_length in window_lengths:
            second_window_length = resolve_second_window_length(
                window_length=window_length,
                second_window_scale=second_window_scale,
            )
            signal_result = compute_similarity_signal_from_signatures(
                laplacian_signatures=laplacian_signatures,
                window_length=window_length,
                n_eigen=n_eigen,
                difference=difference,
                second_window_scale=second_window_scale,
            )
            signal_result["top"] = bool(top)
            signal_result["difference"] = bool(difference)
            signal_result["second_window_scale"] = second_window_scale
            signal_result["second_window_length"] = second_window_length
            signals_by_window[window_length].append(signal_result)

            if save_signals and signals_outdir is not None:
                sample_dir = sample.name if sample.name is not None else f"sample_{sample_idx}"
                signal_outdir = Path(signals_outdir) / sample_dir
                save_signal_result(signal_result, signal_outdir)

    return {
        "n_eigen": n_eigen,
        "window_lengths": np.asarray(window_lengths, dtype=int),
        "signals_by_window": signals_by_window,
        "sample_names": sample_names,
        "top": bool(top),
        "difference": bool(difference),
        "second_window_scale": second_window_scale,
    }


def detect_change_points_from_signal(
    signal: np.ndarray,
    snapshot_indices: Sequence[float],
    n_bkps: int,
) -> list[float]:
    """
    Select change points by directly ranking LAD scores.
    """

    signal = np.asarray(signal, dtype=float).reshape(-1)
    snapshot_indices = np.asarray(snapshot_indices, dtype=float).reshape(-1)

    if signal.shape[0] != snapshot_indices.shape[0]:
        raise ValueError(
            "signal and snapshot_indices must have the same number of points. "
            f"Got {signal.shape[0]} and {snapshot_indices.shape[0]}."
        )

    if signal.size == 0 or n_bkps <= 0:
        return []

    k = min(int(n_bkps), int(signal.size))
    ranked_positions = np.argsort(signal)[-k:][::-1]
    ranked_snapshot_indices = np.sort(snapshot_indices[ranked_positions])
    return ranked_snapshot_indices.tolist()


def evaluate_precomputed_n_eigen_signals(
    samples: Sequence[CPSample],
    n_eigen: int,
    window_lengths: Sequence[int],
    margin: float,
    signals_by_window: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    """
    Evaluate all candidate LAD window lengths for one `n_eigen`.
    """

    n_eigen = int(n_eigen)
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
            )
            window_predicted_change_points.append(pred_cps)
            window_f1_scores.append(f1_score(sample.true_change_points, pred_cps, margin))
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
        "n_eigen": n_eigen,
        "ranking_rule": "top_score",
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


def grid_search_lad(
    samples: Sequence[CPSample],
    window_lengths: Sequence[int],
    n_eigens: Sequence[int],
    margin: float,
    n_jobs: int = 1,
    backend: str = "loky",
    verbose: int = 10,
    outdir: str | Path | None = None,
    normalize: bool = False,
    save_signals: bool = False,
    signals_outdir: str | Path | None = None,
    selection_metric: str = "f1",
    top: bool = True,
    difference: bool = True,
    second_window_scale: float | None = 2.0,
) -> dict[str, Any]:
    """
    Run a parallel grid-search over all LAD `(n_eigen, window_length)` pairs.
    """

    window_lengths = np.asarray(window_lengths, dtype=int)
    n_eigens = np.asarray(n_eigens, dtype=int)

    if len(window_lengths) == 0:
        raise ValueError("window_lengths must contain at least one value.")
    if len(n_eigens) == 0:
        raise ValueError("n_eigens must contain at least one value.")
    if np.any(window_lengths <= 0):
        raise ValueError("All window_lengths must be strictly positive.")
    if np.any(n_eigens <= 0):
        raise ValueError("All n_eigens must be strictly positive.")
    if selection_metric not in {"f1", "hausdorff"}:
        raise ValueError("selection_metric must be either 'f1' or 'hausdorff'.")
    if second_window_scale is not None and float(second_window_scale) <= 1.0:
        raise ValueError("second_window_scale must be greater than 1.0 or None.")

    t0 = time.time()
    t_signal_phase_start = time.time()
    signal_tasks = [
        delayed(compute_and_store_signals_for_n_eigen)(
            samples=samples,
            n_eigen=int(n_eigen),
            window_lengths=window_lengths,
            normalize=normalize,
            save_signals=save_signals,
            signals_outdir=signals_outdir,
            top=top,
            difference=difference,
            second_window_scale=second_window_scale,
        )
        for n_eigen in n_eigens
    ]
    backend_used = backend
    try:
        n_eigen_signal_results = Parallel(
            n_jobs=n_jobs,
            backend=backend,
            verbose=verbose,
        )(signal_tasks)
    except PermissionError:
        if backend != "loky":
            raise
        backend_used = "threading"
        n_eigen_signal_results = Parallel(
            n_jobs=n_jobs,
            backend=backend_used,
            verbose=verbose,
        )(signal_tasks)
    signal_generation_phase_seconds = time.time() - t_signal_phase_start

    t_detection_phase_start = time.time()
    n_eigen_results: list[dict[str, Any]] = []
    score_array = np.empty((len(n_eigens), len(window_lengths)), dtype=float)
    score_array[:] = np.nan
    hausdorff_array = np.empty((len(n_eigens), len(window_lengths)), dtype=float)
    hausdorff_array[:] = np.nan

    results_by_n_eigen: dict[int, dict[str, Any]] = {}
    for i, signal_bundle in enumerate(n_eigen_signal_results):
        result = evaluate_precomputed_n_eigen_signals(
            samples=samples,
            n_eigen=int(signal_bundle["n_eigen"]),
            window_lengths=window_lengths,
            margin=margin,
            signals_by_window=signal_bundle["signals_by_window"],
        )
        n_eigen_results.append(result)
        score_array[i, :] = result["score_array"]
        hausdorff_array[i, :] = np.array(
            [
                result["mean_hausdorff_per_window_length"][int(window_length)]
                for window_length in window_lengths
            ],
            dtype=float,
        )
        results_by_n_eigen[int(result["n_eigen"])] = result
    detection_metrics_phase_seconds = time.time() - t_detection_phase_start
    elapsed = time.time() - t0

    if selection_metric == "f1":
        selection_array = score_array
        if np.all(np.isnan(selection_array)):
            best_index = None
            best_n_eigen = None
            best_window_length = None
            best_score = math.nan
            best_f1 = math.nan
            best_hausdorff = math.nan
        else:
            best_flat = int(np.nanargmax(selection_array))
            best_index = np.unravel_index(best_flat, selection_array.shape)
            best_n_eigen = int(n_eigens[best_index[0]])
            best_window_length = int(window_lengths[best_index[1]])
            best_score = float(selection_array[best_index])
            best_f1 = float(score_array[best_index])
            best_hausdorff = float(hausdorff_array[best_index])
    else:
        selection_array = hausdorff_array.copy()
        if np.all(np.isnan(selection_array)):
            best_index = None
            best_n_eigen = None
            best_window_length = None
            best_score = math.nan
            best_f1 = math.nan
            best_hausdorff = math.nan
        else:
            selection_array[np.isnan(selection_array)] = math.inf
            if np.all(np.isinf(selection_array)):
                best_index = None
                best_n_eigen = None
                best_window_length = None
                best_score = math.nan
                best_f1 = math.nan
                best_hausdorff = math.nan
            else:
                best_flat = int(np.argmin(selection_array))
                best_index = np.unravel_index(best_flat, selection_array.shape)
                best_n_eigen = int(n_eigens[best_index[0]])
                best_window_length = int(window_lengths[best_index[1]])
                best_score = float(hausdorff_array[best_index])
                best_f1 = float(score_array[best_index])
                best_hausdorff = float(hausdorff_array[best_index])

    best_second_window_length = (
        None
        if best_window_length is None
        else resolve_second_window_length(best_window_length, second_window_scale)
    )

    summary = {
        "window_lengths": window_lengths,
        "n_eigens": n_eigens,
        "margin": float(margin),
        "normalize": bool(normalize),
        "ranking_rule": "top_score",
        "top": bool(top),
        "difference": bool(difference),
        "second_window_scale": second_window_scale,
        "best_second_window_length": best_second_window_length,
        "backend_requested": backend,
        "backend_used": backend_used,
        "num_samples": len(samples),
        "num_n_eigen_jobs": len(n_eigens),
        "num_parameter_pairs": len(n_eigens) * len(window_lengths),
        "save_signals": save_signals,
        "signals_outdir": str(signals_outdir) if signals_outdir is not None else None,
        "selection_metric": selection_metric,
        "selection_array": score_array if selection_metric == "f1" else hausdorff_array,
        "score_array": score_array,
        "f1_array": score_array,
        "hausdorff_array": hausdorff_array,
        "signal_generation_phase_seconds": signal_generation_phase_seconds,
        "detection_metrics_phase_seconds": detection_metrics_phase_seconds,
        "n_eigen_results": n_eigen_results,
        "results_by_n_eigen": results_by_n_eigen,
        "best_index": best_index,
        "best_n_eigen": best_n_eigen,
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
        with open(outdir / "gridsearch_results.pkl", "wb") as handle:
            pickle.dump(summary, handle)

    return summary
