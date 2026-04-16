"""
Grid-search training pipeline for snapshot change-point detection using
Laplacian spectrum similarity.

This mirrors the structure of the existing snapshot entropy grid-search:
- precompute signals for one parameter family
- run ruptures on each signal
- score predictions with F1 and Hausdorff distance

Here the parameter pair is `(n_eigen, window_length)` instead of
`(lamda, window)`.
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

from benchmark_methods import (
    compute_laplacian_signatures as benchmark_compute_laplacian_signatures,
    compute_similarity_signal_from_signatures as benchmark_compute_similarity_signal_from_signatures,
    laplacian_spectrum_similarity as benchmark_laplacian_spectrum_similarity,
)
from evaluation_metrics import f1_score, hausdorff_distance


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
        Input temporal network used to generate the signal.
    true_change_points:
        Ground-truth change-point indices in snapshot space.
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
# Signal generation helpers
# -----------------------------------------------------------------------------


def compute_laplacian_signatures(
    data: Any,
    n_eigen: int,
    normalize: bool = False,
) -> np.ndarray:
    """
    Compute the normalized Laplacian signatures for every snapshot using the
    shared benchmark implementation.
    """

    return benchmark_compute_laplacian_signatures(
        data=data,
        n_eigen=n_eigen,
        normalize=normalize,
    )


def compute_similarity_signal_from_signatures(
    laplacian_signatures: np.ndarray,
    window_length: int,
    n_eigen: int,
) -> dict[str, Any]:
    """
    Compute the Laplacian similarity signal from precomputed signatures using
    the shared benchmark implementation.
    """

    return benchmark_compute_similarity_signal_from_signatures(
        laplacian_signatures=laplacian_signatures,
        window_length=window_length,
        n_eigen=n_eigen,
    )


def laplacian_spectrum_similarity(
    data: Any,
    window_length: int,
    normalize: bool = False,
    n_eigen: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Laplacian anomaly-detection statistic of Huang et al. (2020).

    Returns
    -------
    signal:
        Similarity score for each valid snapshot.
    snapshot_indices:
        Snapshot indices corresponding to the entries of `signal`.
    """

    return benchmark_laplacian_spectrum_similarity(
        data=data,
        window_length=window_length,
        normalize=normalize,
        n_eigen=n_eigen,
    )


# -----------------------------------------------------------------------------
# Optional persistence helpers
# -----------------------------------------------------------------------------


def get_signal_result_filename(
    n_eigen: int,
    window_length: int,
    suffix: str = ".pkl",
) -> str:
    """
    Return the canonical filename for one saved Laplacian-similarity signal.
    """

    return (
        f"signal_n_eigen_{int(n_eigen)}"
        f"_window_length_{int(window_length)}{suffix}"
    )


def save_signal_result(result: dict[str, Any], outdir: str | Path) -> Path:
    """
    Save one signal result dictionary to disk inside a sample folder.
    """

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    outfile = outdir / get_signal_result_filename(
        n_eigen=int(result["n_eigen"]),
        window_length=int(result["window_length"]),
    )
    with open(outfile, "wb") as f:
        pickle.dump(result, f)

    return outfile


# -----------------------------------------------------------------------------
# Detection helper
# -----------------------------------------------------------------------------


def detect_change_points_from_signal(
    signal: np.ndarray,
    snapshot_indices: Sequence[float],
    n_bkps: int,
    kernel: str = "linear",
    stopping_rule: str = "n_bkps",
    penalty: float | None = None,
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

    if stopping_rule not in {"n_bkps", "penalty"}:
        raise ValueError("stopping_rule must be either 'n_bkps' or 'penalty'.")

    if signal.shape[0] == 0:
        return []

    try:
        algo = rpt.KernelCPD(kernel=kernel).fit(signal)
        if stopping_rule == "n_bkps":
            if n_bkps <= 0 or signal.shape[0] <= n_bkps:
                return []
            breakpoint_indices = np.asarray(algo.predict(n_bkps=n_bkps), dtype=int)
        else:
            if penalty is None:
                raise ValueError("penalty must be provided when stopping_rule='penalty'.")
            breakpoint_indices = np.asarray(algo.predict(pen=float(penalty)), dtype=int)
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


# -----------------------------------------------------------------------------
# Signal generation phase
# -----------------------------------------------------------------------------


def compute_and_store_signals_for_n_eigen(
    samples: Sequence[CPSample],
    n_eigen: int,
    window_lengths: Sequence[int],
    normalize: bool = False,
    save_signals: bool = False,
    signals_outdir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Compute and optionally store all signals associated with one `n_eigen`.
    """

    n_eigen = int(n_eigen)
    window_lengths = [int(window_length) for window_length in window_lengths]
    signals_by_window = {int(window_length): [] for window_length in window_lengths}
    sample_names = [sample.name for sample in samples]

    for sample_idx, sample in enumerate(samples):
        laplacian_signatures = compute_laplacian_signatures(
            data=sample.data,
            n_eigen=n_eigen,
            normalize=normalize,
        )

        for window_length in window_lengths:
            signal_result = compute_similarity_signal_from_signatures(
                laplacian_signatures=laplacian_signatures,
                window_length=window_length,
                n_eigen=n_eigen,
            )
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
    }


# -----------------------------------------------------------------------------
# Detection and metrics phase
# -----------------------------------------------------------------------------


def evaluate_precomputed_n_eigen_signals(
    samples: Sequence[CPSample],
    n_eigen: int,
    window_lengths: Sequence[int],
    margin: float,
    signals_by_window: dict[int, list[dict[str, Any]]],
    kernel: str = "linear",
    stopping_rule: str = "n_bkps",
    penalty: float | None = None,
) -> dict[str, Any]:
    """
    Evaluate all candidate window lengths for one `n_eigen` using precomputed
    signals.
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
                kernel=kernel,
                stopping_rule=stopping_rule,
                penalty=penalty,
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
        "stopping_rule": stopping_rule,
        "penalty": None if penalty is None else float(penalty),
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


# -----------------------------------------------------------------------------
# Grid-search core
# -----------------------------------------------------------------------------


def grid_search_laplacian_similarity(
    samples: Sequence[CPSample],
    window_lengths: Sequence[int],
    n_eigens: Sequence[int],
    margin: float,
    n_jobs: int = 1,
    backend: str = "loky",
    verbose: int = 10,
    outdir: str | Path | None = None,
    kernel: str = "linear",
    normalize: bool = False,
    save_signals: bool = False,
    signals_outdir: str | Path | None = None,
    selection_metric: str = "f1",
    stopping_rule: str = "n_bkps",
    penalty: float | None = None,
    penalties: Sequence[float] | None = None,
) -> dict[str, Any]:
    """
    Run a parallel grid-search over all `(n_eigen, window_length)` pairs.
    """

    window_lengths = np.asarray(window_lengths, dtype=int)
    n_eigens = np.asarray(n_eigens, dtype=int)

    if len(window_lengths) == 0:
        raise ValueError("window_lengths must contain at least one value.")
    if len(n_eigens) == 0:
        raise ValueError("n_eigens must contain at least one value.")
    if np.any(window_lengths <= 0):
        raise ValueError("All window_lengths must be strictly positive.")
    if np.any(n_eigens < 3):
        raise ValueError(
            "All n_eigens must be at least 3. "
            "n_eigen=2 is excluded because the normalized Laplacian signature "
            "degenerates for connected graphs."
        )
    if selection_metric not in {"f1", "hausdorff"}:
        raise ValueError("selection_metric must be either 'f1' or 'hausdorff'.")
    if stopping_rule not in {"n_bkps", "penalty"}:
        raise ValueError("stopping_rule must be either 'n_bkps' or 'penalty'.")
    if stopping_rule == "penalty":
        if penalties is None:
            if penalty is None:
                raise ValueError(
                    "penalties or penalty must be provided when stopping_rule='penalty'."
                )
            penalties = [penalty]
        penalties = np.asarray(penalties, dtype=float)
        if penalties.size == 0:
            raise ValueError("penalties must contain at least one value.")
        if np.any(penalties <= 0):
            raise ValueError("All penalties must be strictly positive.")
    else:
        penalties = None

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
    if penalties is None:
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
                kernel=kernel,
                stopping_rule=stopping_rule,
                penalty=None,
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
    else:
        score_array = np.empty(
            (len(n_eigens), len(window_lengths), len(penalties)),
            dtype=float,
        )
        score_array[:] = np.nan
        hausdorff_array = np.empty(
            (len(n_eigens), len(window_lengths), len(penalties)),
            dtype=float,
        )
        hausdorff_array[:] = np.nan

        results_by_n_eigen = {int(n_eigen): {} for n_eigen in n_eigens}
        for i, signal_bundle in enumerate(n_eigen_signal_results):
            n_eigen = int(signal_bundle["n_eigen"])
            for penalty_index, penalty_value in enumerate(penalties):
                result = evaluate_precomputed_n_eigen_signals(
                    samples=samples,
                    n_eigen=n_eigen,
                    window_lengths=window_lengths,
                    margin=margin,
                    signals_by_window=signal_bundle["signals_by_window"],
                    kernel=kernel,
                    stopping_rule=stopping_rule,
                    penalty=float(penalty_value),
                )
                n_eigen_results.append(result)
                score_array[i, :, penalty_index] = result["score_array"]
                hausdorff_array[i, :, penalty_index] = np.array(
                    [
                        result["mean_hausdorff_per_window_length"][int(window_length)]
                        for window_length in window_lengths
                    ],
                    dtype=float,
                )
                results_by_n_eigen[n_eigen][float(penalty_value)] = result
    detection_metrics_phase_seconds = time.time() - t_detection_phase_start
    elapsed = time.time() - t0

    if selection_metric == "f1":
        selection_array = score_array
        if np.all(np.isnan(selection_array)):
            best_index = None
            best_n_eigen = None
            best_window_length = None
            best_penalty = None
            best_score = math.nan
            best_f1 = math.nan
            best_hausdorff = math.nan
        else:
            best_flat = int(np.nanargmax(selection_array))
            best_index = np.unravel_index(best_flat, selection_array.shape)
            best_n_eigen = int(n_eigens[best_index[0]])
            best_window_length = int(window_lengths[best_index[1]])
            best_penalty = (
                None if penalties is None else float(penalties[best_index[2]])
            )
            best_score = float(selection_array[best_index])
            best_f1 = float(score_array[best_index])
            best_hausdorff = float(hausdorff_array[best_index])
    else:
        selection_array = hausdorff_array.copy()
        if np.all(np.isnan(selection_array)):
            best_index = None
            best_n_eigen = None
            best_window_length = None
            best_penalty = None
            best_score = math.nan
            best_f1 = math.nan
            best_hausdorff = math.nan
        else:
            selection_array[np.isnan(selection_array)] = math.inf
            if np.all(np.isinf(selection_array)):
                best_index = None
                best_n_eigen = None
                best_window_length = None
                best_penalty = None
                best_score = math.nan
                best_f1 = math.nan
                best_hausdorff = math.nan
            else:
                best_flat = int(np.argmin(selection_array))
                best_index = np.unravel_index(best_flat, selection_array.shape)
                best_n_eigen = int(n_eigens[best_index[0]])
                best_window_length = int(window_lengths[best_index[1]])
                best_penalty = (
                    None if penalties is None else float(penalties[best_index[2]])
                )
                best_score = float(hausdorff_array[best_index])
                best_f1 = float(score_array[best_index])
                best_hausdorff = float(hausdorff_array[best_index])

    summary = {
        "window_lengths": window_lengths,
        "n_eigens": n_eigens,
        "penalties": penalties,
        "margin": float(margin),
        "kernel": kernel,
        "normalize": bool(normalize),
        "stopping_rule": stopping_rule,
        "penalty": best_penalty,
        "best_penalty": best_penalty,
        "backend_requested": backend,
        "backend_used": backend_used,
        "num_samples": len(samples),
        "num_n_eigen_jobs": len(n_eigens),
        "num_penalty_jobs": 0 if penalties is None else len(penalties),
        "num_parameter_pairs": len(n_eigens) * len(window_lengths) * (
            1 if penalties is None else len(penalties)
        ),
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
        with open(outdir / "gridsearch_results.pkl", "wb") as f:
            pickle.dump(summary, f)

    return summary


# -----------------------------------------------------------------------------
# Example usage on the block2activities snapshot dataset
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    with open("data/block2activities_snapshots.pkl", "rb") as f:
        dataset = pickle.load(f)

    first_net = dataset[0]["tnet"]
    num_laplacians = max(1, len(first_net.times) - 1)
    max_window_length = max(1, min(8, num_laplacians - 1))
    max_n_eigen = max(1, int(first_net.num_nodes) - 1)

    # `window_length` is expressed in number of snapshots, not in time units.
    window_lengths = list(range(1, max_window_length + 1))
    n_eigens = [k for k in [3, 4, 6, 8, 10] if k <= max_n_eigen]
    if not n_eigens:
        raise ValueError("No valid n_eigen >= 3 is available for this dataset.")

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

    # One direct call on the first temporal network, matching the original demo.
    demo_window_length = min(5, max_window_length)
    demo_n_eigen = n_eigens[min(2, len(n_eigens) - 1)]
    demo_signal, demo_indices = laplacian_spectrum_similarity(
        first_net,
        window_length=demo_window_length,
        n_eigen=demo_n_eigen,
    )
    print("Demo signal length:", len(demo_signal))
    print("Demo index range:", demo_indices[:3], "...", demo_indices[-3:])

    summary = grid_search_laplacian_similarity(
        samples=training_samples,
        window_lengths=window_lengths,
        n_eigens=n_eigens,
        margin=margin,
        n_jobs=n_jobs,
        outdir="./gridsearch_results/block2activities_snapshots_laplacians",
        kernel="linear",
        normalize=False,
        save_signals=True,
        signals_outdir="./gridsearch_results/block2activities_snapshots_laplacians/signals",
        selection_metric="hausdorff",
        stopping_rule="n_bkps",
    )

    print("Number of samples:", len(training_samples))
    print("Score array shape:", summary["score_array"].shape)
    print("Backend used:", summary["backend_used"])
    print("Selection metric:", summary["selection_metric"])
    print("Stopping rule:", summary["stopping_rule"])
    print("Penalty:", summary["penalty"])
    print("Best n_eigen:", summary["best_n_eigen"])
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
