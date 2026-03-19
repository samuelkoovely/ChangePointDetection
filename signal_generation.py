from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
import pickle
import time

import numpy as np

import compute_S_rate


DEFAULT_WINDOWS = [1, 5, 10, 25]
DEFAULT_SAMPLE_FRACTION = 0.1


# -----------------------------------------------------------------------------
# Loading helpers
# -----------------------------------------------------------------------------

def load_pickled_network(pkl_path: str | Path, index: int = 0, key: str = "tnet"):
    """
    Load one temporal network from a pickle file like the ones used in the
    current project.

    Parameters
    ----------
    pkl_path:
        Path to the pickle file.
    index:
        Index of the entry to load when the pickle stores a list-like object.
    key:
        Key used to retrieve the network inside the selected entry.
    """
    pkl_path = Path(pkl_path)
    with open(pkl_path, "rb") as handle:
        obj = pickle.load(handle)
    return obj[index][key]


# -----------------------------------------------------------------------------
# Core preprocessing helpers
# -----------------------------------------------------------------------------

def ensure_laplacians(net: Any) -> None:
    """
    Ensure the network has Laplacian matrices available.
    """
    if not hasattr(net, "L"):
        net.compute_laplacian_matrices(
            t_start=net.times[0],
            t_stop=net.times[-1],
            random_walk=False,
        )



def compute_inter_transition_matrices_for_lambda(net: Any, lamda: float) -> None:
    """
    Compute the inter-transition matrices for one lambda value.

    This is the expensive lambda-dependent preprocessing step that should be
    reused across all tested window lengths for the same lambda.
    """
    ensure_laplacians(net)
    net.compute_inter_transition_matrices(
        lamda=lamda,
        t_start=net.times[0],
        t_stop=net.times[-1],
        dense_expm=False,
        use_sparse_stoch=False,
        random_walk=False,
    )



def get_sampled_window_indices_and_times(
    net: Any,
    window: float,
    sample_fraction: float = DEFAULT_SAMPLE_FRACTION,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select valid window start indices by sampling approximately a fixed fraction
    of window positions uniformly in time.

    Returns
    -------
    k_samples:
        Integer indices of sampled windows.
    t_samples:
        Times corresponding to the sampled window starts.
    """
    considered_times = net.times[net.times < net.times[-1] - window]
    M = len(considered_times)

    if M <= 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    m = max(1, int(np.ceil(sample_fraction * M)))

    t_targets = np.linspace(float(considered_times[0]), float(considered_times[-1]), m)
    k_samples = np.searchsorted(considered_times, t_targets, side="left")
    k_samples = np.clip(k_samples, 0, M - 1)
    k_samples = np.unique(k_samples).astype(int)

    t_samples = net.times[k_samples]
    return k_samples, t_samples



def compute_window_entropy_signal(
    net: Any,
    lamda: float,
    window: float,
    sample_fraction: float = DEFAULT_SAMPLE_FRACTION,
    p0: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Compute the entropy signal for one (lambda, window) pair.

    Assumes the lambda-dependent inter-transition matrices have already been
    computed, so this function can be called repeatedly for multiple window
    values after a single preprocessing pass.

    Returns a dictionary containing the signal and its associated times.
    """
    k_samples, t_samples = get_sampled_window_indices_and_times(
        net=net,
        window=window,
        sample_fraction=sample_fraction,
    )

    if len(k_samples) == 0:
        return {
            "lamda": float(lamda),
            "window": float(window),
            "k_samples": np.array([], dtype=int),
            "t_samples": np.array([], dtype=float),
            "signal": np.array([], dtype=float),
        }

    if p0 is None:
        p0 = np.ones(net.num_nodes, dtype=float) / net.num_nodes
    else:
        p0 = np.asarray(p0, dtype=float)

    S_arr = np.empty(len(k_samples), dtype=float)
    k_to_pos = {int(k): i for i, k in enumerate(k_samples)}

    def k_to_idx(k: int) -> int:
        return k_to_pos[int(k)]

    on_T = compute_S_rate.make_on_window_matrix_entropy_callback_prealloc(p0, S_arr, k_to_idx)

    net.compute_transition_matrices_sliding_timewindow(
        lamda=lamda,
        reverse_time=False,
        window_timelength=window,
        save_intermediate=False,
        on_window_matrix=on_T,
        force_csr=True,
        k_samples=k_samples,
    )

    return {
        "lamda": float(lamda),
        "window": float(window),
        "k_samples": k_samples,
        "t_samples": np.asarray(t_samples, dtype=float),
        "signal": S_arr,
    }



def compute_signals_for_lambda(
    net: Any,
    lamda: float,
    windows: Sequence[float] = DEFAULT_WINDOWS,
    sample_fraction: float = DEFAULT_SAMPLE_FRACTION,
    p0: np.ndarray | None = None,
) -> dict[float, dict[str, Any]]:
    """
    Compute entropy signals for all windows associated with one lambda value.

    This is the main reusable entry point for the future grid-search code:
    the expensive lambda-specific preprocessing is performed once, then the
    signal is computed for every requested window.
    """
    results, _ = _compute_signals_for_lambda_impl(
        net=net,
        lamda=lamda,
        windows=windows,
        sample_fraction=sample_fraction,
        p0=p0,
        measure_time=False,
    )
    return results


def compute_signals_for_lambda_timed(
    net: Any,
    lamda: float,
    windows: Sequence[float] = DEFAULT_WINDOWS,
    sample_fraction: float = DEFAULT_SAMPLE_FRACTION,
    p0: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Compute entropy signals for one lambda and return a timing breakdown.

    The returned timing separates the shared lambda preprocessing from the
    window-specific signal computation. An attributed per-window timing is also
    provided by spreading the shared preprocessing cost uniformly across the
    requested windows.
    """
    results, timing = _compute_signals_for_lambda_impl(
        net=net,
        lamda=lamda,
        windows=windows,
        sample_fraction=sample_fraction,
        p0=p0,
        measure_time=True,
    )
    return {
        "signals_by_window": results,
        "timing": timing,
    }


def _compute_signals_for_lambda_impl(
    net: Any,
    lamda: float,
    windows: Sequence[float],
    sample_fraction: float,
    p0: np.ndarray | None,
    measure_time: bool,
) -> tuple[dict[float, dict[str, Any]], dict[str, Any] | None]:
    """
    Shared implementation for signal generation with optional timing capture.
    """
    if measure_time:
        t_preprocessing_start = time.perf_counter()
    compute_inter_transition_matrices_for_lambda(net, lamda)
    if measure_time:
        preprocessing_seconds = time.perf_counter() - t_preprocessing_start

    results: dict[float, dict[str, Any]] = {}
    window_signal_seconds: dict[float, float] = {}
    for window in windows:
        window_key = float(window)
        if measure_time:
            t_window_start = time.perf_counter()
        results[window_key] = compute_window_entropy_signal(
            net=net,
            lamda=lamda,
            window=window,
            sample_fraction=sample_fraction,
            p0=p0,
        )
        if measure_time:
            window_signal_seconds[window_key] = time.perf_counter() - t_window_start

    if not measure_time:
        return results, None

    num_windows = len(results)
    shared_preprocessing_seconds = preprocessing_seconds / num_windows if num_windows > 0 else 0.0
    attributed_per_window_seconds = {
        window: window_signal_seconds[window] + shared_preprocessing_seconds
        for window in results
    }

    return results, {
        "preprocessing_seconds": preprocessing_seconds,
        "window_signal_seconds": window_signal_seconds,
        "attributed_per_window_seconds": attributed_per_window_seconds,
        "total_seconds": preprocessing_seconds + float(sum(window_signal_seconds.values())),
    }


# -----------------------------------------------------------------------------
# Optional persistence helpers
# -----------------------------------------------------------------------------

def get_signal_result_filename(lamda: float, window: float, suffix: str = ".pkl") -> str:
    """
    Return the canonical filename for one saved entropy signal.
    """
    return f"signal_lamda_{float(lamda):.11f}_window_{float(window):g}{suffix}"


def get_signal_result_path(outdir: str | Path, lamda: float, window: float) -> Path:
    """
    Return the full path for one saved entropy signal inside a sample folder.
    """
    return Path(outdir) / get_signal_result_filename(lamda=lamda, window=window)


def load_signal_result(outdir: str | Path, lamda: float, window: float) -> dict[str, Any]:
    """
    Load one saved entropy signal from a sample folder.
    """
    filepath = get_signal_result_path(outdir=outdir, lamda=lamda, window=window)
    with open(filepath, "rb") as handle:
        return pickle.load(handle)


def save_signal_result(result: dict[str, Any], outdir: str | Path) -> Path:
    """
    Save one signal result dictionary to disk inside a sample folder.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    lamda = float(result["lamda"])
    window = float(result["window"])
    outfile = get_signal_result_path(outdir=outdir, lamda=lamda, window=window)
    with open(outfile, "wb") as f:
        pickle.dump(result, f)
    return outfile



def save_signals_for_lambda(
    results_by_window: dict[float, dict[str, Any]],
    base: str | Path,
) -> None:
    """
    Save all window-specific signal results for one lambda in a single folder.
    """
    base = Path(base)
    for result in results_by_window.values():
        save_signal_result(result, base)
