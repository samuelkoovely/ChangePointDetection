from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import pickle

import numpy as np

import compute_S_rate
from matrix_segment_tree import OrderedMatrixProductSegmentTree


DEFAULT_WINDOWS = [1, 5, 10, 25]
DEFAULT_SAMPLE_FRACTION = 0.1
SUPPORTED_WINDOW_BACKENDS = {"auto", "naive", "segment_tree"}


@dataclass(frozen=True)
class WindowSamplingPlan:
    """
    Precomputed sampling/index information for one window length on one sample.
    """
    window: float
    k_samples: np.ndarray
    t_samples: np.ndarray
    window_ends: np.ndarray


@dataclass
class PreparedSignalSample:
    """
    Per-sample state reused across all lambdas for signal generation.

    Intended ownership model: one worker owns one prepared sample, then
    processes a block of lambdas sequentially so it can reuse sample-level
    preprocessing and avoid shared mutable state.
    """
    net: Any
    windows: tuple[float, ...]
    sample_fraction: float
    p0: np.ndarray
    window_plans: dict[float, WindowSamplingPlan]


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
    if not hasattr(net, "laplacians"):
        net.compute_laplacian_matrices(
            t_start=net.times[0],
            t_stop=net.times[-1],
            random_walk=False,
        )



def compute_inter_transition_matrices_for_lambda(
    net: Any,
    lamda: float,
    use_linear_approx: bool = False,
    lin_t_s: int = 10,
) -> None:
    """
    Compute the inter-transition matrices for one lambda value.

    This is the expensive lambda-dependent preprocessing step that should be
    reused across all tested window lengths for the same lambda.

    If `use_linear_approx=True`, compute the linearized inter-transition
    matrices with the provided `lin_t_s` value instead of the full matrix
    exponential based matrices.
    """
    ensure_laplacians(net)
    if use_linear_approx:
        net.compute_lin_inter_transition_matrices(
            lamda=lamda,
            t_start=net.times[0],
            t_stop=net.times[-1],
            verbose=False,
            t_s=lin_t_s,
            use_sparse_stoch=False,
        )
    else:
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


def _normalize_windows(windows: Sequence[float]) -> tuple[float, ...]:
    """
    Normalize a window list to ordered unique float values.
    """
    return tuple(dict.fromkeys(float(window) for window in windows))


def _resolve_window_backend(window_backend: str, num_windows: int) -> str:
    """
    Resolve the requested window backend.
    """
    if window_backend not in SUPPORTED_WINDOW_BACKENDS:
        raise ValueError(
            f"window_backend must be one of {sorted(SUPPORTED_WINDOW_BACKENDS)}, "
            f"got {window_backend!r}."
        )
    if window_backend == "auto":
        return "segment_tree" if num_windows > 1 else "naive"
    return window_backend


def _get_uniform_p0(net: Any, p0: np.ndarray | None) -> np.ndarray:
    """
    Return the probability vector used for entropy computations.
    """
    if p0 is None:
        return np.ones(net.num_nodes, dtype=float) / net.num_nodes
    return np.asarray(p0, dtype=float)


def _get_window_ends_vectorized(
    net: Any,
    window: float,
    n_windows: int,
) -> np.ndarray:
    """
    Return window-end indices for all valid window starts.

    Falls back to a local vectorized implementation if the temporal-network
    object does not expose `_get_window_ends_vectorized`.
    """
    if n_windows <= 0:
        return np.array([], dtype=np.int64)

    if hasattr(net, "_get_window_ends_vectorized"):
        return np.asarray(
            net._get_window_ends_vectorized(window, n_windows=n_windows),
            dtype=np.int64,
        )

    times_arr = np.asarray(net.times, dtype=np.float64)
    target_times = times_arr[:n_windows] + float(window)
    window_ends = np.searchsorted(times_arr, target_times, side="right") - 1
    np.clip(window_ends, 0, len(times_arr) - 1, out=window_ends)
    return window_ends.astype(np.int64)


def _empty_signal_result(lamda: float, window: float) -> dict[str, Any]:
    """
    Return an empty signal result for a window with no valid starts.
    """
    return {
        "lamda": float(lamda),
        "window": float(window),
        "k_samples": np.array([], dtype=int),
        "t_samples": np.array([], dtype=float),
        "signal": np.array([], dtype=float),
    }


def prepare_signal_sample(
    net: Any,
    windows: Sequence[float] = DEFAULT_WINDOWS,
    sample_fraction: float = DEFAULT_SAMPLE_FRACTION,
    p0: np.ndarray | None = None,
) -> PreparedSignalSample:
    """
    Prepare per-sample state reused across all lambdas.

    This is the main entry point for the hierarchical execution scheme:
    sample-level work is done once, then one worker can process many lambdas for
    that sample while reusing the prepared window plans and Laplacians.
    """
    ensure_laplacians(net)

    windows_tuple = _normalize_windows(windows)
    p0_arr = _get_uniform_p0(net, p0)
    window_plans: dict[float, WindowSamplingPlan] = {}

    for window in windows_tuple:
        k_samples, t_samples = get_sampled_window_indices_and_times(
            net=net,
            window=window,
            sample_fraction=sample_fraction,
        )
        considered_times = net.times[net.times < net.times[-1] - window]
        window_ends = _get_window_ends_vectorized(
            net=net,
            window=window,
            n_windows=len(considered_times),
        )
        window_plans[window] = WindowSamplingPlan(
            window=float(window),
            k_samples=np.asarray(k_samples, dtype=int),
            t_samples=np.asarray(t_samples, dtype=float),
            window_ends=window_ends,
        )

    return PreparedSignalSample(
        net=net,
        windows=windows_tuple,
        sample_fraction=float(sample_fraction),
        p0=p0_arr,
        window_plans=window_plans,
    )


def _get_inter_transition_matrices(
    net: Any,
    lamda: float,
    use_linear_approx: bool = False,
    lin_t_s: int = 10,
):
    """
    Return the precomputed inter-transition matrices for the requested mode.
    """
    if use_linear_approx:
        return net.inter_T_lin[lamda][lin_t_s]
    return net.inter_T[lamda]


def build_segment_tree_for_lambda(
    net: Any,
    lamda: float,
    use_linear_approx: bool = False,
    lin_t_s: int = 10,
    force_csr: bool = True,
) -> OrderedMatrixProductSegmentTree:
    """
    Build one ordered-product segment tree for all inter-transition matrices of
    a fixed `(sample, lambda, approx_kind)` combination.
    """
    source_mats = _get_inter_transition_matrices(
        net=net,
        lamda=lamda,
        use_linear_approx=use_linear_approx,
        lin_t_s=lin_t_s,
    )
    return OrderedMatrixProductSegmentTree(source_mats, force_csr=force_csr)


def _compute_window_entropy_signal_naive(
    prepared: PreparedSignalSample,
    lamda: float,
    plan: WindowSamplingPlan,
    use_linear_approx: bool = False,
    lin_t_s: int = 10,
) -> dict[str, Any]:
    """
    Compute one window entropy signal via the existing sliding-window method.
    """
    if len(plan.k_samples) == 0:
        return _empty_signal_result(lamda=lamda, window=plan.window)

    S_arr = np.empty(len(plan.k_samples), dtype=float)
    k_to_pos = {int(k): i for i, k in enumerate(plan.k_samples)}

    def k_to_idx(k: int) -> int:
        return k_to_pos[int(k)]

    on_T = compute_S_rate.make_on_window_matrix_entropy_callback_prealloc(
        prepared.p0,
        S_arr,
        k_to_idx,
    )

    prepared.net.compute_transition_matrices_sliding_timewindow(
        lamda=lamda,
        reverse_time=False,
        window_timelength=plan.window,
        save_intermediate=False,
        on_window_matrix=on_T,
        force_csr=True,
        use_linear_inter_T=use_linear_approx,
        lin_t_s=lin_t_s,
        k_samples=plan.k_samples,
    )

    return {
        "lamda": float(lamda),
        "window": float(plan.window),
        "k_samples": np.asarray(plan.k_samples, dtype=int),
        "t_samples": np.asarray(plan.t_samples, dtype=float),
        "signal": S_arr,
    }


def _compute_window_entropy_signal_from_tree(
    tree: OrderedMatrixProductSegmentTree,
    prepared: PreparedSignalSample,
    lamda: float,
    plan: WindowSamplingPlan,
) -> dict[str, Any]:
    """
    Compute one window entropy signal by querying a prebuilt segment tree.
    """
    if len(plan.k_samples) == 0:
        return _empty_signal_result(lamda=lamda, window=plan.window)

    S_arr = np.empty(len(plan.k_samples), dtype=float)
    for idx, k in enumerate(plan.k_samples):
        Tk_window = tree.query(int(k), int(plan.window_ends[int(k)]))
        S_arr[idx] = compute_S_rate.conditional_entropy_of_T(Tk_window, prepared.p0)

    return {
        "lamda": float(lamda),
        "window": float(plan.window),
        "k_samples": np.asarray(plan.k_samples, dtype=int),
        "t_samples": np.asarray(plan.t_samples, dtype=float),
        "signal": S_arr,
    }


def compute_signals_for_lambda_prepared(
    prepared: PreparedSignalSample,
    lamda: float,
    use_linear_approx: bool = False,
    lin_t_s: int = 10,
    window_backend: str = "auto",
) -> dict[float, dict[str, Any]]:
    """
    Compute all window signals for one lambda from a prepared sample context.

    This is the lambda-level unit intended for hierarchical scheduling inside a
    worker that already owns one sample.
    """
    compute_inter_transition_matrices_for_lambda(
        prepared.net,
        lamda,
        use_linear_approx=use_linear_approx,
        lin_t_s=lin_t_s,
    )

    resolved_backend = _resolve_window_backend(window_backend, len(prepared.windows))
    results: dict[float, dict[str, Any]] = {}

    if resolved_backend == "segment_tree":
        tree = build_segment_tree_for_lambda(
            net=prepared.net,
            lamda=lamda,
            use_linear_approx=use_linear_approx,
            lin_t_s=lin_t_s,
            force_csr=True,
        )
        for window in prepared.windows:
            plan = prepared.window_plans[window]
            results[window] = _compute_window_entropy_signal_from_tree(
                tree=tree,
                prepared=prepared,
                lamda=lamda,
                plan=plan,
            )
    else:
        for window in prepared.windows:
            plan = prepared.window_plans[window]
            results[window] = _compute_window_entropy_signal_naive(
                prepared=prepared,
                lamda=lamda,
                plan=plan,
                use_linear_approx=use_linear_approx,
                lin_t_s=lin_t_s,
            )

    return results


def compute_signals_for_lambdas_prepared(
    prepared: PreparedSignalSample,
    lambdas: Sequence[float],
    use_linear_approx: bool = False,
    lin_t_s: int = 10,
    window_backend: str = "auto",
) -> dict[float, dict[float, dict[str, Any]]]:
    """
    Compute signals for a block of lambdas on one prepared sample.

    This is the natural worker payload for a hierarchical sample-owned
    execution model.
    """
    return {
        float(lamda): compute_signals_for_lambda_prepared(
            prepared=prepared,
            lamda=float(lamda),
            use_linear_approx=use_linear_approx,
            lin_t_s=lin_t_s,
            window_backend=window_backend,
        )
        for lamda in lambdas
    }



def compute_window_entropy_signal(
    net: Any,
    lamda: float,
    window: float,
    sample_fraction: float = DEFAULT_SAMPLE_FRACTION,
    p0: np.ndarray | None = None,
    use_linear_approx: bool = False,
    lin_t_s: int = 10,
    window_backend: str = "auto",
) -> dict[str, Any]:
    """
    Compute the entropy signal for one (lambda, window) pair.

    Assumes the lambda-dependent inter-transition matrices have already been
    computed, so this function can be called repeatedly for multiple window
    values after a single preprocessing pass.

    When `use_linear_approx=True`, window products are built from the linearized
    inter-transition matrices associated with `lin_t_s`.

    Returns a dictionary containing the signal and its associated times.
    """
    prepared = prepare_signal_sample(
        net=net,
        windows=[window],
        sample_fraction=sample_fraction,
        p0=p0,
    )
    compute_inter_transition_matrices_for_lambda(
        net,
        lamda,
        use_linear_approx=use_linear_approx,
        lin_t_s=lin_t_s,
    )

    resolved_backend = _resolve_window_backend(window_backend, num_windows=1)
    plan = prepared.window_plans[float(window)]

    if resolved_backend == "segment_tree":
        tree = build_segment_tree_for_lambda(
            net=net,
            lamda=lamda,
            use_linear_approx=use_linear_approx,
            lin_t_s=lin_t_s,
            force_csr=True,
        )
        return _compute_window_entropy_signal_from_tree(
            tree=tree,
            prepared=prepared,
            lamda=lamda,
            plan=plan,
        )

    return _compute_window_entropy_signal_naive(
        prepared=prepared,
        lamda=lamda,
        plan=plan,
        use_linear_approx=use_linear_approx,
        lin_t_s=lin_t_s,
    )



def compute_signals_for_lambda(
    net: Any,
    lamda: float,
    windows: Sequence[float] = DEFAULT_WINDOWS,
    sample_fraction: float = DEFAULT_SAMPLE_FRACTION,
    p0: np.ndarray | None = None,
    use_linear_approx: bool = False,
    lin_t_s: int = 10,
    window_backend: str = "auto",
) -> dict[float, dict[str, Any]]:
    """
    Compute entropy signals for all windows associated with one lambda value.

    This is the main reusable entry point for the future grid-search code:
    the expensive lambda-specific preprocessing is performed once, then the
    signal is computed for every requested window.

    Set `use_linear_approx=True` to use `compute_lin_inter_transition_matrices`
    instead of the full inter-transition matrices.
    """
    prepared = prepare_signal_sample(
        net,
        windows=windows,
        sample_fraction=sample_fraction,
        p0=p0,
    )
    return compute_signals_for_lambda_prepared(
        prepared=prepared,
        lamda=lamda,
        use_linear_approx=use_linear_approx,
        lin_t_s=lin_t_s,
        window_backend=window_backend,
    )


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
