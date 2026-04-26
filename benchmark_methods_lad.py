from __future__ import annotations

import numpy as np
import scipy
import scipy.sparse as ss

from benchmark_utils import laplacian, norm_laplacian


def _is_temporal_network_like(data):
    return hasattr(data, "compute_laplacian_matrices") and hasattr(data, "times")


def _ensure_laplacians(net, normalize=False):
    requested_random_walk = bool(normalize)
    cached_random_walk = getattr(net, "_laplacian_random_walk", None)

    if hasattr(net, "laplacians") and cached_random_walk == requested_random_walk:
        try:
            if len(net.laplacians) > 0:
                return
        except TypeError:
            return

    try:
        net.compute_laplacian_matrices(
            t_start=net.times[0],
            t_stop=net.times[-1],
            random_walk=requested_random_walk,
        )
    except TypeError:
        net.compute_laplacian_matrices(random_walk=requested_random_walk)


def _ensure_adjacencies(net):
    adjacencies = getattr(net, "adjacencies", None)
    if adjacencies is not None:
        try:
            if len(adjacencies) > 0:
                return
        except TypeError:
            return

    kwargs = {"save_adjacencies": True}
    cached_random_walk = getattr(net, "_laplacian_random_walk", None)
    if cached_random_walk is not None:
        kwargs["random_walk"] = bool(cached_random_walk)

    try:
        net.compute_laplacian_matrices(
            t_start=net.times[0],
            t_stop=net.times[-1],
            **kwargs,
        )
    except TypeError:
        net.compute_laplacian_matrices(**kwargs)


def _coerce_snapshot_sequence(data):
    if _is_temporal_network_like(data):
        _ensure_adjacencies(data)
        snapshots = getattr(data, "adjacencies", None)
        return [] if snapshots is None else list(snapshots)

    if isinstance(data, np.ndarray):
        if data.ndim == 2:
            return [np.asarray(data, dtype=float)]
        if data.ndim == 3:
            return [np.asarray(data[i], dtype=float) for i in range(data.shape[0])]
        raise ValueError(
            "Expected a 2D adjacency matrix or a 3D stack of adjacency matrices."
        )

    if ss.isspmatrix(data):
        return [data]

    return list(data)


def _laplacians_from_snapshots(snapshots, normalize=False):
    laplacians = []
    for snapshot in snapshots:
        if ss.isspmatrix(snapshot):
            adjacency = snapshot.tocsr().asfptype()
            if normalize:
                laplacians.append(norm_laplacian(adjacency, sparse=True))
            else:
                laplacians.append(laplacian(adjacency, sparse=True))
        else:
            adjacency = np.asarray(snapshot, dtype=float)
            if normalize:
                laplacians.append(norm_laplacian(adjacency, sparse=False))
            else:
                laplacians.append(laplacian(adjacency, sparse=False))
    return laplacians


def _iter_laplacians(data, normalize=False):
    if _is_temporal_network_like(data):
        _ensure_laplacians(data, normalize=normalize)
        laplacians = getattr(data, "laplacians", None)
        if laplacians is None:
            return []
        if ss.isspmatrix(laplacians):
            return [laplacians]
        return list(laplacians)

    return _laplacians_from_snapshots(
        _coerce_snapshot_sequence(data),
        normalize=normalize,
    )


def _sorted_singular_values(matrix, k, top=True):
    k = int(k)
    if k <= 0:
        raise ValueError("k must be strictly positive.")

    if ss.isspmatrix(matrix):
        matrix = matrix.tocsr().asfptype()
        min_dim = int(min(matrix.shape))
        if min_dim <= 1:
            return np.zeros(k, dtype=float)

        k_eff = min(k, min_dim - 1)
        if k_eff <= 0:
            return np.zeros(k, dtype=float)

        try:
            singular_values = ss.linalg.svds(
                matrix,
                k=k_eff,
                which="LM" if top else "SM",
                return_singular_vectors=False,
            )
        except TypeError:
            _, singular_values, _ = ss.linalg.svds(
                matrix,
                k=k_eff,
                which="LM" if top else "SM",
            )
        except Exception:
            dense_matrix = np.asarray(matrix.toarray(), dtype=float)
            singular_values = scipy.linalg.svdvals(dense_matrix)
    else:
        dense_matrix = np.asarray(matrix, dtype=float)
        min_dim = int(min(dense_matrix.shape))
        if min_dim == 0:
            return np.zeros(k, dtype=float)
        singular_values = scipy.linalg.svdvals(dense_matrix)

    singular_values = np.asarray(singular_values, dtype=float)
    singular_values = np.sort(singular_values)
    if top:
        singular_values = singular_values[::-1]

    singular_values = singular_values[:k]
    if singular_values.size < k:
        singular_values = np.pad(
            singular_values,
            (0, k - singular_values.size),
            mode="constant",
        )

    return singular_values


def _normalize_signature(signature):
    signature = np.asarray(signature, dtype=float)
    norm = float(np.linalg.norm(signature))
    if not np.isfinite(norm) or norm == 0.0:
        return np.zeros_like(signature)
    return signature / norm


def _principal_window_direction(window_signatures):
    window_signatures = np.asarray(window_signatures, dtype=float)
    if window_signatures.ndim != 2:
        raise ValueError("window_signatures must be a 2D array.")

    u, _, _ = scipy.linalg.svd(
        window_signatures.T,
        full_matrices=False,
        compute_uv=True,
    )
    return _normalize_signature(u[:, 0])


def _score_against_window(signature, window_signatures):
    typical_vec = _principal_window_direction(window_signatures)
    signature = _normalize_signature(signature)
    return 1.0 - abs(float(np.dot(signature, typical_vec)))


def _difference_signal(signal):
    signal = np.asarray(signal, dtype=float)
    if signal.size == 0:
        return signal

    diff_signal = np.empty_like(signal)
    diff_signal[0] = 0.0
    diff_signal[1:] = signal[1:] - signal[:-1]
    return diff_signal


def compute_laplacian_signatures(
    data,
    n_eigen,
    normalize=False,
    top=True,
):
    """
    Compute upstream-style LAD signatures from per-snapshot Laplacians.

    Compared with the existing local implementation in `benchmark_methods.py`,
    this variant follows the original LAD repository more closely:
    - use Laplacian singular values rather than the smallest eigenvalues
    - default to the top singular values (`top=True`)
    - L2-normalize each snapshot signature before scoring
    """

    n_eigen = int(n_eigen)
    if n_eigen <= 0:
        raise ValueError("n_eigen must be strictly positive.")

    laplacians = _iter_laplacians(data, normalize=normalize)
    if not laplacians:
        return np.empty((0, n_eigen), dtype=float)

    signatures = np.empty((len(laplacians), n_eigen), dtype=float)
    for i, laplacian_matrix in enumerate(laplacians):
        signatures[i, :] = _normalize_signature(
            _sorted_singular_values(laplacian_matrix, k=n_eigen, top=top)
        )

    return signatures


def compute_similarity_signal_from_signatures(
    laplacian_signatures,
    window_length,
    n_eigen=None,
    difference=False,
    second_window_length=None,
):
    """
    Compute an LAD-style rolling anomaly signal from precomputed signatures.

    Parameters
    ----------
    laplacian_signatures:
        Array of shape `(n_snapshots, n_eigen)`.
    window_length:
        Backward window length. When `second_window_length` is provided this is
        interpreted as the short window.
    n_eigen:
        Optional metadata field kept for compatibility with the existing local
        signal dictionaries.
    difference:
        Apply the first-difference transform used in the upstream LAD scripts.
    second_window_length:
        Optional long window length. When provided, the returned signal is the
        pointwise maximum of the short- and long-window scores, mirroring the
        `change_detection_two_windows` path in the upstream repository.
    """

    window_length = int(window_length)
    if window_length <= 0:
        raise ValueError("window_length must be strictly positive.")

    laplacian_signatures = np.asarray(laplacian_signatures, dtype=float)
    if laplacian_signatures.ndim != 2:
        raise ValueError(
            "laplacian_signatures must be a 2D array with shape "
            "(n_snapshots, n_eigen)."
        )

    if n_eigen is None:
        n_eigen = int(laplacian_signatures.shape[1]) if laplacian_signatures.size else 0
    else:
        n_eigen = int(n_eigen)

    num_snapshots = int(laplacian_signatures.shape[0])

    if second_window_length is None:
        if window_length >= num_snapshots:
            return {
                "window_length": window_length,
                "n_eigen": n_eigen,
                "difference": bool(difference),
                "signal": np.array([], dtype=float),
                "raw_signal": np.array([], dtype=float),
                "snapshot_indices": np.array([], dtype=int),
            }

        snapshot_indices = np.arange(window_length, num_snapshots, dtype=int)
        raw_signal = np.empty(len(snapshot_indices), dtype=float)

        for out_idx, snapshot_index in enumerate(snapshot_indices):
            raw_signal[out_idx] = _score_against_window(
                laplacian_signatures[snapshot_index],
                laplacian_signatures[snapshot_index - window_length : snapshot_index],
            )

        signal = _difference_signal(raw_signal) if difference else raw_signal.copy()
        return {
            "window_length": window_length,
            "n_eigen": n_eigen,
            "difference": bool(difference),
            "signal": signal,
            "raw_signal": raw_signal,
            "snapshot_indices": snapshot_indices,
        }

    second_window_length = int(second_window_length)
    if second_window_length <= 0:
        raise ValueError("second_window_length must be strictly positive.")
    if second_window_length < window_length:
        raise ValueError(
            "second_window_length must be greater than or equal to window_length."
        )
    if second_window_length >= num_snapshots:
        return {
            "window_length": window_length,
            "second_window_length": second_window_length,
            "n_eigen": n_eigen,
            "difference": bool(difference),
            "signal": np.array([], dtype=float),
            "raw_signal": np.array([], dtype=float),
            "short_signal": np.array([], dtype=float),
            "long_signal": np.array([], dtype=float),
            "snapshot_indices": np.array([], dtype=int),
        }

    snapshot_indices = np.arange(second_window_length, num_snapshots, dtype=int)
    short_signal = np.empty(len(snapshot_indices), dtype=float)
    long_signal = np.empty(len(snapshot_indices), dtype=float)
    raw_signal = np.empty(len(snapshot_indices), dtype=float)

    for out_idx, snapshot_index in enumerate(snapshot_indices):
        signature = laplacian_signatures[snapshot_index]
        short_score = _score_against_window(
            signature,
            laplacian_signatures[snapshot_index - window_length : snapshot_index],
        )
        long_score = _score_against_window(
            signature,
            laplacian_signatures[
                snapshot_index - second_window_length : snapshot_index
            ],
        )
        short_signal[out_idx] = short_score
        long_signal[out_idx] = long_score
        raw_signal[out_idx] = max(short_score, long_score)

    signal = _difference_signal(raw_signal) if difference else raw_signal.copy()
    return {
        "window_length": window_length,
        "second_window_length": second_window_length,
        "n_eigen": n_eigen,
        "difference": bool(difference),
        "signal": signal,
        "raw_signal": raw_signal,
        "short_signal": short_signal,
        "long_signal": long_signal,
        "snapshot_indices": snapshot_indices,
    }


def laplacian_spectrum_similarity(
    data,
    window_length,
    normalize=False,
    n_eigen=6,
    top=True,
    difference=False,
    second_window_length=None,
):
    """
    Compute an LAD-style spectrum similarity signal.

    This keeps the same top-level call shape as the current benchmark helper so
    that a test substitution can be done by swapping imports.
    """

    laplacian_signatures = compute_laplacian_signatures(
        data=data,
        n_eigen=n_eigen,
        normalize=normalize,
        top=top,
    )
    result = compute_similarity_signal_from_signatures(
        laplacian_signatures=laplacian_signatures,
        window_length=window_length,
        n_eigen=n_eigen,
        difference=difference,
        second_window_length=second_window_length,
    )
    return result["signal"], result["snapshot_indices"]
