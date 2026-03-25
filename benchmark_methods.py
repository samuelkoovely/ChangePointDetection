import numpy as np
import scipy
import scipy.sparse as ss
from sklearn.cluster import KMeans

from benchmark_distances import distance_frobenius
from benchmark_utils import laplacian, norm_laplacian


def _is_temporal_network_like(data):
    return hasattr(data, "compute_laplacian_matrices") and hasattr(data, "times")


def _ensure_laplacians(net, normalize=True):
    requested_random_walk = bool(normalize)
    cached_random_walk = getattr(net, "_laplacian_random_walk", None)

    if hasattr(net, "laplacians") and cached_random_walk == requested_random_walk:
        return

    try:
        net.compute_laplacian_matrices(
            t_start=net.times[0],
            t_stop=net.times[-1],
            random_walk=requested_random_walk,
        )
    except TypeError:
        net.compute_laplacian_matrices(random_walk=requested_random_walk)


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


def _laplacians_from_snapshots(snapshots, normalize=True):
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


def _iter_laplacians(data, normalize=True):
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


def _prepare_snapshots_for_distances(data):
    snapshots = _coerce_snapshot_sequence(data)
    if not snapshots:
        return []

    if all(isinstance(snapshot, np.ndarray) for snapshot in snapshots):
        return [np.asarray(snapshot, dtype=float) for snapshot in snapshots]

    if all(ss.isspmatrix(snapshot) for snapshot in snapshots):
        return [snapshot.tocsr().asfptype() for snapshot in snapshots]

    dense_snapshots = []
    for snapshot in snapshots:
        if isinstance(snapshot, np.ndarray):
            dense_snapshots.append(np.asarray(snapshot, dtype=float))
        elif ss.isspmatrix(snapshot):
            dense_snapshots.append(np.asarray(snapshot.toarray(), dtype=float))
        elif hasattr(snapshot, "toarray"):
            dense_snapshots.append(np.asarray(snapshot.toarray(), dtype=float))
        else:
            dense_snapshots.append(np.asarray(snapshot, dtype=float))
    return dense_snapshots


def _smallest_eigen_signature(L, n_eigen):
    n_eigen = int(n_eigen)
    if n_eigen <= 0:
        raise ValueError("n_eigen must be strictly positive.")

    if not ss.isspmatrix(L):
        L = ss.csr_matrix(L)

    L = L.asfptype()
    n_nodes = int(L.shape[0])
    if n_nodes <= 1:
        return np.zeros(n_eigen, dtype=float)

    k = min(n_eigen, n_nodes - 1)
    if k <= 0:
        return np.zeros(n_eigen, dtype=float)

    try:
        eigenvalues = ss.linalg.eigsh(
            L,
            return_eigenvectors=False,
            k=k,
            which="SA",
        )
    except Exception:
        return np.zeros(n_eigen, dtype=float)

    eigenvalues = np.sort(np.asarray(eigenvalues, dtype=float))
    if k < n_eigen:
        eigenvalues = np.pad(eigenvalues, (0, n_eigen - k))

    norm = float(np.linalg.norm(eigenvalues))
    if not np.isfinite(norm) or norm == 0.0:
        return np.zeros(n_eigen, dtype=float)

    return eigenvalues / norm


def compute_laplacian_signatures(data, n_eigen, normalize=True):
    """
    Compute one normalized eigenvalue signature per snapshot.
    """

    n_eigen = int(n_eigen)
    if n_eigen <= 0:
        raise ValueError("n_eigen must be strictly positive.")

    laplacians = _iter_laplacians(data, normalize=normalize)
    if not laplacians:
        return np.empty((0, n_eigen), dtype=float)

    signatures = np.empty((len(laplacians), n_eigen), dtype=float)
    for i, L in enumerate(laplacians):
        signatures[i, :] = _smallest_eigen_signature(L, n_eigen=n_eigen)

    return signatures


def compute_similarity_signal_from_signatures(
    laplacian_signatures,
    window_length,
    n_eigen=None,
):
    """
    Compute the Huang et al. signal from precomputed Laplacian signatures.
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
    if window_length >= num_snapshots:
        return {
            "window_length": window_length,
            "n_eigen": n_eigen,
            "signal": np.array([], dtype=float),
            "snapshot_indices": np.array([], dtype=int),
        }

    signal_values = []
    snapshot_indices = np.arange(window_length, num_snapshots, dtype=int)

    for snapshot_index in snapshot_indices:
        window = laplacian_signatures[snapshot_index - window_length : snapshot_index].T
        try:
            u, _, _ = scipy.linalg.svd(window, compute_uv=True, full_matrices=False)
        except Exception:
            return {
                "window_length": window_length,
                "n_eigen": n_eigen,
                "signal": np.array([], dtype=float),
                "snapshot_indices": np.array([], dtype=int),
            }

        score = 1.0 - abs(float(np.dot(u[:, 0], laplacian_signatures[snapshot_index])))
        signal_values.append(score)

    return {
        "window_length": window_length,
        "n_eigen": n_eigen,
        "signal": np.asarray(signal_values, dtype=float),
        "snapshot_indices": snapshot_indices,
    }


def laplacian_spectrum_similarity(data, window_length, normalize=True, n_eigen=6):
    """
    Compute Laplacian anomaly detection statistic [Huang et al. 2020].
    """

    laplacian_signatures = compute_laplacian_signatures(
        data=data,
        n_eigen=n_eigen,
        normalize=normalize,
    )
    result = compute_similarity_signal_from_signatures(
        laplacian_signatures=laplacian_signatures,
        window_length=window_length,
        n_eigen=n_eigen,
    )
    return result["signal"], result["snapshot_indices"]


def NCPD(data, window_length, n_eigen, normalize=False):
    """
    Compute NCPD statistics of the Spectral Clustering method [Cribben et al. 2017]
    :param data (list of DGL graphs): dynamic network sequence:
    :param window_length (int): length of backward window
    :param n_eigen (int): nb of eigenvalues used to compute the node embeddings
    :return:
    """

    if not isinstance(data[0], ss.csr_matrix):
        data = [ss.csr_matrix(data[i].toarray()) for i in range(len(data))]

    gamma = []
    for i in range(window_length, len(data) - window_length):
        avl = sum(data[i - window_length : i]) / window_length
        avr = sum(data[i : i + window_length]) / window_length
        if normalize:
            lapl = norm_laplacian(avl)
            lapr = norm_laplacian(avr)
        else:
            lapl = laplacian(avl)
            lapr = laplacian(avr)

        vl, wl = ss.linalg.eigsh(lapl, n_eigen, which="SA")
        vr, wr = ss.linalg.eigsh(lapr, n_eigen, which="SA")
        xl = KMeans(n_clusters=n_eigen, n_init="auto").fit(wl[:, :n_eigen])
        xr = KMeans(n_clusters=n_eigen, n_init="auto").fit(wr[:, :n_eigen])

        UL = np.stack(
            [xl.cluster_centers_[xl.labels_[j]] for j in range(data[i].shape[0])],
            axis=0,
        )
        UR = np.stack(
            [xr.cluster_centers_[xr.labels_[j]] for j in range(data[i].shape[0])],
            axis=0,
        )

        s = scipy.linalg.svd(UL.transpose().dot(UR), compute_uv=False)
        gamma.append(np.sum(s))

    d_gamma = np.maximum(
        np.abs(np.array(gamma)[1:-1] - np.array(gamma)[:-2]),
        np.abs(np.array(gamma)[1:-1] - np.array(gamma)[2:]),
    )

    return d_gamma, 1 + np.arange(window_length, len(data) - window_length - 2)


def CUMSUM(data, window_length):
    """
    CUMSUM statistics from Optimal network online change point localisation
    [Yu, Padilla, Wang & Rinaldo 2021] (without USVT)

    :param data:
    :param window_length:
    :return:
    """

    if not isinstance(data[0], ss.csr_matrix):
        data = [ss.csr_matrix(data[i].toarray()) for i in range(len(data))]

    A = [data[2 * i].toarray() for i in range(len(data) // 2)]
    B = [data[2 * i + 1].toarray() for i in range(len(data) // 2)]

    Y = []
    for i in range(window_length, len(A) - window_length):
        C_A = 1.0 / (np.sqrt(2 * window_length)) * (
            np.sum(np.stack(A[i - window_length : i], axis=2), axis=2)
            - np.sum(np.stack(A[i : i + window_length], axis=2), axis=2)
        )
        C_B = 1.0 / (np.sqrt(2 * window_length)) * (
            np.sum(np.stack(B[i - window_length : i], axis=2), axis=2)
            - np.sum(np.stack(B[i : i + window_length], axis=2), axis=2)
        )

        C_B = C_B / np.linalg.norm(C_B)
        Y.append(np.sum(C_A * C_B))

    times = np.arange(2 * window_length, len(data) - 2 * window_length, 2)
    return Y, times


def CUMSUM_2(data, window_length):
    """
    CUMSUM statistics from Change-Point Detection in Dynamic Networks with
    Missing Links [Enikeeva & Klopp 2021]

    :param data:
    :param window_length:
    :return:
    """

    if not isinstance(data[0], ss.csr_matrix):
        data = [ss.csr_matrix(data[i].toarray()) for i in range(len(data))]

    data = [data[i].toarray() for i in range(len(data))]
    stat = []

    for i in range(window_length, len(data) - window_length):
        csum = 1.0 / (np.sqrt(2 * window_length)) * (
            np.sum(np.stack(data[i - window_length : i], axis=2), axis=2)
            - np.sum(np.stack(data[i : i + window_length], axis=2), axis=2)
        )
        stat.append(np.linalg.norm(csum, ord=2))

    return stat, np.arange(window_length, len(data) - window_length)


def avg_frobenius_distance(data, window_length, diff=False):
    """
    Compute averaged Frobenius distance statistics over a backward window.
    """

    window_length = int(window_length)
    if window_length <= 0:
        raise ValueError("window_length must be strictly positive.")

    snapshots = _prepare_snapshots_for_distances(data)
    num_snapshots = len(snapshots)
    if window_length >= num_snapshots:
        return np.array([], dtype=float), np.array([], dtype=int)

    avg_dist = []
    for i in range(window_length, num_snapshots):
        dist_t = []
        for j in range(1, window_length + 1):
            dist_t.append(distance_frobenius(snapshots[i], snapshots[i - j]))
        avg_dist.append(np.mean(dist_t))

    avg_dist = np.asarray(avg_dist, dtype=float)
    if diff:
        if avg_dist.size <= 1:
            return np.array([], dtype=float), np.array([], dtype=int)
        d_avg_dist = np.abs(avg_dist[1:] - avg_dist[:-1])
        return d_avg_dist, np.arange(window_length + 1, num_snapshots, dtype=int)

    return avg_dist, np.arange(window_length, num_snapshots, dtype=int)
