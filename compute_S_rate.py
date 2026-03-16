import os
import numpy as np
from scipy.sparse import csr_matrix, isspmatrix_csr
import time


def compute_conditional_entropy(net=None, list_T=None, lamda=None, t_start=None, t_stop=None,
                                    verbose=False,
                                    # save_intermediate=True,
                                    reverse_time=False,
                                    force_csr=True,
                                    time_domain=None,
                                    p0 = None):
    """
    Compute conditional entropy values from a sequence of transition matrices.

    For each transition matrix T in `list_T[time_domain[k]]`, this computes
    -sum_i p0_i * sum_j T_ij * log(T_ij), and returns the running list of
    conditional entropies (one per transition matrix, plus an initial 0).

    Parameters
    ----------
    net : object, optional
        Temporal network object used for default time_domain/p0 sizing.
    list_T : sequence or mapping of sparse matrices
        Transition matrices indexed by time step.
    lamda : float, optional
        Label for the returned dict key (formatted to 11 decimals).
    t_start : float or int, optional
        Unused; kept for API compatibility.
    t_stop : float or int, optional
        Unused; kept for API compatibility.
    verbose : bool, optional
        Print progress every 1000 steps.
    reverse_time : bool, optional
        If True, iterate through `time_domain` in reverse order.
    force_csr : bool, optional
        Unused; kept for API compatibility.
    time_domain : sequence of int, optional
        Indices into `list_T`. Defaults to 1..len(net.times())-1.
    p0 : array-like, optional
        Initial distribution over nodes. Defaults to uniform.

    Returns
    -------
    dict
        Dictionary keyed by formatted `lamda` containing the entropy list.

    Notes
    -----
    This implementation avoids constructing the intermediate sparse matrix
    T * log(T). For each row i, it directly computes
        h_i = -sum_j T_ij log(T_ij)
    from the CSR data arrays, and then returns p0 @ h.
    """
    
    if time_domain is None:
        time_domain = list(range(1, len(net.times())))

    if reverse_time:
        k_range = range(len(time_domain) - 1, -1, -1)
        if verbose:
            print('PID ', os.getpid(), ' : reversed time computation.')
    else:
        k_range = range(len(time_domain))

    conditional_S = dict()

    if p0 is None:
        p0 = np.full(net.num_nodes, 1 / net.num_nodes, dtype=float)
    else:
        p0 = np.asarray(p0, dtype=float)

    t0 = time.time()
    key = f'{lamda:.11f}'
    conditional_S[key] = [0.0]
    s_list = conditional_S[key]

    # One entropy value for each Transition Matrix
    for k in k_range:
        if verbose and k % 1000 == 0:
            print('PID ', os.getpid(), ' : ', k, ' over ', len(time_domain))
            print(f'PID {os.getpid()} : {time.time()-t0:.2f}s')

        T = list_T[time_domain[k]]
        if not isspmatrix_csr(T):
            T = T.tocsr()

        s_list.append(conditional_entropy_of_T(T, p0))
        
    t_end = time.time()-t0
    if verbose:
        print('PID ', os.getpid(), ' : ', f'finished in {t_end:.2f}s') 
    
    return conditional_S

def conditional_entropy_of_T(T, p0):
    """
    Returns
        - sum_i p0[i] * sum_j T[i,j] log T[i,j]

    Parameters
    ----------
    T : scipy sparse matrix
        Transition matrix. CSR format is preferred.
    p0 : (n,) numpy array
        Probability vector over rows/nodes.

    Notes
    -----
    This implementation avoids constructing the intermediate sparse matrix
    TlogT. It works directly on the CSR structure:

    1. compute entrywise values x log x only on positive entries,
    2. sum them row-wise using np.add.reduceat on CSR row boundaries,
    3. take the weighted sum with p0.

    This is both faster and lighter on memory than building a new sparse matrix.
    """
    if not isspmatrix_csr(T):
        T = T.tocsr()

    p0 = np.asarray(p0, dtype=float)

    data = T.data
    indptr = T.indptr
    n_rows = T.shape[0]

    if data.size == 0:
        return 0.0

    # Safe x log x on positive entries only.
    xlogx = np.zeros_like(data, dtype=float)
    mask = data > 0
    xlogx[mask] = data[mask] * np.log(data[mask])

    # Row sums of x log x using CSR row boundaries.
    row_lengths = np.diff(indptr)
    row_sums = np.zeros(n_rows, dtype=float)
    nonempty = row_lengths > 0
    if np.any(nonempty):
        starts = indptr[:-1][nonempty]
        row_sums[nonempty] = np.add.reduceat(xlogx, starts)

    return float(-np.dot(p0, row_sums))

def make_on_window_matrix_entropy_callback(p0, S_vals):
    p0 = np.asarray(p0)

    def on_window_matrix(k, Tk_window):
        S_vals.append(conditional_entropy_of_T(Tk_window, p0))

    return on_window_matrix

def make_on_window_matrix_entropy_callback_prealloc(p0, S_arr, k_to_idx):
    """
    p0: (n,) probability vector
    S_arr: preallocated numpy array of length num_windows (+1 if you keep leading 0)
    k_to_idx: function mapping k -> storage index in S_arr

    Requires conditional_entropy_of_T(T, p0) to be defined in the same module.
    """
    p0 = np.asarray(p0)

    def on_window_matrix(k, Tk_window):
        S_arr[k_to_idx(k)] = conditional_entropy_of_T(Tk_window, p0)

    return on_window_matrix