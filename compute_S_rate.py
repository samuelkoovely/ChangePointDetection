import os
import numpy as np
from scipy.sparse import csr_matrix
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

    """
    
    if time_domain is None:
        time_domain = list(range(1,len(net.times())))
    if reverse_time:
        k_init = len(time_domain)-1
        k_range = reversed(range(0, k_init))
        if verbose:
            print('PID ', os.getpid(), ' : reversed time computation.')
    else:
        k_init = 0
        k_range = range(len(time_domain))



    conditional_S = dict() 
    
    if p0 is None:
        p0 = 1/net.num_nodes*np.ones(net.num_nodes)
    
    t0 = time.time()
    # Initial conditional entropy
    conditional_S[f'{lamda:.11f}'] = [0]
    

    #One entropy value for each Transition Matrix
    for k in k_range:
        if verbose and not k%1000:
            print('PID ', os.getpid(), ' : ',k, ' over ' , len())
            print(f'PID {os.getpid()} : {time.time()-t0:.2f}s')
        
        T = list_T[time_domain[k]].tocsr()
        #p = p0 @ T
        logTdata = np.log(np.where(T.data > 0, T.data, 1))
        TlogTdata = T.data * logTdata
        # there shouldn't be need for this
        # TlogT[TlogT>0]=0
        TlogT = csr_matrix((TlogTdata, T.indices, T.indptr), shape=T.shape)
        conditional_S[f'{lamda:.11f}'].append(-np.sum(p0 @ TlogT, where= np.isfinite(p0 @ TlogT)))
        
    t_end = time.time()-t0
    if verbose:
        print('PID ', os.getpid(), ' : ', f'finished in {t_end:.2f}s') 
    
    return conditional_S

def conditional_entropy_of_T(T, p0):
    """
    Returns - sum_i p0[i] * sum_j T[i,j] log T[i,j]
    T: scipy sparse matrix (CSR recommended)
    p0: (n,) numpy array
    """
    T = T.tocsr()

    # log(T.data) safely
    logdata = np.zeros_like(T.data)
    mask = T.data > 0
    logdata[mask] = np.log(T.data[mask])

    TlogT = csr_matrix((T.data * logdata, T.indices, T.indptr), shape=T.shape)
    v = p0 @ TlogT
    return -np.sum(v[np.isfinite(v)])

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