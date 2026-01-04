import os
import pandas as pd
import numpy as np
from scipy.sparse import (lil_matrix, dok_matrix, diags, eye, isspmatrix_csr, isspmatrix,
                          csr_matrix, coo_matrix, csc_matrix)
from scipy.sparse.linalg import expm, eigsh
from scipy.linalg import expm as d_expm
from scipy.linalg import logm as d_logm
from scipy.sparse.csgraph import connected_components
import gzip
from SparseStochMat import sparse_stoch_mat, inplace_csr_row_normalize

from parallel_expm import compute_subspace_expm_parallel

from functools import partial

import time
import pickle
from math import exp

def compute_conditional_entropy(net=None, list_T=None, lamda=None, t_start=None, t_stop=None,
                                    verbose=False,
                                    # save_intermediate=True,
                                    reverse_time=False,
                                    force_csr=True,
                                    time_domain=None,
                                    p0 = None):
    """

    Computes interevent transition matrices as T_k(lamda) = expm(-tau_k*lamda*L_k).
    
    The transition matrix T_k is saved in `self.inter_T[lamda][k]`, where 
    self.inter_T is a dictionary with lamda as keys and lists of transition
    matrices as values.
    
    will compute from self.times[self._k_start_laplacians] until 
    self.times[self._k_stop_laplacians-1]
    
    the transition matrix at step k, is the probability transition matrix
    between times[k] and times[k+1]
    
    Parameters
    ----------
    lamda : float, optional
        Random walk rate, dynamical resolution parameter. The default (None)
        is 1 over the median inter event time.
    t_start : float or int, optional
        Starting time, passed to `compute_laplacian_matrices` if the 
        Laplacians have not yet been computed.
        Otherwise is not used.
        The computation starts at self.times[self._k_start_laplacians].
        The default is None, i.e. starts at the beginning of times.
    t_stop : float or int, optional
        Same than `t_start` but for the ending time of computations.
        Computations stop at self.times[self._k_stop_laplacians-1].
        Default is end of times.
    verbose : bool, optional
        The default is False.
    fix_tau_k : bool, optional
        If true, all interevent times (tau_k) in the formula above are set to 1. 
        This decouples the dynamic scale from the length of event which
        is useful for temporal networks with instantaneous events.
        The default is False.
    use_sparse_stoch : bool, optional
        Whether to use custom sparse stochastic matrix format to save the
        inter transition matrices. Especially useful for large networks as 
        the matrix exponential is then computed on each connected component 
        separately (more memory efficient). The default is False.
    dense_expm : bool, optional
        Whether to use the dense version of the matrix exponential algorithm
        at each time steps. Recommended for not too large networks. 
        The inter trans. matrices are still saved as sparse scipy matrices
        as they usually have many zero values. The default is True. Has no
        effect is use_sparse_stoch is True.

    Returns
    -------
    None.

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