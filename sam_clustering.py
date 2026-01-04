from joblib import Parallel, delayed
import numpy as np
from SynthTempNetwork import Individual, SynthTempNetwork
from TemporalNetwork import ContTempNetwork, StaticTempNetwork
from FlowStability import FlowIntegralClustering
import pickle

import matplotlib.pyplot as plt
import matplotlib

from scipy.sparse import (lil_matrix, dok_matrix, diags, eye, isspmatrix_csr, isspmatrix,
                          csr_matrix, coo_matrix, csc_matrix)
from SparseStochMat import sparse_autocov_mat
from FlowStability import Clustering, SparseClustering
import parallel_clustering    


# net = ContTempNetwork.load('fig3_network')
net_heat = ContTempNetwork.load('fig3_network250_2scales_heat')
p1 = p2 = 1/net_heat.num_nodes * np.ones((net_heat.num_nodes))
P = eye(net_heat.num_nodes, dtype=np.float64).tocsr()
P.data = p1

def worker(lamda):
    multi_res_sam = {}

    with open(f'//scratch/tmp/180/skoove/growing_experiment300_static_weighted_heat/Tplot/T{lamda:.11f}', 'rb') as f:
        T = pickle.load(f)['T'][-1]

    PT = P @ T
    sam = sparse_autocov_mat(PT, p1, p2, PT_symmetric=True)
    S = sam.from_T(T = T, p1 = p1, p2 = p2)
    clustering = SparseClustering(p1=p1, p2=p2,
                        T=T, S=S)
    clusters, stabilites, seeds = parallel_clustering.compute_parallel_clustering(clustering, num_repeat=50, nproc=10, 
                                verbose=False, n_meta_iter_max=1000, 
                                n_sub_iter_max=1000, 
                                clust_verbose=False, print_num_loops=False)
    multi_res_sam[lamda] = clusters

    print(lamda)

    file=f'//scratch/tmp/180/skoove/growing_experiment300_static_weighted_heat/clusters_sam_slow'
    with open(file, 'wb') as fopen:
        pickle.dump(multi_res_sam, fopen)

lambdas = np.logspace(-2.5,0,6)

n_cpu = 50 #some appropriate number
Parallel(n_jobs=n_cpu)(delayed(worker)(l) for l in lambdas)