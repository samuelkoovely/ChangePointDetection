from joblib import Parallel, delayed
import numpy as np
from SynthTempNetwork import Individual, SynthTempNetwork
from TemporalNetwork import ContTempNetwork, StaticTempNetwork
from FlowStability import SparseClustering, FlowIntegralClustering, run_multi_louvain, avg_norm_var_information
import pickle

import matplotlib.pyplot as plt
import matplotlib

from scipy.sparse import (lil_matrix, dok_matrix, diags, eye, isspmatrix_csr, isspmatrix,
                          csr_matrix, coo_matrix, csc_matrix)
import parallel_clustering


# net = ContTempNetwork.load('fig3_network')
net_heat = ContTempNetwork.load('/home/b/skoove/Desktop/entropy/paper_data/socio_pat_primary_school/primaryschoolnet')
folder = '//scratch/tmp/180/skoove/primaryschoolnet_rw/'

flag_start = 1320
flag_stop = 1556

def worker(lamda):
    with open(folder + f'Tplot1320_1556/T{lamda:.11f}', 'rb') as f: #open(folder + f'Tplot_bw{flag_start}_{flag_stop}/T{lamda:.11f}', 'rb') as f:
        T_list = pickle.load(f)['T']

    flowintegralclustering = FlowIntegralClustering(T_list=T_list,
                                  time_list=net_heat.times[flag_start:flag_stop],
                                  verbose=False,
                                  reverse_time=False)
    
    clustering = SparseClustering(p1=flowintegralclustering.p1, p2=None,
                         T=flowintegralclustering.T_list[-1], S=flowintegralclustering.I_list[0])
    
    n_loops, cluster_lists, stabilities, seeds = run_multi_louvain(clustering, num_repeat=100)

    file= folder + f'clustersplot1320_1556/cluster{lamda:.11f}'
    # file= folder + f'clustersplot_extended{flag_start}_{flag_stop}/cluster{lamda:.11f}'
    with open(file, 'wb') as fopen:
        pickle.dump(cluster_lists, fopen)

    print(lamda)

lambdas = np.logspace(-5,0,200)

n_cpu = 60 #some appropriate number
Parallel(n_jobs=n_cpu)(delayed(worker)(l) for l in lambdas)