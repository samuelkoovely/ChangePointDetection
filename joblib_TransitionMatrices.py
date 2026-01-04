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


net = ContTempNetwork.load('evolving_SBM_net0')
# net = ContTempNetwork.load('/home/b/skoove/Desktop/entropy/paper_data/socio_pat_primary_school/primaryschoolnet')

folder = '//scratch/tmp/180/skoove/evolving_SBM_heat/net0/'
#folder = '//scratch/tmp/180/skoove/primaryschoolnet_rw/'

flag_start = 0
flag_stop = -1

net.compute_laplacian_matrices(t_start= net.times[flag_start], t_stop=net.times[flag_stop], random_walk=False)
# Laplacians={'L': net.laplacians}
# file= folder + 'L'
# with open(file, 'wb') as fopen:
#     pickle.dump(Laplacians, fopen)

def worker(lamda):

    # net.compute_inter_transition_matrices(lamda=lamda, dense_expm=False, use_sparse_stoch=False)
    # net.compute_transition_matrices(lamda=lamda)
    
    net.compute_inter_transition_matrices(lamda=lamda, t_start= net.times[flag_start], t_stop=net.times[flag_stop], dense_expm=False, use_sparse_stoch=False, random_walk=False)
    net.compute_transition_matrices(lamda=lamda, reverse_time=False)

    
    # inter_T={'lamda': lamda, 'inter_T': net.inter_T[lamda]}
    # file=f'//scratch/tmp/180/skoove/experiment/inter_T/inter_T{lamda:.06f}'
    # with open(file, 'wb') as fopen:
    #     pickle.dump(inter_T, fopen)
    
    # T={'lamda': lamda, 'T': net.T[lamda]}
    # file=f'//scratch/tmp/180/skoove/experiment/T/T{lamda:.06f}'
    # with open(file, 'wb') as fopen:
    #     pickle.dump(T, fopen)
    
    # inter_T_heat={'lamda': lamda, 'inter_T': net.inter_T[lamda]}
    # file= folder + f'inter_Tselected/inter_T{lamda:.11f}'
    # with open(file, 'wb') as fopen:
    #     pickle.dump(inter_T_heat, fopen)

    T_heat={'lamda': lamda, 'T': net.T[lamda]}
    
    #file= folder + f'T{flag_start}_{flag_stop}/T{lamda:.11f}'
    file= folder + f'T/T{lamda:.11f}'
    with open(file, 'wb') as fopen:
        pickle.dump(T_heat, fopen)

    print(lamda)

lambdas = np.logspace(-5,0,10)

n_cpu = 10 #some appropriate number
Parallel(n_jobs=n_cpu)(delayed(worker)(l) for l in lambdas)