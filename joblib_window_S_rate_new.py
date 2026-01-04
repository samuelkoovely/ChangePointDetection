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

import compute_S_rate

#net = ContTempNetwork.load('fig3_growing_network300')
#net = ContTempNetwork.load('/home/b/skoove/Desktop/entropy/paper_data/socio_pat_primary_school/primaryschoolnet')
#net = ContTempNetwork.load('periodic_SBM_net')
#net = ContTempNetwork.load('evolving_SBM_net2_c')
#net = ContTempNetwork.load('dynamic_SBM')
net = ContTempNetwork.load('evolving_SBM_net_1activity')

#folder = '//scratch/tmp/180/skoove/growing_experiment300_temporal_rw/'
#folder = '//scratch/tmp/180/skoove/primaryschoolnet_rw_day2/'
#folder = '//scratch/tmp/180/skoove/periodic_SBM_heat/'
#folder = '//scratch/tmp/180/skoove/evolving_SBM_heat/net2_c/'
#folder = '/scratch/tmp/180/skoove/dynamic_SBM_heat/'
folder = '/scratch/tmp/180/skoove/evolving_SBM_1activity/'


# with open(folder + 'L', 'rb') as f:
#     dict_L = pickle.load(f)
window = 25

def worker(lamda):
    with open(folder + f'window_T_selected_new/{window}/window_T{lamda:.11f}', 'rb') as f:
        dict_T = pickle.load(f)

    
    try:
        considered_times = net.times[net.times < net.times[-1] - window]
        range = len(considered_times)
        S = compute_S_rate.compute_conditional_entropy(net=net, list_T=dict_T['window_T'],
                                    lamda=lamda, force_csr=True,
                                    time_domain= list(np.arange(0,range))) #list(np.arange(0,len(net.times) -1556 -180 -1))) #list(np.arange(0, len(net.times)-2))) #[len(net.times)-2]) #list(np.arange(0, len(net.times), 10)))

        
        S_rate={'lamda': f'{lamda:.11f}', 'window_S': S}
        file= folder + f'window_S_selected_new/{window}/window_S{lamda:.11f}'
        with open(file, 'wb') as fopen:
            pickle.dump(S_rate, fopen)
    except ValueError:
        S_rate={'lamda': f'{lamda:.11f}', 'window_S': 10}
        print('error with lamda')
    
    print(lamda)

lambdas = np.logspace(-5,0,10) #np.logspace(-3,1,10) #np.logspace(-5,0,10)

n_cpu = 10 #some appropriate number
Parallel(n_jobs=n_cpu)(delayed(worker)(l) for l in lambdas)