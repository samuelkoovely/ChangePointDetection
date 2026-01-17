from joblib import Parallel, delayed
import numpy as np
from TemporalNetwork import ContTempNetwork
import pickle




net = ContTempNetwork.load('evolving_SBM_net_1activity')


folder = '/scratch/tmp/180/skoove/evolving_SBM_1activity/'

flag_start = 0
flag_stop = -1

window = 25

net.compute_laplacian_matrices(t_start= net.times[flag_start], t_stop=net.times[flag_stop], random_walk=False)
Laplacians={'L': net.laplacians}
file= folder + 'L'
# with open(file, 'wb') as fopen:
#     pickle.dump(Laplacians, fopen)

def worker(lamda):   
    net.compute_inter_transition_matrices(lamda=lamda, t_start= net.times[flag_start], t_stop=net.times[flag_stop], dense_expm=False, use_sparse_stoch=False, random_walk=False)
    #net.compute_transition_matrices_sliding_window(lamda=lamda, reverse_time=False, window_size=window)
    net.compute_transition_matrices_sliding_timewindow(lamda=lamda, reverse_time=False, window_timelength=window)

    window_T_heat={'lamda': lamda, 'window_T': net.window_T[lamda]}
    file= folder + f'window_T_selected_new/{window}/window_T{lamda:.11f}'
    with open(file, 'wb') as fopen:
        pickle.dump(window_T_heat, fopen)

    print(lamda)

lambdas = np.logspace(-5,0,10)

n_cpu = 20 #some appropriate number
Parallel(n_jobs=n_cpu)(delayed(worker)(l) for l in lambdas)