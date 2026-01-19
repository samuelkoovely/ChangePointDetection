from joblib import Parallel, delayed
import numpy as np
from TemporalNetwork import ContTempNetwork
import pickle
import compute_S_rate

folder = '/scratch/tmp/180/skoove/evolving_SBM_1activity/'
window = 25

flag_start = 0
flag_stop = -1

def worker(lamda):
    # IMPORTANT: load inside worker to avoid joblib pickling the whole net to each process
    net = ContTempNetwork.load('evolving_SBM_net_1activity')

    net.compute_laplacian_matrices(t_start=net.times[flag_start],
                                  t_stop=net.times[flag_stop],
                                  random_walk=False)

    net.compute_inter_transition_matrices(lamda=lamda,
                                         t_start=net.times[flag_start],
                                         t_stop=net.times[flag_stop],
                                         dense_expm=False,
                                         use_sparse_stoch=False,
                                         random_walk=False)

    p0 = np.ones(net.num_nodes) / net.num_nodes
    S_vals = [0.0]  # keep your convention if you want

    on_T = compute_S_rate.make_on_window_matrix_entropy_callback(p0, S_vals)

    net.compute_transition_matrices_sliding_timewindow(lamda=lamda,
                                                      reverse_time=False,
                                                      window_timelength=window,
                                                      save_intermediate=False,   # <-- don't store matrices
                                                      on_window_matrix=on_T,     # <-- compute entropy on the fly
                                                      force_csr=True)

    out = {'lamda': f'{lamda:.11f}', 'window_S': S_vals}
    file = folder + f'window_S_selected_new/{window}/window_S{lamda:.11f}'
    with open(file, 'wb') as f:
        pickle.dump(out, f)

    print(lamda)

lambdas = np.logspace(-5, 0, 10)
n_cpu = 20
Parallel(n_jobs=n_cpu)(delayed(worker)(l) for l in lambdas)