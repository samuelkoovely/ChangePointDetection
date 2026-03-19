from joblib import Parallel, delayed
import numpy as np
from TemporalNetwork import ContTempNetwork
import pickle
import compute_S_rate


net = ContTempNetwork.load('evolving_SBM_net_1activity')
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
                                    time_domain= list(np.arange(0,range)))
 
        S_rate={'lamda': f'{lamda:.11f}', 'signal': S}
        file= folder + f'window_S_selected_new/{window}/window_S{lamda:.11f}'
        with open(file, 'wb') as fopen:
            pickle.dump(S_rate, fopen)
    except ValueError:
        S_rate={'lamda': f'{lamda:.11f}', 'signal': 10}
        print('error with lamda')
    
    print(lamda)

lambdas = np.logspace(-5,0,10)

n_cpu = 10 #some appropriate number
Parallel(n_jobs=n_cpu)(delayed(worker)(l) for l in lambdas)
