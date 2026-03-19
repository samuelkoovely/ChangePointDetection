import EDLDE
import random as rd
import numpy as np
import matplotlib.pyplot as plt
import pickle

tnets = []

n_per_group = 100
n_groups = 1
t_start = 0
t_end = 200

basis_num_communities = 2
powers_num_communities = [0]
list_p_within_community = [45/50] * len(powers_num_communities)

inter_tau_1 = 2
density_1 = 100

inter_tau_2 = 2
density_2 = 150

rd.seed(34)
for i in range(5):
    t_split = rd.randint(50, 150)
    # Phase 1: [t0, t_split]
    n1, s1, e1 = EDLDE.EDLDE(
        density=density_1, inter_tau=inter_tau_1, t_start=t_start, t_end=t_split, seed=1415+i)
    # Phase 2: [t_split, t_end]
    n2, s2, e2 = EDLDE.EDLDE(
        density=density_2, inter_tau=inter_tau_2, t_start=t_split, t_end=t_end, seed=8281+i
    )
    # Combine
    starts = np.concatenate([s1, s2])
    ends   = np.concatenate([e1, e2])

    tnet = EDLDE.generate_smooth_SBM(inter_tau =  0, density = 0,
                          n_per_group = n_per_group, n_groups =n_groups,
                          t_start = t_start, t_end =t_end,
                          basis_num_communities = basis_num_communities, powers_num_communities = powers_num_communities, list_p_within_community = list_p_within_community,
                          number_of_events = n1+n2, starting_times=starts, ending_times=ends, seed=271+i)
    
    tnet_trimmed, t_0 = EDLDE.trim_temporal_network_head_tail(tnet, density=density_1, inter_tau=inter_tau_1, tail_start_time=t_end)

    tnets.append({'tnet': tnet_trimmed, 'bkp': t_split - t_0, 'starts': tnet_trimmed.events_table['starting_times'], 'ends': tnet_trimmed.events_table['ending_times']})

pickle.dump(tnets, open('block2activities.pkl', 'wb'))