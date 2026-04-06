import EDLDE
import random as rd
import pickle
import numpy as np

n_per_group = 25
n_groups = 4
t_start = 0
t_end = 300

basis_num_communities = 2
powers_num_communities = [2, 1, 0]
list_p_within_community = [45/50] * len(powers_num_communities)

inter_tau_1 = 10
density_1 = 5

inter_tau_2 = 15
density_2 = 5

inter_tau_3 = 20
density_3 = 5

rd.seed(34)

t_split_1 = 100
t_split_2 = 200
# Phase 1: [t0, t_split]
n1, s1, e1 = EDLDE.EDLDE(
    density=density_1, inter_tau=inter_tau_1, t_start=t_start, t_end=t_split_1, seed=1415)
# Phase 2: [t_split, t_end]
n2, s2, e2 = EDLDE.EDLDE(
    density=density_2, inter_tau=inter_tau_2, t_start=t_split_1, t_end=t_split_2, seed=8281
)
# Phase 3: [t_split_2, t_end]
n3, s3, e3 = EDLDE.EDLDE(
    density=density_3, inter_tau=inter_tau_3, t_start=t_split_2, t_end=t_end, seed=77
)
# Combine
starts = np.concatenate([s1, s2, s3])
ends   = np.concatenate([e1, e2, e3])

tnet = EDLDE.generate_smooth_SBM(inter_tau =  0, density = 0,
                        n_per_group = n_per_group, n_groups =n_groups,
                        t_start = t_start, t_end =t_end,
                        basis_num_communities = basis_num_communities, powers_num_communities = powers_num_communities, list_p_within_community = list_p_within_community,
                        number_of_events = n1+n2+n3, starting_times=starts, ending_times=ends, seed=271)

pickle.dump(tnet, open('data/merge_merge.pkl', 'wb'))


basis_num_communities = 2
powers_num_communities = [2, 0, 1]
list_p_within_community = [45/50] * len(powers_num_communities)

inter_tau_1 = 10
density_1 = 5

inter_tau_2 = 20
density_2 = 5

inter_tau_3 = 15
density_3 = 5

rd.seed(34)

t_split_1 = 100
t_split_2 = 200
# Phase 1: [t0, t_split]
n1, s1, e1 = EDLDE.EDLDE(
    density=density_1, inter_tau=inter_tau_1, t_start=t_start, t_end=t_split_1, seed=1415)
# Phase 2: [t_split, t_end]
n2, s2, e2 = EDLDE.EDLDE(
    density=density_2, inter_tau=inter_tau_2, t_start=t_split_1, t_end=t_split_2, seed=8281
)
# Phase 3: [t_split_2, t_end]
n3, s3, e3 = EDLDE.EDLDE(
    density=density_3, inter_tau=inter_tau_3, t_start=t_split_2, t_end=t_end, seed=77
)
# Combine
starts = np.concatenate([s1, s2, s3])
ends   = np.concatenate([e1, e2, e3])

tnet = EDLDE.generate_smooth_SBM(inter_tau =  0, density = 0,
                        n_per_group = n_per_group, n_groups =n_groups,
                        t_start = t_start, t_end =t_end,
                        basis_num_communities = basis_num_communities, powers_num_communities = powers_num_communities, list_p_within_community = list_p_within_community,
                        number_of_events = n1+n2+n3, starting_times=starts, ending_times=ends, seed=271)


pickle.dump(tnet, open('data/merge_split.pkl', 'wb'))


basis_num_communities = 2
powers_num_communities = [1, 2, 0]
list_p_within_community = [45/50] * len(powers_num_communities)

inter_tau_1 = 15
density_1 = 5

inter_tau_2 = 10
density_2 = 5

inter_tau_3 = 20
density_3 = 5

rd.seed(34)

t_split_1 = 100
t_split_2 = 200
# Phase 1: [t0, t_split]
n1, s1, e1 = EDLDE.EDLDE(
    density=density_1, inter_tau=inter_tau_1, t_start=t_start, t_end=t_split_1, seed=1415)
# Phase 2: [t_split, t_end]
n2, s2, e2 = EDLDE.EDLDE(
    density=density_2, inter_tau=inter_tau_2, t_start=t_split_1, t_end=t_split_2, seed=8281
)
# Phase 3: [t_split_2, t_end]
n3, s3, e3 = EDLDE.EDLDE(
    density=density_3, inter_tau=inter_tau_3, t_start=t_split_2, t_end=t_end, seed=77
)
# Combine
starts = np.concatenate([s1, s2, s3])
ends   = np.concatenate([e1, e2, e3])

tnet = EDLDE.generate_smooth_SBM(inter_tau =  0, density = 0,
                        n_per_group = n_per_group, n_groups =n_groups,
                        t_start = t_start, t_end =t_end,
                        basis_num_communities = basis_num_communities, powers_num_communities = powers_num_communities, list_p_within_community = list_p_within_community,
                        number_of_events = n1+n2+n3, starting_times=starts, ending_times=ends, seed=271)

pickle.dump(tnet, open('data/split_merge.pkl', 'wb'))