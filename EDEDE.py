from scipy.stats import uniform, expon, poisson
import numpy as np
from TemporalNetwork import ContTempNetwork
from typing import Sequence, List, Tuple, Any

def EDEDE(density = 2, inter_tau = 2, t_start = 0, t_end = 300, seed = 314):

    number_of_events = poisson.rvs(size = 1, mu = (t_end - t_start) * density, random_state = seed)[0]
    starting_times = np.sort(uniform.rvs(size=number_of_events, loc=t_start, scale=t_end - t_start, random_state=seed))
    ending_times = expon.rvs(size=number_of_events, scale= inter_tau, random_state=seed)
    ending_times = ending_times + starting_times
    return number_of_events, starting_times, ending_times


def activity_EDEDE(starting_times: Sequence[Any],
                    ending_times:   Sequence[Any]) -> Tuple[List[Any], List[int]]:
    """
    Compute the piecewise-constant active-event count over time for intervals [start, end).

    Parameters
    ----------
    starting_times : sequence of comparable timestamps (float/int/datetime, etc.)
    ending_times   : sequence of comparable timestamps, same length as starting_times

    Returns
    -------
    change_times : list
        Sorted timestamps where the count value changes.
    counts_after : list of int
        Active count immediately AFTER each timestamp in change_times (right-continuous).

    Notes
    -----
    - Intervals are treated as half-open [start, end): starts add +1 at 'start', ends add -1 at 'end'.
    - If multiple starts/ends share the same timestamp, their net effect is combined;
      if the net change at a time is zero, that time is omitted (no change in the value).
    """
    if len(starting_times) != len(ending_times):
        raise ValueError("starting_times and ending_times must have the same length.")

    # Optional sanity check (kept on by default for safety)
    for i, (s, e) in enumerate(zip(starting_times, ending_times)):
        if not (s < e):
            raise ValueError(f"Invalid interval at index {i}: start={s} must be < end={e}.")

    # Accumulate +1 at starts and -1 at ends
    delta_by_time = {}
    for s in starting_times:
        delta_by_time[s] = delta_by_time.get(s, 0) + 1
    for e in ending_times:
        delta_by_time[e] = delta_by_time.get(e, 0) - 1

    change_times: List[Any] = []
    counts_after: List[int] = []
    current = 0

    for t in sorted(delta_by_time):
        delta = delta_by_time[t]
        if delta != 0:  # skip timestamps with no net change
            current += delta
            change_times.append(t)
            counts_after.append(current)

    return change_times, counts_after

def make_step_block_probs(
    deltat1,
    n_groups=27, n_per_group=3,
    basis_num_communities=3,
    powers_num_communities=[3, 2, 1],
    list_p_within_community=None,
):
    """
    Returns a function that generates the block probability matrix as a function of time.

    - You can include 0 in powers_num_communities; 0 -> single community.
    """

    if list_p_within_community is None:
        list_p_within_community = [49/50] * len(powers_num_communities)

    list_num_communities = list(np.power(basis_num_communities, powers_num_communities))

    def calculate_pout(p_within, community_size, total_size):
        # Single-community case: no out-group; pout will not be used.
        if community_size == total_size:
            return 0.0
        k = (1 / p_within - community_size) / (total_size - community_size)
        return k * p_within

    def generate_block_matrix(num_communities, p_within_base):
        community_size = n_groups // num_communities
        total_size = n_groups

        p_within = p_within_base / community_size
        p_within_outgroup = p_within_base / (community_size * n_per_group - 1)

        pout = calculate_pout(p_within, community_size, total_size) / n_per_group

        n_nodes = n_groups * n_per_group
        block_matrix = np.zeros((n_nodes, n_nodes), dtype=float)

        # Fill within-community blocks
        for i in range(num_communities):
            start = i * community_size * n_per_group
            end = start + community_size * n_per_group
            block_matrix[start:end, start:end] = p_within_outgroup

        # If num_communities == 1, the whole matrix (except diag) is already filled;
        # for >1 we set the remaining entries to pout.
        if num_communities > 1:
            block_matrix[block_matrix == 0] = pout

        # Remove self-events
        np.fill_diagonal(block_matrix, 0.0)

        # --- Row normalisation to make each row sum to 1 ---
        row_sums = block_matrix.sum(axis=1, keepdims=True)

        # Handle possible zero rows by assigning uniform over all *other* nodes
        zero_rows = (row_sums == 0.0).flatten()
        if np.any(zero_rows):
            for idx in np.where(zero_rows)[0]:
                block_matrix[idx, :] = 1.0
                block_matrix[idx, idx] = 0.0  # no self
            row_sums = block_matrix.sum(axis=1, keepdims=True)

        block_matrix /= row_sums

        return block_matrix

    # Precompute matrices
    stage_matrices = []
    for num_communities, p_within_community in zip(list_num_communities, list_p_within_community):
        stage_matrices.append(generate_block_matrix(num_communities, p_within_community))

    def block_mod_func(t):
        total_stages = len(stage_matrices)
        stage_duration = deltat1

        for stage_index in range(total_stages):
            if stage_index * stage_duration <= t < (stage_index + 1) * stage_duration:
                return stage_matrices[stage_index]

        print("Warning: t is out of bounds. Returning identity matrix.")
        n_nodes = n_groups * n_per_group
        return np.eye(n_nodes)

    return block_mod_func

# def make_step_block_probs(deltat1,
#                           n_groups=27, n_per_group = 3,
#                           basis_num_communities = 3, powers_num_communities = [3, 2, 1], list_p_within_community = [49/50] * len([27, 9, 3])):
#     """
#     Returns a function that generates the block probability matrix as a function of time.

#     Parameters:
#     - deltat1 (int): Length of each temporal step.
#     - n_groups (int): Total number of cores in the network.
#     - basis_num_communities (int): The base number of communities for symmetry.
#     - powers_num_communities (list of int): Powers of the base defining the number of communities in each stage.
#     - list_p_within_community (list of float): List of within-community interaction probabilities for each stage.

#     """
    
#     list_num_communities=list(np.power(basis_num_communities, powers_num_communities))
    
#     def calculate_pout(p_within, community_size, total_size):
#         k = (1 / p_within - community_size) / (total_size - community_size)
#         return k * p_within

#     def generate_block_matrix(num_communities, p_within_base):
#         # Determine community sizes
#         community_size = n_groups // num_communities
        
#         p_within = p_within_base / community_size
#         p_within_outgroup = p_within_base / (community_size * n_per_group -1)  # Adjust to avoid self_events
        
#         pout = calculate_pout(p_within, community_size, n_groups) / n_per_group

#         # Create block matrix
#         block_matrix = np.zeros((n_groups*n_per_group, n_groups*n_per_group))
#         for i in range(num_communities):
#             start = i * community_size*n_per_group
#             end = start + community_size*n_per_group
#             block_matrix[start:end, start:end] = p_within_outgroup 
 
#         block_matrix[block_matrix == 0] = pout
#         np.fill_diagonal(block_matrix, 0 , wrap=False) # Correction for self_events
        
#         return block_matrix

#     # Precompute the matrices for different stages
#     stage_matrices = []
#     for num_communities, p_within_community in zip(list_num_communities, list_p_within_community):
#         stage_matrices.append(generate_block_matrix(num_communities, p_within_community))

#     def block_mod_func(t):
#         total_stages = len(stage_matrices)
#         stage_duration = deltat1

#         for stage_index in range(total_stages):
#             if t >= stage_index * stage_duration and t < (stage_index + 1) * stage_duration:
#                 return stage_matrices[stage_index]

#         print("Warning: t is out of bounds. Returning identity matrix.")
#         return np.eye(n_groups)

#     return block_mod_func


def generate_smooth_SBM(
    density=2, inter_tau=2,
    n_per_group=10, n_groups=27,
    t_start=0, t_end=300,
    basis_num_communities=3,
    powers_num_communities=[3, 2, 1],
    list_p_within_community=None,
    number_of_events=None, starting_times=None, ending_times=None, seed=271):

    # Create a dedicated RNG for this function call
    rng = np.random.default_rng(seed)

    if number_of_events is None or starting_times is None or ending_times is None:
        number_of_events, starting_times, ending_times = EDEDE(
            density=density,
            inter_tau=inter_tau,
            t_start=t_start,
            t_end=t_end,
            seed=rng)

    if list_p_within_community is None:
        list_p_within_community = [49/50] * len(powers_num_communities)
    
    source_nodes = rng.choice(n_groups * n_per_group, number_of_events, replace=True)

    block_mod_func = make_step_block_probs(
        deltat1=(t_end - t_start) / len(powers_num_communities),
        n_groups=n_groups, n_per_group=n_per_group,
        basis_num_communities=basis_num_communities,
        powers_num_communities=powers_num_communities,
        list_p_within_community=list_p_within_community)

    target_nodes = []
    for i, source in enumerate(source_nodes):
        probs = block_mod_func(starting_times[i])[source]

        # Numerical safety: ensure non-negative and row-sum 1
        probs = np.maximum(probs, 0.0)
        s = probs.sum()
        if s <= 0:
            # Fallback: uniform over all other nodes
            probs = np.ones_like(probs)
            probs[source] = 0.0
            s = probs.sum()
        probs = probs / s

        target_nodes.append(
            rng.choice(n_groups * n_per_group, 1, p=probs)[0]
        )

    temporal_net = ContTempNetwork(
        source_nodes=source_nodes,
        target_nodes=target_nodes,
        starting_times=starting_times,
        ending_times=ending_times,
        merge_overlapping_events=True)

    return temporal_net

# def generate_evolving_SBM(inter_tau = 2, density = 2,
#                           n_per_group = 10, n_groups = 27,
#                           t_start = 0, t_end = 300,
#                           basis_num_communities = 3, powers_num_communities = [3, 2, 1], list_p_within_community = [49/50] * len([27, 9, 3])):

#     number_of_events, starting_times, ending_times = EDEDE(density = density, inter_tau = inter_tau, t_start = t_start, t_end = t_end)

#     source_nodes = np.random.choice(n_groups * n_per_group, number_of_events, replace=True)
#     block_mod_func = make_step_block_probs(deltat1 = (t_end - t_start) / len(powers_num_communities),
#                                            n_groups=n_groups, n_per_group=n_per_group,
#                                            basis_num_communities=basis_num_communities, powers_num_communities=powers_num_communities, list_p_within_community=list_p_within_community)


#     target_nodes = []
#     for i, source in enumerate(source_nodes):
        
#         target_nodes.append(np.random.choice(n_groups * n_per_group, 1, p=block_mod_func(starting_times[i])[source])[0])
    
#     temporal_net = ContTempNetwork(source_nodes=source_nodes,
#                             target_nodes=target_nodes,
#                             starting_times=starting_times,
#                             ending_times=ending_times,
#                             merge_overlapping_events=True)
    
#     return temporal_net