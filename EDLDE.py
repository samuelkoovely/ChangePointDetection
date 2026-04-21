from scipy.stats import uniform, expon, poisson
import numpy as np
from TemporalNetwork import ContTempNetwork
from typing import Sequence, List, Tuple, Any, Optional

def EDLDE(density = 2, inter_tau = 2, t_start = 0, t_end = 300, seed = 314):

    number_of_events = poisson.rvs(size = 1, mu = (t_end - t_start) * density, random_state = seed)[0]
    starting_times = np.sort(uniform.rvs(size=number_of_events, loc=t_start, scale=t_end - t_start, random_state=seed))
    ending_times = expon.rvs(size=number_of_events, scale= inter_tau, random_state=seed)
    ending_times = ending_times + starting_times
    return number_of_events, starting_times, ending_times


def activity_EDLDE(starting_times: Sequence[Any],
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
    deltat1=None,
    n_groups=27, n_per_group=3,
    basis_num_communities=3,
    powers_num_communities=[3, 2, 1],
    list_p_within_community=None,
    t_start=0,
    t_end=None,
    breakpoints=None,
):
    """
    Returns a function that generates the block probability matrix as a function of time.

    - You can include 0 in powers_num_communities; 0 -> single community.
    - If `breakpoints` is provided, it should contain the internal stage boundaries.
      Otherwise the stages are split evenly between `t_start` and `t_end`.
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

    total_stages = len(stage_matrices)
    if breakpoints is None:
        if t_end is not None:
            stage_boundaries = np.linspace(t_start, t_end, total_stages + 1, dtype=float)
        elif deltat1 is not None:
            stage_boundaries = np.asarray(
                [t_start + stage_index * deltat1 for stage_index in range(total_stages + 1)],
                dtype=float)
        else:
            raise ValueError("Either t_end or deltat1 must be provided when breakpoints is None.")
    else:
        if t_end is None:
            raise ValueError("t_end must be provided when breakpoints are specified.")
        if len(breakpoints) != total_stages - 1:
            raise ValueError(
                "breakpoints must contain exactly len(powers_num_communities) - 1 values.")
        stage_boundaries = np.asarray([t_start, *breakpoints, t_end], dtype=float)
        if not np.all(np.diff(stage_boundaries) > 0):
            raise ValueError("breakpoints must be strictly increasing and lie within (t_start, t_end).")

    def block_mod_func(t):
        if stage_boundaries[0] <= t < stage_boundaries[-1]:
            stage_index = np.searchsorted(stage_boundaries, t, side="right") - 1
            return stage_matrices[stage_index]
        if np.isclose(t, stage_boundaries[-1]):
            return stage_matrices[-1]

        print("Warning: t is out of bounds. Returning identity matrix.")
        n_nodes = n_groups * n_per_group
        return np.eye(n_nodes)

    return block_mod_func


def generate_smooth_SBM(
    density=2, inter_tau=2,
    n_per_group=10, n_groups=27,
    t_start=0, t_end=300,
    basis_num_communities=3,
    powers_num_communities=[3, 2, 1],
    list_p_within_community=None,
    breakpoints=None,
    number_of_events=None, starting_times=None, ending_times=None, seed=271):

    # Create a dedicated RNG for this function call
    rng = np.random.default_rng(seed)

    if number_of_events is None or starting_times is None or ending_times is None:
        number_of_events, starting_times, ending_times = EDLDE(
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
        t_start=t_start, t_end=t_end,
        basis_num_communities=basis_num_communities,
        powers_num_communities=powers_num_communities,
        list_p_within_community=list_p_within_community,
        breakpoints=breakpoints)

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


def trim_temporal_network_head_tail(
    temporal_net: ContTempNetwork,
    density: float,
    inter_tau: float,
    tail_start_time: float,
    head_start_time: Optional[float] = None,
    clip_ending_times: bool = True,
    align_first_event_to_zero: bool = True,
    merge_overlapping_events: Optional[bool] = None,
) -> Tuple[ContTempNetwork, float]:
    """Trim the head and tail of a continuous-time temporal network.

    This is useful when the early/late parts of a synthetic time series contain
    transient effects that you want to discard.

    The kept events are those whose lifetimes overlap the kept window
    ``[head, tail_start_time)``:
        ending_times > head and starting_times < tail_start_time

    Events already active at ``head`` are preserved by clipping their starting
    time to ``head``. By default, events extending past ``tail_start_time`` are
    also clipped on the right.

    where the default head cutoff is computed from the exact formula
    with λ = density and μ = 1 / inter_tau:
        head = -inter_tau * np.log(
            (-1 + np.sqrt(1 + 4 * density * inter_tau))
            / (2 * density * inter_tau)
        )

    After filtering and clipping, times are translated so that the first kept
    event starts at 0. By default we shift by the *first kept starting time*
    (which can be > head if no event overlaps the head cutoff), i.e.
        t0 = min(starting_times of kept events)
        starting_times <- starting_times - t0
        ending_times   <- ending_times   - t0

    If `align_first_event_to_zero=False`, we instead shift by `head`.

    Parameters
    ----------
    temporal_net : ContTempNetwork
        Input temporal network.
    density : float
        Event density used to generate the network.
    inter_tau : float
        Mean event duration parameter.
    tail_start_time : float
        Absolute time at which the tail begins. Events with starting times
        before this instant can be kept, but events extending past it are
        clipped on the right when `clip_ending_times=True`.
    head_start_time : float, optional
        If provided, overrides the default head formula.
    clip_ending_times : bool
        If True, clip ending times to `tail_start_time` before shifting.
    align_first_event_to_zero : bool
        If True (default), shift times by the first kept starting time so the
        earliest event starts exactly at 0. If False, shift by `head`.
    merge_overlapping_events : bool, optional
        If True, merges overlapping events in the returned network. If None,
        reuses the input network's merge status when available.

    Returns
    -------
    (ContTempNetwork, float)
        Returns a tuple `(new_net, t0)`, where `new_net` is the trimmed temporal
        network and `t0` is the time shift applied (either the first kept
        starting time, or `head`).

    Raises
    ------
    ValueError
        If parameters are invalid or the trimming removes all events.
    """

    if density <= 0 or inter_tau <= 0:
        raise ValueError("density and inter_tau must be positive.")

    if head_start_time is None:
        val = density * inter_tau
        if val <= 0:
            raise ValueError("density * inter_tau must be > 0 to compute the head cutoff.")
        ratio = (-1.0 + float(np.sqrt(1.0 + 4.0 * val))) / (2.0 * val)
        head = -float(inter_tau) * float(np.log(ratio))
    else:
        head = float(head_start_time)

    tail_start_time = float(tail_start_time)
    if not np.isfinite(tail_start_time):
        raise ValueError("tail_start_time must be finite.")
    if tail_start_time <= head:
        raise ValueError(
            f"tail_start_time must be greater than head (tail_start_time={tail_start_time}, head={head})."
        )

    # Reuse input merge setting if requested
    if merge_overlapping_events is None:
        merge_overlapping_events = bool(getattr(temporal_net, "_overlapping_events_merged", False))

    et = temporal_net.events_table
    mask = (et["ending_times"] > head) & (et["starting_times"] < tail_start_time)
    trimmed = et.loc[mask].copy()

    if trimmed.shape[0] == 0:
        raise ValueError(
            "No events remain after trimming. "
            "Check head/tail cutoffs or the time span of the input network."
        )

    trimmed["starting_times"] = np.maximum(trimmed["starting_times"].to_numpy(), head)

    if clip_ending_times:
        trimmed["ending_times"] = np.minimum(trimmed["ending_times"].to_numpy(), tail_start_time)

    # Shift time so the (kept) series starts at 0.
    # If the network is sparse, there may be no event exactly at `head`, so by
    # default we align the first kept event to 0.
    if align_first_event_to_zero:
        t0 = float(trimmed["starting_times"].min())
    else:
        t0 = head

    trimmed["starting_times"] = trimmed["starting_times"].to_numpy() - t0
    trimmed["ending_times"] = trimmed["ending_times"].to_numpy() - t0

    # Keep the same node labels as the input network
    node_to_label_dict = getattr(temporal_net, "node_to_label_dict", None)

    new_net = ContTempNetwork(
        events_table=trimmed.reset_index(drop=True),
        relabel_nodes=False,
        node_to_label_dict=node_to_label_dict,
        merge_overlapping_events=merge_overlapping_events,
    )

    return new_net, t0
