from pathlib import Path
import pickle

import numpy as np

import EDLDE


N_PER_GROUP = 25
N_GROUPS = 4
T_START = 0
T_END = 300
T_SPLIT_1 = 100
T_SPLIT_2 = 200

BASIS_NUM_COMMUNITIES = 2
LIST_P_WITHIN_COMMUNITY = [45 / 50] * 3

PHASE_SEEDS = [1415, 8281, 77]
NETWORK_SEED = 271
OUTPUT_DIR = Path("data")

PHASE_INTERVALS = [
    (T_START, T_SPLIT_1),
    (T_SPLIT_1, T_SPLIT_2),
    (T_SPLIT_2, T_END),
]

MOTIF_CONFIGS = [
    {
        "name": "merge_merge",
        "powers_num_communities": [2, 1, 0],
        "phases": [
            {"inter_tau": 5, "density": 10},
            {"inter_tau": 5, "density": 15},
            {"inter_tau": 5, "density": 20},
        ],
    },
    {
        "name": "merge_split",
        "powers_num_communities": [2, 0, 1],
        "phases": [
            {"inter_tau": 5, "density": 10},
            {"inter_tau": 5, "density": 20},
            {"inter_tau": 5, "density": 15},
        ],
    },
    {
        "name": "split_merge",
        "powers_num_communities": [1, 2, 0],
        "phases": [
            {"inter_tau": 5, "density": 15},
            {"inter_tau": 5, "density": 10},
            {"inter_tau": 5, "density": 20},
        ],
    },
]


def build_motif(config):
    counts = []
    starts = []
    ends = []

    for phase_config, phase_seed, (phase_start, phase_end) in zip(
        config["phases"], PHASE_SEEDS, PHASE_INTERVALS
    ):
        number_of_events, phase_starts, phase_ends = EDLDE.EDLDE(
            density=phase_config["density"],
            inter_tau=phase_config["inter_tau"],
            t_start=phase_start,
            t_end=phase_end,
            seed=phase_seed,
        )
        counts.append(number_of_events)
        starts.append(phase_starts)
        ends.append(phase_ends)

    return EDLDE.generate_smooth_SBM(
        inter_tau=0,
        density=0,
        n_per_group=N_PER_GROUP,
        n_groups=N_GROUPS,
        t_start=T_START,
        t_end=T_END,
        basis_num_communities=BASIS_NUM_COMMUNITIES,
        powers_num_communities=config["powers_num_communities"],
        list_p_within_community=LIST_P_WITHIN_COMMUNITY,
        number_of_events=sum(counts),
        starting_times=np.concatenate(starts),
        ending_times=np.concatenate(ends),
        seed=NETWORK_SEED,
    )


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    for config in MOTIF_CONFIGS:
        tnet = build_motif(config)
        output_path = OUTPUT_DIR / f"{config['name']}.pkl"
        with output_path.open("wb") as handle:
            pickle.dump(tnet, handle)


if __name__ == "__main__":
    main()
