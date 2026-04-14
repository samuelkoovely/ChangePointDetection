from __future__ import annotations

import pickle
import random as rd
from pathlib import Path

import numpy as np

import EDLDE


TRAIN_NUM_SAMPLES = 5
TEST_NUM_SAMPLES = 10

TRAIN_RANDOM_SEED = 34
TEST_RANDOM_SEED = 1034
TEST_SAMPLE_INDEX_OFFSET = 1000

n_per_group = 100
n_groups = 1
t_start = 0
t_end = 200

basis_num_communities = 2
powers_num_communities = [0]
list_p_within_community = [45 / 50] * len(powers_num_communities)

REGIMES = (
    {"inter_tau": 2, "density": 100},
    {"inter_tau": 2, "density": 150},
)

MIN_NUM_BREAKPOINTS = 1
MAX_NUM_BREAKPOINTS = 4
BREAKPOINT_LOWER_BOUND = 30
BREAKPOINT_UPPER_BOUND = 170

EVENT_SEED_BASE = 1415
SBM_SEED_BASE = 271

TRAIN_OUTPUT_PATH = Path("data/multibkps_block2activities.pkl")
TEST_OUTPUT_PATH = Path("data/multibkps_block2activities_test.pkl")


def sample_breakpoints(rng: rd.Random) -> list[int]:
    num_breakpoints = rng.randint(MIN_NUM_BREAKPOINTS, MAX_NUM_BREAKPOINTS)
    return sorted(
        rng.sample(
            range(BREAKPOINT_LOWER_BOUND, BREAKPOINT_UPPER_BOUND + 1),
            num_breakpoints,
        )
    )


def generate_piecewise_activity(
    sample_index: int,
    breakpoints: list[int],
) -> tuple[int, np.ndarray, np.ndarray]:
    boundaries = [t_start, *breakpoints, t_end]
    number_of_events = 0
    all_starts: list[np.ndarray] = []
    all_ends: list[np.ndarray] = []

    for segment_index, (segment_start, segment_end) in enumerate(
        zip(boundaries[:-1], boundaries[1:])
    ):
        regime = REGIMES[segment_index % len(REGIMES)]
        seed = EVENT_SEED_BASE + sample_index * 100 + segment_index
        n_events, starts, ends = EDLDE.EDLDE(
            density=regime["density"],
            inter_tau=regime["inter_tau"],
            t_start=segment_start,
            t_end=segment_end,
            seed=seed,
        )
        number_of_events += n_events
        all_starts.append(starts)
        all_ends.append(ends)

    return number_of_events, np.concatenate(all_starts), np.concatenate(all_ends)


def generate_sample(sample_index: int, breakpoints: list[int]) -> dict[str, object]:
    number_of_events, starts, ends = generate_piecewise_activity(sample_index, breakpoints)

    tnet = EDLDE.generate_smooth_SBM(
        inter_tau=0,
        density=0,
        n_per_group=n_per_group,
        n_groups=n_groups,
        t_start=t_start,
        t_end=t_end,
        basis_num_communities=basis_num_communities,
        powers_num_communities=powers_num_communities,
        list_p_within_community=list_p_within_community,
        number_of_events=number_of_events,
        starting_times=starts,
        ending_times=ends,
        seed=SBM_SEED_BASE + sample_index,
    )

    first_regime = REGIMES[0]
    tnet_trimmed, t_0 = EDLDE.trim_temporal_network_head_tail(
        tnet,
        density=first_regime["density"],
        inter_tau=first_regime["inter_tau"],
        tail_start_time=t_end,
    )

    trimmed_breakpoints = [
        float(breakpoint - t_0)
        for breakpoint in breakpoints
        if t_0 < breakpoint < t_end
    ]

    return {
        "tnet": tnet_trimmed,
        "bkps": trimmed_breakpoints,
        "n_bkps": len(trimmed_breakpoints),
        "starts": tnet_trimmed.events_table["starting_times"],
        "ends": tnet_trimmed.events_table["ending_times"],
    }


def generate_dataset(
    num_samples: int,
    random_seed: int,
    sample_index_offset: int = 0,
) -> list[dict[str, object]]:
    rng = rd.Random(random_seed)
    return [
        generate_sample(
            sample_index=sample_index_offset + sample_number,
            breakpoints=sample_breakpoints(rng),
        )
        for sample_number in range(num_samples)
    ]


def write_dataset(dataset: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as handle:
        pickle.dump(dataset, handle)


def main() -> None:
    training_dataset = generate_dataset(
        num_samples=TRAIN_NUM_SAMPLES,
        random_seed=TRAIN_RANDOM_SEED,
    )
    test_dataset = generate_dataset(
        num_samples=TEST_NUM_SAMPLES,
        random_seed=TEST_RANDOM_SEED,
        sample_index_offset=TEST_SAMPLE_INDEX_OFFSET,
    )

    write_dataset(training_dataset, TRAIN_OUTPUT_PATH)
    write_dataset(test_dataset, TEST_OUTPUT_PATH)


if __name__ == "__main__":
    main()
