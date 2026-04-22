from __future__ import annotations

import pickle
import random as rd
from pathlib import Path

import EDLDE


TRAIN_NUM_SAMPLES = 10
TEST_NUM_SAMPLES = 50

TRAIN_RANDOM_SEED = 34
TEST_RANDOM_SEED = 1034
TEST_SAMPLE_INDEX_OFFSET = 1000

n_per_group = 50
n_groups = 2
t_start = 0
t_end = 200

basis_num_communities = 2
powers_num_communities = [1, 0]
list_p_within_community = [45 / 50] * len(powers_num_communities)

inter_tau = 2
density = 100

BREAKPOINT_LOWER_BOUND = 50
BREAKPOINT_UPPER_BOUND = 150
EVENT_SEED_BASE = 1415
SBM_SEED_BASE = 271

TRAIN_OUTPUT_PATH = Path("data/block1activity_train.pkl")
TEST_OUTPUT_PATH = Path("data/block1activity_test.pkl")


def generate_sample(sample_index: int, t_split: int) -> dict[str, object]:
    number_of_events, starting_times, ending_times = EDLDE.EDLDE(
        density=density,
        inter_tau=inter_tau,
        t_start=t_start,
        t_end=t_end,
        seed=EVENT_SEED_BASE + sample_index,
    )

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
        breakpoints=[t_split],
        number_of_events=number_of_events,
        starting_times=starting_times,
        ending_times=ending_times,
        seed=SBM_SEED_BASE + sample_index,
    )

    tnet_trimmed, t_0 = EDLDE.trim_temporal_network_head_tail(
        tnet,
        density=density,
        inter_tau=inter_tau,
        tail_start_time=t_end,
    )

    breakpoint = float(t_split - t_0)
    return {
        "tnet": tnet_trimmed,
        "bkps": [breakpoint],
        "n_bkps": 1,
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
            t_split=rng.randint(BREAKPOINT_LOWER_BOUND, BREAKPOINT_UPPER_BOUND),
        )
        for sample_number in range(num_samples)
    ]


def write_dataset(dataset: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as handle:
        pickle.dump(dataset, handle)


def generate_training_dataset() -> list[dict[str, object]]:
    return generate_dataset(
        num_samples=TRAIN_NUM_SAMPLES,
        random_seed=TRAIN_RANDOM_SEED,
    )


def generate_test_dataset() -> list[dict[str, object]]:
    return generate_dataset(
        num_samples=TEST_NUM_SAMPLES,
        random_seed=TEST_RANDOM_SEED,
        sample_index_offset=TEST_SAMPLE_INDEX_OFFSET,
    )


def main() -> None:
    training_dataset = generate_training_dataset()
    test_dataset = generate_test_dataset()

    write_dataset(training_dataset, TRAIN_OUTPUT_PATH)
    write_dataset(test_dataset, TEST_OUTPUT_PATH)


if __name__ == "__main__":
    main()
