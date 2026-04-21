from __future__ import annotations

import argparse
from pathlib import Path

from sparse_block_activity_common import (
    DEFAULT_CT_NUM_SAMPLES,
    DEFAULT_CT_RANDOM_SEED,
    generate_block2_sparse_ct_dataset,
    save_pickle,
)


DEFAULT_OUTPUT_PATH = Path("data/ct_block2activities.pkl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate sparse continuous-time block-2 activity networks."
        )
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_CT_NUM_SAMPLES,
        help="Number of continuous-time networks to generate.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_CT_RANDOM_SEED,
        help="Seed used to sample breakpoint locations.",
    )
    parser.add_argument(
        "--sample-index-offset",
        type=int,
        default=0,
        help="Offset added to per-sample event/SBM seeds.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Where to save the generated dataset pickle.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = generate_block2_sparse_ct_dataset(
        num_samples=args.num_samples,
        random_seed=args.random_seed,
        sample_index_offset=args.sample_index_offset,
    )
    save_pickle(dataset, args.output_path)
    print(args.output_path)


if __name__ == "__main__":
    main()
