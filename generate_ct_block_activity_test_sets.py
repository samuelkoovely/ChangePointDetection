from __future__ import annotations

import argparse
from pathlib import Path

from ct_examples_common import (
    DEFAULT_CT_TEST_NUM_SAMPLES,
    DEFAULT_CT_TEST_RANDOM_SEED,
    DEFAULT_CT_TEST_SAMPLE_INDEX_OFFSET,
    generate_block1_sparse_ct_dataset,
    generate_block2_sparse_ct_dataset,
    save_pickle,
)


DEFAULT_OUTPUT_PATHS = {
    "block1": Path("data/ct_block1activity_test.pkl"),
    "block2": Path("data/ct_block2activities_test.pkl"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate continuous-time sparse block-activity test datasets."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["block1", "block2"],
        choices=sorted(DEFAULT_OUTPUT_PATHS),
        help="Subset of CT datasets to generate.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_CT_TEST_NUM_SAMPLES,
        help="Number of networks to generate per requested dataset.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_CT_TEST_RANDOM_SEED,
        help="Seed used to sample breakpoint locations.",
    )
    parser.add_argument(
        "--sample-index-offset",
        type=int,
        default=DEFAULT_CT_TEST_SAMPLE_INDEX_OFFSET,
        help="Offset added to per-sample event/SBM seeds.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional override for the output directory. Filenames remain "
            "ct_block1activity_test.pkl and ct_block2activities_test.pkl."
        ),
    )
    return parser.parse_args()


def build_dataset(dataset_key: str, *, num_samples: int, random_seed: int, sample_index_offset: int) -> list[dict]:
    if dataset_key == "block1":
        return generate_block1_sparse_ct_dataset(
            num_samples=num_samples,
            random_seed=random_seed,
            sample_index_offset=sample_index_offset,
        )
    if dataset_key == "block2":
        return generate_block2_sparse_ct_dataset(
            num_samples=num_samples,
            random_seed=random_seed,
            sample_index_offset=sample_index_offset,
        )
    raise ValueError(f"Unsupported dataset key: {dataset_key!r}")


def main() -> None:
    args = parse_args()

    for dataset_key in args.datasets:
        dataset = build_dataset(
            dataset_key,
            num_samples=args.num_samples,
            random_seed=args.random_seed,
            sample_index_offset=args.sample_index_offset,
        )
        output_path = DEFAULT_OUTPUT_PATHS[dataset_key]
        if args.output_dir is not None:
            output_path = Path(args.output_dir) / output_path.name
        save_pickle(dataset, output_path)
        print(output_path)


if __name__ == "__main__":
    main()
