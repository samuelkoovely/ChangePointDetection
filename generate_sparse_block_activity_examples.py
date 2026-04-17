from __future__ import annotations

import argparse
from pathlib import Path

from sparse_block_activity_common import (
    DEFAULT_DATA_DIR,
    SPECS,
    generate_sparse_sample,
    get_specs,
    sample_path,
    save_pickle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the sparse block-activity sample pickles."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[spec.key for spec in SPECS],
        choices=[spec.key for spec in SPECS],
        help="Subset of sparse examples to generate.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory where sparse sample pickles will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for spec in get_specs(args.datasets):
        print(f"Generating sparse {spec.key} sample")
        sample = generate_sparse_sample(spec.key)
        output_path = sample_path(spec, args.data_dir)
        save_pickle(sample, output_path)
        print(output_path)


if __name__ == "__main__":
    main()
