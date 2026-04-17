from __future__ import annotations

import argparse
from pathlib import Path

from sparse_block_activity_common import (
    DEFAULT_DATA_DIR,
    DEFAULT_ENTROPY_WINDOWS,
    DEFAULT_LAMDA,
    SPECS,
    build_signal_bundle,
    compute_entropy_signals,
    get_specs,
    load_pickle,
    normalize_windows,
    sample_path,
    save_pickle,
    signal_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the sparse block-activity local-entropy signal bundles."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[spec.key for spec in SPECS],
        choices=[spec.key for spec in SPECS],
        help="Subset of sparse examples to process.",
    )
    parser.add_argument(
        "--lamda",
        type=float,
        default=DEFAULT_LAMDA,
        help="Entropy scale to evaluate.",
    )
    parser.add_argument(
        "--entropy-windows",
        nargs="+",
        type=float,
        default=DEFAULT_ENTROPY_WINDOWS,
        help="Local-entropy window lengths in seconds.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory where sparse samples and signal bundles are stored.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entropy_windows = normalize_windows(args.entropy_windows)

    for spec in get_specs(args.datasets):
        input_path = sample_path(spec, args.data_dir)
        if not input_path.exists():
            raise FileNotFoundError(
                f"Missing sparse sample {input_path}. Run generate_sparse_block_activity_examples.py first."
            )

        print(
            f"Computing sparse {spec.key} signals with "
            f"lambda={float(args.lamda):.2e}, windows={entropy_windows}"
        )
        sample = load_pickle(input_path)
        signals_by_window = compute_entropy_signals(
            sample=sample,
            lamda=float(args.lamda),
            windows=entropy_windows,
        )
        bundle = build_signal_bundle(
            signals_by_window=signals_by_window,
            lamda=float(args.lamda),
            windows=entropy_windows,
        )
        output_path = signal_path(spec, args.data_dir)
        save_pickle(bundle, output_path)
        print(output_path)


if __name__ == "__main__":
    main()
