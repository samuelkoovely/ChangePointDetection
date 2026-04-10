import argparse
from pathlib import Path

from edlde_global_entropy_upper_bound_common import (
    DEFAULT_NETWORK_DIR,
    DEFAULT_RATES,
    ensure_sampled_network,
    write_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and cache the EDLDE upper-bound temporal networks."
    )
    parser.add_argument(
        "--rates",
        nargs="+",
        type=int,
        default=list(DEFAULT_RATES),
        help="Logical panel identifiers to generate.",
    )
    parser.add_argument(
        "--network-dir",
        type=Path,
        default=DEFAULT_NETWORK_DIR,
        help="Directory where sampled temporal networks will be stored.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("gridsearch_results/edlde_upper_bound"),
        help="Directory where the shared summary file will be updated.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate cached network samples.",
    )
    return parser.parse_args()


def generate_networks(rates, network_dir: Path, results_dir: Path, force: bool):
    network_dir.mkdir(parents=True, exist_ok=True)
    metadata_by_rate = {}
    for rate in rates:
        _, metadata = ensure_sampled_network(
            rate=int(rate),
            network_dir=network_dir,
            force=bool(force),
        )
        metadata_by_rate[str(int(rate))] = metadata
    return write_summary(
        rates=rates,
        lambdas=[],
        network_dir=network_dir,
        results_dir=results_dir,
        metadata_by_rate=metadata_by_rate,
    )


def main() -> None:
    args = parse_args()
    summary_filepath = generate_networks(
        rates=[int(rate) for rate in args.rates],
        network_dir=args.network_dir,
        results_dir=args.results_dir,
        force=bool(args.force),
    )
    print(f"Saved summary to {summary_filepath}")


if __name__ == "__main__":
    main()
