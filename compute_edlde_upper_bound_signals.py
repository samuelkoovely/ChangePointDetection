import argparse
import pickle
from pathlib import Path

from compute_global_entropy_motifs import clear_lambda_cache, compute_global_entropy_curve
from edlde_global_entropy_upper_bound_common import (
    DEFAULT_LAMBDAS,
    DEFAULT_NETWORK_DIR,
    DEFAULT_RATES,
    DEFAULT_RESULTS_DIR,
    ensure_plotting_env,
    load_rate_network,
    rate_limit_path,
    rate_results_dir,
    rate_signal_dir,
    signal_filename,
    write_summary,
)
from signal_generation import ensure_laplacians

ensure_plotting_env()

from fig_global_entropy import compute_global_limit_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute cached forward entropy curves and upper bounds for the EDLDE upper-bound networks."
    )
    parser.add_argument(
        "--rates",
        nargs="+",
        type=int,
        default=list(DEFAULT_RATES),
        help="Logical panel identifiers to process.",
    )
    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=list(DEFAULT_LAMBDAS),
        help="Diffusion-rate values to evaluate.",
    )
    parser.add_argument(
        "--network-dir",
        type=Path,
        default=DEFAULT_NETWORK_DIR,
        help="Directory containing cached temporal networks.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory where entropy curves and upper bounds will be stored.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute cached curve and limit payloads.",
    )
    return parser.parse_args()


def compute_rate_curves(
    rate: int,
    net,
    network_path: str,
    lambdas,
    results_dir: Path,
    force: bool,
) -> None:
    rate_dir = rate_results_dir(rate=rate, results_dir=results_dir)
    signal_dir = rate_signal_dir(rate=rate, results_dir=results_dir)
    signal_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for stale_path in signal_dir.glob("global_S*"):
            stale_path.unlink()

    metadata_path = rate_dir / "metadata_forward.pkl"
    shared_metadata_path = rate_dir / "metadata.pkl"

    ensure_laplacians(net)
    for lamda in lambdas:
        signal_path = signal_dir / signal_filename(lamda)
        if signal_path.exists() and not force:
            continue
        payload = compute_global_entropy_curve(net=net, lamda=float(lamda), reverse_time=False)
        with signal_path.open("wb") as handle:
            pickle.dump(payload, handle)
        clear_lambda_cache(net, float(lamda))

    metadata = {
        "rate": int(rate),
        "network_path": str(network_path),
        "forward_lambdas": [float(lamda) for lamda in lambdas],
        "reverse_time": False,
        "direction": "forward",
        "curve_kind": "global_entropy",
        "signal_subdir": "global_S",
        "num_nodes": int(net.num_nodes),
        "num_events": int(net.num_events),
        "num_times": int(len(net.times)),
    }
    with metadata_path.open("wb") as handle:
        pickle.dump(metadata, handle)
    with shared_metadata_path.open("wb") as handle:
        pickle.dump(metadata, handle)


def compute_rate_limit(rate: int, net, results_dir: Path, force: bool) -> None:
    limit_path = rate_limit_path(rate=rate, results_dir=results_dir)
    limit_path.parent.mkdir(parents=True, exist_ok=True)
    if limit_path.exists() and not force:
        return
    payload = compute_global_limit_payload(net, motif_name=f"rate_{float(rate):g}".replace(".", "p"))
    with limit_path.open("wb") as handle:
        pickle.dump(payload, handle)


def compute_signals(rates, lambdas, network_dir: Path, results_dir: Path, force: bool):
    results_dir.mkdir(parents=True, exist_ok=True)
    metadata_by_rate = {}
    for rate in rates:
        net, metadata = load_rate_network(rate=rate, network_dir=network_dir)
        metadata_by_rate[str(int(rate))] = metadata
        compute_rate_curves(
            rate=rate,
            net=net,
            network_path=metadata["network_path"],
            lambdas=lambdas,
            results_dir=results_dir,
            force=bool(force),
        )
        compute_rate_limit(
            rate=rate,
            net=net,
            results_dir=results_dir,
            force=bool(force),
        )

    return write_summary(
        rates=rates,
        lambdas=lambdas,
        network_dir=network_dir,
        results_dir=results_dir,
        metadata_by_rate=metadata_by_rate,
    )


def main() -> None:
    args = parse_args()
    summary_filepath = compute_signals(
        rates=[int(rate) for rate in args.rates],
        lambdas=[float(lamda) for lamda in args.lambdas],
        network_dir=args.network_dir,
        results_dir=args.results_dir,
        force=bool(args.force),
    )
    print(f"Saved summary to {summary_filepath}")


if __name__ == "__main__":
    main()
