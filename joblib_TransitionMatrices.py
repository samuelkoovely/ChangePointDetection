from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path
from typing import Sequence

import numpy as np
from joblib import Parallel, delayed

from integral_clustering import (
    build_entropy_intervals,
    interval_transition_subdir,
    load_ruptures_results,
)
from primary_school_compute import load_primary_school_day1_network


DEFAULT_NETWORK_PATH = Path("./data/primaryschoolnet")
DEFAULT_OUTPUT_BASE = Path("//scratch/tmp/180/skoove/primaryschoolnet_heat/")
DEFAULT_LAMBDAS = np.logspace(-5, 0, 200)
DEFAULT_INTER_T_SUBDIR = "inter_Tselected"
DEFAULT_T_SUBDIR = "T"
DEFAULT_RUPTURES_RESULTS_PATH = Path(
    "./gridsearch_results/primaryschool_day1_ruptures/forward/window_3600/lamda_1.00000000000/ruptures_results.pkl"
)
DEFAULT_PENALTY = 60.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute and save day-1 primary-school inter-transition matrices "
            "and cumulative transition matrices for a range of lambda values."
        )
    )
    parser.add_argument(
        "--network-path",
        type=Path,
        default=DEFAULT_NETWORK_PATH,
        help="Primary-school network path.",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=DEFAULT_OUTPUT_BASE,
        help="Base directory where inter_T and T outputs will be saved.",
    )
    parser.add_argument(
        "--inter-t-subdir",
        type=str,
        default=DEFAULT_INTER_T_SUBDIR,
        help="Subdirectory under output-base used for saved inter_T payloads.",
    )
    parser.add_argument(
        "--t-subdir",
        type=str,
        default=DEFAULT_T_SUBDIR,
        help="Subdirectory under output-base used for saved cumulative T payloads.",
    )
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[float(lamda) for lamda in DEFAULT_LAMBDAS],
        help="Lambda values to compute.",
    )
    parser.add_argument(
        "--ruptures-results-path",
        type=Path,
        default=DEFAULT_RUPTURES_RESULTS_PATH,
        help="Saved primary-school ruptures results used to define entropy intervals.",
    )
    parser.add_argument(
        "--penalty",
        type=float,
        default=DEFAULT_PENALTY,
        help="Penalty value whose detected change points define the subintervals.",
    )
    parser.add_argument(
        "--include-full-interval",
        action="store_true",
        help="Also save the full-day transition list under the legacy Tplot folder.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=min(4, len(DEFAULT_LAMBDAS), os.cpu_count() or 1),
        help="Number of parallel lambda-block workers.",
    )
    parser.add_argument(
        "--dense-expm",
        action="store_true",
        help="Use the dense matrix exponential inside inter_T computations.",
    )
    parser.add_argument(
        "--use-sparse-stoch",
        action="store_true",
        help=(
            "Store inter_T using the custom sparse stochastic format instead of "
            "CSR matrices with delta compression."
        ),
    )
    parser.add_argument(
        "--compress-inter-t",
        action="store_true",
        help="Compress saved inter_T payloads with gzip.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing per-lambda outputs.",
    )
    return parser.parse_args()


def lambda_blocks(lambdas: Sequence[float], n_jobs: int) -> list[list[float]]:
    effective_jobs = max(1, min(int(n_jobs), len(lambdas)))
    blocks = np.array_split(np.asarray(lambdas, dtype=float), effective_jobs)
    return [block.tolist() for block in blocks if len(block) > 0]


def inter_t_stem(output_base: Path, subdir: str, lamda: float) -> Path:
    return output_base / subdir / f"inter_T{float(lamda):.11f}"


def t_path(output_base: Path, subdir: str, lamda: float) -> Path:
    return output_base / subdir / f"T{float(lamda):.11f}"


def inter_t_exists(
    output_base: Path,
    subdir: str,
    lamda: float,
) -> bool:
    stem = inter_t_stem(output_base, subdir, lamda)
    candidates = [
        stem,
        Path(f"{stem}.pickle"),
        Path(f"{stem}.gz"),
        Path(f"{stem}.pickle.gz"),
    ]
    return any(candidate.exists() for candidate in candidates)


def outputs_exist(
    output_base: Path,
    inter_t_subdir: str,
    t_subdir: str,
    lamda: float,
) -> bool:
    return inter_t_exists(output_base, inter_t_subdir, lamda) and t_path(
        output_base,
        t_subdir,
        lamda,
    ).exists()


def interval_t_path(output_base: Path, interval: dict, lamda: float) -> Path:
    return output_base / interval_transition_subdir(interval) / f"T{float(lamda):.11f}"


def interval_t_outputs_exist(
    output_base: Path,
    intervals: Sequence[dict],
    lamda: float,
) -> bool:
    return all(interval_t_path(output_base, interval, lamda).exists() for interval in intervals)


def save_t_payload(
    net,
    lamda: float,
    outfile: Path,
    overwrite: bool = False,
    t_list: Sequence | None = None,
    time_array: np.ndarray | None = None,
    k_start: int | None = None,
    k_stop: int | None = None,
    interval_label: str | None = None,
) -> None:
    if outfile.exists() and not overwrite:
        return

    outfile.parent.mkdir(parents=True, exist_ok=True)
    resolved_t_list = net.T[lamda] if t_list is None else list(t_list)
    resolved_k_start = net._k_start_laplacians if k_start is None else int(k_start)
    resolved_k_stop = net._k_stop_laplacians if k_stop is None else int(k_stop)
    resolved_time_array = (
        np.asarray(
            net.times.values[net._k_start_laplacians : net._k_stop_laplacians + 1],
            dtype=float,
        )
        if time_array is None
        else np.asarray(time_array, dtype=float)
    )
    payload = {
        "lamda": float(lamda),
        "T": resolved_t_list,
        "_k_start_laplacians": resolved_k_start,
        "_k_stop_laplacians": resolved_k_stop,
        "_t_start_laplacians": float(resolved_time_array[0]),
        "_t_stop_laplacians": float(resolved_time_array[-1]),
        "times_k_start_to_k_stop+1": resolved_time_array,
        "num_nodes": int(net.num_nodes),
        "_compute_times": dict(net._compute_times),
        "interval_label": interval_label,
    }
    with open(outfile, "wb") as handle:
        pickle.dump(payload, handle)


def prepare_day1_network(network_path: Path):
    net, day1_stop_time = load_primary_school_day1_network(network_path=network_path)
    net.compute_laplacian_matrices(
        t_start=net.times[0],
        t_stop=net.times[-1],
        random_walk=False,
    )
    return net, day1_stop_time


def compute_interval_transition_lists(
    net,
    lamda: float,
    intervals: Sequence[dict],
) -> dict[str, dict]:
    full_inter_t = list(net.inter_T[lamda])
    original_t = None
    if hasattr(net, "T") and lamda in net.T:
        original_t = net.T[lamda]

    interval_payloads: dict[str, dict] = {}
    for interval in intervals:
        start = int(interval["start"])
        stop = int(interval["stop"])
        interval_inter_t = full_inter_t[start : stop - 1]
        interval_times = np.asarray(net.times.values[start:stop], dtype=float)

        if len(interval_inter_t) == 0 or len(interval_times) < 2:
            continue

        net.inter_T[lamda] = list(interval_inter_t)
        if hasattr(net, "T") and lamda in net.T:
            del net.T[lamda]
        net.compute_transition_matrices(lamda=lamda, reverse_time=False)

        interval_payloads[interval["label"]] = {
            "interval": interval,
            "t_list": list(net.T[lamda]),
            "time_array": interval_times,
        }

        if hasattr(net, "T") and lamda in net.T:
            del net.T[lamda]

    net.inter_T[lamda] = full_inter_t
    if original_t is not None:
        if not hasattr(net, "T"):
            net.T = {}
        net.T[lamda] = original_t

    return interval_payloads


def worker(
    lamda_block: Sequence[float],
    network_path: Path,
    output_base: Path,
    inter_t_subdir: str,
    t_subdir: str,
    intervals: Sequence[dict],
    dense_expm: bool = False,
    use_sparse_stoch: bool = False,
    compress_inter_t: bool = False,
    overwrite: bool = False,
) -> list[float]:
    net, _ = prepare_day1_network(network_path=network_path)
    completed: list[float] = []

    for lamda in [float(value) for value in lamda_block]:
        if (
            not overwrite
            and outputs_exist(output_base, inter_t_subdir, t_subdir, lamda)
            and interval_t_outputs_exist(output_base, intervals, lamda)
        ):
            print(f"skipping lambda={lamda:.11f}: outputs already exist")
            completed.append(lamda)
            continue

        net.compute_inter_transition_matrices(
            lamda=lamda,
            t_start=net.times[0],
            t_stop=net.times[-1],
            dense_expm=bool(dense_expm),
            use_sparse_stoch=bool(use_sparse_stoch),
            random_walk=False,
        )
        net.compute_transition_matrices(
            lamda=lamda,
            reverse_time=False,
        )

        inter_t_outdir = output_base / inter_t_subdir
        inter_t_outdir.mkdir(parents=True, exist_ok=True)
        net.save_inter_T(
            str(inter_t_stem(output_base, inter_t_subdir, lamda)),
            lamda=lamda,
            compressed=bool(compress_inter_t),
            save_delta=not bool(use_sparse_stoch),
            replace_existing=bool(overwrite),
        )

        save_t_payload(
            net=net,
            lamda=lamda,
            outfile=t_path(output_base, t_subdir, lamda),
            overwrite=bool(overwrite),
        )

        interval_payloads = compute_interval_transition_lists(
            net=net,
            lamda=lamda,
            intervals=intervals,
        )
        for interval in intervals:
            interval_result = interval_payloads.get(interval["label"])
            if interval_result is None:
                continue
            save_t_payload(
                net=net,
                lamda=lamda,
                outfile=interval_t_path(output_base, interval, lamda),
                overwrite=bool(overwrite),
                t_list=interval_result["t_list"],
                time_array=interval_result["time_array"],
                k_start=int(interval["start"]),
                k_stop=int(interval["stop"] - 1),
                interval_label=str(interval["label"]),
            )

        if hasattr(net, "inter_T") and lamda in net.inter_T:
            del net.inter_T[lamda]
        if hasattr(net, "T") and lamda in net.T:
            del net.T[lamda]

        print(f"finished lambda={lamda:.11f}")
        completed.append(lamda)

    return completed


def save_metadata(
    output_base: Path,
    metadata: dict,
) -> None:
    output_base.mkdir(parents=True, exist_ok=True)
    with open(output_base / "metadata.pkl", "wb") as handle:
        pickle.dump(metadata, handle)


def main() -> None:
    args = parse_args()
    lambdas = [float(lamda) for lamda in args.lambdas]
    blocks = lambda_blocks(lambdas, n_jobs=args.n_jobs)

    preview_net, day1_stop_time = prepare_day1_network(network_path=args.network_path)
    ruptures_results = load_ruptures_results(args.ruptures_results_path)
    intervals = build_entropy_intervals(
        results=ruptures_results,
        penalty=float(args.penalty),
        final_stop_index=int(len(preview_net.times)),
        time_array=np.asarray(preview_net.times, dtype=float),
        include_full_interval=bool(args.include_full_interval),
    )
    metadata = {
        "network_path": str(args.network_path),
        "ruptures_results_path": str(args.ruptures_results_path),
        "penalty": float(args.penalty),
        "output_base": str(args.output_base),
        "inter_t_subdir": args.inter_t_subdir,
        "t_subdir": args.t_subdir,
        "lambdas": np.asarray(lambdas, dtype=float),
        "n_jobs": int(args.n_jobs),
        "dense_expm": bool(args.dense_expm),
        "use_sparse_stoch": bool(args.use_sparse_stoch),
        "compress_inter_t": bool(args.compress_inter_t),
        "overwrite": bool(args.overwrite),
        "random_walk": False,
        "reverse_time": False,
        "day1_num_nodes": int(preview_net.num_nodes),
        "day1_num_events": int(preview_net.num_events),
        "day1_num_times": int(len(preview_net.times)),
        "day1_start_time": float(preview_net.times[0]),
        "day1_stop_time": float(preview_net.times[-1]),
        "day1_stop_time_reference": float(day1_stop_time),
        "intervals": intervals,
        "output_layout": {
            "inter_T": f"{args.inter_t_subdir}/inter_T<lambda>.pickle or .gz",
            "T": f"{args.t_subdir}/T<lambda>",
            "interval_T": "Tplot<start>_<stop>/T<lambda>",
        },
        "elapsed_seconds": None,
    }
    save_metadata(args.output_base, metadata)

    t0 = time.time()
    Parallel(n_jobs=len(blocks), backend="loky")(
        delayed(worker)(
            lamda_block=block,
            network_path=args.network_path,
            output_base=args.output_base,
            inter_t_subdir=args.inter_t_subdir,
            t_subdir=args.t_subdir,
            intervals=intervals,
            dense_expm=bool(args.dense_expm),
            use_sparse_stoch=bool(args.use_sparse_stoch),
            compress_inter_t=bool(args.compress_inter_t),
            overwrite=bool(args.overwrite),
        )
        for block in blocks
    )
    metadata["elapsed_seconds"] = time.time() - t0
    save_metadata(args.output_base, metadata)

    print(
        f"finished {len(lambdas)} lambdas in {metadata['elapsed_seconds']:.2f} s"
    )
    print(
        "inter_T directory:",
        args.output_base / args.inter_t_subdir,
    )
    print(
        "T directory:",
        args.output_base / args.t_subdir,
    )
    print("interval-local T directories rooted at:", args.output_base)


if __name__ == "__main__":
    main()
