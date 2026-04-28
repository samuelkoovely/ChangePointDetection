from __future__ import annotations

import argparse
import gzip
import math
import os
import pickle
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from joblib import Parallel, delayed

from FlowStability import (
    FlowIntegralClustering,
    SparseClustering,
    avg_norm_var_information,
    run_multi_louvain,
)
from TemporalNetwork import ContTempNetwork
from primary_school_compute import load_primary_school_day1_network


DEFAULT_NETWORK_PATH = Path("./data/primaryschoolnet")
DEFAULT_RUPTURES_RESULTS_PATH = Path(
    "./gridsearch_results/primaryschool_day1_ruptures/forward/window_3600/lamda_1.00000000000/ruptures_results.pkl"
)
DEFAULT_OUTPUT_BASE = Path("//scratch/tmp/180/skoove/primaryschoolnet_heat/primaryschool_day1_flow_clustering")
DEFAULT_PENALTY = 60.0
DEFAULT_LAMBDAS = np.logspace(-5, 0, 200)
DEFAULT_NUM_REPEAT = 100
DEFAULT_N_META_ITER_MAX = 1000
DEFAULT_N_SUB_ITER_MAX = 1000
DEFAULT_TRANSITION_PREFIX = "T"
DEFAULT_INTER_T_PREFIX = "inter_T"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run primary-school day-1 flow-stability clustering on all "
            "subintervals induced by entropy-signal change points."
        )
    )
    parser.add_argument(
        "--matrix-source",
        choices=("T", "inter_T"),
        default="T",
        help=(
            "Which precomputed matrix family to load. "
            "`T` uses saved transition matrices directly; `inter_T` rebuilds "
            "interval-local T inside FlowIntegralClustering."
        ),
    )
    parser.add_argument(
        "--transition-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing saved transition payloads. For `T`, this can be "
            "either a directory with full-day files like T<lambda> or a parent "
            "directory containing interval-local subfolders like Tplot146_566/."
        ),
    )
    parser.add_argument(
        "--transition-prefix",
        type=str,
        default=DEFAULT_TRANSITION_PREFIX,
        help="Filename prefix used before the lambda value inside transition-dir.",
    )
    parser.add_argument(
        "--transition-layout",
        choices=("auto", "full_day_cumulative", "interval_local_subdirs"),
        default="auto",
        help=(
            "How saved `T` payloads are organized. "
            "`full_day_cumulative` expects one full-day file per lambda. "
            "`interval_local_subdirs` expects old-style Tplot<start>_<stop>/T<lambda> "
            "files for each interval. `auto` tries interval-local first, then full-day."
        ),
    )
    parser.add_argument(
        "--inter-transition-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing one saved full-day inter-transition payload "
            "per lambda, usually named like inter_T<lambda>. "
            "Used only when --matrix-source=inter_T."
        ),
    )
    parser.add_argument(
        "--inter-transition-prefix",
        type=str,
        default=DEFAULT_INTER_T_PREFIX,
        help="Filename prefix used before the lambda value inside inter-transition-dir.",
    )
    parser.add_argument(
        "--network-path",
        type=Path,
        default=DEFAULT_NETWORK_PATH,
        help="Primary-school network path used to recover the day-1 time grid.",
    )
    parser.add_argument(
        "--ruptures-results-path",
        type=Path,
        default=DEFAULT_RUPTURES_RESULTS_PATH,
        help="Saved primary-school ruptures results used to define the intervals.",
    )
    parser.add_argument(
        "--penalty",
        type=float,
        default=DEFAULT_PENALTY,
        help="Penalty value whose detected change points define the subintervals.",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=DEFAULT_OUTPUT_BASE,
        help="Directory where clustering results and metadata will be saved.",
    )
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[float(lamda) for lamda in DEFAULT_LAMBDAS],
        help="Lambda values to scan.",
    )
    parser.add_argument(
        "--num-repeat",
        type=int,
        default=DEFAULT_NUM_REPEAT,
        help="Number of repeated Louvain runs per lambda and per interval.",
    )
    parser.add_argument(
        "--n-meta-iter-max",
        type=int,
        default=DEFAULT_N_META_ITER_MAX,
        help="Maximum number of Louvain meta-iterations.",
    )
    parser.add_argument(
        "--n-sub-iter-max",
        type=int,
        default=DEFAULT_N_SUB_ITER_MAX,
        help="Maximum number of Louvain sub-iterations.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=min(os.cpu_count() or 1, len(DEFAULT_LAMBDAS)),
        help="Number of parallel lambda workers.",
    )
    parser.add_argument(
        "--include-full-interval",
        action="store_true",
        help="Also compute the full-day clustering under the legacy folder name clustersplot.",
    )
    parser.add_argument(
        "--reverse-time",
        action="store_true",
        help="Pass reverse_time=True to FlowIntegralClustering.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing per-lambda clustering files.",
    )
    return parser.parse_args()


def load_pickle(path: Path) -> Any:
    if path.suffix == ".gz" or path.name.endswith(".pickle.gz"):
        with gzip.open(path, "rb") as handle:
            return pickle.load(handle)
    with open(path, "rb") as handle:
        return pickle.load(handle)


def candidate_matrix_files(stem: Path) -> list[Path]:
    return [
        stem,
        Path(f"{stem}.pickle"),
        Path(f"{stem}.gz"),
        Path(f"{stem}.pickle.gz"),
    ]


def find_matching_float_key(mapping: dict[Any, Any], target: float, name: str) -> Any:
    for key in mapping:
        if np.isclose(float(key), float(target)):
            return key
    raise ValueError(f"Could not find {name}={target} in saved data.")


def load_ruptures_results(results_path: Path) -> dict[str, Any]:
    with open(results_path, "rb") as handle:
        return pickle.load(handle)


def get_penalty_result(results: dict[str, Any], penalty: float) -> dict[str, Any]:
    for result in results["lambda_results"]:
        if np.isclose(float(result["penalty"]), float(penalty)):
            return result
    raise ValueError(f"Could not find penalty={penalty} in {results}.")


def build_entropy_intervals(
    results: dict[str, Any],
    penalty: float,
    final_stop_index: int,
    time_array: np.ndarray,
    include_full_interval: bool = False,
) -> list[dict[str, Any]]:
    penalty_result = get_penalty_result(results, penalty=penalty)
    k_samples = np.asarray(results["k_samples"], dtype=int)
    cp_signal_indices = np.asarray(penalty_result["change_point_indices"], dtype=int)
    cp_k_samples = (
        k_samples[cp_signal_indices]
        if len(cp_signal_indices) > 0
        else np.empty(0, dtype=int)
    )

    boundaries = [0]
    boundaries.extend(int(value) for value in cp_k_samples.tolist())
    boundaries.append(int(final_stop_index))
    boundaries = sorted(set(boundaries))

    intervals: list[dict[str, Any]] = []
    if include_full_interval:
        intervals.append(
            {
                "label": "full",
                "folder_name": "clustersplot",
                "start": 0,
                "stop": int(final_stop_index),
                "start_time": float(time_array[0]),
                "stop_time": float(time_array[final_stop_index - 1]),
                "start_hour": float(time_array[0]) / 3600.0,
                "stop_hour": float(time_array[final_stop_index - 1]) / 3600.0,
            }
        )

    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if stop - start < 2:
            continue
        intervals.append(
            {
                "label": f"{start}_{stop}",
                "folder_name": f"clustersplot{start}_{stop}",
                "start": int(start),
                "stop": int(stop),
                "start_time": float(time_array[start]),
                "stop_time": float(time_array[stop - 1]),
                "start_hour": float(time_array[start]) / 3600.0,
                "stop_hour": float(time_array[stop - 1]) / 3600.0,
            }
        )

    return intervals


def load_inter_transition_payload(
    inter_transition_dir: Path,
    prefix: str,
    lamda: float,
) -> dict[str, Any]:
    stem = inter_transition_dir / f"{prefix}{float(lamda):.11f}"

    try:
        return ContTempNetwork.load_inter_T(str(stem))
    except Exception:
        pass

    for candidate in candidate_matrix_files(stem):
        if candidate.exists():
            raw = load_pickle(candidate)
            if isinstance(raw, dict) and "inter_T" in raw:
                return raw
            raise ValueError(
                f"Unsupported raw inter-transition format in {candidate}."
            )

    raise FileNotFoundError(
        f"Could not find saved inter-transition matrices for lambda={lamda:.11f} "
        f"under {stem}."
    )


def load_transition_payload(
    transition_dir: Path,
    prefix: str,
    lamda: float,
) -> dict[str, Any]:
    stem = transition_dir / f"{prefix}{float(lamda):.11f}"

    try:
        return ContTempNetwork.load_T(str(stem))
    except Exception:
        pass

    for candidate in candidate_matrix_files(stem):
        if candidate.exists():
            raw = load_pickle(candidate)
            if isinstance(raw, dict) and "T" in raw:
                return raw
            raise ValueError(f"Unsupported raw transition format in {candidate}.")

    raise FileNotFoundError(
        f"Could not find saved transition matrices for lambda={lamda:.11f} "
        f"under {stem}."
    )


def extract_inter_transition_list(
    payload: dict[str, Any],
    lamda: float,
) -> list[Any]:
    inter_t = payload["inter_T"]
    if isinstance(inter_t, list):
        return inter_t
    if isinstance(inter_t, dict):
        key = find_matching_float_key(inter_t, target=lamda, name="lambda")
        return inter_t[key]
    raise TypeError("Unsupported inter_T payload structure.")


def extract_transition_list(
    payload: dict[str, Any],
    lamda: float,
) -> list[Any]:
    transitions = payload["T"]
    if isinstance(transitions, list):
        return transitions
    if isinstance(transitions, dict):
        key = find_matching_float_key(transitions, target=lamda, name="lambda")
        return transitions[key]
    raise TypeError("Unsupported T payload structure.")


def extract_time_array(
    payload: dict[str, Any],
    fallback_time_array: np.ndarray,
) -> np.ndarray:
    if "times_k_start_to_k_stop+1" in payload:
        return np.asarray(payload["times_k_start_to_k_stop+1"], dtype=float)
    return np.asarray(fallback_time_array, dtype=float)


def interval_transition_subdir(interval: dict[str, Any]) -> str:
    if interval["label"] == "full":
        return "Tplot"
    return f"Tplot{interval['start']}_{interval['stop']}"


def has_any_matrix_file(stem: Path) -> bool:
    return any(candidate.exists() for candidate in candidate_matrix_files(stem))


def resolve_transition_payload_for_interval(
    transition_dir: Path,
    transition_prefix: str,
    transition_layout: str,
    interval: dict[str, Any],
    lamda: float,
) -> tuple[dict[str, Any], str]:
    interval_dir = transition_dir / interval_transition_subdir(interval)
    interval_stem = interval_dir / f"{transition_prefix}{float(lamda):.11f}"
    full_day_stem = transition_dir / f"{transition_prefix}{float(lamda):.11f}"

    if transition_layout in {"auto", "interval_local_subdirs"} and has_any_matrix_file(interval_stem):
        return load_transition_payload(interval_dir, transition_prefix, lamda), "interval_local"

    if transition_layout == "interval_local_subdirs":
        raise FileNotFoundError(
            f"Missing interval-local T file for interval {interval['label']} and "
            f"lambda={lamda:.11f} under {interval_stem}."
        )

    if transition_layout in {"auto", "full_day_cumulative"} and has_any_matrix_file(full_day_stem):
        return load_transition_payload(transition_dir, transition_prefix, lamda), "full_day"

    raise FileNotFoundError(
        f"Could not find any saved T payload for interval {interval['label']} and "
        f"lambda={lamda:.11f} under either {interval_stem} or {full_day_stem}."
    )


def cluster_output_path(output_base: Path, folder_name: str, lamda: float) -> Path:
    return output_base / folder_name / f"cluster{float(lamda):.11f}"


def interval_outputs_exist(
    output_base: Path,
    intervals: Sequence[dict[str, Any]],
    lamda: float,
) -> bool:
    return all(
        cluster_output_path(output_base, interval["folder_name"], lamda).exists()
        for interval in intervals
    )


def validate_interval_coverage(
    inter_t_list: Sequence[Any],
    time_array: np.ndarray,
    intervals: Sequence[dict[str, Any]],
) -> None:
    max_stop = max(interval["stop"] for interval in intervals)
    if len(time_array) < max_stop:
        raise ValueError(
            f"Time array has length {len(time_array)} but intervals require "
            f"at least {max_stop} samples."
        )
    if len(inter_t_list) < max_stop - 1:
        raise ValueError(
            f"inter_T list has length {len(inter_t_list)} but intervals require "
            f"at least {max_stop - 1} inter-transition matrices."
        )


def validate_transition_coverage(
    t_list: Sequence[Any],
    time_array: np.ndarray,
    interval: dict[str, Any],
    mode: str,
) -> None:
    start = int(interval["start"])
    stop = int(interval["stop"])

    if mode == "interval_local":
        required_t_len = stop - start - 1
        required_time_len = stop - start
        if len(t_list) < required_t_len:
            raise ValueError(
                f"Interval-local T list for {interval['label']} has length {len(t_list)} "
                f"but requires at least {required_t_len} matrices."
            )
        if len(time_array) < required_time_len:
            raise ValueError(
                f"Interval-local time array for {interval['label']} has length {len(time_array)} "
                f"but requires at least {required_time_len} entries."
            )
        return

    if start != 0:
        raise ValueError(
            "A full-day cumulative T payload cannot be reused directly for "
            f"subinterval {interval['label']} because T(t0, tk) does not determine "
            "T(t_start, tk) without additional information. Use interval-local T "
            "files or keep using saved inter_T."
        )

    if len(t_list) < stop - 1:
        raise ValueError(
            f"Full-day T list has length {len(t_list)} but interval {interval['label']} "
            f"requires at least {stop - 1} matrices."
        )
    if len(time_array) < stop:
        raise ValueError(
            f"Full-day time array has length {len(time_array)} but interval {interval['label']} "
            f"requires at least {stop} entries."
        )


def get_interval_transition_views(
    t_list: Sequence[Any],
    time_array: np.ndarray,
    interval: dict[str, Any],
    mode: str,
) -> tuple[list[Any], np.ndarray]:
    validate_transition_coverage(
        t_list=t_list,
        time_array=time_array,
        interval=interval,
        mode=mode,
    )

    start = int(interval["start"])
    stop = int(interval["stop"])

    if mode == "interval_local":
        return list(t_list[: stop - start - 1]), np.asarray(time_array[: stop - start], dtype=float)

    return list(t_list[: stop - 1]), np.asarray(time_array[:stop], dtype=float)


def compute_interval_result(
    lamda: float,
    interval: dict[str, Any],
    inter_t_list: Sequence[Any],
    time_array: np.ndarray,
    num_repeat: int,
    n_meta_iter_max: int,
    n_sub_iter_max: int,
    reverse_time: bool = False,
) -> dict[str, Any]:
    start = int(interval["start"])
    stop = int(interval["stop"])
    interval_inter_t = list(inter_t_list[start : stop - 1])
    interval_times = np.asarray(time_array[start:stop], dtype=float)

    if len(interval_inter_t) == 0 or len(interval_times) < 2:
        raise ValueError(
            f"Interval {interval['label']} is too short for clustering: "
            f"{len(interval_inter_t)} inter-transition matrices, "
            f"{len(interval_times)} times."
        )

    flow_integral_clustering = FlowIntegralClustering(
        T_inter_list=interval_inter_t,
        time_list=interval_times,
        verbose=False,
        reverse_time=reverse_time,
    )

    clustering = SparseClustering(
        p1=flow_integral_clustering.p1,
        p2=None,
        T=flow_integral_clustering.T_list[-1],
        S=flow_integral_clustering.I_list[0],
    )

    n_loops, cluster_lists, stabilities, seeds = run_multi_louvain(
        clustering,
        num_repeat=num_repeat,
        n_meta_iter_max=n_meta_iter_max,
        n_sub_iter_max=n_sub_iter_max,
        verbose=False,
        print_num_loops=False,
    )

    stabilities_array = np.asarray(stabilities, dtype=float)
    best_index = int(np.argmax(stabilities_array))
    avg_num_clusters = float(
        np.mean([len(cluster_list) for cluster_list in cluster_lists])
    )
    avg_nvi = (
        float(avg_norm_var_information(cluster_lists))
        if len(cluster_lists) > 1
        else math.nan
    )

    return {
        "lamda": float(lamda),
        "interval_label": interval["label"],
        "interval_start": start,
        "interval_stop": stop,
        "interval_start_time": float(interval_times[0]),
        "interval_stop_time": float(interval_times[-1]),
        "interval_start_hour": float(interval_times[0]) / 3600.0,
        "interval_stop_hour": float(interval_times[-1]) / 3600.0,
        "num_repeat": int(num_repeat),
        "reverse_time": bool(reverse_time),
        "cluster_lists": cluster_lists,
        "stabilities": stabilities_array,
        "seeds": list(seeds),
        "n_loops": list(n_loops),
        "best_index": best_index,
        "best_cluster": cluster_lists[best_index],
        "best_stability": float(stabilities_array[best_index]),
        "best_seed": int(seeds[best_index]),
        "best_n_loop": int(n_loops[best_index]),
        "avg_stability": float(np.mean(stabilities_array)),
        "avg_num_clusters": avg_num_clusters,
        "avg_nvi": avg_nvi,
        "time_list": interval_times,
        "compatibility_cluster_list": cluster_lists[best_index],
        "matrix_source": "inter_T",
    }


def compute_interval_result_from_transition_list(
    lamda: float,
    interval: dict[str, Any],
    transition_list: Sequence[Any],
    time_array: np.ndarray,
    num_repeat: int,
    n_meta_iter_max: int,
    n_sub_iter_max: int,
    reverse_time: bool = False,
) -> dict[str, Any]:
    transition_list = list(transition_list)
    interval_times = np.asarray(time_array, dtype=float)

    if len(transition_list) == 0 or len(interval_times) < 2:
        raise ValueError(
            f"Interval {interval['label']} is too short for clustering: "
            f"{len(transition_list)} transition matrices, "
            f"{len(interval_times)} times."
        )

    flow_integral_clustering = FlowIntegralClustering(
        T_list=transition_list,
        time_list=interval_times,
        verbose=False,
        reverse_time=reverse_time,
    )

    clustering = SparseClustering(
        p1=flow_integral_clustering.p1,
        p2=None,
        T=flow_integral_clustering.T_list[-1],
        S=flow_integral_clustering.I_list[0],
    )

    n_loops, cluster_lists, stabilities, seeds = run_multi_louvain(
        clustering,
        num_repeat=num_repeat,
        n_meta_iter_max=n_meta_iter_max,
        n_sub_iter_max=n_sub_iter_max,
        verbose=False,
        print_num_loops=False,
    )

    stabilities_array = np.asarray(stabilities, dtype=float)
    best_index = int(np.argmax(stabilities_array))
    avg_num_clusters = float(
        np.mean([len(cluster_list) for cluster_list in cluster_lists])
    )
    avg_nvi = (
        float(avg_norm_var_information(cluster_lists))
        if len(cluster_lists) > 1
        else math.nan
    )

    return {
        "lamda": float(lamda),
        "interval_label": interval["label"],
        "interval_start": int(interval["start"]),
        "interval_stop": int(interval["stop"]),
        "interval_start_time": float(interval_times[0]),
        "interval_stop_time": float(interval_times[-1]),
        "interval_start_hour": float(interval_times[0]) / 3600.0,
        "interval_stop_hour": float(interval_times[-1]) / 3600.0,
        "num_repeat": int(num_repeat),
        "reverse_time": bool(reverse_time),
        "cluster_lists": cluster_lists,
        "stabilities": stabilities_array,
        "seeds": list(seeds),
        "n_loops": list(n_loops),
        "best_index": best_index,
        "best_cluster": cluster_lists[best_index],
        "best_stability": float(stabilities_array[best_index]),
        "best_seed": int(seeds[best_index]),
        "best_n_loop": int(n_loops[best_index]),
        "avg_stability": float(np.mean(stabilities_array)),
        "avg_num_clusters": avg_num_clusters,
        "avg_nvi": avg_nvi,
        "time_list": interval_times,
        "compatibility_cluster_list": cluster_lists[best_index],
        "matrix_source": "T",
    }


def save_interval_result(
    result: dict[str, Any],
    output_base: Path,
    folder_name: str,
) -> Path:
    outdir = output_base / folder_name
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"cluster{float(result['lamda']):.11f}"
    with open(outfile, "wb") as handle:
        pickle.dump(result, handle)
    return outfile


def worker(
    lamda: float,
    matrix_source: str,
    transition_dir: Path | None,
    transition_prefix: str,
    transition_layout: str,
    inter_transition_dir: Path | None,
    inter_transition_prefix: str,
    intervals: Sequence[dict[str, Any]],
    fallback_time_array: np.ndarray,
    output_base: Path,
    num_repeat: int,
    n_meta_iter_max: int,
    n_sub_iter_max: int,
    reverse_time: bool = False,
    overwrite: bool = False,
) -> float:
    lamda = float(lamda)

    if not overwrite and interval_outputs_exist(output_base, intervals, lamda):
        print(f"skipping lambda={lamda:.11f}: all interval outputs already exist")
        return lamda

    if matrix_source == "inter_T":
        assert inter_transition_dir is not None
        payload = load_inter_transition_payload(
            inter_transition_dir=inter_transition_dir,
            prefix=inter_transition_prefix,
            lamda=lamda,
        )
        inter_t_list = extract_inter_transition_list(payload, lamda=lamda)
        time_array = extract_time_array(payload, fallback_time_array=fallback_time_array)
        validate_interval_coverage(
            inter_t_list=inter_t_list,
            time_array=time_array,
            intervals=intervals,
        )

        for interval in intervals:
            result = compute_interval_result(
                lamda=lamda,
                interval=interval,
                inter_t_list=inter_t_list,
                time_array=time_array,
                num_repeat=num_repeat,
                n_meta_iter_max=n_meta_iter_max,
                n_sub_iter_max=n_sub_iter_max,
                reverse_time=reverse_time,
            )
            save_interval_result(
                result=result,
                output_base=output_base,
                folder_name=interval["folder_name"],
            )
    else:
        assert transition_dir is not None
        for interval in intervals:
            payload, mode = resolve_transition_payload_for_interval(
                transition_dir=transition_dir,
                transition_prefix=transition_prefix,
                transition_layout=transition_layout,
                interval=interval,
                lamda=lamda,
            )
            transition_list = extract_transition_list(payload, lamda=lamda)
            time_array = extract_time_array(
                payload,
                fallback_time_array=fallback_time_array,
            )
            interval_transition_list, interval_times = get_interval_transition_views(
                t_list=transition_list,
                time_array=time_array,
                interval=interval,
                mode=mode,
            )
            result = compute_interval_result_from_transition_list(
                lamda=lamda,
                interval=interval,
                transition_list=interval_transition_list,
                time_array=interval_times,
                num_repeat=num_repeat,
                n_meta_iter_max=n_meta_iter_max,
                n_sub_iter_max=n_sub_iter_max,
                reverse_time=reverse_time,
            )
            result["transition_layout_used"] = mode
            save_interval_result(
                result=result,
                output_base=output_base,
                folder_name=interval["folder_name"],
            )

    print(f"finished lambda={lamda:.11f}")
    return lamda


def save_metadata(
    output_base: Path,
    metadata: dict[str, Any],
) -> None:
    output_base.mkdir(parents=True, exist_ok=True)
    with open(output_base / "metadata.pkl", "wb") as handle:
        pickle.dump(metadata, handle)


def main() -> None:
    args = parse_args()
    lambdas = [float(lamda) for lamda in args.lambdas]

    if args.matrix_source == "T" and args.transition_dir is None:
        raise ValueError("--transition-dir is required when --matrix-source=T.")
    if args.matrix_source == "inter_T" and args.inter_transition_dir is None:
        raise ValueError(
            "--inter-transition-dir is required when --matrix-source=inter_T."
        )

    day1_net, _ = load_primary_school_day1_network(network_path=args.network_path)
    day1_times = np.asarray(day1_net.times, dtype=float)
    final_stop_index = int(len(day1_times))

    ruptures_results = load_ruptures_results(args.ruptures_results_path)
    intervals = build_entropy_intervals(
        results=ruptures_results,
        penalty=float(args.penalty),
        final_stop_index=final_stop_index,
        time_array=day1_times,
        include_full_interval=bool(args.include_full_interval),
    )

    penalty_result = get_penalty_result(ruptures_results, penalty=float(args.penalty))

    metadata = {
        "network_path": str(args.network_path),
        "ruptures_results_path": str(args.ruptures_results_path),
        "matrix_source": args.matrix_source,
        "transition_dir": None if args.transition_dir is None else str(args.transition_dir),
        "transition_prefix": args.transition_prefix,
        "transition_layout": args.transition_layout,
        "inter_transition_dir": (
            None if args.inter_transition_dir is None else str(args.inter_transition_dir)
        ),
        "inter_transition_prefix": args.inter_transition_prefix,
        "output_base": str(args.output_base),
        "penalty": float(args.penalty),
        "lambdas": np.asarray(lambdas, dtype=float),
        "num_repeat": int(args.num_repeat),
        "n_meta_iter_max": int(args.n_meta_iter_max),
        "n_sub_iter_max": int(args.n_sub_iter_max),
        "n_jobs": int(args.n_jobs),
        "reverse_time": bool(args.reverse_time),
        "include_full_interval": bool(args.include_full_interval),
        "intervals": intervals,
        "change_point_signal_indices": np.asarray(
            penalty_result["change_point_indices"],
            dtype=int,
        ),
        "change_point_hours": np.asarray(
            penalty_result["change_point_t_hours"],
            dtype=float,
        ),
        "signal_lambda": float(ruptures_results["lamda"]),
        "signal_window_minutes": float(ruptures_results["window_minutes"]),
        "elapsed_seconds": None,
        "output_layout": "output_base/clustersplot<start>_<stop>/cluster<lambda>",
    }
    save_metadata(args.output_base, metadata)

    t0 = time.time()
    Parallel(n_jobs=int(args.n_jobs), backend="loky")(
        delayed(worker)(
            lamda=lamda,
            matrix_source=args.matrix_source,
            transition_dir=args.transition_dir,
            transition_prefix=args.transition_prefix,
            transition_layout=args.transition_layout,
            inter_transition_dir=args.inter_transition_dir,
            inter_transition_prefix=args.inter_transition_prefix,
            intervals=intervals,
            fallback_time_array=day1_times,
            output_base=args.output_base,
            num_repeat=int(args.num_repeat),
            n_meta_iter_max=int(args.n_meta_iter_max),
            n_sub_iter_max=int(args.n_sub_iter_max),
            reverse_time=bool(args.reverse_time),
            overwrite=bool(args.overwrite),
        )
        for lamda in lambdas
    )

    metadata["elapsed_seconds"] = time.time() - t0
    save_metadata(args.output_base, metadata)
    print(
        f"finished {len(lambdas)} lambdas across {len(intervals)} intervals in "
        f"{metadata['elapsed_seconds']:.2f} s"
    )


if __name__ == "__main__":
    main()
