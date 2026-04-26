from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from gridsearch_score_snapshots import detect_change_points_from_signal
from primary_school_compute import LAMBDAS as DEFAULT_LAMBDAS
from primary_school_compute import OUTPUT_BASE as DEFAULT_SIGNAL_BASE
from primary_school_compute import WINDOWS_SECONDS as DEFAULT_WINDOWS_SECONDS


DEFAULT_OUTPUT_BASE = Path("./gridsearch_results/primaryschool_day1_ruptures")
DEFAULT_WINDOW_SECONDS = float(DEFAULT_WINDOWS_SECONDS[-1])
DEFAULT_PENALTIES = np.asarray(
    [50.0, 55.0, 60.0, 65.0],
    dtype=float,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run penalized ruptures change-point detection on the saved "
            "primary-school day-1 entropy signal."
        )
    )
    parser.add_argument(
        "--signal-base",
        type=Path,
        default=DEFAULT_SIGNAL_BASE,
        help=(
            "Base directory containing the saved primary-school entropy signals. "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=DEFAULT_OUTPUT_BASE,
        help="Directory where the ruptures outputs will be saved.",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=DEFAULT_WINDOW_SECONDS,
        help="Window length in seconds. Default uses the 60-minute signal.",
    )
    parser.add_argument(
        "--lamda",
        type=float,
        default=None,
        help=(
            "Diffusion rate used to select the entropy signal. "
            "Defaults to the largest available lambda."
        ),
    )
    parser.add_argument(
        "--penalties",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Penalty grid passed to ruptures. Default keeps only the "
            "high-penalty primary-school range 50, 55, 60, 65."
        ),
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="linear",
        help="Kernel passed to ruptures.KernelCPD.",
    )
    parser.add_argument(
        "--reverse-time",
        action="store_true",
        help="Use the backward-time primary-school entropy signal layout.",
    )
    return parser.parse_args()


def load_metadata(base: Path) -> dict[str, Any] | None:
    metadata_path = base / "metadata.pkl"
    if not metadata_path.exists():
        return None

    with open(metadata_path, "rb") as handle:
        return pickle.load(handle)


def available_lambdas(metadata: dict[str, Any] | None) -> np.ndarray:
    if metadata is not None and "lambdas" in metadata:
        return np.asarray(metadata["lambdas"], dtype=float)
    return np.asarray(DEFAULT_LAMBDAS, dtype=float)


def resolve_lambda(requested: float | None, metadata: dict[str, Any] | None) -> float:
    lambdas = available_lambdas(metadata)

    if requested is None:
        return float(np.max(lambdas))

    matches = np.flatnonzero(np.isclose(lambdas, float(requested)))
    if matches.size == 0:
        available_text = ", ".join(f"{lamda:.11f}" for lamda in lambdas)
        raise ValueError(
            f"Requested lambda {float(requested):.11f} is not available. "
            f"Available lambdas: {available_text}"
        )

    return float(lambdas[matches[0]])


def load_signal_payload(
    signal_base: Path,
    window_seconds: float,
    lamda: float,
    reverse_time: bool = False,
) -> tuple[dict[str, Any], Path]:
    signal_subdir = "window_S_selected_rev" if reverse_time else "window_S_selected"
    lamda_key = f"{float(lamda):.11f}"
    signal_path = signal_base / signal_subdir / str(int(window_seconds)) / f"window_S{lamda_key}"

    if not signal_path.exists():
        reverse_flag = " --reverse-time" if reverse_time else ""
        raise FileNotFoundError(
            f"Missing signal file {signal_path}. "
            f"Run primary_school_compute.py{reverse_flag} first."
        )

    with open(signal_path, "rb") as handle:
        payload = pickle.load(handle)

    if isinstance(payload, dict) and "error" in payload:
        raise RuntimeError(
            f"Signal file {signal_path} contains an error payload: {payload['error']}"
        )

    return payload, signal_path


def extract_signal_array(payload: dict[str, Any], lamda: float) -> np.ndarray:
    if "signal_array" in payload:
        return np.asarray(payload["signal_array"], dtype=float)

    lamda_key = f"{float(lamda):.11f}"
    signal = payload["signal"]
    if isinstance(signal, dict):
        return np.asarray(signal[lamda_key], dtype=float)

    return np.asarray(signal, dtype=float)


def detect_for_penalties(
    signal: np.ndarray,
    k_samples: np.ndarray,
    t_samples: np.ndarray,
    penalties: Sequence[float],
    kernel: str,
    lamda: float,
    window_seconds: float,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for penalty in penalties:
        change_point_indices = detect_change_points_from_signal(
            signal=signal,
            n_bkps=1,
            kernel=kernel,
            stopping_rule="penalty",
            penalty=float(penalty),
        )
        cp_idx_array = np.asarray(change_point_indices, dtype=int)
        cp_k_samples = k_samples[cp_idx_array].astype(int).tolist()
        cp_t_seconds = t_samples[cp_idx_array].astype(float).tolist()
        cp_t_hours = (t_samples[cp_idx_array] / 3600.0).astype(float).tolist()

        results.append(
            {
                "lamda": float(lamda),
                "window": float(window_seconds),
                "penalty": float(penalty),
                "change_point_indices": change_point_indices,
                "change_point_k_samples": cp_k_samples,
                "change_point_t_seconds": cp_t_seconds,
                "change_point_t_hours": cp_t_hours,
                "num_change_points": int(len(change_point_indices)),
                "predicted_change_points": {
                    float(window_seconds): change_point_indices,
                },
                "predicted_k_samples": {
                    float(window_seconds): cp_k_samples,
                },
                "predicted_t_samples": {
                    float(window_seconds): cp_t_seconds,
                },
                "predicted_t_hours": {
                    float(window_seconds): cp_t_hours,
                },
            }
        )

    return results


def build_output_dir(
    output_base: Path,
    lamda: float,
    window_seconds: float,
    reverse_time: bool = False,
) -> Path:
    direction = "backward" if reverse_time else "forward"
    return (
        output_base
        / direction
        / f"window_{int(window_seconds)}"
        / f"lamda_{float(lamda):.11f}"
    )


def save_penalty_summary_csv(results: Sequence[dict[str, Any]], output_dir: Path) -> Path:
    outfile = output_dir / "penalty_summary.csv"
    with open(outfile, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "penalty",
                "num_change_points",
                "change_point_indices",
                "change_point_k_samples",
                "change_point_t_seconds",
                "change_point_t_hours",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "penalty": f"{float(result['penalty']):.12g}",
                    "num_change_points": int(result["num_change_points"]),
                    "change_point_indices": " ".join(
                        str(idx) for idx in result["change_point_indices"]
                    ),
                    "change_point_k_samples": " ".join(
                        str(idx) for idx in result["change_point_k_samples"]
                    ),
                    "change_point_t_seconds": " ".join(
                        f"{value:.6f}" for value in result["change_point_t_seconds"]
                    ),
                    "change_point_t_hours": " ".join(
                        f"{value:.6f}" for value in result["change_point_t_hours"]
                    ),
                }
            )
    return outfile


def save_change_points_long_csv(
    results: Sequence[dict[str, Any]],
    output_dir: Path,
) -> Path:
    outfile = output_dir / "change_points_long.csv"
    with open(outfile, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "penalty",
                "change_point_rank",
                "signal_index",
                "k_sample",
                "t_seconds",
                "t_hours",
            ],
        )
        writer.writeheader()
        for result in results:
            for rank, (
                signal_index,
                k_sample,
                t_seconds,
                t_hours,
            ) in enumerate(
                zip(
                    result["change_point_indices"],
                    result["change_point_k_samples"],
                    result["change_point_t_seconds"],
                    result["change_point_t_hours"],
                ),
                start=1,
            ):
                writer.writerow(
                    {
                        "penalty": f"{float(result['penalty']):.12g}",
                        "change_point_rank": rank,
                        "signal_index": int(signal_index),
                        "k_sample": int(k_sample),
                        "t_seconds": f"{float(t_seconds):.6f}",
                        "t_hours": f"{float(t_hours):.6f}",
                    }
                )
    return outfile


def save_summary_pickle(summary: dict[str, Any], output_dir: Path) -> Path:
    outfile = output_dir / "ruptures_results.pkl"
    with open(outfile, "wb") as handle:
        pickle.dump(summary, handle)
    return outfile


def main() -> None:
    args = parse_args()
    metadata = load_metadata(args.signal_base)
    lamda = resolve_lambda(args.lamda, metadata)
    penalties = np.asarray(
        DEFAULT_PENALTIES if args.penalties is None else args.penalties,
        dtype=float,
    )

    if penalties.size == 0:
        raise ValueError("At least one penalty value must be provided.")
    if np.any(penalties <= 0):
        raise ValueError("All penalties must be strictly positive.")

    payload, signal_path = load_signal_payload(
        signal_base=args.signal_base,
        window_seconds=float(args.window_seconds),
        lamda=lamda,
        reverse_time=bool(args.reverse_time),
    )

    signal = extract_signal_array(payload, lamda=lamda)
    k_samples = np.asarray(payload["k_samples"], dtype=int)
    t_samples = np.asarray(payload["t_samples"], dtype=float)

    if signal.ndim != 1:
        raise ValueError(
            f"Expected a 1D entropy signal, got shape {signal.shape} from {signal_path}."
        )
    if len(signal) != len(k_samples) or len(signal) != len(t_samples):
        raise ValueError(
            "Signal, k_samples, and t_samples must all have the same length."
        )

    penalty_results = detect_for_penalties(
        signal=signal,
        k_samples=k_samples,
        t_samples=t_samples,
        penalties=penalties,
        kernel=args.kernel,
        lamda=lamda,
        window_seconds=float(args.window_seconds),
    )

    output_dir = build_output_dir(
        output_base=args.output_base,
        lamda=lamda,
        window_seconds=float(args.window_seconds),
        reverse_time=bool(args.reverse_time),
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "signal_base": str(args.signal_base),
        "signal_path": str(signal_path),
        "output_dir": str(output_dir),
        "lamda": float(lamda),
        "window_seconds": float(args.window_seconds),
        "window_minutes": float(args.window_seconds) / 60.0,
        "kernel": args.kernel,
        "stopping_rule": "penalty",
        "penalties": penalties,
        "reverse_time": bool(args.reverse_time),
        "direction": payload.get(
            "direction",
            "backward" if args.reverse_time else "forward",
        ),
        "signal_length": int(len(signal)),
        "k_samples": k_samples,
        "t_samples": t_samples,
        "signal_array": signal,
        "lambda_results": penalty_results,
        "results_by_lambda": {
            float(lamda): {
                float(result["penalty"]): result for result in penalty_results
            }
        },
    }

    pickle_path = save_summary_pickle(summary, output_dir=output_dir)
    summary_csv_path = save_penalty_summary_csv(penalty_results, output_dir=output_dir)
    long_csv_path = save_change_points_long_csv(penalty_results, output_dir=output_dir)

    print(f"Signal path: {signal_path}")
    print(f"Lambda: {lamda:.11f}")
    print(f"Window: {float(args.window_seconds):g} s")
    print(f"Kernel: {args.kernel}")
    print(f"Signal length: {len(signal)}")
    print("Penalty scan:")
    for result in penalty_results:
        print(
            f"  pen={float(result['penalty']):.6g} -> "
            f"{int(result['num_change_points'])} change points at "
            f"indices={result['change_point_indices']} "
            f"times_hours={[round(value, 6) for value in result['change_point_t_hours']]}"
        )
    print(f"Saved pickle: {pickle_path}")
    print(f"Saved summary CSV: {summary_csv_path}")
    print(f"Saved long CSV: {long_csv_path}")


if __name__ == "__main__":
    main()
