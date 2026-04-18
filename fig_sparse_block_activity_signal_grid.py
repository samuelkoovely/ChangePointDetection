from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np

from signal_generation import load_signal_result
from sparse_block_activity_common import load_pickle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a grid overview of sparse block-activity entropy signals for "
            "all requested (lambda, window) pairs in one results directory."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help=(
            "Dataset-specific results directory containing metadata.pkl and "
            "signals/, for example ./gridsearch_results/ct_block2/."
        ),
    )
    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=None,
        help="Optional subset of lambda values to display.",
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        type=float,
        default=None,
        help="Optional subset of window values to display.",
    )
    parser.add_argument(
        "--sample-path",
        type=Path,
        default=None,
        help="Optional override for the sample pickle path stored in metadata.pkl.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional path for the overview image. Defaults inside results-dir.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=250,
        help="Raster resolution used when saving PNG output.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure after saving it.",
    )
    return parser.parse_args()


def load_metadata(results_dir: Path) -> dict[str, Any]:
    metadata_path = results_dir / "metadata.pkl"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing metadata file {metadata_path}. "
            "Expected a dataset-specific signal-grid directory."
        )
    return load_pickle(metadata_path)


def resolve_requested_values(
    requested: Sequence[float] | None,
    available: Sequence[float],
    name: str,
) -> list[float]:
    available_arr = np.asarray(available, dtype=float)
    if requested is None:
        return [float(value) for value in available_arr]

    resolved: list[float] = []
    for value in requested:
        matches = np.flatnonzero(np.isclose(available_arr, float(value), rtol=1e-12, atol=1e-15))
        if len(matches) == 0:
            raise ValueError(
                f"Requested {name}={float(value):g} is not available. "
                f"Available {name}s: {[float(v) for v in available_arr]}"
            )
        resolved_value = float(available_arr[int(matches[0])])
        if resolved_value not in resolved:
            resolved.append(resolved_value)
    return resolved


def default_output_path(results_dir: Path, reverse_time: bool = False) -> Path:
    suffix = "_rev" if reverse_time else ""
    return results_dir / f"signal_grid_overview{suffix}.png"


def pad_limits(values: Sequence[float], lower_pad: float = 0.05, upper_pad: float = 0.05) -> tuple[float, float]:
    values_arr = np.asarray(values, dtype=float)
    finite_values = values_arr[np.isfinite(values_arr)]
    if finite_values.size == 0:
        return 0.0, 1.0

    vmin = float(np.min(finite_values))
    vmax = float(np.max(finite_values))
    if np.isclose(vmin, vmax):
        delta = max(abs(vmin) * 0.1, 0.1)
        return vmin - delta, vmax + delta

    span = vmax - vmin
    return vmin - lower_pad * span, vmax + upper_pad * span


def load_sample(metadata: dict[str, Any], sample_path_override: Path | None = None) -> dict[str, Any]:
    sample_path = sample_path_override
    if sample_path is None:
        sample_path_raw = metadata.get("sample_path")
        if sample_path_raw is None:
            raise KeyError(
                "metadata.pkl does not contain 'sample_path'. "
                "Pass --sample-path explicitly."
            )
        sample_path = Path(sample_path_raw)

    if not sample_path.exists():
        raise FileNotFoundError(
            f"Sample pickle {sample_path} does not exist. "
            "Pass --sample-path explicitly if the metadata path is stale."
        )
    return load_pickle(sample_path)


def load_signal_grid(
    results_dir: Path,
    lambdas: Sequence[float],
    windows: Sequence[float],
    reverse_time: bool = False,
) -> dict[tuple[float, float], dict[str, Any]]:
    signal_dir = results_dir / "signals"
    if not signal_dir.exists():
        raise FileNotFoundError(
            f"Missing signal directory {signal_dir}. "
            "Expected per-(lambda, window) signal pickle files there."
        )

    loaded: dict[tuple[float, float], dict[str, Any]] = {}
    for lamda in lambdas:
        for window in windows:
            loaded[(float(lamda), float(window))] = load_signal_result(
                outdir=signal_dir,
                lamda=float(lamda),
                window=float(window),
                reverse_time=reverse_time,
            )
    return loaded


def dataset_title(metadata: dict[str, Any], results_dir: Path) -> str:
    dataset = metadata.get("dataset")
    if dataset is not None:
        return str(dataset)
    return results_dir.name


def format_lamda(value: float) -> str:
    return f"{float(value):.2e}"


def plot_signal_grid(
    results_dir: Path,
    metadata: dict[str, Any],
    sample: dict[str, Any],
    signals: dict[tuple[float, float], dict[str, Any]],
    lambdas: Sequence[float],
    windows: Sequence[float],
    output_path: Path,
    dpi: int,
    show: bool,
) -> Path:
    net = sample["tnet"]
    t_min = float(net.times[0])
    t_max = float(net.times[-1])
    breakpoints = [float(value) for value in sample.get("bkps", [])]

    signal_arrays = [
        np.asarray(payload["signal"], dtype=float)
        for payload in signals.values()
        if len(np.asarray(payload["signal"], dtype=float)) > 0
    ]
    all_signal_values = (
        np.concatenate(signal_arrays)
        if len(signal_arrays) > 0
        else np.array([], dtype=float)
    )
    y_limits = pad_limits(all_signal_values)

    n_rows = len(windows)
    n_cols = len(lambdas)
    fig_width = max(4.0, 2.8 * n_cols)
    fig_height = max(3.0, 2.1 * n_rows)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for row_index, window in enumerate(windows):
        for col_index, lamda in enumerate(lambdas):
            axis = axes[row_index][col_index]
            payload = signals[(float(lamda), float(window))]
            times = np.asarray(payload["t_samples"], dtype=float)
            values = np.asarray(payload["signal"], dtype=float)
            use_markers = len(times) <= 200

            axis.plot(
                times,
                values,
                color="tab:blue",
                linewidth=1.2,
                marker="o" if use_markers else None,
                markersize=2.2 if use_markers else 0.0,
            )
            for breakpoint in breakpoints:
                axis.axvline(
                    breakpoint,
                    color="black",
                    linestyle="--",
                    linewidth=0.9,
                    alpha=0.8,
                )

            axis.set_xlim(t_min, t_max)
            axis.set_ylim(*y_limits)
            axis.tick_params(labelsize=8)
            axis.text(
                0.98,
                0.96,
                f"n={len(times)}",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=7,
                color="0.35",
            )

            if row_index == 0:
                axis.set_title(f"lamda={format_lamda(lamda)}", fontsize=9)
            if col_index == 0:
                axis.set_ylabel(f"window={float(window):g}\nS", fontsize=9)
            if row_index == n_rows - 1:
                axis.set_xlabel("t [s]", fontsize=9)

    fig.suptitle(
        (
            f"{dataset_title(metadata, results_dir)} signal overview\n"
            f"nodes={int(metadata['num_nodes'])}, events={int(metadata['num_events'])}, "
            f"sample_fraction={float(metadata.get('sample_fraction', 1.0)):g}, "
            f"direction={metadata.get('direction', 'forward')}"
        ),
        fontsize=11,
    )
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.08, top=0.86, wspace=0.12, hspace=0.22)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    metadata = load_metadata(results_dir)
    reverse_time = bool(metadata.get("reverse_time", False))
    available_lambdas = np.asarray(metadata["lambdas"], dtype=float)
    available_windows = np.asarray(metadata["windows"], dtype=float)

    lambdas = resolve_requested_values(
        requested=args.lambdas,
        available=available_lambdas,
        name="lambda",
    )
    windows = resolve_requested_values(
        requested=args.windows,
        available=available_windows,
        name="window",
    )

    sample = load_sample(
        metadata=metadata,
        sample_path_override=args.sample_path,
    )
    signals = load_signal_grid(
        results_dir=results_dir,
        lambdas=lambdas,
        windows=windows,
        reverse_time=reverse_time,
    )
    output_path = args.output_path
    if output_path is None:
        output_path = default_output_path(results_dir, reverse_time=reverse_time)

    saved_path = plot_signal_grid(
        results_dir=results_dir,
        metadata=metadata,
        sample=sample,
        signals=signals,
        lambdas=lambdas,
        windows=windows,
        output_path=output_path,
        dpi=int(args.dpi),
        show=bool(args.show),
    )
    print(saved_path)


if __name__ == "__main__":
    main()
