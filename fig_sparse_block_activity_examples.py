from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import EDLDE
from sparse_block_activity_common import (
    DEFAULT_DATA_DIR,
    DEFAULT_FIGURE_DIR,
    SPECS,
    figure_path,
    get_specs,
    load_pickle,
    sample_path,
    signal_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the sparse block-activity figures from saved sample and signal pickles."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[spec.key for spec in SPECS],
        choices=[spec.key for spec in SPECS],
        help="Subset of sparse figures to render.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing sparse samples and signal bundles.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help="Directory where the output figures will be saved.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster resolution used when saving PNG output.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display each figure after saving it.",
    )
    return parser.parse_args()


def compute_active_event_signal(sample: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    starts = np.asarray(sample["starts"], dtype=float)
    ends = np.asarray(sample["ends"], dtype=float)
    change_times, counts_after = EDLDE.activity_EDLDE(
        starting_times=starts,
        ending_times=ends,
    )
    return np.asarray(change_times, dtype=float), np.asarray(counts_after, dtype=float)


def pad_limits(values: np.ndarray, lower_pad: float = 0.05, upper_pad: float = 0.05) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return 0.0, 1.0

    vmin = float(np.min(finite_values))
    vmax = float(np.max(finite_values))
    if np.isclose(vmin, vmax):
        delta = max(abs(vmin) * 0.1, 0.1)
        return vmin - delta, vmax + delta

    span = vmax - vmin
    return vmin - lower_pad * span, vmax + upper_pad * span


def compute_interval_matrices(network, intervals: list[tuple[float, float]]) -> list[np.ndarray]:
    return [
        network.compute_static_adjacency_matrix(
            start_time=float(start),
            end_time=float(end),
        ).toarray()
        for start, end in intervals
    ]


def build_inset_intervals(
    t_min: float,
    t_max: float,
    breakpoint: float,
    two_interval_split: bool = False,
) -> list[tuple[float, float]]:
    span = float(t_max - t_min)
    if span <= 0:
        return [(t_min, t_max)]

    if two_interval_split:
        intervals = [
            (float(t_min), float(breakpoint)),
            (float(breakpoint), float(t_max)),
        ]
        cleaned_intervals = []
        for start, end in intervals:
            if end <= start:
                end = min(t_max, start + max(1e-6, 0.05 * span))
            cleaned_intervals.append((float(start), float(end)))
        return cleaned_intervals

    transition_width = min(max(0.12 * span, 6.0), 12.0)
    half_width = 0.5 * transition_width

    first_end = max(t_min + 1e-6, breakpoint - half_width)
    middle_start = max(t_min, breakpoint - half_width)
    middle_end = min(t_max, breakpoint + half_width)
    third_start = min(t_max - 1e-6, breakpoint + half_width)

    intervals = [
        (t_min, first_end),
        (middle_start, middle_end),
        (third_start, t_max),
    ]
    cleaned_intervals = []
    for start, end in intervals:
        if end <= start:
            end = min(t_max, start + max(1e-6, 0.05 * span))
        cleaned_intervals.append((float(start), float(end)))
    return cleaned_intervals


def make_inset_cmap():
    cmap = colormaps["inferno"].copy()
    cmap.set_bad(color="white")
    return cmap


def draw_matrix_insets(
    host_ax,
    matrices: list[np.ndarray],
    intervals: list[tuple[float, float]],
    inset_positions: list[float],
    inset_cmap,
) -> None:
    for matrix, pos, (start, end) in zip(matrices, inset_positions, intervals):
        inset_ax = inset_axes(
            host_ax,
            width="18%",
            height="18%",
            loc="lower left",
            bbox_to_anchor=(pos, 0.05, 1, 1),
            bbox_transform=host_ax.transAxes,
        )
        masked_matrix = np.ma.masked_where(matrix == 0, matrix)
        positive_entries = matrix[matrix > 0]
        vmax = positive_entries.max() if positive_entries.size else 1.0

        inset_ax.matshow(
            masked_matrix,
            cmap=inset_cmap,
            aspect="equal",
            vmin=0,
            vmax=vmax,
            interpolation="nearest",
        )
        inset_ax.set_facecolor("white")
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        inset_ax.set_title(f"{start:.0f} <= t < {end:.0f}", fontsize=7)
        for spine in inset_ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_edgecolor("black")


def plot_sample(
    spec,
    sample: dict[str, Any],
    signal_bundle: dict[str, Any],
    figure_dir: Path,
    dpi: int,
    show: bool,
) -> Path:
    net = sample["tnet"]
    breakpoint = float(sample["bkps"][0])
    t_min = float(net.times[0])
    t_max = float(net.times[-1])
    windows = [float(window) for window in signal_bundle["windows"]]
    signals_by_window = signal_bundle["signals_by_window"]
    lamda = float(signal_bundle["lamda"])
    inset_intervals = build_inset_intervals(
        t_min=t_min,
        t_max=t_max,
        breakpoint=breakpoint,
        two_interval_split=True,
    )
    interval_matrices = compute_interval_matrices(net, inset_intervals)
    inset_positions = [0.14, 0.62]
    inset_cmap = make_inset_cmap()

    active_times, active_counts = compute_active_event_signal(sample)
    active_ylim = (0.0, max(1.0, 1.05 * float(np.max(active_counts))))

    fig, axes = plt.subplots(1, len(windows), figsize=(5.0 * len(windows), 4.6), sharex=True)
    if len(windows) == 1:
        axes = [axes]

    twin_axes = []
    for axis, window in zip(axes, windows):
        signal_payload = signals_by_window[float(window)]
        entropy_times = np.asarray(signal_payload["t_samples"], dtype=float)
        entropy_values = np.asarray(signal_payload["signal"], dtype=float)

        axis.step(
            active_times,
            active_counts,
            where="post",
            color="tab:red",
            linewidth=1.4,
        )
        axis.set_xlim(t_min, t_max)
        axis.set_ylim(*active_ylim)
        axis.set_xlabel("t [s]")
        axis.tick_params(axis="y", labelcolor="tab:red")
        axis.axvline(
            breakpoint,
            color="black",
            linestyle="--",
            linewidth=1.2,
            alpha=0.85,
        )

        entropy_axis = axis.twinx()
        entropy_axis.plot(
            entropy_times,
            entropy_values,
            color="tab:blue",
            linewidth=1.4,
            alpha=0.95,
        )
        entropy_axis.set_ylim(*pad_limits(entropy_values))
        entropy_axis.tick_params(axis="y", labelcolor="tab:blue")
        twin_axes.append(entropy_axis)

        axis.set_title(f"Entropy window = {window:g}s", fontsize=11)
        draw_matrix_insets(
            host_ax=axis,
            matrices=interval_matrices,
            intervals=inset_intervals,
            inset_positions=inset_positions,
            inset_cmap=inset_cmap,
        )

    axes[0].set_ylabel("Active events", color="tab:red")
    twin_axes[-1].set_ylabel("Local entropy", color="tab:blue")
    axes[0].legend(
        handles=[
            Line2D([0], [0], color="tab:red", linewidth=1.6, label="Active events"),
            Line2D([0], [0], color="tab:blue", linewidth=1.6, label="Local entropy"),
            Line2D([0], [0], color="black", linestyle="--", linewidth=1.2, label="Breakpoint"),
        ],
        loc="upper left",
        frameon=False,
    )

    fig.suptitle(
        (
            f"{spec.title}\n"
            f"lambda={lamda:.2e}, entropy windows={', '.join(f'{window:g}s' for window in windows)}, "
            f"nodes={net.num_nodes}, events={net.num_events}"
        ),
        fontsize=12,
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.12, wspace=0.28)

    figure_dir.mkdir(parents=True, exist_ok=True)
    output_path = figure_path(spec, figure_dir)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def main() -> None:
    args = parse_args()

    for spec in get_specs(args.datasets):
        sample_input_path = sample_path(spec, args.data_dir)
        signal_input_path = signal_path(spec, args.data_dir)

        if not sample_input_path.exists():
            raise FileNotFoundError(
                f"Missing sparse sample {sample_input_path}. Run generate_sparse_block_activity_examples.py first."
            )
        if not signal_input_path.exists():
            raise FileNotFoundError(
                f"Missing sparse signal bundle {signal_input_path}. Run compute_sparse_block_activity_signals.py first."
            )

        print(f"Plotting sparse {spec.key} figure")
        sample = load_pickle(sample_input_path)
        signal_bundle = load_pickle(signal_input_path)
        output_path = plot_sample(
            spec=spec,
            sample=sample,
            signal_bundle=signal_bundle,
            figure_dir=args.figure_dir,
            dpi=int(args.dpi),
            show=bool(args.show),
        )
        print(output_path)


if __name__ == "__main__":
    main()
