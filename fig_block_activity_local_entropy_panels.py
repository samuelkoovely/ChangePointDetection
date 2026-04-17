from __future__ import annotations

import argparse
import importlib
import os
import pickle
import random as rd
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import EDLDE
from signal_generation import compute_signals_for_lambda


DEFAULT_LAMBDAS = np.logspace(-5, 0, 10)
DEFAULT_ENTROPY_WINDOW = 2.0
DEFAULT_SAMPLE_FRACTION = 0.1
DEFAULT_PANEL_WIDTH = 30.0
DEFAULT_OUTPUT_DIR = Path("figures")


@dataclass(frozen=True)
class FigureSpec:
    key: str
    module_name: str
    title: str
    output_stem: str


FIGURE_SPECS = (
    FigureSpec(
        key="block1",
        module_name="generate_block1activity",
        title="Block-1 activity",
        output_stem="block1activity_local_entropy_panels",
    ),
    FigureSpec(
        key="block2",
        module_name="generate_block2activities",
        title="Block-2 activities",
        output_stem="block2activities_local_entropy_panels",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the first training sample from the block-activity generators "
            "with active-event counts and one local-entropy curve."
        )
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Training-sample index to regenerate from each generator module.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[figure_spec.key for figure_spec in FIGURE_SPECS],
        choices=[figure_spec.key for figure_spec in FIGURE_SPECS],
        help="Subset of figures to render.",
    )
    parser.add_argument(
        "--lamda",
        type=float,
        default=None,
        help=(
            "Entropy scale to plot. When omitted, the script interprets the "
            "'highest frequency' request as the largest value from --lambdas."
        ),
    )
    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=DEFAULT_LAMBDAS.tolist(),
        help="Lambda grid used to resolve the default highest-frequency curve.",
    )
    parser.add_argument(
        "--entropy-window",
        type=float,
        default=DEFAULT_ENTROPY_WINDOW,
        help="Local-entropy window length in seconds.",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=DEFAULT_SAMPLE_FRACTION,
        help=(
            "Fraction of valid entropy-window centers to evaluate. Use 1.0 for "
            "a full scan."
        ),
    )
    parser.add_argument(
        "--use-linear-approx",
        action="store_true",
        help=(
            "Use the repository's linearized transition matrices instead of the "
            "exact matrix exponential pipeline. This is faster but approximate."
        ),
    )
    parser.add_argument(
        "--panel-width",
        type=float,
        default=DEFAULT_PANEL_WIDTH,
        help="Width in seconds of each of the three zoomed panels.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the figures will be saved.",
    )
    parser.add_argument(
        "--format",
        choices=("png", "pdf"),
        default="png",
        help="Output figure format.",
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
        help="Display the figures interactively after saving them.",
    )
    return parser.parse_args()


def resolve_lamda(args: argparse.Namespace) -> float:
    if args.lamda is not None:
        return float(args.lamda)
    return float(np.max(np.asarray(args.lambdas, dtype=float)))


def load_training_sample(module_name: str, sample_index: int) -> dict[str, Any]:
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative.")

    module = importlib.import_module(module_name)
    num_samples = int(module.TRAIN_NUM_SAMPLES)
    if sample_index >= num_samples:
        raise ValueError(
            f"{module_name} only defines {num_samples} training samples; "
            f"requested sample_index={sample_index}."
        )

    rng = rd.Random(int(module.TRAIN_RANDOM_SEED))
    t_split = None
    for _ in range(sample_index + 1):
        t_split = rng.randint(
            int(module.BREAKPOINT_LOWER_BOUND),
            int(module.BREAKPOINT_UPPER_BOUND),
        )

    return module.generate_sample(sample_index=sample_index, t_split=int(t_split))


def entropy_cache_path(
    cache_dir: Path,
    figure_spec: FigureSpec,
    sample_index: int,
    lamda: float,
    window: float,
    sample_fraction: float,
    use_linear_approx: bool,
) -> Path:
    mode = "linear" if use_linear_approx else "exact"
    filename = (
        f"{figure_spec.key}_sample_{sample_index}"
        f"_lamda_{float(lamda):.11f}"
        f"_window_{float(window):g}"
        f"_sample_fraction_{float(sample_fraction):g}"
        f"_{mode}.pkl"
    )
    return cache_dir / filename


def compute_active_event_signal(sample: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    starts = np.asarray(sample["starts"], dtype=float)
    ends = np.asarray(sample["ends"], dtype=float)
    change_times, counts_after = EDLDE.activity_EDLDE(
        starting_times=starts,
        ending_times=ends,
    )
    return np.asarray(change_times, dtype=float), np.asarray(counts_after, dtype=float)


def compute_local_entropy_signal(
    net: Any,
    lamda: float,
    window: float,
    sample_fraction: float,
    use_linear_approx: bool,
    cache_path: Path | None = None,
) -> dict[str, Any]:
    if cache_path is not None and cache_path.exists():
        with cache_path.open("rb") as handle:
            return pickle.load(handle)

    if use_linear_approx:
        net.compute_laplacian_matrices(
            t_start=net.times[0],
            t_stop=net.times[-1],
            random_walk=True,
        )
    else:
        net.compute_laplacian_matrices(
            t_start=net.times[0],
            t_stop=net.times[-1],
            random_walk=False,
        )
        net.compute_inter_transition_matrices(
            lamda=float(lamda),
            t_start=net.times[0],
            t_stop=net.times[-1],
            dense_expm=True,
            use_sparse_stoch=False,
            random_walk=False,
        )

    results = compute_signals_for_lambda(
        net=net,
        lamda=float(lamda),
        windows=[float(window)],
        sample_fraction=float(sample_fraction),
        p0=None,
        use_linear_approx=bool(use_linear_approx),
        lin_t_s=10,
        window_backend="segment_tree",
        reverse_time=False,
    )
    result = results[float(window)]

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as handle:
            pickle.dump(result, handle)

    return result


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


def centered_interval(
    center: float,
    width: float,
    t_min: float,
    t_max: float,
) -> tuple[float, float]:
    total_width = min(float(width), float(t_max - t_min))
    left = float(center) - 0.5 * total_width
    right = float(center) + 0.5 * total_width

    if left < t_min:
        right += t_min - left
        left = t_min
    if right > t_max:
        left -= right - t_max
        right = t_max

    return max(left, t_min), min(right, t_max)


def build_panel_windows(
    t_min: float,
    t_max: float,
    breakpoint: float,
    panel_width: float,
) -> list[tuple[str, tuple[float, float]]]:
    pre_center = 0.5 * (t_min + breakpoint)
    post_center = 0.5 * (breakpoint + t_max)
    return [
        ("Pre-change", centered_interval(pre_center, panel_width, t_min, t_max)),
        ("Around breakpoint", centered_interval(breakpoint, panel_width, t_min, t_max)),
        ("Post-change", centered_interval(post_center, panel_width, t_min, t_max)),
    ]


def plot_dataset_figure(
    figure_spec: FigureSpec,
    sample: dict[str, Any],
    sample_index: int,
    lamda: float,
    entropy_window: float,
    sample_fraction: float,
    use_linear_approx: bool,
    panel_width: float,
    output_dir: Path,
    output_format: str,
    dpi: int,
    show: bool,
) -> Path:
    net = sample["tnet"]
    breakpoint = float(sample["bkps"][0])
    t_min = float(net.times[0])
    t_max = float(net.times[-1])
    cache_dir = output_dir / ".entropy_cache"
    cache_path = entropy_cache_path(
        cache_dir=cache_dir,
        figure_spec=figure_spec,
        sample_index=sample_index,
        lamda=lamda,
        window=entropy_window,
        sample_fraction=sample_fraction,
        use_linear_approx=use_linear_approx,
    )

    active_times, active_counts = compute_active_event_signal(sample)
    entropy_result = compute_local_entropy_signal(
        net=net,
        lamda=lamda,
        window=entropy_window,
        sample_fraction=sample_fraction,
        use_linear_approx=use_linear_approx,
        cache_path=cache_path,
    )
    entropy_times = np.asarray(entropy_result["t_samples"], dtype=float)
    entropy_values = np.asarray(entropy_result["signal"], dtype=float)

    panel_windows = build_panel_windows(
        t_min=t_min,
        t_max=t_max,
        breakpoint=breakpoint,
        panel_width=panel_width,
    )

    active_ylim = (0.0, max(1.0, 1.05 * float(np.max(active_counts))))
    entropy_ylim = pad_limits(entropy_values)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=False)
    twin_axes = []

    for axis, (panel_title, (left, right)) in zip(axes, panel_windows):
        axis.step(
            active_times,
            active_counts,
            where="post",
            color="tab:red",
            linewidth=1.4,
        )
        axis.set_xlim(left, right)
        axis.set_ylim(*active_ylim)
        axis.set_xlabel("t [s]")
        axis.tick_params(axis="y", labelcolor="tab:red")
        axis.grid(axis="x", alpha=0.2)

        if left <= breakpoint <= right:
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
        entropy_axis.set_ylim(*entropy_ylim)
        entropy_axis.tick_params(axis="y", labelcolor="tab:blue")
        twin_axes.append(entropy_axis)

        axis.set_title(f"{panel_title}\n[{left:.1f}, {right:.1f}] s", fontsize=11)

    axes[0].set_ylabel("Active events", color="tab:red")
    twin_axes[-1].set_ylabel("Local entropy", color="tab:blue")

    legend_handles = [
        Line2D([0], [0], color="tab:red", linewidth=1.6, label="Active events"),
        Line2D([0], [0], color="tab:blue", linewidth=1.6, label="Local entropy"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.2, label="Breakpoint"),
    ]
    axes[0].legend(handles=legend_handles, loc="upper left", frameon=False)

    fig.suptitle(
        (
            f"{figure_spec.title}, training sample {sample_index}\n"
            f"lambda={lamda:.2e}, entropy window={entropy_window:g}s, "
            f"sample_fraction={sample_fraction:g}, "
            f"mode={'linear approx' if use_linear_approx else 'exact'}"
        ),
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"{figure_spec.output_stem}_sample_{sample_index}.{output_format}"
    )
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def main() -> None:
    args = parse_args()
    lamda = resolve_lamda(args)
    selected_dataset_keys = set(args.datasets)

    for figure_spec in FIGURE_SPECS:
        if figure_spec.key not in selected_dataset_keys:
            continue

        print(
            f"Rendering {figure_spec.key}: sample={args.sample_index}, "
            f"lambda={lamda:.2e}, window={float(args.entropy_window):g}, "
            f"mode={'linear approx' if args.use_linear_approx else 'exact'}"
        )
        sample = load_training_sample(
            module_name=figure_spec.module_name,
            sample_index=int(args.sample_index),
        )
        output_path = plot_dataset_figure(
            figure_spec=figure_spec,
            sample=sample,
            sample_index=int(args.sample_index),
            lamda=lamda,
            entropy_window=float(args.entropy_window),
            sample_fraction=float(args.sample_fraction),
            use_linear_approx=bool(args.use_linear_approx),
            panel_width=float(args.panel_width),
            output_dir=args.output_dir,
            output_format=args.format,
            dpi=int(args.dpi),
            show=bool(args.show),
        )
        print(output_path)


if __name__ == "__main__":
    main()
