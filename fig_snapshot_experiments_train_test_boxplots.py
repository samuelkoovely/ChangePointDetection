from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from fig_snapshot_experiments_boxplots import (
    EXPERIMENT_RESULT_PATHS,
    METHOD_COLORS,
    extract_metric_arrays,
    load_pickle,
    style_axes,
)


SPLIT_ORDER = ("Train", "Test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot paired train/test boxplots of per-sample Hausdorff performance "
            "for the three snapshot experiments."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path. If omitted, show the figure.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Figure DPI when saving to disk.",
    )
    parser.add_argument(
        "--showfliers",
        action="store_true",
        help="Show boxplot fliers.",
    )
    return parser.parse_args()


def extract_test_metric_arrays(summary: dict) -> dict[str, np.ndarray]:
    per_sample_results = summary.get("per_sample_results")
    if not per_sample_results:
        raise ValueError("Test summary is missing 'per_sample_results'.")

    return {
        "f1": np.asarray([entry["f1"] for entry in per_sample_results], dtype=float),
        "hausdorff": np.asarray(
            [entry["hausdorff"] for entry in per_sample_results],
            dtype=float,
        ),
    }


def load_all_split_metric_arrays() -> dict[str, dict[str, dict[str, dict[str, np.ndarray]]]]:
    metrics_by_experiment: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]] = {}
    for experiment_name, method_paths in EXPERIMENT_RESULT_PATHS.items():
        metrics_by_experiment[experiment_name] = {}
        for method_name, result_paths in method_paths.items():
            train_summary = load_pickle(result_paths["train"])
            test_summary = load_pickle(result_paths["test"])
            metrics_by_experiment[experiment_name][method_name] = {
                "Train": extract_metric_arrays(train_summary),
                "Test": extract_test_metric_arrays(test_summary),
            }
    return metrics_by_experiment


def _compute_log_floor(
    metrics_by_experiment: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
) -> float:
    positive_values: list[float] = []
    for methods in metrics_by_experiment.values():
        for splits in methods.values():
            for split_metrics in splits.values():
                values = np.asarray(split_metrics["hausdorff"], dtype=float)
                finite_positive = values[np.isfinite(values) & (values > 0)]
                positive_values.extend(finite_positive.tolist())

    if not positive_values:
        return 1e-3
    return min(positive_values) / 2.0


def _prepare_boxplot_values(
    values: np.ndarray,
    hausdorff_cap: float,
    log_floor: float,
) -> np.ndarray:
    display_values = np.asarray(values, dtype=float).copy()
    display_values[np.isposinf(display_values)] = hausdorff_cap
    display_values[np.isneginf(display_values)] = log_floor
    display_values[display_values <= 0] = log_floor
    display_values = display_values[~np.isnan(display_values)]
    if display_values.size == 0:
        raise ValueError("No finite values available to plot.")
    return display_values


def _draw_boxplot(
    ax: plt.Axes,
    *,
    values: np.ndarray,
    position: float,
    width: float,
    method_color: str,
    split_name: str,
    hausdorff_cap: float,
    log_floor: float,
    showfliers: bool,
) -> None:
    display_values = _prepare_boxplot_values(values, hausdorff_cap, log_floor)
    boxplot = ax.boxplot(
        [display_values],
        positions=[position],
        widths=width,
        patch_artist=True,
        showfliers=showfliers,
        whis=1.5,
        medianprops={"color": "#303030", "linewidth": 1.5},
        whiskerprops={"color": method_color, "linewidth": 1.1},
        capprops={"color": method_color, "linewidth": 1.1},
        boxprops={"edgecolor": method_color, "linewidth": 1.2},
    )
    box = boxplot["boxes"][0]
    if split_name == "Train":
        box.set_facecolor(method_color)
        box.set_alpha(0.88)
    else:
        box.set_facecolor("white")
        box.set_alpha(1.0)
        box.set_hatch("///")

    if np.any(np.isposinf(values)):
        ax.text(
            position,
            hausdorff_cap,
            r"$\infty$",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#404040",
        )


def draw_grouped_boxplots(
    ax: plt.Axes,
    metrics_by_experiment: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
    showfliers: bool,
) -> None:
    experiment_names = list(metrics_by_experiment.keys())
    method_names = list(METHOD_COLORS.keys())

    centers = np.arange(len(experiment_names), dtype=float) * 4.1
    method_offsets = np.linspace(-1.15, 1.15, len(method_names))
    split_offsets = {"Train": -0.10, "Test": 0.10}
    width = 0.17

    style_axes(ax)
    finite_values: list[float] = []
    has_infinite_values = False
    for experiment_name in experiment_names:
        for method_name in method_names:
            for split_name in SPLIT_ORDER:
                values = metrics_by_experiment[experiment_name][method_name][split_name][
                    "hausdorff"
                ]
                finite_values.extend(values[np.isfinite(values)].tolist())
                if np.any(~np.isfinite(values)):
                    has_infinite_values = True

    finite_max = max(finite_values) if finite_values else 1.0
    hausdorff_cap = 1.1 * finite_max if finite_max > 0 else 1.0
    log_floor = _compute_log_floor(metrics_by_experiment)

    for experiment_idx, experiment_name in enumerate(experiment_names):
        center = centers[experiment_idx]
        for method_idx, method_name in enumerate(method_names):
            method_center = center + method_offsets[method_idx]
            method_color = METHOD_COLORS[method_name]
            for split_name in SPLIT_ORDER:
                position = method_center + split_offsets[split_name]
                values = metrics_by_experiment[experiment_name][method_name][split_name][
                    "hausdorff"
                ]
                _draw_boxplot(
                    ax,
                    values=values,
                    position=position,
                    width=width,
                    method_color=method_color,
                    split_name=split_name,
                    hausdorff_cap=hausdorff_cap,
                    log_floor=log_floor,
                    showfliers=showfliers,
                )

    ax.set_yscale("log")
    ax.set_xticks(centers)
    ax.set_xticklabels(experiment_names)
    ax.set_xlim(centers[0] - 1.75, centers[-1] + 1.75)
    ax.set_ylim(
        bottom=log_floor / 1.15,
        top=hausdorff_cap * 1.12 if has_infinite_values else None,
    )
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)


def build_figure(
    metrics_by_experiment: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
    showfliers: bool,
) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(11.6, 4.2))
    draw_grouped_boxplots(
        ax=ax,
        metrics_by_experiment=metrics_by_experiment,
        showfliers=showfliers,
    )
    ax.set_ylabel("Hausdorff distance (log scale)")
    ax.set_title("Train and test Hausdorff performance across snapshot experiments")

    legend_handles = [
        Patch(facecolor=METHOD_COLORS[method_name], edgecolor=METHOD_COLORS[method_name], label=method_name)
        for method_name in METHOD_COLORS
    ]
    legend_handles.extend(
        [
            Patch(facecolor="#808080", edgecolor="#505050", alpha=0.88, label="Train"),
            Patch(facecolor="white", edgecolor="#505050", hatch="///", label="Test"),
        ]
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=5,
        frameon=False,
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.88))
    return fig


def main() -> None:
    args = parse_args()
    metrics_by_experiment = load_all_split_metric_arrays()
    fig = build_figure(
        metrics_by_experiment=metrics_by_experiment,
        showfliers=args.showfliers,
    )

    if args.output is None:
        plt.show()
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
