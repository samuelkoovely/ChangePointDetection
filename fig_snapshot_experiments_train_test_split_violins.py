from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

from fig_snapshot_experiments_boxplots import METHOD_COLORS, style_axes
from fig_snapshot_experiments_train_test_boxplots import (
    _compute_log_floor,
    _prepare_boxplot_values,
    load_all_split_metric_arrays,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    BASE_DIR / "figures" / "fig_snapshot_experiments_train_test_split_violins.pdf"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot paired train/test split violins of per-sample Hausdorff "
            "performance for the snapshot experiments."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output PDF path. Use --show to display interactively as well.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Figure DPI when saving to disk.",
    )
    parser.add_argument(
        "--include-multibkps",
        action="store_true",
        help=(
            "Include the multi-bkps experiment. It is omitted by default because "
            "some current distances are non-finite."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively after rendering.",
    )
    return parser.parse_args()


def _lighten_color(
    color: str,
    mix_with_white: float = 0.58,
) -> tuple[float, float, float]:
    base = np.asarray(to_rgb(color), dtype=float)
    return tuple(base + (1.0 - base) * mix_with_white)


def _select_experiments(
    metrics_by_experiment: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
    *,
    include_multibkps: bool,
) -> dict[str, dict[str, dict[str, dict[str, np.ndarray]]]]:
    if include_multibkps:
        return metrics_by_experiment

    filtered = {
        experiment_name: methods
        for experiment_name, methods in metrics_by_experiment.items()
        if experiment_name != "Multi-bkps snapshots"
    }
    if not filtered:
        raise ValueError("No experiments available to plot after filtering.")
    return filtered


def _compute_hausdorff_cap(
    metrics_by_experiment: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
) -> tuple[float, bool]:
    finite_values: list[float] = []
    has_infinite_values = False
    for methods in metrics_by_experiment.values():
        for splits in methods.values():
            for split_metrics in splits.values():
                values = np.asarray(split_metrics["hausdorff"], dtype=float)
                finite_values.extend(values[np.isfinite(values)].tolist())
                if np.any(~np.isfinite(values)):
                    has_infinite_values = True

    finite_max = max(finite_values) if finite_values else 1.0
    hausdorff_cap = 1.1 * finite_max if finite_max > 0 else 1.0
    return hausdorff_cap, has_infinite_values


def _clip_violin_half(body: PolyCollection, *, position: float, side: str) -> None:
    vertices = body.get_paths()[0].vertices
    if side == "left":
        vertices[:, 0] = np.minimum(vertices[:, 0], position)
        return
    if side == "right":
        vertices[:, 0] = np.maximum(vertices[:, 0], position)
        return
    raise ValueError(f"Unsupported violin side: {side}")


def _draw_half_violin(
    ax: plt.Axes,
    *,
    values: np.ndarray,
    original_values: np.ndarray,
    position: float,
    width: float,
    side: str,
    facecolor: tuple[float, float, float] | str,
    edgecolor: str,
    hausdorff_cap: float,
    log_floor: float,
) -> None:
    display_values = _prepare_boxplot_values(values, hausdorff_cap, log_floor)
    violin = ax.violinplot(
        [display_values],
        positions=[position],
        widths=width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    body = violin["bodies"][0]
    body.set_facecolor(facecolor)
    body.set_edgecolor(edgecolor)
    body.set_alpha(0.9)
    body.set_linewidth(1.1)
    _clip_violin_half(body, position=position, side=side)

    median = float(np.median(display_values))
    median_span = width * 0.22
    median_margin = width * 0.03
    if side == "left":
        ax.hlines(
            median,
            position - median_span,
            position - median_margin,
            color="#2f2f2f",
            linewidth=1.6,
            zorder=4,
        )
        text_x = position - width * 0.18
    else:
        ax.hlines(
            median,
            position + median_margin,
            position + median_span,
            color="#2f2f2f",
            linewidth=1.6,
            zorder=4,
        )
        text_x = position + width * 0.18

    if np.any(np.isposinf(original_values)):
        ax.text(
            text_x,
            hausdorff_cap,
            r"$\infty$",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#404040",
        )


def draw_grouped_split_violins(
    ax: plt.Axes,
    metrics_by_experiment: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
) -> None:
    experiment_names = list(metrics_by_experiment.keys())
    method_names = list(METHOD_COLORS.keys())

    centers = np.arange(len(experiment_names), dtype=float) * 3.8
    method_offsets = np.linspace(-0.95, 0.95, len(method_names))
    width = 0.50

    style_axes(ax)
    hausdorff_cap, has_infinite_values = _compute_hausdorff_cap(metrics_by_experiment)
    log_floor = _compute_log_floor(metrics_by_experiment)

    for experiment_idx, experiment_name in enumerate(experiment_names):
        center = centers[experiment_idx]
        for method_idx, method_name in enumerate(method_names):
            position = center + method_offsets[method_idx]
            method_color = METHOD_COLORS[method_name]
            splits = metrics_by_experiment[experiment_name][method_name]

            train_values = np.asarray(splits["Train"]["hausdorff"], dtype=float)
            test_values = np.asarray(splits["Test"]["hausdorff"], dtype=float)
            combined_display_values = np.concatenate(
                [
                    _prepare_boxplot_values(train_values, hausdorff_cap, log_floor),
                    _prepare_boxplot_values(test_values, hausdorff_cap, log_floor),
                ]
            )
            ax.vlines(
                position,
                float(np.min(combined_display_values)),
                float(np.max(combined_display_values)),
                color="#565656",
                linewidth=0.9,
                alpha=0.85,
                zorder=3,
            )

            _draw_half_violin(
                ax,
                values=train_values,
                original_values=train_values,
                position=position,
                width=width,
                side="left",
                facecolor=method_color,
                edgecolor=method_color,
                hausdorff_cap=hausdorff_cap,
                log_floor=log_floor,
            )
            _draw_half_violin(
                ax,
                values=test_values,
                original_values=test_values,
                position=position,
                width=width,
                side="right",
                facecolor=_lighten_color(method_color),
                edgecolor=method_color,
                hausdorff_cap=hausdorff_cap,
                log_floor=log_floor,
            )

    #ax.set_yscale("log")
    ax.set_xticks(centers)
    ax.set_xticklabels(experiment_names)
    ax.set_xlim(centers[0] - 1.55, centers[-1] + 1.55)
    ax.set_ylim(
        bottom=log_floor / 1.15,
        top=hausdorff_cap * 1.12 if has_infinite_values else None,
    )
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)


def build_figure(
    metrics_by_experiment: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
) -> plt.Figure:
    fig_width = 3.8 + 2.6 * len(metrics_by_experiment)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 4.3))
    draw_grouped_split_violins(ax=ax, metrics_by_experiment=metrics_by_experiment)
    ax.set_ylabel("Hausdorff distance (log scale)")
    ax.set_title("Train/test Hausdorff distributions across snapshot experiments")

    legend_handles = [
        Patch(
            facecolor=METHOD_COLORS[method_name],
            edgecolor=METHOD_COLORS[method_name],
            label=method_name,
        )
        for method_name in METHOD_COLORS
    ]
    legend_handles.extend(
        [
            Patch(
                facecolor="#707070",
                edgecolor="#505050",
                alpha=0.9,
                label="Train (left)",
            ),
            Patch(
                facecolor="#d9d9d9",
                edgecolor="#505050",
                alpha=0.9,
                label="Test (right)",
            ),
        ]
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=5,
        frameon=False,
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.90))
    return fig


def main() -> None:
    args = parse_args()
    metrics_by_experiment = _select_experiments(
        load_all_split_metric_arrays(),
        include_multibkps=args.include_multibkps,
    )
    fig = build_figure(metrics_by_experiment=metrics_by_experiment)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
        print(args.output)

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
