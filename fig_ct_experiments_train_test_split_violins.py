from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from fig_ct_experiments_train_test_boxplots import load_all_split_metric_arrays
from fig_snapshot_experiments_boxplots import METHOD_COLORS, style_axes
from fig_snapshot_experiments_train_test_split_violins import (
    _clip_violin_half,
    _compute_hausdorff_cap,
)


BASE_DIR = Path(__file__).resolve().parent
ENTROPY_COLOR = METHOD_COLORS["Entropy"]
DEFAULT_OUTPUT = (
    BASE_DIR / "figures" / "fig_ct_experiments_train_test_split_violins.pdf"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot paired train/test split violins of per-sample Hausdorff "
            "performance for the continuous-time block-activity experiments."
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


def _prepare_violin_values(
    values: np.ndarray,
    *,
    hausdorff_cap: float,
) -> np.ndarray:
    display_values = np.asarray(values, dtype=float).copy()
    display_values[np.isposinf(display_values)] = hausdorff_cap
    display_values[np.isneginf(display_values)] = 0.0
    display_values[display_values < 0] = 0.0
    display_values = display_values[~np.isnan(display_values)]
    if display_values.size == 0:
        raise ValueError("No finite values available to plot.")
    return display_values


def _draw_half_violin(
    ax: plt.Axes,
    *,
    values: np.ndarray,
    position: float,
    width: float,
    side: str,
    facecolor: tuple[float, float, float] | str,
    edgecolor: str,
    hausdorff_cap: float,
) -> None:
    display_values = _prepare_violin_values(
        values,
        hausdorff_cap=hausdorff_cap,
    )
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

    if np.any(np.isposinf(values)):
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
    metrics_by_experiment: dict[str, dict[str, dict[str, np.ndarray]]],
) -> None:
    experiment_names = list(metrics_by_experiment.keys())
    centers = np.arange(len(experiment_names), dtype=float) * 2.9
    width = 0.56

    style_axes(ax)
    hausdorff_cap, has_infinite_values = _compute_hausdorff_cap(
        {
            experiment_name: {"Entropy": splits}
            for experiment_name, splits in metrics_by_experiment.items()
        }
    )

    for experiment_idx, experiment_name in enumerate(experiment_names):
        position = centers[experiment_idx]
        train_values = np.asarray(
            metrics_by_experiment[experiment_name]["Train"]["hausdorff"],
            dtype=float,
        )
        test_values = np.asarray(
            metrics_by_experiment[experiment_name]["Test"]["hausdorff"],
            dtype=float,
        )
        combined_display_values = np.concatenate(
            [
                _prepare_violin_values(
                    train_values,
                    hausdorff_cap=hausdorff_cap,
                ),
                _prepare_violin_values(
                    test_values,
                    hausdorff_cap=hausdorff_cap,
                ),
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
            position=position,
            width=width,
            side="left",
            facecolor=ENTROPY_COLOR,
            edgecolor=ENTROPY_COLOR,
            hausdorff_cap=hausdorff_cap,
        )
        _draw_half_violin(
            ax,
            values=test_values,
            position=position,
            width=width,
            side="right",
            facecolor=_lighten_color(ENTROPY_COLOR),
            edgecolor=ENTROPY_COLOR,
            hausdorff_cap=hausdorff_cap,
        )

    ax.set_xticks(centers)
    ax.set_xticklabels(experiment_names)
    ax.set_xlim(centers[0] - 0.95, centers[-1] + 0.95)
    ax.set_ylim(
        bottom=0.0,
        top=hausdorff_cap * 1.12 if has_infinite_values else None,
    )
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)


def build_figure(
    metrics_by_experiment: dict[str, dict[str, dict[str, np.ndarray]]],
) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(7.4, 4.1))
    draw_grouped_split_violins(ax=ax, metrics_by_experiment=metrics_by_experiment)
    ax.set_ylabel("Hausdorff distance")
    ax.set_title("Train/test Hausdorff distributions across CT experiments")
    fig.tight_layout()
    return fig


def main() -> None:
    args = parse_args()
    metrics_by_experiment = load_all_split_metric_arrays()
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
