from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fig_entropy_inf_community import (
    format_hour_label,
    load_clustering_panels_context,
    plot_community_evolution_panel,
    plot_primary_school_panel,
)


START_TIME = time.perf_counter()
OUTPUT_PATH = Path("figures/fig_entropy_inf_community_wide_sankey.pdf")


def style_primary_school_panel(
    ax,
    *,
    title_fontsize=14,
    axis_label_fontsize=12,
    tick_labelsize=11,
    tick_rotation=45,
):
    ax.title.set_fontsize(title_fontsize)
    ax.xaxis.label.set_size(axis_label_fontsize)
    ax.yaxis.label.set_size(axis_label_fontsize)

    if ax.lines:
        t_hours = np.asarray(ax.lines[0].get_xdata(), dtype=float)
        min_hour = int(np.floor(np.min(t_hours)))
        max_hour = int(np.ceil(np.max(t_hours)))
        hour_ticks = np.arange(min_hour, max_hour + 1, dtype=int)
        ax.set_xticks(hour_ticks)
        ax.set_xticklabels(
            [format_hour_label(hour_tick) for hour_tick in hour_ticks],
            rotation=tick_rotation,
            ha="right" if tick_rotation else "center",
        )

    ax.tick_params(axis="both", labelsize=tick_labelsize)


def style_community_evolution_panel(
    ax,
    *,
    title_fontsize=14,
    text_fontsize=11,
    legend_fontsize=11,
    legend_title_fontsize=12,
    legend_markersize=11,
):
    ax.title.set_fontsize(title_fontsize)

    for text in ax.texts:
        text.set_fontsize(text_fontsize)

    legend = ax.get_legend()
    if legend is None:
        return

    legend.get_title().set_fontsize(legend_title_fontsize)
    for text in legend.get_texts():
        text.set_fontsize(legend_fontsize)

    legend_handles = getattr(legend, "legend_handles", None)
    if legend_handles is None:
        legend_handles = getattr(legend, "legendHandles", [])
    for handle in legend_handles:
        if hasattr(handle, "set_markersize"):
            handle.set_markersize(legend_markersize)


def main(output_path: Path = OUTPUT_PATH):
    clustering_context = load_clustering_panels_context()

    fig = plt.figure(figsize=(16.5, 5.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.35, 1.35])

    ax_a = fig.add_subplot(gs[0, 0])
    plot_primary_school_panel(ax_a)
    style_primary_school_panel(ax_a)

    ax_c = fig.add_subplot(gs[0, 1:])
    plot_community_evolution_panel(
        ax_c,
        clustering_context,
        title="(B) Community Evolution - Primary School - Day 1",
        title_loc="center",
    )
    style_community_evolution_panel(ax_c)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Total runtime: {time.perf_counter() - START_TIME:.2f} s")


if __name__ == "__main__":
    main()
