from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt

from fig_entropy_inf_community import (
    load_clustering_panels_context,
    plot_community_evolution_panel,
    plot_primary_school_panel,
)


START_TIME = time.perf_counter()
OUTPUT_PATH = Path("figures/fig_entropy_inf_community_wide_sankey.pdf")


def main(output_path: Path = OUTPUT_PATH):
    clustering_context = load_clustering_panels_context()

    fig = plt.figure(figsize=(16.5, 5.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.35, 1.35])

    ax_a = fig.add_subplot(gs[0, 0])
    plot_primary_school_panel(
        ax_a,
        title_fontsize=14,
        axis_label_fontsize=12,
        tick_labelsize=11,
        show_hourly_xticks=True,
        hourly_tick_label_rotation=45,
    )

    ax_c = fig.add_subplot(gs[0, 1:])
    plot_community_evolution_panel(
        ax_c,
        clustering_context,
        title="(B) Community Evolution - Primary School - Day 1",
        title_loc="center",
        title_fontsize=14,
        time_label_fontsize=11,
        legend_fontsize=11,
        legend_title_fontsize=12,
        legend_markersize=11,
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Total runtime: {time.perf_counter() - START_TIME:.2f} s")


if __name__ == "__main__":
    main()
