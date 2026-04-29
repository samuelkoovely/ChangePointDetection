"""
Plot interval-wise clustering diagnostics for the primary-school day-1 analysis
and write explicit panel-C lambda selections.

The saved CSV is intended to be the handoff into `fig_entropy_inf_community.py`:
that figure will use the `best_cluster` at each interval's selected lambda when
the CSV is present.
"""

import math
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CLUSTER_DIR = Path(
    "//scratch/tmp/180/skoove/primaryschoolnet_heat/primaryschool_day1_flow_clustering"
)
OUTPUT_FIGURE_PATH = Path("figures/primaryschool_interval_nvi_selection.pdf")
OUTPUT_SELECTIONS_PATH = Path(
    "gridsearch_results/primaryschool_day1/panel_c_selected_lambdas.csv"
)


def load_cluster_result(folder_name, lamda):
    path = CLUSTER_DIR / folder_name / f"cluster{float(lamda):.11f}"
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_cluster_results(folder_name, lambdas):
    results = {}
    for lamda in np.asarray(lambdas, dtype=float):
        path = CLUSTER_DIR / folder_name / f"cluster{float(lamda):.11f}"
        if not path.exists():
            continue
        results[float(lamda)] = load_cluster_result(folder_name, float(lamda))
    if not results:
        raise FileNotFoundError(
            f"Could not find any clustering outputs under {CLUSTER_DIR / folder_name}."
        )
    return results


def summarize_clusters(cluster_results):
    sorted_lambdas = np.array(sorted(cluster_results), dtype=float)
    avg_num_clusters = np.array(
        [cluster_results[lamda]["avg_num_clusters"] for lamda in sorted_lambdas],
        dtype=float,
    )
    avg_nvi = np.array(
        [cluster_results[lamda]["avg_nvi"] for lamda in sorted_lambdas],
        dtype=float,
    )
    best_stabilities = np.array(
        [cluster_results[lamda]["best_stability"] for lamda in sorted_lambdas],
        dtype=float,
    )
    return sorted_lambdas, avg_num_clusters, avg_nvi, best_stabilities


def load_clustering_metadata():
    metadata_path = CLUSTER_DIR / "metadata.pkl"
    with metadata_path.open("rb") as handle:
        return pickle.load(handle)


def get_panel_c_intervals(metadata):
    return [
        interval
        for interval in sorted(
            metadata["intervals"],
            key=lambda interval: (int(interval["start"]), int(interval["stop"])),
        )
        if str(interval["label"]) != "full"
    ]


def format_hour_label(hour_value):
    total_minutes = int(round(float(hour_value) * 60.0))
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours:02d}:{minutes:02d}"


def select_lambda(sorted_lambdas, avg_nvi, best_stabilities):
    finite_nvi_mask = np.isfinite(avg_nvi)
    if finite_nvi_mask.any():
        finite_indices = np.flatnonzero(finite_nvi_mask)
        selected_index = int(finite_indices[np.argmin(avg_nvi[finite_nvi_mask])])
        selection_mode = "min_avg_nvi"
    else:
        selected_index = int(np.argmax(best_stabilities))
        selection_mode = "max_best_stability_fallback"

    return selected_index, selection_mode, float(sorted_lambdas[selected_index])


def build_selection_rows(panel_c_intervals, metadata):
    rows = []
    interval_summaries = []

    for interval in panel_c_intervals:
        cluster_results = load_cluster_results(
            interval["folder_name"],
            metadata["lambdas"],
        )
        (
            sorted_lambdas,
            avg_num_clusters,
            avg_nvi,
            best_stabilities,
        ) = summarize_clusters(cluster_results)

        selected_index, selection_mode, selected_lambda = select_lambda(
            sorted_lambdas=sorted_lambdas,
            avg_nvi=avg_nvi,
            best_stabilities=best_stabilities,
        )
        selected_result = cluster_results[selected_lambda]

        rows.append(
            {
                "interval_label": str(interval["label"]),
                "folder_name": str(interval["folder_name"]),
                "interval_start": int(interval["start"]),
                "interval_stop": int(interval["stop"]),
                "start_hour": float(interval["start_hour"]),
                "stop_hour": float(interval["stop_hour"]),
                "selected_lambda": selected_lambda,
                "selection_mode": selection_mode,
                "selected_avg_nvi": float(avg_nvi[selected_index]),
                "selected_avg_num_clusters": float(avg_num_clusters[selected_index]),
                "selected_best_stability": float(best_stabilities[selected_index]),
                "selected_partition_key": "best_cluster",
                "selected_cluster_file": str(
                    CLUSTER_DIR
                    / interval["folder_name"]
                    / f"cluster{selected_lambda:.11f}"
                ),
                "best_seed": int(selected_result["best_seed"]),
            }
        )
        interval_summaries.append(
            {
                "interval": interval,
                "sorted_lambdas": sorted_lambdas,
                "avg_num_clusters": avg_num_clusters,
                "avg_nvi": avg_nvi,
                "selected_index": selected_index,
                "selected_lambda": selected_lambda,
            }
        )

    return rows, interval_summaries


def plot_interval_summaries(interval_summaries):
    num_intervals = len(interval_summaries)
    num_cols = 2
    num_rows = int(math.ceil(num_intervals / num_cols))
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(12, 3.5 * num_rows),
        squeeze=False,
    )
    flat_axes = axes.ravel()

    for ax, summary in zip(flat_axes, interval_summaries):
        interval = summary["interval"]
        sorted_lambdas = summary["sorted_lambdas"]
        avg_num_clusters = summary["avg_num_clusters"]
        avg_nvi = summary["avg_nvi"]
        selected_lambda = summary["selected_lambda"]

        ax.plot(sorted_lambdas, avg_nvi, color="tab:red")
        ax.axvline(
            selected_lambda,
            color="black",
            linestyle=":",
            linewidth=1.0,
            alpha=0.8,
        )
        ax.set_xscale("log")
        ax.set_xlabel(r"$\lambda$ [s]")
        ax.set_ylabel("Avg. Norm. Var. Inf.", color="tab:red")
        ax.tick_params(axis="y", labelcolor="tab:red")
        ax.set_title(
            (
                f"{interval['label']} | "
                f"{format_hour_label(interval['start_hour'])} to "
                f"{format_hour_label(interval['stop_hour'])}"
            ),
            fontsize=11,
        )

        ax_right = ax.twinx()
        ax_right.plot(sorted_lambdas, avg_num_clusters, color="tab:blue")
        ax_right.set_ylabel("Avg. no. clusters", color="tab:blue")
        ax_right.tick_params(axis="y", labelcolor="tab:blue")

    for ax in flat_axes[num_intervals:]:
        ax.set_visible(False)

    fig.suptitle(
        "Primary School Day 1 Interval Selection for Panel C",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def main():
    metadata = load_clustering_metadata()
    panel_c_intervals = get_panel_c_intervals(metadata)
    selection_rows, interval_summaries = build_selection_rows(
        panel_c_intervals=panel_c_intervals,
        metadata=metadata,
    )

    fig = plot_interval_summaries(interval_summaries)

    OUTPUT_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIGURE_PATH, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    selections = pd.DataFrame(selection_rows)
    OUTPUT_SELECTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    selections.to_csv(OUTPUT_SELECTIONS_PATH, index=False)

    print(f"Saved figure to {OUTPUT_FIGURE_PATH}")
    print(f"Saved selections to {OUTPUT_SELECTIONS_PATH}")


if __name__ == "__main__":
    main()
