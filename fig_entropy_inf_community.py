import pickle
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import auxiliary_functions
from TemporalNetwork import ContTempNetwork
from primary_school_ruptures_defaults import (
    PRIMARY_SCHOOL_DEFAULT_PENALTY,
    build_primary_school_ruptures_results_path,
)
try:
    from sankeyflow import Sankey
except ModuleNotFoundError:
    Sankey = None


START_TIME = time.perf_counter()
NODE_LABEL_PATTERN = re.compile(r"L(\d+)_C(\d+)$")


OUTPUT_PATH = Path("figures/fig_entropy_inf_community.pdf")
CLUSTER_DIR = Path(
    "//scratch/tmp/180/skoove/primaryschoolnet_heat/primaryschool_day1_flow_clustering"
)
NETWORK_PATH = (
    "data/primaryschoolnet"
)
PRIMARY_SCHOOL_RUPTURES_RESULTS_PATH = build_primary_school_ruptures_results_path(
    Path(".")
)
PRIMARY_SCHOOL_SELECTED_PENALTY = PRIMARY_SCHOOL_DEFAULT_PENALTY
PANEL_C_SELECTED_LAMBDAS_PATH = Path(
    "gridsearch_results/primaryschool_day1/panel_c_selected_lambdas.csv"
)

SUMMARY_INTERVAL_LABEL = "960_1320"


def load_cluster_result(folder_name, lamda):
    path = CLUSTER_DIR / folder_name / f"cluster{lamda:.11f}"
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
    nvi_values = np.array(
        [cluster_results[lamda]["avg_nvi"] for lamda in sorted_lambdas],
        dtype=float,
    )
    return sorted_lambdas, avg_num_clusters, nvi_values


def load_clustering_metadata():
    metadata_path = CLUSTER_DIR / "metadata.pkl"
    with metadata_path.open("rb") as handle:
        return pickle.load(handle)


def get_interval_map(metadata):
    intervals = sorted(
        metadata["intervals"],
        key=lambda interval: (int(interval["start"]), int(interval["stop"])),
    )
    return {str(interval["label"]): interval for interval in intervals}


def resolve_summary_interval(interval_map):
    if SUMMARY_INTERVAL_LABEL in interval_map:
        return interval_map[SUMMARY_INTERVAL_LABEL]
    non_full_intervals = [
        interval for interval in interval_map.values() if str(interval["label"]) != "full"
    ]
    if not non_full_intervals:
        raise ValueError("No non-full intervals found in clustering metadata.")
    return non_full_intervals[-1]


def select_best_interval_result(cluster_results):
    return max(
        cluster_results.values(),
        key=lambda result: (float(result["best_stability"]), -float(result["lamda"])),
    )


def find_matching_lambda(available_lambdas, requested_lambda):
    for lamda in available_lambdas:
        if np.isclose(float(lamda), float(requested_lambda)):
            return float(lamda)
    raise ValueError(
        f"Could not match selected lambda={requested_lambda} "
        f"against available lambdas: {sorted(float(lamda) for lamda in available_lambdas)}"
    )


def load_panel_c_selected_lambdas(
    selection_path: Path = PANEL_C_SELECTED_LAMBDAS_PATH,
):
    if not selection_path.exists():
        return None

    selections = pd.read_csv(selection_path)
    required_columns = {"interval_label", "selected_lambda"}
    missing_columns = required_columns.difference(selections.columns)
    if missing_columns:
        raise ValueError(
            f"{selection_path} is missing required columns: {sorted(missing_columns)}"
        )

    duplicated_labels = selections["interval_label"][
        selections["interval_label"].duplicated()
    ]
    if not duplicated_labels.empty:
        raise ValueError(
            f"{selection_path} contains duplicated interval labels: "
            f"{sorted(duplicated_labels.unique())}"
        )

    return {
        str(row.interval_label): float(row.selected_lambda)
        for row in selections.itertuples()
    }


def resolve_panel_c_cluster_selection(
    panel_c_intervals,
    metadata,
    selected_lambdas=None,
):
    bestclusters = []
    resolved_lambdas = {}
    available_lambdas = metadata["lambdas"]

    for interval in panel_c_intervals:
        cluster_results = load_cluster_results(
            interval["folder_name"],
            available_lambdas,
        )
        interval_label = str(interval["label"])

        if selected_lambdas is None:
            selected_result = select_best_interval_result(cluster_results)
        else:
            if interval_label not in selected_lambdas:
                raise ValueError(
                    f"Missing panel-C lambda selection for interval {interval_label} "
                    f"in {PANEL_C_SELECTED_LAMBDAS_PATH}."
                )
            matched_lambda = find_matching_lambda(
                cluster_results,
                selected_lambdas[interval_label],
            )
            selected_result = cluster_results[matched_lambda]

        resolved_lambdas[interval_label] = float(selected_result["lamda"])
        bestclusters.append(selected_result["best_cluster"])

    return bestclusters, resolved_lambdas


def format_hour_label(hour_value):
    total_minutes = int(round(float(hour_value) * 60.0))
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours:02d}:{minutes:02d}"


def build_time_labels(intervals):
    labels = [
        f"{format_hour_label(interval['start_hour'])}-\n{format_hour_label(interval['stop_hour'])}"
        for interval in intervals
    ]
    if len(labels) <= 1:
        positions = np.array([0.5], dtype=float)
    else:
        # Anchor each time label at the center of its Sankey column.
        positions = (np.arange(len(labels), dtype=float) + 0.5) / float(len(labels))
    return labels, positions


def school_class_sort_key(school_class):
    school_class_str = str(school_class)
    normalized = school_class_str.strip()
    if "teacher" in normalized.lower():
        return (float("inf"), "ZZZ", normalized)

    match = re.fullmatch(r"(\d+)\s*([A-Za-z]+)", normalized)
    if match is not None:
        grade = int(match.group(1))
        section = match.group(2).upper()
        return (grade, section, normalized)

    return (float("inf"), normalized.upper(), normalized)


def get_ordered_flow_types(flow_values):
    unique_flow_types = {str(flow_type) for flow_type in flow_values}
    return sorted(unique_flow_types, key=school_class_sort_key)


def parse_node_label(node_label):
    match = NODE_LABEL_PATTERN.fullmatch(str(node_label))
    if match is None:
        return (float("inf"), float("inf"))
    return (int(match.group(1)), int(match.group(2)))


def build_class_dict(net):
    return {
        school_class: set(net.node_array[net.node_class_array == school_class])
        for school_class in np.unique(net.node_class_array)
    }


def build_node_to_cluster_dict(comms):
    node_to_cluster = {}
    for cluster_idx, community in enumerate(comms):
        for node in community:
            node_to_cluster[int(node)] = cluster_idx
    return node_to_cluster


def build_node_class_lookup(class_dict):
    node_to_class = {}
    for school_class, class_nodes in class_dict.items():
        for node in class_nodes:
            node_to_class[int(node)] = school_class
    return node_to_class


def build_flow_dataframe(source_comms, target_comms, class_dict):
    source_node_to_cluster = build_node_to_cluster_dict(source_comms)
    target_node_to_cluster = build_node_to_cluster_dict(target_comms)
    node_to_class = build_node_class_lookup(class_dict)

    flow_counts = {}
    shared_nodes = sorted(
        set(source_node_to_cluster).intersection(target_node_to_cluster)
    )
    for node in shared_nodes:
        school_class = node_to_class.get(node)
        if school_class is None:
            continue
        key = (
            int(source_node_to_cluster[node]),
            int(target_node_to_cluster[node]),
            school_class,
        )
        flow_counts[key] = flow_counts.get(key, 0) + 1

    sorted_flow_items = sorted(
        flow_counts.items(),
        key=lambda item: (
            int(item[0][0]),
            int(item[0][1]),
            school_class_sort_key(item[0][2]),
        ),
    )
    flows = [
        {
            "source": source_idx,
            "target": target_idx,
            "type": school_class,
            "value": value,
        }
        for (source_idx, target_idx, school_class), value in sorted_flow_items
        if value > 0
    ]
    return pd.DataFrame(flows, columns=["source", "target", "type", "value"])


def offset_flow_columns(flow_frames):
    adjusted_frames = []
    current_offset = 0

    for frame in flow_frames:
        adjusted_frame = frame.copy()
        adjusted_frame["source"] += current_offset
        target_offset = int(adjusted_frame["source"].max() + 1)
        adjusted_frame["target_label"] = adjusted_frame["target"] + target_offset
        adjusted_frames.append(adjusted_frame)
        # The next layer's source ids must start where this layer's targets start.
        current_offset = target_offset

    return adjusted_frames


def add_sankey_node_labels(flow_frames):
    labeled_frames = []
    for level_index, frame in enumerate(flow_frames):
        labeled_frame = frame.copy()
        labeled_frame["source_label"] = labeled_frame["source"].map(
            lambda value: f"L{level_index}_C{int(value)}"
        )
        labeled_frame["target_label"] = labeled_frame["target"].map(
            lambda value: f"L{level_index + 1}_C{int(value)}"
        )
        labeled_frames.append(labeled_frame)
    return labeled_frames


def build_node_class_profiles(flow_frames):
    node_profiles = {}
    for frame in flow_frames:
        for row in frame.itertuples():
            flow_type = str(row.type)
            for node_label in (row.source_label, row.target_label):
                profile = node_profiles.setdefault(str(node_label), {})
                profile[flow_type] = profile.get(flow_type, 0) + int(row.value)
    return node_profiles


def build_sankey_node_sort_key(node_label, node_profiles, ordered_flow_types):
    node_profile = node_profiles.get(str(node_label), {})
    class_counts = tuple(
        -int(node_profile.get(flow_type, 0)) for flow_type in ordered_flow_types
    )
    _, cluster_index = parse_node_label(node_label)
    return class_counts + (cluster_index, str(node_label))


def build_sankey_nodes(flow_frames, ordered_flow_types):
    node_profiles = build_node_class_profiles(flow_frames)
    layer_totals = [
        frame.groupby("source_label", sort=False)["value"].sum()
        for frame in flow_frames
    ]
    layer_totals.append(
        flow_frames[-1].groupby("target_label", sort=False)["value"].sum()
    )

    layer_node_labels = []
    for totals in layer_totals:
        ordered_node_labels = sorted(
            (str(node_label) for node_label in totals.index),
            key=lambda node_label: build_sankey_node_sort_key(
                node_label,
                node_profiles,
                ordered_flow_types,
            ),
        )
        layer_node_labels.append(ordered_node_labels)

    node_order_lookup = {
        node_label: order_index
        for ordered_node_labels in layer_node_labels
        for order_index, node_label in enumerate(ordered_node_labels)
    }
    nodes = [
        [
            (
                node_label,
                float(totals.loc[node_label]),
                {"color": "black"},
            )
            for node_label in ordered_node_labels
        ]
        for totals, ordered_node_labels in zip(layer_totals, layer_node_labels)
    ]
    return nodes, node_order_lookup


def has_valid_sankey_flow_frames(flow_frames):
    return bool(flow_frames) and all(
        (not frame.empty) and (frame["value"].sum() > 0) for frame in flow_frames
    )


def load_primary_school_penalty_plot_data(
    results_path: Path = PRIMARY_SCHOOL_RUPTURES_RESULTS_PATH,
    penalty: float = PRIMARY_SCHOOL_SELECTED_PENALTY,
):
    with open(results_path, "rb") as handle:
        ruptures_results = pickle.load(handle)

    signal = np.asarray(ruptures_results["signal_array"], dtype=float)
    t_hours = np.asarray(ruptures_results["t_samples"], dtype=float) / 3600.0

    for result in ruptures_results["lambda_results"]:
        if np.isclose(float(result["penalty"]), float(penalty)):
            return {
                "signal": signal,
                "t_hours": t_hours,
                "change_point_indices": np.asarray(
                    result["change_point_indices"],
                    dtype=int,
                ),
                "change_point_hours": np.asarray(
                    result["change_point_t_hours"],
                    dtype=float,
                ),
                "lamda": float(ruptures_results["lamda"]),
                "window_minutes": float(ruptures_results["window_minutes"]),
                "penalty": float(result["penalty"]),
            }

    raise ValueError(f"Could not find penalty={penalty} in {results_path}.")


def load_clustering_panels_context():
    try:
        clustering_metadata = load_clustering_metadata()
        interval_map = get_interval_map(clustering_metadata)
        summary_interval = resolve_summary_interval(interval_map)
        summary_cluster_results = load_cluster_results(
            summary_interval["folder_name"],
            clustering_metadata["lambdas"],
        )
        summary_lambdas, summary_avg_num_clusters, summary_avg_nvi = summarize_clusters(
            summary_cluster_results
        )

        panel_c_intervals = [
            interval
            for interval in sorted(
                interval_map.values(),
                key=lambda interval: (int(interval["start"]), int(interval["stop"])),
            )
            if str(interval["label"]) != "full"
        ]
        panel_c_selected_lambdas = load_panel_c_selected_lambdas()
        bestclusters, resolved_panel_c_lambdas = resolve_panel_c_cluster_selection(
            panel_c_intervals=panel_c_intervals,
            metadata=clustering_metadata,
            selected_lambdas=panel_c_selected_lambdas,
        )

        time_labels, time_label_positions = build_time_labels(panel_c_intervals)

        net_rw = ContTempNetwork.load(
            NETWORK_PATH,
            attributes_list=[
                "node_to_label_dict",
                "events_table",
                "node_class_array",
            ],
        )

        class_dict = build_class_dict(net_rw)
        flow_frames = [
            build_flow_dataframe(source_comms, target_comms, class_dict)
            for source_comms, target_comms in zip(bestclusters[:-1], bestclusters[1:])
        ]
        sankey_ready = has_valid_sankey_flow_frames(flow_frames)
        if sankey_ready:
            flow_frames = add_sankey_node_labels(flow_frames)
            df_flows = pd.concat(flow_frames, ignore_index=True)
        else:
            df_flows = pd.DataFrame(
                columns=["source", "target", "type", "value", "source_label", "target_label"]
            )

        return {
            "summary_interval": summary_interval,
            "summary_lambdas": summary_lambdas,
            "summary_avg_num_clusters": summary_avg_num_clusters,
            "summary_avg_nvi": summary_avg_nvi,
            "resolved_panel_c_lambdas": resolved_panel_c_lambdas,
            "time_labels": time_labels,
            "time_label_positions": time_label_positions,
            "flow_frames": flow_frames,
            "df_flows": df_flows,
            "sankey_ready": sankey_ready,
        }
    except FileNotFoundError:
        return None


def plot_primary_school_panel(ax, primary_school_panel=None):
    if primary_school_panel is None:
        primary_school_panel = load_primary_school_penalty_plot_data()

    ax.plot(
        primary_school_panel["t_hours"],
        primary_school_panel["signal"],
        color="black",
        alpha=0.75,
    )
    for cp_hour in primary_school_panel["change_point_hours"]:
        ax.axvline(
            cp_hour,
            color="red",
            linestyle="--",
            linewidth=1.1,
            alpha=0.9,
        )
    if len(primary_school_panel["change_point_indices"]) > 0:
        ax.scatter(
            primary_school_panel["change_point_hours"],
            primary_school_panel["signal"][primary_school_panel["change_point_indices"]],
            color="red",
            s=14,
            zorder=3,
        )

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Entropy")
    ax.set_title("(A) Local Conditional Entropy", loc="left", fontsize=12)


def plot_flow_stability_panel(ax, clustering_context):
    if clustering_context is None:
        ax.text(
            0.5,
            0.5,
            "Missing flow-clustering\noutputs under\n`primaryschool_day1_flow_clustering`.",
            ha="center",
            va="center",
            fontsize=11,
            transform=ax.transAxes,
        )
        ax.set_title("(B) Flow Stability", loc="left", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        return

    summary_interval = clustering_context["summary_interval"]
    summary_lambdas = clustering_context["summary_lambdas"]
    summary_avg_num_clusters = clustering_context["summary_avg_num_clusters"]
    summary_avg_nvi = clustering_context["summary_avg_nvi"]
    resolved_panel_c_lambdas = clustering_context["resolved_panel_c_lambdas"]

    ax.plot(summary_lambdas, summary_avg_nvi, color="tab:red", label="static norm NVI")
    summary_interval_label = str(summary_interval["label"])
    if summary_interval_label in resolved_panel_c_lambdas:
        ax.axvline(
            resolved_panel_c_lambdas[summary_interval_label],
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
            f"(B) Flow Stability - {format_hour_label(summary_interval['start_hour'])} "
            f"to {format_hour_label(summary_interval['stop_hour'])}"
        ),
        loc="left",
        fontsize=12,
    )

    ax_right = ax.twinx()
    ax_right.plot(summary_lambdas, summary_avg_num_clusters, color="tab:blue")
    ax_right.set_xlabel(r"$\lambda$ [s]")
    ax_right.set_ylabel("Avg. no. clusters", color="tab:blue")
    ax_right.tick_params(axis="y", labelcolor="tab:blue")


def plot_community_evolution_panel(
    ax,
    clustering_context,
    title="(C) Community Evolution - Primary School - Day 1",
    title_loc="left",
):
    if clustering_context is not None and clustering_context["sankey_ready"] and Sankey is not None:
        flow_frames = clustering_context["flow_frames"]
        df_flows = clustering_context["df_flows"]
        time_labels = clustering_context["time_labels"]
        time_label_positions = clustering_context["time_label_positions"]
        flow_types = get_ordered_flow_types(df_flows["type"])
        nodes, node_order_lookup = build_sankey_nodes(flow_frames, flow_types)
        flow_type_order = {
            flow_type: index for index, flow_type in enumerate(flow_types)
        }
        color_list = auxiliary_functions.generate_plasma_colors(len(flow_types))
        dict_color = {
            flow_type: color_list[i] for i, flow_type in enumerate(flow_types)
        }
        ordered_rows = sorted(
            df_flows.itertuples(),
            key=lambda row: (
                parse_node_label(row.source_label)[0],
                node_order_lookup.get(str(row.source_label), float("inf")),
                node_order_lookup.get(str(row.target_label), float("inf")),
                flow_type_order[str(row.type)],
                int(row.source),
                int(row.target),
            ),
        )
        flows = [
            (
                row.source_label,
                row.target_label,
                row.value,
                {"color": dict_color[str(row.type)]},
            )
            for row in ordered_rows
        ]
        try:
            sankey = Sankey(
                flows=flows,
                nodes=nodes,
                node_opts={"label_format": "", "color": "black"},
            )
            sankey.draw(ax)

            for x_pos, label in zip(time_label_positions, time_labels):
                ax.text(
                    x_pos,
                    -0.07,
                    label,
                    fontsize=10,
                    ha="center",
                    va="top",
                    transform=ax.transAxes,
                    clip_on=False,
                )

            handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=dict_color[flow_type],
                    markersize=10,
                )
                for flow_type in flow_types
            ]
            ax.legend(
                handles,
                flow_types,
                title="Group",
                loc="center left",
                bbox_to_anchor=(1, 0.5),
            )
        except ZeroDivisionError:
            clustering_context["sankey_ready"] = False
    if clustering_context is None:
        ax.text(
            0.5,
            0.5,
            "Missing flow-clustering\noutputs under\n`primaryschool_day1_flow_clustering`.",
            ha="center",
            va="center",
            fontsize=11,
            transform=ax.transAxes,
        )
    elif Sankey is None:
        ax.text(
            0.5,
            0.5,
            "Install `sankeyflow`\nto render this panel.",
            ha="center",
            va="center",
            fontsize=11,
            transform=ax.transAxes,
        )
    elif not clustering_context["sankey_ready"]:
        ax.text(
            0.5,
            0.5,
            "No non-empty\ncommunity overlaps\nwere found across\nadjacent intervals.",
            ha="center",
            va="center",
            fontsize=11,
            transform=ax.transAxes,
        )
    ax.set_title(title, loc=title_loc, fontsize=12)
    ax.set_yticks([])
    ax.set_frame_on(False)


def main(output_path: Path = OUTPUT_PATH):
    clustering_context = load_clustering_panels_context()

    fig = plt.figure(figsize=(12, 4))
    gs = fig.add_gridspec(1, 3)

    ax_a = fig.add_subplot(gs[0, 0])
    plot_primary_school_panel(ax_a)

    ax_b = fig.add_subplot(gs[0, 1])
    plot_flow_stability_panel(ax_b, clustering_context)

    ax_c = fig.add_subplot(gs[0, 2])
    plot_community_evolution_panel(ax_c, clustering_context)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Total runtime: {time.perf_counter() - START_TIME:.2f} s")


if __name__ == "__main__":
    main()
