import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import auxiliary_functions
from TemporalNetwork import ContTempNetwork
from sankeyflow import Sankey


START_TIME = time.perf_counter()


OUTPUT_PATH = Path("figures/fig_entropy_inf_community.pdf")
CLUSTER_DIR = Path(
    "//scratch/tmp/180/skoove/primaryschoolnet_heat/primaryschool_day1_flow_clustering"
)
NETWORK_PATH = (
    "data/primaryschoolnet"
)
PRIMARY_SCHOOL_SIGNAL_PATH = Path(
    "gridsearch_results/primaryschool_day1/window_S_selected/3600/window_S1.00000000000"
)
PRIMARY_SCHOOL_RUPTURES_RESULTS_PATH = Path(
    "gridsearch_results/primaryschool_day1_ruptures/forward/window_3600/lamda_1.00000000000/ruptures_results.pkl"
)
PRIMARY_SCHOOL_SELECTED_PENALTY = 60.0

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


def format_hour_label(hour_value):
    total_minutes = int(round(float(hour_value) * 60.0))
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours:02d}:{minutes:02d}"


def build_time_labels(intervals):
    labels = [format_hour_label(interval["start_hour"]) for interval in intervals]
    positions = np.linspace(0.1, 0.9, len(labels))
    return labels, positions


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

    flows = [
        {
            "source": source_idx,
            "target": target_idx,
            "type": school_class,
            "value": value,
        }
        for (source_idx, target_idx, school_class), value in sorted(flow_counts.items())
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


def build_sankey_nodes(flow_frames):
    nodes = []
    for frame in flow_frames:
        source_totals = frame.groupby("source", sort=False)["value"].sum()
        nodes.append(
            [
                (int(node), value, {"color": "black"})
                for node, value in source_totals.items()
            ]
        )

    target_totals = flow_frames[-1].groupby("target_label", sort=False)["value"].sum()
    nodes.append(
        [
            (
                int(node),
                value,
                {"color": "black"},
            )
            for node, value in target_totals.items()
        ]
    )
    return nodes


def has_valid_sankey_flow_frames(flow_frames):
    return bool(flow_frames) and all(not frame.empty for frame in flow_frames)


def load_primary_school_penalty_plot_data(
    signal_path: Path = PRIMARY_SCHOOL_SIGNAL_PATH,
    results_path: Path = PRIMARY_SCHOOL_RUPTURES_RESULTS_PATH,
    penalty: float = PRIMARY_SCHOOL_SELECTED_PENALTY,
):
    with open(signal_path, "rb") as handle:
        signal_payload = pickle.load(handle)

    signal = np.asarray(signal_payload["signal_array"], dtype=float)
    t_hours = np.asarray(signal_payload["t_samples"], dtype=float) / 3600.0

    with open(results_path, "rb") as handle:
        ruptures_results = pickle.load(handle)

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
bestclusters = []
for interval in panel_c_intervals:
    cluster_results = load_cluster_results(
        interval["folder_name"],
        clustering_metadata["lambdas"],
    )
    bestclusters.append(select_best_interval_result(cluster_results)["best_cluster"])

TIME_LABELS, TIME_LABEL_POSITIONS = build_time_labels(panel_c_intervals)

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
    flow_frames = offset_flow_columns(flow_frames)
    df_flows = pd.concat(flow_frames, ignore_index=True)
else:
    df_flows = pd.DataFrame(columns=["source", "target", "type", "value", "target_label"])


fig = plt.figure(figsize=(12, 4))
gs = fig.add_gridspec(1, 3)

ax_a = fig.add_subplot(gs[0, 0])
primary_school_panel = load_primary_school_penalty_plot_data()
ax_a.plot(
    primary_school_panel["t_hours"],
    primary_school_panel["signal"],
    color="black",
    alpha=0.75,
)
for cp_hour in primary_school_panel["change_point_hours"]:
    ax_a.axvline(
        cp_hour,
        color="red",
        linestyle="--",
        linewidth=1.1,
        alpha=0.9,
    )
if len(primary_school_panel["change_point_indices"]) > 0:
    ax_a.scatter(
        primary_school_panel["change_point_hours"],
        primary_school_panel["signal"][primary_school_panel["change_point_indices"]],
        color="red",
        s=14,
        zorder=3,
    )

ax_a.set_xlabel("Time (hours)")
ax_a.set_ylabel("Entropy")
    
ax_a.set_title("(A) Local Conditional Entropy", loc="left", fontsize=12)

ax_b = fig.add_subplot(gs[0, 1])
ax_b.plot(summary_lambdas, summary_avg_nvi, color="tab:red", label="static norm NVI")
ax_b.set_xscale("log")
ax_b.set_xlabel(r"$\lambda$ [s]")
ax_b.set_ylabel("Avg. Norm. Var. Inf.", color="tab:red")
ax_b.tick_params(axis="y", labelcolor="tab:red")
ax_b.set_title(
    (
        f"(B) Flow Stability - {format_hour_label(summary_interval['start_hour'])} "
        f"to {format_hour_label(summary_interval['stop_hour'])}"
    ),
    loc="left",
    fontsize=12,
)

ax_b_right = ax_b.twinx()
ax_b_right.plot(summary_lambdas, summary_avg_num_clusters, color="tab:blue")
ax_b_right.set_xlabel(r"$\lambda$ [s]")
ax_b_right.set_ylabel("Avg. no. clusters", color="tab:blue")
ax_b_right.tick_params(axis="y", labelcolor="tab:blue")

ax_c = fig.add_subplot(gs[0, 2])
if sankey_ready:
    nodes = build_sankey_nodes(flow_frames)
    flow_types = df_flows["type"].unique()
    color_list = auxiliary_functions.generate_plasma_colors(len(flow_types))
    dict_color = {
        flow_type: color_list[i] for i, flow_type in enumerate(flow_types)
    }
    flows = [
        (
            row.source,
            row.target_label,
            row.value,
            {"color": dict_color[row.type]},
        )
        for row in df_flows.itertuples()
    ]

    sankey = Sankey(flows=flows, nodes=nodes, node_opts={"label_format": ""})
    sankey.draw()

    for x_pos, label in zip(TIME_LABEL_POSITIONS, TIME_LABELS):
        ax_c.text(x_pos, -0.05, label, fontsize=10, ha="center", transform=ax_c.transAxes)

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
    ax_c.legend(
        handles,
        flow_types,
        title="Group",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )
else:
    ax_c.text(
        0.5,
        0.5,
        "No non-empty\ncommunity overlaps\nwere found across\nadjacent intervals.",
        ha="center",
        va="center",
        fontsize=11,
        transform=ax_c.transAxes,
    )
ax_c.set_title(
    "(C) Community Evolution - Primary School - Day 1",
    loc="left",
    fontsize=12,
)
ax_c.set_yticks([])
ax_c.set_frame_on(False)

plt.tight_layout()
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_PATH, format="pdf", dpi=300, bbox_inches="tight")
plt.show()
print(f"Total runtime: {time.perf_counter() - START_TIME:.2f} s")
