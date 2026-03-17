import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import auxiliary_functions
from FlowStability import avg_norm_var_information
from TemporalNetwork import ContTempNetwork
from sankeyflow import Sankey


START_TIME = time.perf_counter()


ENTROPY_DIR = "//scratch/tmp/180/skoove/primaryschoolnet_rw/conditional_S_selected_hr"
CLUSTER_DIR = "//scratch/tmp/180/skoove/primaryschoolnet_rw"
NETWORK_PATH = (
    "/home/b/skoove/compute/entropy/paper_data/socio_pat_primary_school/primaryschoolnet"
)
CSV_PATH = "/home/b/skoove/compute/entropy/paper_data/socio_pat_primary_school/primaryschool.csv"

SELECTED_LAMBDAS = np.logspace(-5, 0, 10)
LAMBDAS_GROWING = np.logspace(-5, 0, 200)
INTERVAL_CONFIGS = [
    ("full", "clustersplot"),
    ("0_240", "clustersplot0_240"),
    ("240_600", "clustersplot240_600"),
    ("600_960", "clustersplot600_960"),
    ("960_1320", "clustersplot960_1320"),
    ("1320_1556", "clustersplot1320_1556"),
]
BEST_CLUSTER_INDICES = {
    "0_240": 170,
    "240_600": 135,
    "600_960": 180,
    "960_1320": 125,
    "1320_1556": 90,
}
TIME_LABELS = ["08:30", "10:30", "12:00", "14:00", "16:00"]
TIME_LABEL_POSITIONS = [0.1, 0.3, 0.5, 0.7, 0.9]


def load_conditional_entropies(lambdas):
    conditional_entropies = []
    for lamda in lambdas:
        path = f"{ENTROPY_DIR}/S_rate{lamda:.11f}"
        with open(path, "rb") as handle:
            s_rate = pickle.load(handle)
        conditional_entropies.append(s_rate["S_rate"][f"{lamda:.11f}"])
    return conditional_entropies


def load_cluster_results(folder_name, lambdas):
    cluster_results = {}
    for lamda in lambdas:
        path = f"{CLUSTER_DIR}/{folder_name}/cluster{lamda:.11f}"
        with open(path, "rb") as handle:
            cluster_results[lamda] = pickle.load(handle)
    return cluster_results


def summarize_clusters(cluster_results, lambdas):
    avg_cluster_sizes = [
        np.mean([len(cluster) for cluster in cluster_results[lamda] if len(cluster) > 1])
        for lamda in lambdas
    ]
    nvi_values = [avg_norm_var_information(cluster_results[lamda]) for lamda in lambdas]
    return avg_cluster_sizes, nvi_values


def build_class_dict(net):
    return {
        school_class: set(net.node_array[net.node_class_array == school_class])
        for school_class in np.unique(net.node_class_array)
    }


def build_flow_dataframe(source_comms, target_comms, class_dict):
    flows = []
    for school_class, class_nodes in class_dict.items():
        for source_idx, source_comm in enumerate(source_comms):
            for target_idx, target_comm in enumerate(target_comms):
                value = len(class_nodes.intersection(source_comm).intersection(target_comm))
                if value > 0:
                    flows.append(
                        {
                            "source": source_idx,
                            "target": target_idx,
                            "type": school_class,
                            "value": value,
                        }
                    )
    return pd.DataFrame(flows)


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
        nodes.append(
            [
                (node, frame.loc[frame["source"] == node, "value"].sum(), {"color": "black"})
                for node in frame["source"].unique()
            ]
        )

    last_frame = flow_frames[-1]
    nodes.append(
        [
            (
                node,
                last_frame.loc[last_frame["target_label"] == node, "value"].sum(),
                {"color": "black"},
            )
            for node in last_frame["target_label"].unique()
        ]
    )
    return nodes


Conditional_S_selected_hr = load_conditional_entropies(SELECTED_LAMBDAS)

net_rw = ContTempNetwork.load(
    NETWORK_PATH,
    attributes_list=[
        "node_to_label_dict",
        "events_table",
        "times",
        "time_grid",
        "num_nodes",
        "_overlapping_events_merged",
        "start_date",
        "node_label_array",
        "male_array",
        "female_array",
        "node_first_start_array",
        "node_last_end_array",
        "node_class_array",
        "datetimes",
    ],
)

df = pd.read_csv(
    CSV_PATH,
    header=None,
    sep="\t",
    names=["time", "id1", "id2", "class1", "class2"],
)
df["hour"] = df["time"] // 3600
df["minute"] = (df["time"] % 3600) / 60

net_times_hours = net_rw.times / 3600
flag10 = np.argmax(net_times_hours > 10)
flag12 = np.argmax(net_times_hours > 12)
flag14 = np.argmax(net_times_hours > 14)
flag16 = np.argmax(net_times_hours > 16)
flagday1 = np.argmax(net_times_hours > 18)
print(flag10, flag12, flag14, flag16, flagday1)


interval_results = {
    label: load_cluster_results(folder_name, LAMBDAS_GROWING)
    for label, folder_name in INTERVAL_CONFIGS
}
interval_summaries = {
    label: summarize_clusters(cluster_results, LAMBDAS_GROWING)
    for label, cluster_results in interval_results.items()
}

bestclusters = [
    interval_results[label][LAMBDAS_GROWING[best_index]][0]
    for label, best_index in BEST_CLUSTER_INDICES.items()
]

class_dict = build_class_dict(net_rw)
flow_frames = [
    build_flow_dataframe(source_comms, target_comms, class_dict)
    for source_comms, target_comms in zip(bestclusters[:-1], bestclusters[1:])
]
flow_frames = offset_flow_columns(flow_frames)
df_flows = pd.concat(flow_frames, ignore_index=True)


fig = plt.figure(figsize=(12, 4))
gs = fig.add_gridspec(1, 3)

ax_a = fig.add_subplot(gs[0, 0])
list_colors = auxiliary_functions.generate_plasma_colors(len(SELECTED_LAMBDAS))
for color, lamda, entropy in zip(list_colors, SELECTED_LAMBDAS, Conditional_S_selected_hr):
    ax_a.plot(
        net_times_hours[:1556],
        entropy[:1556],
        color=color,
        alpha=0.30,
        label=f"lamda = {lamda:.11f}",
    )
ax_a.set_xlabel("t")
ax_a.set_title("(A) Conditional entropy H(pt | p0)", loc="left", fontsize=12)

ax_b = fig.add_subplot(gs[0, 1])
nclusters_960_1320, nvi_960_1320 = interval_summaries["960_1320"]
ax_b.plot(LAMBDAS_GROWING, nvi_960_1320, color="tab:red", label="static norm NVI")
ax_b.set_xscale("log")
ax_b.set_xlabel(r"$\lambda$ [s]")
ax_b.set_ylabel("Avg. Norm. Var. Inf.", color="tab:red")
ax_b.tick_params(axis="y", labelcolor="tab:red")
ax_b.set_title("(B) Flow Stability - Sub-Interval", loc="left", fontsize=12)

ax_b_right = ax_b.twinx()
ax_b_right.plot(
    LAMBDAS_GROWING,
    nclusters_960_1320,
    color="tab:blue",
    label="edge-centric",
)
ax_b_right.set_xlabel(r"$\lambda$ [s]")
ax_b_right.set_ylabel("Avg. no. clusters", color="tab:blue")
ax_b_right.tick_params(axis="y", labelcolor="tab:blue")

nodes = build_sankey_nodes(flow_frames)
color_list = auxiliary_functions.generate_plasma_colors(11)
dict_color = {
    flow_type: color_list[i] for i, flow_type in enumerate(df_flows["type"].unique())
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

ax_c = fig.add_subplot(gs[0, 2])
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
    for flow_type in df_flows["type"].unique()
]
ax_c.legend(
    handles,
    df_flows["type"].unique(),
    title="Flow Types",
    loc="center left",
    bbox_to_anchor=(1, 0.5),
)
ax_c.set_title("(C) Community Evolution - Primary School", loc="left", fontsize=12)
ax_c.set_yticks([])
ax_c.set_frame_on(False)

plt.tight_layout()
# plt.savefig('/home/b/skoove/Desktop/ChangePointDetection/fig_entropy_inf_community.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()
print(f"Total runtime: {time.perf_counter() - START_TIME:.2f} s")