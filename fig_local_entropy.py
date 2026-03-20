import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
import auxiliary_functions
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


with open('data/merge_merge.pkl', 'rb') as handle:
    merge_merge = pickle.load(handle)
with open('data/merge_split.pkl', 'rb') as handle:
    merge_split = pickle.load(handle)
with open('data/split_merge.pkl', 'rb') as handle:
    split_merge = pickle.load(handle)


def compute_interval_matrices(network, intervals):
    return [
        network.compute_static_adjacency_matrix(start_time=start, end_time=end).toarray()
        for start, end in intervals
    ]


def load_entropy_curves(base_dir, lamdas, subdir="window_S/5"):
    curves = []
    for lamda in lamdas:
        lamda_str = f"{lamda:.11f}"
        filepath = f"//scratch/tmp/180/skoove/{base_dir}/{subdir}/window_S{lamda_str}"
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        curves.append([data['t_samples'], data['window_S']])
    return curves


def make_inset_cmap():
    cmap = cm.get_cmap('inferno').copy()
    cmap.set_bad(color='white')
    return cmap


def plot_network_panel(ax, network, forward_curves, matrices, panel_title,
                       curve_indices, forward_colors, time_intervals, inset_positions, inset_cmap):
    for color_idx, curve_idx in enumerate(curve_indices):
        ax.plot(forward_curves[curve_idx][0], forward_curves[curve_idx][1], color=forward_colors[color_idx], alpha=1)


    ax.set_xlim(-5, 310)
    ax.set_xlabel("t [s]")
    ax.set_title(panel_title, loc='left', fontsize=14)

    for matrix, pos, (start, end) in zip(matrices, inset_positions, time_intervals):
        inset_ax = inset_axes(
            ax,
            width="20%",
            height="20%",
            loc="lower left",
            bbox_to_anchor=(pos, 0.05, 1, 1),
            bbox_transform=ax.transAxes,
        )
        masked_matrix = np.ma.masked_where(matrix == 0, matrix)
        positive_entries = matrix[matrix > 0]
        vmax = positive_entries.max() if positive_entries.size else 1.0

        inset_ax.matshow(
            masked_matrix,
            cmap=inset_cmap,
            aspect='equal',
            vmin=0,
            vmax=vmax,
            interpolation='nearest',
        )
        inset_ax.set_facecolor('white')
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        inset_ax.set_title(f"{start} ≤ t < {end}", fontsize=8)
        for spine in inset_ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_edgecolor('black')


networks = {
    'merge_merge': merge_merge,
    'merge_split': merge_split,
    'split_merge': split_merge,
}

panel_specs = [
    {
        'key': 'merge_merge',
        'title': '(A)',
        'backward_index': 7,
    },
    {
        'key': 'merge_split',
        'title': '(B)',
        'backward_index': 5,
    },
    {
        'key': 'split_merge',
        'title': '(C)',
        'backward_index': 6,
    },
]

curve_indices = [5, 6, 7, 8]
inset_positions = [0.06, 0.37, 0.68]

time_intervals = [(0, 100), (100, 200), (200, 300)]

lamdas = np.logspace(-4,-1,10)

interval_matrices = {
    key: compute_interval_matrices(network, time_intervals)
    for key, network in networks.items()
}

forward_entropy = {
    key: load_entropy_curves(key, lamdas, subdir='window_S/5')
    for key in networks
}


fig = plt.figure(figsize=(12, 4))
gs = fig.add_gridspec(1, 3)
axes = gs.subplots(sharey=True)

color = auxiliary_functions.generate_plasma_colors(len(curve_indices))
inset_cmap = make_inset_cmap()

for ax, spec in zip(np.atleast_1d(axes), panel_specs):
    key = spec['key']
    plot_network_panel(
        ax=ax,
        network=networks[key],
        forward_curves=forward_entropy[key],
        matrices=interval_matrices[key],
        panel_title=spec['title'],
        curve_indices=curve_indices,
        forward_colors=color,
        time_intervals=time_intervals,
        inset_positions=inset_positions,
        inset_cmap=inset_cmap,
    )

axes = np.atleast_1d(axes)
axes[0].set_ylabel("Entropy")
for ax in axes[1:]:
    ax.tick_params(labelleft=False)

# Adjust layout and display
plt.tight_layout()
plt.savefig('/home/b/skoove/Desktop/ChangePointDetection/fig_local_entropy.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()