import numpy as np

import pickle

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


import auxiliary_functions

from mpl_toolkits.axes_grid1.inset_locator import inset_axes


with open('merge_merge.pkl', 'rb') as handle:
    merge_merge = pickle.load(handle)
with open('merge_split.pkl', 'rb') as handle:
    merge_split = pickle.load(handle)
with open('split_merge.pkl', 'rb') as handle:
    split_merge = pickle.load(handle)


# Example matrices for demonstration
merge_merge1 = merge_merge.compute_static_adjacency_matrix(start_time=0, end_time=100).toarray()
merge_merge2 = merge_merge.compute_static_adjacency_matrix(start_time=100, end_time=200).toarray()
merge_merge3 = merge_merge.compute_static_adjacency_matrix(start_time=200, end_time=300).toarray()

merge_split1 = merge_split.compute_static_adjacency_matrix(start_time=0, end_time=100).toarray()
merge_split2 = merge_split.compute_static_adjacency_matrix(start_time=100, end_time=200).toarray()
merge_split3 = merge_split.compute_static_adjacency_matrix(start_time=200, end_time=300).toarray()

split_merge1 = split_merge.compute_static_adjacency_matrix(start_time=0, end_time=100).toarray()
split_merge2 = split_merge.compute_static_adjacency_matrix(start_time=100, end_time=200).toarray()
split_merge3 = split_merge.compute_static_adjacency_matrix(start_time=200, end_time=300).toarray()

# Define the time intervals for the x-axis
time_intervals = [(0, 100), (100, 200), (200, 300)]

lamdas = np.logspace(-4,-1,10)

Conditional_S_merge_merge = []
for i, lamda in enumerate(lamdas):
    with open(f'//scratch/tmp/180/skoove/merge_merge/S/S{lamda:.11f}', 'rb') as f:
        S = pickle.load(f)
        Conditional_S_merge_merge.append(S['S'][f'{lamda:.11f}'])

Conditional_S_merge_split = []
for i, lamda in enumerate(lamdas):
    with open(f'//scratch/tmp/180/skoove/merge_split/S/S{lamda:.11f}', 'rb') as f:
        S = pickle.load(f)
        Conditional_S_merge_split.append(S['S'][f'{lamda:.11f}'])

Conditional_S_split_merge = []
for i, lamda in enumerate(lamdas):
    with open(f'//scratch/tmp/180/skoove/split_merge/S/S{lamda:.11f}', 'rb') as f:
        S = pickle.load(f)
        Conditional_S_split_merge.append(S['S'][f'{lamda:.11f}'])

Conditional_S_merge_merge_rev = []
for i, lamda in enumerate(lamdas):
    with open(f'//scratch/tmp/180/skoove/merge_merge/S_rev/S{lamda:.11f}', 'rb') as f:
        S = pickle.load(f)
        Conditional_S_merge_merge_rev.append(S['S'][f'{lamda:.11f}'])

Conditional_S_merge_split_rev = []
for i, lamda in enumerate(lamdas):
    with open(f'//scratch/tmp/180/skoove/merge_split/S_rev/S{lamda:.11f}', 'rb') as f:
        S = pickle.load(f)
        Conditional_S_merge_split_rev.append(S['S'][f'{lamda:.11f}'])

Conditional_S_split_merge_rev = []
for i, lamda in enumerate(lamdas):
    with open(f'//scratch/tmp/180/skoove/split_merge/S_rev/S{lamda:.11f}', 'rb') as f:
        S = pickle.load(f)
        Conditional_S_split_merge_rev.append(S['S'][f'{lamda:.11f}'])


fig = plt.figure(figsize=(12, 4))
gs = fig.add_gridspec(1, 3)

color=auxiliary_functions.generate_plasma_colors(4)

# Column 1: Plot A
ax_a = fig.add_subplot(gs[0, 0])
for i in range(5,9):
    S = Conditional_S_merge_merge[i]
    ax_a.plot(merge_merge.times, S, color=color[i-5], alpha=1)


ax_a.plot(merge_merge.times[-1::-1], Conditional_S_merge_merge_rev[7], '--', color = 'black', alpha=1)

ax_a.set_xlim(-5,310)
ax_a.set_xlabel("t [s]")
ax_a.set_ylabel("Entropy")
ax_a.set_title("(A)", loc='left', fontsize=14)

# Add inset plots for the matrices inside Plot A
matrices = [merge_merge1, merge_merge2, merge_merge3]
positions = [0.06, 0.37, 0.68]  # Horizontal positions for the insets
for i, (matrix, pos) in enumerate(zip(matrices, positions)):
    inset_ax = inset_axes(ax_a, width="20%", height="20%", loc="lower left", 
                          bbox_to_anchor=(pos, 0.05, 1, 1), bbox_transform=ax_a.transAxes)
    inset_ax.matshow(matrix, cmap='plasma', aspect='equal')  # Keep aspect ratio equal to preserve square shape
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    inset_ax.set_title(f"{time_intervals[i][0]} ≤ t < {time_intervals[i][1]}", fontsize=8)

# Column 2: Plot B
ax_b = fig.add_subplot(gs[0, 1])

for i in range(5,9):
    S = Conditional_S_merge_split[i]
    ax_b.plot(merge_split.times, S, color=color[i-5], alpha=1)

ax_b.plot(merge_split.times[-1::-1], Conditional_S_merge_split_rev[5], '--', color = 'black', alpha=1)

ax_b.set_xlim(-5,310)
ax_b.set_xlabel("t [s]")
ax_b.set_ylabel("Entropy")
ax_b.set_title("(B)", loc='left', fontsize=14)

# Add inset plots for the matrices inside Plot A
matrices = [merge_split1, merge_split2, merge_split3]
positions = [0.06, 0.37, 0.68]  # Horizontal positions for the insets
for i, (matrix, pos) in enumerate(zip(matrices, positions)):
    inset_ax = inset_axes(ax_b, width="20%", height="20%", loc="lower left", 
                          bbox_to_anchor=(pos, 0.05, 1, 1), bbox_transform=ax_b.transAxes)
    inset_ax.matshow(matrix, cmap='plasma', aspect='equal')  # Keep aspect ratio equal to preserve square shape
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    inset_ax.set_title(f"{time_intervals[i][0]} ≤ t < {time_intervals[i][1]}", fontsize=8)


# Column 3: Plot C
ax_c = fig.add_subplot(gs[0, 2])

for i in range(5,9):
    S = Conditional_S_split_merge[i]
    ax_c.plot(split_merge.times, S, color=color[i-5], alpha=1)

ax_c.plot(split_merge.times[-1::-1], Conditional_S_split_merge_rev[6], '--', color = 'black', alpha=1)

ax_c.set_xlim(-5,310)
ax_c.set_xlabel("t [s]")
ax_c.set_ylabel("Entropy")
ax_c.set_title("(C)", loc='left', fontsize=14)

# Add inset plots for the matrices inside Plot A
matrices = [split_merge1, split_merge2, split_merge3]
positions = [0.06, 0.37, 0.68]  # Horizontal positions for the insets
for i, (matrix, pos) in enumerate(zip(matrices, positions)):
    inset_ax = inset_axes(ax_c, width="20%", height="20%", loc="lower left", 
                          bbox_to_anchor=(pos, 0.05, 1, 1), bbox_transform=ax_c.transAxes)
    inset_ax.matshow(matrix, cmap='plasma', aspect='equal')  # Keep aspect ratio equal to preserve square shape
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    inset_ax.set_title(f"{time_intervals[i][0]} ≤ t < {time_intervals[i][1]}", fontsize=8)


# Adjust layout and display
plt.tight_layout()
#plt.savefig('/home/b/skoove/Desktop/growing300/GCE_filter_fig.png', format='png', dpi=300, bbox_inches='tight')
plt.show()