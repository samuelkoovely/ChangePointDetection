import numpy as np
from TemporalNetwork import ContTempNetwork
import pickle
import matplotlib.pyplot as plt
import auxiliary_functions


net_rw = ContTempNetwork.load('./data/primaryschoolnet',
                              attributes_list=['node_to_label_dict',
                      'events_table',
                      'times',
                      'time_grid',
                      'num_nodes',
                      '_overlapping_events_merged',
                      'start_date',
                      'node_label_array',
                      'male_array',
                      'female_array',
                      'node_first_start_array',
                      'node_last_end_array',
                      'node_class_array',
                      'datetimes'])


net_times_hours = net_rw.times / 3600

windows = [6, 90, 180]
selected_lamdas = np.logspace(-5,0,10)
selected_lamdas = selected_lamdas[range(0, len(selected_lamdas), 2)]

#Plot 5
list_colors = auxiliary_functions.generate_plasma_colors(6)
indices_plot = [0, 1, 2]


fig, ax = plt.subplots(1, 3, figsize = (10, 4))
for j, window in enumerate(windows):
    window_S_selected = []
    for i, lamda in enumerate(selected_lamdas):
        with open(f'//scratch/tmp/180/skoove/primaryschoolnet_rw/window_S_selected/{window}/window_S{lamda:.11f}', 'rb') as f:
            S_rate = pickle.load(f)
            window_S_selected.append(S_rate['window_S'][f'{lamda:.11f}'])

    if j == 0:
        for i, lamda in enumerate(selected_lamdas):
            S = window_S_selected[i]
            ax[indices_plot[j]].plot(net_times_hours[(window // 2)+1 :1556- (window // 2)], S[1:1556-window], color = list_colors[i], alpha = 0.75, label='lamda = ' + f'{lamda:.5f}')
            ax[indices_plot[j]].set_xlabel('Time')
    else:
        for i, lamda in enumerate(selected_lamdas):
            S = window_S_selected[i]
            ax[indices_plot[j]].plot(net_times_hours[(window // 2)+1 :1556- (window // 2)], S[1:1556-window], color = list_colors[i], alpha = 0.75)
            ax[indices_plot[j]].set_xlabel('Time')

    ax[indices_plot[j]].set_title(f' {window // 3} mins window')
    

lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
#fig.legend(lines, labels, fontsize='xx-small', bbox_to_anchor=(0.85,0), ncol=5)

plt.tight_layout()  # otherwise the right y-label is slightly clipped
#plt.savefig('/home/b/skoove/Desktop/ChangePointDetection/fig_primaryschool_window.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()