import pickle
import numpy as np
from TemporalNetwork import ContTempNetwork


with open('block2activities.pkl', 'rb') as handle:
    dataset = pickle.load(handle)


aggregation_window = 10 
sanpshots_dataset = []

for data_index in range(len(dataset)):
    tnet = dataset[data_index]
    net = tnet['tnet']
    t_split = tnet['bkp']

    #snapshots = []

    source_nodes = []
    target_nodes = []
    starting_times = []
    ending_times = []

    for i in range(0, int(net.times[-1] - aggregation_window), aggregation_window): # - aggregation_window to avoid tail
        matrix_snapshot = net.compute_static_adjacency_matrix(start_time=i, end_time=i+aggregation_window).toarray()
        #snapshots.append(matrix_snapshot)

        matrix_snapshot = (matrix_snapshot > 0).astype(int)
        source_nodes_snapshot = np.nonzero(matrix_snapshot)[0]
        target_nodes_snapshot = np.nonzero(matrix_snapshot)[1]
        starting_times_snapshot = [i] * len(source_nodes_snapshot)
        ending_times_snapshot = [i+aggregation_window] * len(source_nodes_snapshot)

        source_nodes += list(source_nodes_snapshot)
        target_nodes += list(target_nodes_snapshot)
        starting_times += starting_times_snapshot
        ending_times += ending_times_snapshot

    snap_net = ContTempNetwork(source_nodes=source_nodes,
                        target_nodes=target_nodes,
                        starting_times=starting_times,
                        ending_times=ending_times,
                        merge_overlapping_events=True)

    tnet = {}
    tnet['tnet'] = snap_net
    #tnet['snapshots'] = snapshots
    tnet['aggregation_window'] = aggregation_window
    tnet['bkp'] = t_split // aggregation_window
    sanpshots_dataset.append(tnet)


pickle.dump(sanpshots_dataset, open('block2activities_snapshots.pkl', 'wb'))