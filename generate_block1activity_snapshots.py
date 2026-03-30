import pickle
import numpy as np
from TemporalNetwork import ContTempNetwork


with open('data/block1activity.pkl', 'rb') as handle:
    dataset = pickle.load(handle)


aggregation_window = 4
snapshots_to_skip_at_start = 1
sanpshots_dataset = []

for data_index in range(len(dataset)):
    tnet = dataset[data_index]
    net = tnet['tnet']
    breakpoints = [float(breakpoint) for breakpoint in tnet['bkps']]

    #snapshots = []

    source_nodes = []
    target_nodes = []
    starting_times = []
    ending_times = []

    snapshot_starts = list(
        range(
            snapshots_to_skip_at_start * aggregation_window,
            int(net.times[-1] - aggregation_window),
            aggregation_window,
        )
    )

    for i in snapshot_starts:  # - aggregation_window to avoid tail
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
    num_snapshots = len(snapshot_starts)
    aggregated_breakpoints = sorted({
        int(breakpoint // aggregation_window) - snapshots_to_skip_at_start
        for breakpoint in breakpoints
        if 0 <= int(breakpoint // aggregation_window) - snapshots_to_skip_at_start < num_snapshots
    })
    tnet['bkps'] = aggregated_breakpoints
    tnet['n_bkps'] = len(aggregated_breakpoints)
    sanpshots_dataset.append(tnet)


pickle.dump(sanpshots_dataset, open('data/block1activity_snapshots.pkl', 'wb'))
