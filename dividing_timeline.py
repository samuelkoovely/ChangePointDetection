import numpy as np
from TemporalNetwork import ContTempNetwork


starting_times = [1, 1.5, 2, 3, 4.5, 7]
ending_times = [1.25, 1.75, 2.25, 3.25, 4.75, 7.25]
source_nodes = [1, 2, 5, 6, 1, 5]
target_nodes = [2, 3, 7, 8, 4, 8]

net = ContTempNetwork(source_nodes=source_nodes, target_nodes=target_nodes, starting_times=starting_times, ending_times=ending_times)
net._compute_time_grid()

time_window = 3
starting_time = 0
ending_time = net.times[-1]


for interval_left in np.arange(starting_time, ending_time, time_window):
    interval_right = interval_left + time_window

    # Binary search in sorted net.times
    index_left = np.searchsorted(net.times, interval_left, side='right') - 1
    index_right = np.searchsorted(net.times, interval_right, side='right')

    # Clamp index_left to -1 if it's before the array starts
    if index_left < 0:
        index_left = None  # Special case: nothing before interval_left
    # Clamp index_right if it goes past the array
    if index_right >= len(net.times):
        index_right = None  # Special case: nothing after interval_right

    print('left:' + str(interval_left) + 'right:' + str(interval_right))
    smallest_t = net.times[index_left] if index_left is not None else interval_left
    print('smallest_t =' + str(smallest_t))
    biggest_t = net.times[index_right] if index_right is not None else min(interval_right, net.times[-1])
    print('biggest_t =' + str(biggest_t))
    # # Slice the elements within the window
    # in_window_indices = np.arange(index_left + 1 if index_left is not None else 0,
    #                               index_right if index_right is not None else len(net.times))

    # # Optionally include the neighbors if they exist
    # neighbors = []
    # if index_left is not None:
    #     neighbors.append(index_left)
    # if index_right is not None:
    #     neighbors.append(index_right)

    # all_indices = np.concatenate([neighbors[:1], in_window_indices, neighbors[1:]]) if neighbors else in_window_indices

    # # Do something with all_indices
    # print(f"Window [{interval_left}, {interval_right}): indices {all_indices}")
