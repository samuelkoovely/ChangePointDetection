from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from TemporalNetwork import ContTempNetwork


DATASET_PATHS = (
    (
        Path("data/multibkps_block2activities.pkl"),
        Path("data/multibkps_block2activities_snapshots.pkl"),
    ),
    (
        Path("data/multibkps_block2activities_test.pkl"),
        Path("data/multibkps_block2activities_snapshots_test.pkl"),
    ),
)
AGGREGATION_WINDOW = 4
SNAPSHOTS_TO_SKIP_AT_START = 0


def aggregate_breakpoints(
    breakpoints: list[float],
    aggregation_window: int,
    num_snapshots: int,
) -> list[int]:
    # This benchmark alternates between two activity regimes. If multiple
    # regime flips land in the same snapshot bin, only the parity matters:
    # an even number of flips cancels out, while an odd number leaves one
    # effective change inside that snapshot.
    breakpoint_counts_by_snapshot: dict[int, int] = {}
    for breakpoint in breakpoints:
        snapshot_index = (
            int(float(breakpoint) // aggregation_window) - SNAPSHOTS_TO_SKIP_AT_START
        )
        breakpoint_counts_by_snapshot[snapshot_index] = (
            breakpoint_counts_by_snapshot.get(snapshot_index, 0) + 1
        )

    aggregated_breakpoints = sorted(
        snapshot_index
        for snapshot_index, count in breakpoint_counts_by_snapshot.items()
        if count % 2 == 1
    )
    return [
        breakpoint
        for breakpoint in aggregated_breakpoints
        if 0 <= breakpoint < num_snapshots
    ]


def build_snapshot_network(
    net: ContTempNetwork,
    aggregation_window: int,
) -> tuple[ContTempNetwork, int]:
    source_nodes = []
    target_nodes = []
    starting_times = []
    ending_times = []

    snapshot_starts = list(
        range(
            SNAPSHOTS_TO_SKIP_AT_START * aggregation_window,
            int(net.times[-1] - aggregation_window),
            aggregation_window,
        )
    )

    for snapshot_start in snapshot_starts:
        snapshot_end = snapshot_start + aggregation_window
        matrix_snapshot = net.compute_static_adjacency_matrix(
            start_time=snapshot_start,
            end_time=snapshot_end,
        ).toarray()

        matrix_snapshot = (matrix_snapshot > 0).astype(int)
        source_nodes_snapshot = np.nonzero(matrix_snapshot)[0]
        target_nodes_snapshot = np.nonzero(matrix_snapshot)[1]
        starting_times_snapshot = [snapshot_start] * len(source_nodes_snapshot)
        ending_times_snapshot = [snapshot_end] * len(source_nodes_snapshot)

        source_nodes += list(source_nodes_snapshot)
        target_nodes += list(target_nodes_snapshot)
        starting_times += starting_times_snapshot
        ending_times += ending_times_snapshot

    snap_net = ContTempNetwork(
        source_nodes=source_nodes,
        target_nodes=target_nodes,
        starting_times=starting_times,
        ending_times=ending_times,
        merge_overlapping_events=True,
    )

    return snap_net, len(snapshot_starts)


def convert_dataset(dataset: list[dict[str, object]]) -> list[dict[str, object]]:
    snapshots_dataset = []

    for entry in dataset:
        net = entry["tnet"]
        breakpoints = [float(breakpoint) for breakpoint in entry["bkps"]]

        snap_net, num_snapshots = build_snapshot_network(
            net=net,
            aggregation_window=AGGREGATION_WINDOW,
        )
        aggregated_breakpoints = aggregate_breakpoints(
            breakpoints=breakpoints,
            aggregation_window=AGGREGATION_WINDOW,
            num_snapshots=num_snapshots,
        )

        snapshots_dataset.append(
            {
                "tnet": snap_net,
                "aggregation_window": AGGREGATION_WINDOW,
                "bkps": aggregated_breakpoints,
                "n_bkps": len(aggregated_breakpoints),
            }
        )

    return snapshots_dataset


def main() -> None:
    for input_path, output_path in DATASET_PATHS:
        with open(input_path, "rb") as handle:
            dataset = pickle.load(handle)

        snapshots_dataset = convert_dataset(dataset)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as handle:
            pickle.dump(snapshots_dataset, handle)


if __name__ == "__main__":
    main()
