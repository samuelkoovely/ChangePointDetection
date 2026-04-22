from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import numpy as np

from TemporalNetwork import ContTempNetwork


AGGREGATION_WINDOW = 4
SNAPSHOTS_TO_SKIP_AT_START = 0


def aggregate_breakpoints(
    breakpoints: list[float],
    aggregation_window: int,
    num_snapshots: int,
    snapshots_to_skip_at_start: int = SNAPSHOTS_TO_SKIP_AT_START,
) -> list[int]:
    aggregated_breakpoints = sorted(
        {
            int(float(breakpoint) // aggregation_window) - snapshots_to_skip_at_start
            for breakpoint in breakpoints
        }
    )
    return [
        breakpoint
        for breakpoint in aggregated_breakpoints
        if 0 <= breakpoint < num_snapshots
    ]


def build_snapshot_network(
    net: ContTempNetwork,
    aggregation_window: int,
    snapshots_to_skip_at_start: int = SNAPSHOTS_TO_SKIP_AT_START,
) -> tuple[ContTempNetwork, int]:
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


def convert_dataset(
    dataset: list[dict[str, object]],
    aggregation_window: int = AGGREGATION_WINDOW,
    snapshots_to_skip_at_start: int = SNAPSHOTS_TO_SKIP_AT_START,
) -> list[dict[str, object]]:
    snapshots_dataset = []

    for entry in dataset:
        net = entry["tnet"]
        breakpoints = [float(breakpoint) for breakpoint in entry["bkps"]]

        snap_net, num_snapshots = build_snapshot_network(
            net=net,
            aggregation_window=aggregation_window,
            snapshots_to_skip_at_start=snapshots_to_skip_at_start,
        )
        aggregated_breakpoints = aggregate_breakpoints(
            breakpoints=breakpoints,
            aggregation_window=aggregation_window,
            num_snapshots=num_snapshots,
            snapshots_to_skip_at_start=snapshots_to_skip_at_start,
        )

        snapshots_dataset.append(
            {
                "tnet": snap_net,
                "aggregation_window": aggregation_window,
                "bkps": aggregated_breakpoints,
                "n_bkps": len(aggregated_breakpoints),
            }
        )

    return snapshots_dataset


def write_dataset(dataset: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as handle:
        pickle.dump(dataset, handle)


def convert_dataset_file(
    input_path: Path,
    output_paths: Sequence[Path],
    aggregation_window: int = AGGREGATION_WINDOW,
    snapshots_to_skip_at_start: int = SNAPSHOTS_TO_SKIP_AT_START,
) -> None:
    with open(input_path, "rb") as handle:
        dataset = pickle.load(handle)

    snapshots_dataset = convert_dataset(
        dataset=dataset,
        aggregation_window=aggregation_window,
        snapshots_to_skip_at_start=snapshots_to_skip_at_start,
    )

    for output_path in output_paths:
        write_dataset(snapshots_dataset, output_path)


def convert_dataset_files(
    dataset_specs: Sequence[tuple[Path, Sequence[Path]]],
    aggregation_window: int = AGGREGATION_WINDOW,
    snapshots_to_skip_at_start: int = SNAPSHOTS_TO_SKIP_AT_START,
) -> None:
    for input_path, output_paths in dataset_specs:
        convert_dataset_file(
            input_path=input_path,
            output_paths=output_paths,
            aggregation_window=aggregation_window,
            snapshots_to_skip_at_start=snapshots_to_skip_at_start,
        )
