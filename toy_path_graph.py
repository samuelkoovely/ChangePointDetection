from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from TemporalNetwork import ContTempNetwork


@dataclass(frozen=True)
class TemporalPathSpec:
    """
    Specification for a path graph with a tunable merged prefix of links.

    Let `num_edges = num_nodes - 1` and label the links
        (0,1), (1,2), ..., (num_edges - 1, num_edges).

    `overlap` is an integer depth:
    - `overlap = 0` gives the fully asymmetric case:
      [0,1], [1,2], [2,3], ...
    - `overlap = 1` makes the first two links share [0,2]
    - `overlap = 2` makes the first three links share [0,3]
    - ...
    - `overlap = num_edges - 1` makes all links share [0, num_edges]

    The optional `time_unit` rescales these natural integer supports.
    """

    num_nodes: int = 4
    overlap: int = 0
    time_unit: float = 1.0
    start_time: float = 0.0
    total_span: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic temporal path-graph examples with a "
            "tunable merged prefix of links."
        )
    )
    parser.add_argument(
        "--overlaps",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Merged-prefix overlap depths. For a path with N nodes, the "
            "natural range is 0 to N-2."
        ),
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=4,
        help="Number of nodes in the path graph.",
    )
    parser.add_argument(
        "--time-unit",
        type=float,
        default=1.0,
        help="Length of one natural time slot before optional rescaling.",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=0.0,
        help="Starting time of the first link.",
    )
    parser.add_argument(
        "--total-span",
        type=float,
        default=None,
        help=(
            "Optional final observation span. When provided, all timestamps are "
            "linearly rescaled to [start_time, start_time + total_span]."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory where each network is saved as a pickle. A CSV "
            "copy of the event table is written alongside it."
        ),
    )
    parser.add_argument(
        "--print-table",
        action="store_true",
        help="Print the generated event table(s) to stdout.",
    )
    return parser.parse_args()


def validate_spec(spec: TemporalPathSpec) -> None:
    if spec.num_nodes < 2:
        raise ValueError("num_nodes must be at least 2.")
    if int(spec.overlap) < 0:
        raise ValueError("overlap must be non-negative.")
    if int(spec.overlap) > int(spec.num_nodes) - 2:
        raise ValueError(
            "overlap must be at most num_nodes - 2 so that the merged prefix "
            "does not exceed the number of links."
        )
    if float(spec.time_unit) <= 0.0:
        raise ValueError("time_unit must be strictly positive.")
    if spec.total_span is not None and float(spec.total_span) <= 0.0:
        raise ValueError("total_span must be strictly positive when provided.")


def build_temporal_path_time_table(spec: TemporalPathSpec) -> pd.DataFrame:
    """
    Build the event table for a path graph 0-1-2-...-(n-1).

    In natural units, link `j` occupies [j, j+1] in the disjoint case.
    For a chosen overlap depth `k`, all links with `j <= k` are reassigned
    to the common support [0, k+1], while links with `j > k` keep their
    original slots [j, j+1].
    """

    validate_spec(spec)

    num_edges = int(spec.num_nodes) - 1
    source_nodes = np.arange(num_edges, dtype=int)
    target_nodes = source_nodes + 1
    starting_times = np.arange(num_edges, dtype=float)
    ending_times = starting_times + 1.0

    overlap = int(spec.overlap)
    merged_mask = source_nodes <= overlap
    starting_times[merged_mask] = 0.0
    ending_times[merged_mask] = float(overlap + 1)

    starting_times = float(spec.start_time) + float(spec.time_unit) * starting_times
    ending_times = float(spec.start_time) + float(spec.time_unit) * ending_times

    if spec.total_span is not None:
        raw_span = float(ending_times[-1] - spec.start_time)
        scale = float(spec.total_span) / raw_span
        starting_times = float(spec.start_time) + scale * (
            starting_times - float(spec.start_time)
        )
        ending_times = float(spec.start_time) + scale * (
            ending_times - float(spec.start_time)
        )

    table = pd.DataFrame(
        {
            "source_nodes": source_nodes,
            "target_nodes": target_nodes,
            "starting_times": starting_times,
            "ending_times": ending_times,
        }
    )
    return table.sort_values(
        by=["starting_times", "ending_times", "source_nodes", "target_nodes"]
    ).reset_index(drop=True)


def build_temporal_path_network(spec: TemporalPathSpec) -> ContTempNetwork:
    """
    Instantiate a `ContTempNetwork` from the deterministic path event table.
    """

    table = build_temporal_path_time_table(spec)

    net = ContTempNetwork(
        source_nodes=table["source_nodes"].tolist(),
        target_nodes=table["target_nodes"].tolist(),
        starting_times=table["starting_times"].tolist(),
        ending_times=table["ending_times"].tolist(),
        relabel_nodes=False,
        node_to_label_dict={idx: idx for idx in range(int(spec.num_nodes))},
    )
    net._compute_time_grid()
    return net


def format_overlap_tag(overlap: int) -> str:
    return f"{int(overlap):02d}"


def save_outputs(
    spec: TemporalPathSpec,
    table: pd.DataFrame,
    net: ContTempNetwork,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = f"path_n{int(spec.num_nodes)}_overlap_{format_overlap_tag(spec.overlap)}"
    csv_path = output_dir / f"{stem}.csv"
    pkl_path = output_dir / f"{stem}.pkl"

    table.to_csv(csv_path, index=False)
    with pkl_path.open("wb") as handle:
        pickle.dump(net, handle)

    return csv_path, pkl_path


def main() -> None:
    args = parse_args()

    overlaps = (
        list(args.overlaps)
        if args.overlaps is not None
        else list(range(max(1, int(args.num_nodes) - 1)))
    )

    for overlap in overlaps:
        spec = TemporalPathSpec(
            num_nodes=int(args.num_nodes),
            overlap=int(overlap),
            time_unit=float(args.time_unit),
            start_time=float(args.start_time),
            total_span=(
                None if args.total_span is None else float(args.total_span)
            ),
        )

        table = build_temporal_path_time_table(spec)
        net = build_temporal_path_network(spec)

        if args.print_table or args.output_dir is None:
            print(f"overlap={int(spec.overlap)}")
            print(table.to_string(index=False))
            print()

        if args.output_dir is not None:
            csv_path, pkl_path = save_outputs(
                spec=spec,
                table=table,
                net=net,
                output_dir=args.output_dir,
            )
            print(f"saved {csv_path}")
            print(f"saved {pkl_path}")


if __name__ == "__main__":
    main()
