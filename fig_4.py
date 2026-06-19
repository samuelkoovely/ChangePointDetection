from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compute_global_entropy_motifs import clear_lambda_cache, compute_global_entropy_curve
from toy_path_graph import TemporalPathSpec, build_temporal_path_network


plt.style.use(Path(__file__).with_name("paper.mplstyle"))


DEFAULT_OUTPUT_PATH = Path("figures/fig_toy_path_final_global_entropy.pdf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the final global entropy at one diffusion rate for a family "
            "of deterministic temporal path graphs with a tunable merged "
            "prefix of links."
        )
    )
    parser.add_argument(
        "--lamda",
        type=float,
        default=10.0,
        help="Diffusion rate used to compute the global entropy curve.",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=10,
        help="Number of nodes in the temporal path graph.",
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
            "Optional common observation span. When provided, every temporal "
            "path example is rescaled to this total duration."
        ),
    )
    parser.add_argument(
        "--overlaps",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Explicit merged-prefix overlap depths. By default the script uses "
            "the full natural range 0, 1, ..., N-2."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output figure path.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV path for the plotted summary table.",
    )
    return parser.parse_args()


def resolve_overlaps(
    overlaps: list[int] | None,
    num_nodes: int,
) -> np.ndarray:
    if overlaps is not None:
        values = np.asarray(overlaps, dtype=int)
    else:
        values = np.arange(max(1, int(num_nodes) - 1), dtype=int)

    if np.any(values < 0):
        raise ValueError("All overlap values must be non-negative.")
    if np.any(values > int(num_nodes) - 2):
        raise ValueError(
            "All overlap values must be at most num_nodes - 2."
        )

    return np.unique(values.astype(int))


def compute_summary_table(
    overlaps: np.ndarray,
    *,
    lamda: float,
    num_nodes: int,
    time_unit: float,
    start_time: float,
    total_span: float | None,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []

    for overlap in overlaps:
        spec = TemporalPathSpec(
            num_nodes=int(num_nodes),
            overlap=int(overlap),
            time_unit=float(time_unit),
            start_time=float(start_time),
            total_span=None if total_span is None else float(total_span),
        )
        net = build_temporal_path_network(spec)
        result = compute_global_entropy_curve(net=net, lamda=float(lamda))
        signal = np.asarray(result["signal_array"], dtype=float)

        if len(signal) == 0:
            final_entropy = float("nan")
        else:
            final_entropy = float(signal[-1])

        upper_bound = float(np.log(net.num_nodes))
        rows.append(
            {
                "overlap": int(overlap),
                "shared_prefix_links": int(overlap) + 1,
                "final_global_entropy": final_entropy,
                "upper_bound_log_n": upper_bound,
                "gap_to_upper_bound": float(upper_bound - final_entropy),
                "num_nodes": int(net.num_nodes),
                "num_events": int(net.num_events),
                "lamda": float(lamda),
            }
        )
        clear_lambda_cache(net, float(lamda))

    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, output_path: Path) -> None:
    overlaps = summary["overlap"].to_numpy(dtype=int)
    entropies = summary["final_global_entropy"].to_numpy(dtype=float)
    upper_bounds = summary["upper_bound_log_n"].to_numpy(dtype=float)
    upper_bound = float(upper_bounds[0])

    fig, ax = plt.subplots(figsize=(7.2, 3.0))

    ax.plot(
        overlaps,
        entropies,
        color="tab:blue",
        marker="o",
        linewidth=2.0,
        markersize=4.5,
    )
    ax.axhline(
        upper_bound,
        color="black",
        linestyle="--",
        linewidth=1.6,
    )

    ax.set_xlim(float(overlaps.min()) - 0.15, float(overlaps.max()) + 0.15)
    ax.set_xticks(overlaps)
    ax.set_xlabel("Overlap Depth")
    ax.set_ylabel("Final Entropy Value")

    ymin = float(np.nanmin(entropies))
    ymax = float(np.nanmax(np.r_[entropies, upper_bound]))
    pad = max(0.02, 0.08 * (ymax - ymin))
    ax.set_ylim(ymin - pad, ymax + 0.5 * pad)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    overlaps = resolve_overlaps(args.overlaps, args.num_nodes)
    summary = compute_summary_table(
        overlaps=overlaps,
        lamda=float(args.lamda),
        num_nodes=int(args.num_nodes),
        time_unit=float(args.time_unit),
        start_time=float(args.start_time),
        total_span=None if args.total_span is None else float(args.total_span),
    )

    plot_summary(
        summary=summary,
        output_path=args.output,
    )

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output_csv, index=False)

    print(summary.to_string(index=False))
    print()
    print(f"saved {args.output}")
    if args.output_csv is not None:
        print(f"saved {args.output_csv}")


if __name__ == "__main__":
    main()
