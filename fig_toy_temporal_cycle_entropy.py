from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import auxiliary_functions
from TemporalNetwork import ContTempNetwork
from compute_global_entropy_motifs import clear_lambda_cache, compute_global_entropy_curve


plt.style.use(Path(__file__).with_name("paper.mplstyle"))


DEFAULT_LAMBDAS = (1.0, 10.0)
DEFAULT_OUTPUT_PATH = Path("figures/fig_toy_temporal_cycle_entropy.pdf")
DEFAULT_PNG_OUTPUT_PATH = Path("figures/fig_toy_temporal_cycle_entropy.png")
SNAPSHOT_EDGES = ((1, 2), (2, 3), (3, 1))
SNAPSHOT_INTERVAL_LABELS = (
    r"$0 \leq t < 1$",
    r"$1 \leq t < 2$",
    r"$2 \leq t < 3$",
)
NODE_POSITIONS = {
    1: (-0.8, -0.55),
    2: (0.8, -0.55),
    3: (0.0, 0.85),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a toy global-entropy figure for a three-node temporal cycle "
            "with snapshot links (1,2), then (2,3), then (3,1)."
        )
    )
    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=list(DEFAULT_LAMBDAS),
        help="Diffusion rates to plot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output PDF path.",
    )
    parser.add_argument(
        "--png-output",
        type=Path,
        default=None,
        help="Optional PNG export path.",
    )
    return parser.parse_args()


def build_temporal_cycle_network() -> ContTempNetwork:
    net = ContTempNetwork(
        source_nodes=[1, 2, 3],
        target_nodes=[2, 3, 1],
        starting_times=[0.0, 1.0, 2.0],
        ending_times=[1.0, 2.0, 3.0],
        relabel_nodes=True,
    )
    net._compute_time_grid()
    return net


def compute_entropy_curves(
    net: ContTempNetwork,
    lambdas: list[float],
) -> list[dict[str, np.ndarray | float]]:
    curves: list[dict[str, np.ndarray | float]] = []
    initial_time = float(net.times[0])

    for lamda in lambdas:
        result = compute_global_entropy_curve(net=net, lamda=float(lamda))
        entropy = np.asarray(result["signal_array"], dtype=float)
        end_times = np.asarray(net.times[1 : len(entropy) + 1], dtype=float)
        curves.append(
            {
                "lamda": float(lamda),
                "times": np.concatenate(([initial_time], end_times)),
                "entropy": np.concatenate(([0.0], entropy)),
            }
        )
        clear_lambda_cache(net, float(lamda))

    return curves


def draw_snapshot_inset(
    ax: plt.Axes,
    active_edge: tuple[int, int],
    interval_label: str,
    anchor_x: float,
) -> None:
    inset_ax = ax.inset_axes([anchor_x, 0.68, 0.22, 0.26])

    graph = nx.Graph()
    graph.add_nodes_from(NODE_POSITIONS)

    nx.draw_networkx_edges(
        graph,
        pos=NODE_POSITIONS,
        edgelist=list(SNAPSHOT_EDGES),
        ax=inset_ax,
        edge_color="#d6d6d6",
        width=1.2,
        style="dashed",
        alpha=0.9,
    )
    nx.draw_networkx_edges(
        graph,
        pos=NODE_POSITIONS,
        edgelist=[active_edge],
        ax=inset_ax,
        edge_color="#d94841",
        width=3.2,
    )
    nx.draw_networkx_nodes(
        graph,
        pos=NODE_POSITIONS,
        ax=inset_ax,
        node_color="white",
        edgecolors="black",
        linewidths=1.2,
        node_size=420,
    )
    nx.draw_networkx_labels(
        graph,
        pos=NODE_POSITIONS,
        ax=inset_ax,
        font_size=10,
        font_color="black",
    )

    inset_ax.set_title(interval_label, fontsize=9, pad=2.0)
    inset_ax.set_xlim(-1.05, 1.05)
    inset_ax.set_ylim(-0.9, 1.0)
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    inset_ax.set_facecolor("white")
    for spine in inset_ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_edgecolor("black")


def plot_toy_temporal_cycle_entropy(
    curves: list[dict[str, np.ndarray | float]],
    output_path: Path,
    png_output_path: Path | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    colors = auxiliary_functions.generate_plasma_colors(len(curves))
    markers = ("o", "s", "^", "D")

    for idx, curve in enumerate(curves):
        lamda = float(curve["lamda"])
        times = np.asarray(curve["times"], dtype=float)
        entropy = np.asarray(curve["entropy"], dtype=float)
        ax.plot(
            times,
            entropy,
            color=colors[idx],
            marker=markers[idx % len(markers)],
            linewidth=2.2,
            markersize=5.5,
            label=rf"$\lambda = {lamda:g}$",
            zorder=3 + idx,
        )

    for anchor_x, active_edge, interval_label in zip(
        (0.05, 0.39, 0.73),
        SNAPSHOT_EDGES,
        SNAPSHOT_INTERVAL_LABELS,
    ):
        draw_snapshot_inset(
            ax=ax,
            active_edge=active_edge,
            interval_label=interval_label,
            anchor_x=anchor_x,
        )

    max_entropy = max(
        float(np.max(np.asarray(curve["entropy"], dtype=float)))
        for curve in curves
    )

    ax.set_xlim(-0.05, 3.05)
    ax.set_ylim(-0.02, max_entropy + 0.16)
    ax.set_xticks([0.0, 1.0, 2.0, 3.0])
    ax.set_xlabel(r"Observation time $t$")
    ax.set_ylabel("Global entropy")
    ax.set_title("(A) Three-node temporal cycle", loc="left", fontsize=14)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(loc="lower right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format=output_path.suffix.lstrip(".") or "pdf", bbox_inches="tight")
    if png_output_path is not None:
        png_output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_curve_summary(curves: list[dict[str, np.ndarray | float]]) -> None:
    for curve in curves:
        lamda = float(curve["lamda"])
        times = np.asarray(curve["times"], dtype=float)
        entropy = np.asarray(curve["entropy"], dtype=float)
        print(f"lambda={lamda:g}")
        print("times   :", np.array2string(times, precision=3))
        print("entropy :", np.array2string(entropy, precision=6))
        print()


def main() -> None:
    args = parse_args()
    lambdas = [float(lamda) for lamda in args.lambdas]
    net = build_temporal_cycle_network()
    curves = compute_entropy_curves(net=net, lambdas=lambdas)
    plot_toy_temporal_cycle_entropy(
        curves=curves,
        output_path=args.output,
        png_output_path=args.png_output,
    )
    print_curve_summary(curves)
    print(f"saved {args.output}")
    if args.png_output is not None:
        print(f"saved {args.png_output}")


if __name__ == "__main__":
    main()
