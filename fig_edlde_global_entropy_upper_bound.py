import argparse
import pickle
from pathlib import Path

import numpy as np

from edlde_global_entropy_upper_bound_common import (
    DEFAULT_FIGURE_PATH,
    DEFAULT_LAMBDAS,
    DEFAULT_NETWORK_DIR,
    DEFAULT_RATES,
    DEFAULT_RESULTS_DIR,
    DEFAULT_TIME_INTERVALS,
    INSET_POSITIONS,
    N_GROUPS,
    N_PER_GROUP,
    T_END,
    T_START,
    ensure_plotting_env,
    load_networks_and_metadata,
    panel_title,
    rate_limit_path,
    rate_signal_dir,
    signal_filename,
    write_summary,
)

ensure_plotting_env()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import auxiliary_functions
from fig_global_entropy import (
    LIMIT_STYLE,
    compute_interval_matrices,
    extract_limit_array,
    format_lambda_label,
    make_inset_cmap,
)


TOTAL_NODES = N_GROUPS * N_PER_GROUP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the EDLDE global-entropy upper-bound comparison figure from cached network and signal data."
    )
    parser.add_argument(
        "--rates",
        nargs="+",
        type=int,
        default=list(DEFAULT_RATES),
        help="Logical panel identifiers to plot.",
    )
    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=list(DEFAULT_LAMBDAS),
        help="Diffusion-rate values to plot.",
    )
    parser.add_argument(
        "--network-dir",
        type=Path,
        default=DEFAULT_NETWORK_DIR,
        help="Directory containing cached temporal networks.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing cached entropy curves and upper bounds.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_FIGURE_PATH,
        help="Output PDF path for the figure.",
    )
    return parser.parse_args()


def load_rate_curves(rate: int, lambdas, results_dir: Path):
    signal_dir = rate_signal_dir(rate=rate, results_dir=results_dir)
    if not signal_dir.exists():
        raise FileNotFoundError(
            f"Missing signal directory {signal_dir}. Run compute_edlde_upper_bound_signals.py first."
        )

    curves = []
    for lamda in lambdas:
        signal_path = signal_dir / signal_filename(lamda)
        if not signal_path.exists():
            raise FileNotFoundError(
                f"Missing signal file {signal_path}. Run compute_edlde_upper_bound_signals.py first."
            )
        with signal_path.open("rb") as handle:
            payload = pickle.load(handle)
        curves.append(
            (
                np.asarray(payload["t_samples"], dtype=float),
                np.asarray(payload["signal_array"], dtype=float),
            )
        )
    return curves


def load_rate_limit(rate: int, results_dir: Path) -> np.ndarray:
    limit_payload_path = rate_limit_path(rate=rate, results_dir=results_dir)
    if not limit_payload_path.exists():
        raise FileNotFoundError(
            f"Missing limit payload {limit_payload_path}. Run compute_edlde_upper_bound_signals.py first."
        )
    with limit_payload_path.open("rb") as handle:
        payload = pickle.load(handle)
    return extract_limit_array(payload)


def plot_rate_panel(
    ax,
    forward_curves,
    limit_curve,
    matrices,
    title,
    forward_colors,
    time_intervals,
    inset_positions,
    inset_cmap,
):
    def interval_label(start, end):
        if np.isclose(float(start), T_START) and np.isclose(float(end), T_END):
            return f"{int(T_START)} ≤ t ≤ {int(T_END)}"
        return f"{start} ≤ t < {end}"

    for color, curve in zip(forward_colors, forward_curves):
        ax.plot(curve[0], curve[1], color=color, linewidth=1.8, alpha=1.0)

    ax.plot(limit_curve[:, 0], limit_curve[:, 1], **LIMIT_STYLE)
    ax.set_xlim(T_START, T_END)
    ax.set_ylim(-0.1, np.log(TOTAL_NODES) + 0.25)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("t [s]")
    ax.set_title(title, loc="left", fontsize=14)
    ax.set_box_aspect(1)

    for matrix, pos, (start, end) in zip(matrices, inset_positions, time_intervals):
        inset_ax = inset_axes(
            ax,
            width="20%",
            height="20%",
            loc="lower left",
            bbox_to_anchor=(pos, 0.05, 1, 1),
            bbox_transform=ax.transAxes,
        )
        masked_matrix = np.ma.masked_where(matrix == 0, matrix)
        positive_entries = matrix[matrix > 0]
        vmax = positive_entries.max() if positive_entries.size else 1.0
        inset_ax.matshow(
            masked_matrix,
            cmap=inset_cmap,
            aspect="equal",
            vmin=0,
            vmax=vmax,
            interpolation="nearest",
        )
        inset_ax.set_facecolor("white")
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        inset_ax.set_title(interval_label(start, end), fontsize=8)
        for spine in inset_ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_edgecolor("black")


def generate_figure(rates, lambdas, network_dir: Path, results_dir: Path, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    networks, metadata_by_rate = load_networks_and_metadata(rates=rates, network_dir=network_dir)
    interval_matrices = {
        int(rate): compute_interval_matrices(networks[int(rate)], DEFAULT_TIME_INTERVALS)
        for rate in rates
    }
    forward_curves = {
        int(rate): load_rate_curves(rate=int(rate), lambdas=lambdas, results_dir=results_dir)
        for rate in rates
    }
    limit_curves = {
        int(rate): load_rate_limit(rate=int(rate), results_dir=results_dir)
        for rate in rates
    }

    forward_colors = auxiliary_functions.generate_plasma_colors(len(lambdas))
    inset_cmap = make_inset_cmap()

    fig = plt.figure(figsize=(13, 5.4))
    gs = fig.add_gridspec(1, len(rates))
    axes = np.atleast_1d(gs.subplots(sharey=True))

    for ax, rate in zip(axes, rates):
        plot_rate_panel(
            ax=ax,
            forward_curves=forward_curves[int(rate)],
            limit_curve=limit_curves[int(rate)],
            matrices=interval_matrices[int(rate)],
            title=panel_title(int(rate)),
            forward_colors=forward_colors,
            time_intervals=DEFAULT_TIME_INTERVALS,
            inset_positions=INSET_POSITIONS,
            inset_cmap=inset_cmap,
        )

    axes[0].set_ylabel("Conditional Entropy")
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    legend_handles = [
        Line2D([0], [0], color=color, linewidth=1.8, label=format_lambda_label(lamda))
        for color, lamda in zip(forward_colors, lambdas)
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color=LIMIT_STYLE["color"],
            linestyle=LIMIT_STYLE["linestyle"],
            linewidth=LIMIT_STYLE["linewidth"],
            label="Upper Bound",
        )
    )
    fig.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.02, 0.96, 0.08),
        mode="expand",
        ncol=len(legend_handles),
        fontsize="medium",
        frameon=False,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(left=0.055, right=0.995, top=0.93, bottom=0.2, wspace=0.08)
    fig.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return write_summary(
        rates=rates,
        lambdas=lambdas,
        network_dir=network_dir,
        results_dir=results_dir,
        figure_path=output_path,
        metadata_by_rate=metadata_by_rate,
    )


def main() -> None:
    args = parse_args()
    summary_filepath = generate_figure(
        rates=[int(rate) for rate in args.rates],
        lambdas=[float(lamda) for lamda in args.lambdas],
        network_dir=args.network_dir,
        results_dir=args.results_dir,
        output_path=args.output,
    )
    print(f"Saved figure to {args.output}")
    print(f"Saved summary to {summary_filepath}")


if __name__ == "__main__":
    main()
