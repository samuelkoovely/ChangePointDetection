from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import auxiliary_functions
import fig_gridsearch_split_merge_limit as split_merge_limit
import fig_local_entropy as local_entropy

plt.style.use(Path(__file__).with_name("paper.mplstyle"))


DEFAULT_LOCAL_WINDOW = 5.0
DEFAULT_GRID_WINDOWS = [1.0, 5.0, 10.0]
DEFAULT_OUTPUT = Path("./figures/fig_local_entropy_combined.pdf")
TOP_PANEL_SPECS = [
    {"key": "merge_merge", "title": "(A)"},
    {"key": "merge_split", "title": "(B)"},
    {"key": "split_merge", "title": "(C)"},
]
BOTTOM_PANEL_LABELS = ["(D)", "(E)", "(F)"]
PANEL_TITLE_FONTSIZE = 16
AXIS_LABEL_FONTSIZE = 14
TICK_LABEL_FONTSIZE = 13
LEGEND_FONTSIZE = 14
INSET_TITLE_FONTSIZE = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine the local-entropy motif figure and the split-merge "
            "upper-bound grid-search figure in a shared 2x3 layout."
        )
    )
    parser.add_argument(
        "--motif-base",
        type=Path,
        default=None,
        help=(
            "Optional forward motif results base. When omitted, the script "
            "searches the standard locations."
        ),
    )
    parser.add_argument(
        "--signal-base",
        type=Path,
        default=None,
        help=(
            "Optional split-merge signal results directory. When omitted, the "
            "script searches the standard locations."
        ),
    )
    parser.add_argument(
        "--limit-base",
        type=Path,
        default=None,
        help=(
            "Optional split-merge limit results directory. When omitted, the "
            "script searches the standard locations."
        ),
    )
    parser.add_argument(
        "--local-window",
        type=float,
        default=DEFAULT_LOCAL_WINDOW,
        help="Window length used for the top-row local-entropy panels.",
    )
    parser.add_argument(
        "--grid-windows",
        nargs=3,
        type=float,
        default=DEFAULT_GRID_WINDOWS,
        help="Exactly three window lengths for the bottom-row split-merge panels.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to the output PDF.",
    )
    return parser.parse_args()


def resolve_motif_base(requested_base: Path | None = None) -> Path:
    if requested_base is None:
        return local_entropy.resolve_results_base_for_direction(reverse_time=False)

    expected_subdir = local_entropy.entropy_subdir(reverse_time=False)
    if all(
        (requested_base / motif_name / "metadata.pkl").exists()
        and (requested_base / motif_name / expected_subdir).exists()
        for motif_name in local_entropy.MOTIF_NAMES
    ):
        return requested_base

    raise FileNotFoundError(
        f"Explicit motif base {requested_base} is missing one or more motif "
        f"directories with metadata.pkl and {expected_subdir}."
    )


def select_available_lambdas(
    available_lambdas: np.ndarray,
    requested_lambdas: np.ndarray,
) -> np.ndarray:
    selected = []
    for lamda in np.asarray(requested_lambdas, dtype=float):
        matches = np.flatnonzero(
            np.isclose(np.asarray(available_lambdas, dtype=float), lamda, rtol=1e-9, atol=1e-12)
        )
        if len(matches) == 0:
            available_text = ", ".join(f"{value:g}" for value in available_lambdas)
            raise ValueError(
                f"Lambda {lamda:g} is not available in the local-entropy results. "
                f"Available lambdas: {available_text}"
            )
        selected.append(float(available_lambdas[matches[0]]))
    return np.asarray(selected, dtype=float)


def select_curves_for_lambdas(
    curves: list[list[np.ndarray]],
    available_lambdas: np.ndarray,
    selected_lambdas: np.ndarray,
) -> list[list[np.ndarray]]:
    selected_curves = []
    available_lambdas = np.asarray(available_lambdas, dtype=float)
    for lamda in np.asarray(selected_lambdas, dtype=float):
        matches = np.flatnonzero(
            np.isclose(available_lambdas, lamda, rtol=1e-9, atol=1e-12)
        )
        if len(matches) == 0:
            raise ValueError(f"Missing curve for lambda {lamda:g}.")
        selected_curves.append(curves[int(matches[0])])
    return selected_curves


def plot_split_merge_limit_panel(
    ax,
    panel_title: str,
    window: float,
    lambdas: np.ndarray,
    colors,
    signal_base: Path,
    signal_subdir: str,
    limit_base: Path,
    network,
    matrices,
    inset_cmap,
) -> None:
    for color, lamda in zip(colors, lambdas):
        payload = split_merge_limit.load_signal_payload(
            window=window,
            lamda=float(lamda),
            signal_base=signal_base,
            signal_subdir=signal_subdir,
        )
        t_samples = np.asarray(payload["t_samples"], dtype=float)
        signal = split_merge_limit.extract_signal_array(payload, lamda=float(lamda))
        ax.plot(t_samples, signal, color=color, alpha=0.85)

    limit_payload = split_merge_limit.load_or_compute_limit_payload(
        window=window,
        limit_base=limit_base,
        network=network,
    )
    limit_curve = split_merge_limit.extract_limit_array(limit_payload)
    ax.plot(limit_curve[:, 0], limit_curve[:, 1], **split_merge_limit.LIMIT_STYLE)

    ax.set_xlim(-5, 310)
    ax.set_ylim(-2, 5)
    ax.set_xlabel("t [s]", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(panel_title, loc="left", fontsize=PANEL_TITLE_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.set_box_aspect(1)

    for matrix, pos, (start, end) in zip(
        matrices,
        split_merge_limit.INSET_POSITIONS,
        split_merge_limit.TIME_INTERVALS,
    ):
        inset_ax = split_merge_limit.inset_axes(
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
        inset_ax.set_title(f"{start} ≤ t < {end}", fontsize=INSET_TITLE_FONTSIZE)
        for spine in inset_ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_edgecolor("black")


def build_legend_handles(colors, lambdas: np.ndarray) -> list[Line2D]:
    handles = [
        Line2D(
            [0],
            [0],
            color=color,
            linewidth=1.8,
            label=split_merge_limit.format_lambda_label(float(lamda)),
        )
        for color, lamda in zip(colors, lambdas)
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color=split_merge_limit.LIMIT_STYLE["color"],
            linestyle=split_merge_limit.LIMIT_STYLE["linestyle"],
            linewidth=split_merge_limit.LIMIT_STYLE["linewidth"],
            label="Upper bound",
        )
    )
    return handles


def main() -> None:
    args = parse_args()

    motif_base = resolve_motif_base(requested_base=args.motif_base)
    signal_base = split_merge_limit.resolve_signal_base(requested_base=args.signal_base)
    signal_metadata = split_merge_limit.load_metadata(signal_base)
    signal_subdir = split_merge_limit.resolve_signal_subdir(signal_base, signal_metadata)
    limit_base = split_merge_limit.resolve_limit_base(
        signal_base,
        requested_base=args.limit_base,
    )

    selected_lambdas = split_merge_limit.get_selected_lambdas(signal_metadata)
    local_available_lambdas_by_motif = {}
    for spec in TOP_PANEL_SPECS:
        motif_metadata = local_entropy.load_metadata(
            spec["key"],
            results_base=motif_base,
        )
        motif_lambdas = np.sort(np.asarray(motif_metadata["lambdas"], dtype=float))
        local_available_lambdas_by_motif[spec["key"]] = select_available_lambdas(
            available_lambdas=motif_lambdas,
            requested_lambdas=selected_lambdas,
        )

    selected_lambdas = local_available_lambdas_by_motif["split_merge"]

    grid_windows = [float(window) for window in args.grid_windows]
    split_merge_limit.ensure_signal_payloads(
        signal_base=signal_base,
        signal_metadata=signal_metadata,
        signal_subdir=signal_subdir,
        windows=grid_windows,
        lambdas=selected_lambdas,
    )

    top_networks = local_entropy.load_networks(results_base=motif_base)
    top_interval_matrices = {
        key: local_entropy.compute_interval_matrices(
            network=top_networks[key],
            intervals=split_merge_limit.TIME_INTERVALS,
        )
        for key in top_networks
    }
    split_merge_network = split_merge_limit.load_split_merge_network(signal_metadata)
    bottom_interval_matrices = split_merge_limit.compute_interval_matrices(
        split_merge_network,
        split_merge_limit.TIME_INTERVALS,
    )

    colors = auxiliary_functions.generate_plasma_colors(len(selected_lambdas))
    inset_cmap = local_entropy.make_inset_cmap()

    fig = plt.figure(figsize=(13.0, 9.0))
    gs = fig.add_gridspec(2, 3)
    axes = np.asarray(gs.subplots(sharey=True))

    for ax, spec in zip(axes[0], TOP_PANEL_SPECS):
        curves = local_entropy.load_entropy_curves(
            motif_name=spec["key"],
            results_base=motif_base,
            window=float(args.local_window),
            reverse_time=False,
            network=top_networks[spec["key"]],
        )
        selected_curves = select_curves_for_lambdas(
            curves=curves,
            available_lambdas=local_available_lambdas_by_motif[spec["key"]],
            selected_lambdas=selected_lambdas,
        )
        local_entropy.plot_network_panel(
            ax=ax,
            forward_curves=selected_curves,
            matrices=top_interval_matrices[spec["key"]],
            panel_title=spec["title"],
            forward_colors=colors,
            time_intervals=split_merge_limit.TIME_INTERVALS,
            inset_positions=split_merge_limit.INSET_POSITIONS,
            inset_cmap=inset_cmap,
        )

    for ax, label, window in zip(axes[1], BOTTOM_PANEL_LABELS, grid_windows):
        plot_split_merge_limit_panel(
            ax=ax,
            panel_title=f"{label} {split_merge_limit.window_title(window)}",
            window=window,
            lambdas=selected_lambdas,
            colors=colors,
            signal_base=signal_base,
            signal_subdir=signal_subdir,
            limit_base=limit_base,
            network=split_merge_network,
            matrices=bottom_interval_matrices,
            inset_cmap=inset_cmap,
        )

    axes[0, 0].set_ylabel("Entropy", fontsize=AXIS_LABEL_FONTSIZE)
    axes[1, 0].set_ylabel("Entropy", fontsize=AXIS_LABEL_FONTSIZE)
    for row in axes:
        for ax in row[1:]:
            ax.tick_params(labelleft=False)

    legend_handles = build_legend_handles(colors=colors, lambdas=selected_lambdas)
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(legend_handles),
        fontsize=LEGEND_FONTSIZE,
        frameon=False,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.995,
        top=0.96,
        bottom=0.12,
        wspace=0.18,
        hspace=0.28,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format="pdf", dpi=300, bbox_inches="tight")
    if "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
