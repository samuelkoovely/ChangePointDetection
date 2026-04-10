import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
TMP_ROOT = Path(os.environ.get("TMPDIR", "/tmp")).resolve()
MPL_CACHE_DIR = TMP_ROOT / "changepointdetection_mplconfig"
XDG_CACHE_DIR = TMP_ROOT / "changepointdetection_xdg_cache"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import auxiliary_functions
from compute_global_entropy_motifs import clear_lambda_cache, compute_global_entropy_curve
from EDLDE import generate_smooth_SBM, trim_temporal_network_head_tail
from fig_global_entropy import (
    LIMIT_STYLE,
    compute_global_limit_payload,
    compute_interval_matrices,
    extract_limit_array,
    format_lambda_label,
    make_inset_cmap,
)
from signal_generation import ensure_laplacians


DEFAULT_RATES = (2, 5, 10)
DEFAULT_LAMBDAS = (0.1, 0.464158883, 2.15443469, 10.0)
DEFAULT_NETWORK_DIR = Path("data/edlde_upper_bound")
DEFAULT_RESULTS_DIR = Path("gridsearch_results/edlde_upper_bound")
DEFAULT_FIGURE_PATH = Path("figures/fig_global_entropy_edlde_upper_bound.pdf")
DEFAULT_TIME_INTERVALS = ((0, 100),)
DEFAULT_SEED_HINTS = {
    2: 4,
    5: 1,
    10: 1,
}
RATE_CONFIGS = {
    2: {
        "density": 5.0,
        "inter_tau": 5.0,
        "list_p_within_community": [1.0],
    },
    5: {
        "density": 5.0,
        "inter_tau": 5.0,
    },
    10: {
        "density": 10.0,
        "inter_tau": 10.0,
    },
}
N_GROUPS = 4
N_PER_GROUP = 25
TOTAL_NODES = N_GROUPS * N_PER_GROUP
T_START = 0.0
T_END = 100.0
INTER_TAU = 5.0
POWER_CONFIG = [1]
BASIS_NUM_COMMUNITIES = 4
PANEL_TITLES = {
    2: "(A) 5 links/s, no noise",
    5: "(B) 5 links/s",
    10: "(C) 10 links/s",
}
INSET_POSITIONS = (0.37,)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample EDLDE temporal networks with one fixed 4-cluster community "
            "structure, compute forward global entropy curves, and plot them "
            "against the connected-component upper bound."
        )
    )
    parser.add_argument(
        "--rates",
        nargs="+",
        type=int,
        default=list(DEFAULT_RATES),
        help="Event spawning rates (links per second) to simulate.",
    )
    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=list(DEFAULT_LAMBDAS),
        help="Diffusion-rate values to evaluate.",
    )
    parser.add_argument(
        "--network-dir",
        type=Path,
        default=DEFAULT_NETWORK_DIR,
        help="Directory where sampled temporal networks will be stored.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory where entropy curves and upper bounds will be stored.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_FIGURE_PATH,
        help="Output PDF path for the figure.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate cached networks, curves, and upper bounds.",
    )
    return parser.parse_args()


def rate_slug(rate: int | float) -> str:
    return f"rate_{float(rate):g}".replace(".", "p")


def preferred_seed_order(rate: int) -> list[int]:
    seeds = []
    hint = DEFAULT_SEED_HINTS.get(int(rate))
    if hint is not None:
        seeds.append(int(hint))
    seeds.extend(seed for seed in range(1, 101) if seed != hint)
    return seeds


def rate_config(rate: int) -> dict:
    config = dict(RATE_CONFIGS.get(int(rate), {}))
    config.setdefault("density", float(rate))
    config.setdefault("inter_tau", INTER_TAU)
    return config


def sample_rate_network(rate: int, seed: int):
    config = rate_config(rate)
    net = generate_smooth_SBM(
        density=float(config["density"]),
        inter_tau=float(config["inter_tau"]),
        n_per_group=N_PER_GROUP,
        n_groups=N_GROUPS,
        t_start=T_START,
        t_end=T_END,
        basis_num_communities=BASIS_NUM_COMMUNITIES,
        powers_num_communities=POWER_CONFIG,
        list_p_within_community=config.get("list_p_within_community"),
        seed=int(seed),
    )
    trimmed_net, _ = trim_temporal_network_head_tail(
        temporal_net=net,
        density=float(config["density"]),
        inter_tau=float(config["inter_tau"]),
        tail_start_time=T_END,
        head_start_time=T_START,
        clip_ending_times=True,
        align_first_event_to_zero=False,
    )
    return trimmed_net


def ensure_sampled_network(rate: int, network_dir: Path, force: bool):
    rate_dir = network_dir / rate_slug(rate)
    network_path = rate_dir / "network.pkl"
    metadata_path = rate_dir / "metadata.json"

    if network_path.exists() and metadata_path.exists() and not force:
        with network_path.open("rb") as handle:
            net = pickle.load(handle)
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        return net, metadata

    rate_dir.mkdir(parents=True, exist_ok=True)
    selected_seed = None
    net = None
    for seed in preferred_seed_order(rate):
        candidate = sample_rate_network(rate=rate, seed=seed)
        if candidate.num_nodes == TOTAL_NODES and np.isclose(float(candidate.times[-1]), T_END):
            selected_seed = seed
            net = candidate
            break

    if net is None:
        raise RuntimeError(
            f"Could not sample a {TOTAL_NODES}-node network for rate={rate} "
            "within the tested seed range."
        )

    metadata = {
        "rate": int(rate),
        "density": float(rate_config(rate)["density"]),
        "seed": int(selected_seed),
        "t_start": T_START,
        "t_end": T_END,
        "inter_tau": float(rate_config(rate)["inter_tau"]),
        "list_p_within_community": rate_config(rate).get("list_p_within_community"),
        "n_groups": N_GROUPS,
        "n_per_group": N_PER_GROUP,
        "basis_num_communities": BASIS_NUM_COMMUNITIES,
        "powers_num_communities": POWER_CONFIG,
        "num_nodes": int(net.num_nodes),
        "num_events": int(net.num_events),
        "num_times": int(len(net.times)),
        "network_path": str(network_path),
    }

    with network_path.open("wb") as handle:
        pickle.dump(net, handle)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return net, metadata


def signal_filename(lamda: float) -> str:
    return f"global_S{float(lamda):.11f}"


def ensure_rate_curves(
    rate: int,
    net,
    network_path: str,
    lambdas: list[float],
    results_dir: Path,
    force: bool,
):
    rate_dir = results_dir / rate_slug(rate)
    signal_dir = rate_dir / "global_S"
    signal_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for stale_path in signal_dir.glob("global_S*"):
            stale_path.unlink()
    metadata_path = rate_dir / "metadata_forward.pkl"
    shared_metadata_path = rate_dir / "metadata.pkl"

    ensure_laplacians(net)
    curves = []
    for lamda in lambdas:
        signal_path = signal_dir / signal_filename(lamda)
        if signal_path.exists() and not force:
            with signal_path.open("rb") as handle:
                payload = pickle.load(handle)
        else:
            payload = compute_global_entropy_curve(net=net, lamda=float(lamda), reverse_time=False)
            with signal_path.open("wb") as handle:
                pickle.dump(payload, handle)
        curves.append(
            (
                np.asarray(payload["t_samples"], dtype=float),
                np.asarray(payload["signal_array"], dtype=float),
            )
        )
        clear_lambda_cache(net, float(lamda))

    metadata = {
        "rate": int(rate),
        "network_path": str(network_path),
        "forward_lambdas": np.asarray(lambdas, dtype=float),
        "reverse_time": False,
        "direction": "forward",
        "curve_kind": "global_entropy",
        "signal_subdir": "global_S",
        "num_nodes": int(net.num_nodes),
        "num_events": int(net.num_events),
        "num_times": int(len(net.times)),
    }
    with metadata_path.open("wb") as handle:
        pickle.dump(metadata, handle)
    with shared_metadata_path.open("wb") as handle:
        pickle.dump(metadata, handle)

    return curves


def ensure_rate_limit(rate: int, net, results_dir: Path, force: bool) -> np.ndarray:
    rate_dir = results_dir / rate_slug(rate)
    limit_path = rate_dir / "global_limit_selected" / "global_limit.pkl"
    limit_path.parent.mkdir(parents=True, exist_ok=True)

    if limit_path.exists() and not force:
        with limit_path.open("rb") as handle:
            payload = pickle.load(handle)
    else:
        payload = compute_global_limit_payload(net, motif_name=rate_slug(rate))
        with limit_path.open("wb") as handle:
            pickle.dump(payload, handle)
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


def main() -> None:
    args = parse_args()
    rates = [int(rate) for rate in args.rates]
    lambdas = [float(lamda) for lamda in args.lambdas]
    network_dir = args.network_dir
    results_dir = args.results_dir
    output_path = args.output

    network_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    networks = {}
    metadata_by_rate = {}
    forward_curves = {}
    limit_curves = {}
    interval_matrices = {}

    for rate in rates:
        net, metadata = ensure_sampled_network(
            rate=rate,
            network_dir=network_dir,
            force=bool(args.force),
        )
        networks[rate] = net
        metadata_by_rate[rate] = metadata
        forward_curves[rate] = ensure_rate_curves(
            rate=rate,
            net=net,
            network_path=metadata["network_path"],
            lambdas=lambdas,
            results_dir=results_dir,
            force=bool(args.force),
        )
        limit_curves[rate] = ensure_rate_limit(
            rate=rate,
            net=net,
            results_dir=results_dir,
            force=bool(args.force),
        )
        interval_matrices[rate] = compute_interval_matrices(net, DEFAULT_TIME_INTERVALS)

    forward_colors = auxiliary_functions.generate_plasma_colors(len(lambdas))
    inset_cmap = make_inset_cmap()

    fig = plt.figure(figsize=(13, 5.4))
    gs = fig.add_gridspec(1, len(rates))
    axes = np.atleast_1d(gs.subplots(sharey=True))

    for ax, rate in zip(axes, rates):
        plot_rate_panel(
            ax=ax,
            forward_curves=forward_curves[rate],
            limit_curve=limit_curves[rate],
            matrices=interval_matrices[rate],
            title=PANEL_TITLES.get(rate, f"{rate:g} links/s"),
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
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(legend_handles),
        fontsize="small",
    )

    fig.subplots_adjust(left=0.055, right=0.995, top=0.93, bottom=0.2, wspace=0.08)
    fig.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary_path = results_dir / "summary.json"
    summary = {
        "rates": rates,
        "lambdas": lambdas,
        "rate_configs": {str(rate): rate_config(rate) for rate in rates},
        "network_dir": str(network_dir),
        "results_dir": str(results_dir),
        "figure_path": str(output_path),
        "networks": metadata_by_rate,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved figure to {output_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
