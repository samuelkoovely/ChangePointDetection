import argparse
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.sparse.csgraph import connected_components

import auxiliary_functions


GLOBAL_RESULTS_BASE_CANDIDATES = (
    Path("gridsearch_results/motifs_global"),
    Path("gridsearch_results/motifs_global_f"),
    Path("gridsearch_results/motifs_global_b"),
)
LOCAL_RESULTS_BASE_CANDIDATES = (
    Path("gridsearch_results/motifs_f"),
    Path("gridsearch_results/motifs"),
    Path("gridsearch_results/motifs_run1"),
    Path("gridsearch_results/motifs_b"),
)
DEFAULT_OUTPUT_PATH = Path("./figures/fig_global_entropy.pdf")
MOTIF_NAMES = ("merge_merge", "merge_split", "split_merge")
DEFAULT_CURVE_INDICES = [5, 6, 7, 8]
BACKWARD_CURVE_STYLE = {
    "color": "red",
    "linewidth": 1.8,
    "zorder": 6,
}
LIMIT_STYLE = {
    "color": "black",
    "linestyle": "--",
    "linewidth": 1.8,
    "zorder": 5,
}
GLOBAL_LIMIT_START_TIME = 0.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot motif entropy curves. Use --curve-kind local together with "
            "--window to visualize local entropy signals."
        )
    )
    parser.add_argument(
        "--curve-kind",
        choices=("global", "local"),
        default="global",
        help="Which family of entropy curves to plot.",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=10.0,
        help="Local window length in seconds. Ignored for global curves.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output PDF path.",
    )
    return parser.parse_args()


def default_output_path(curve_kind, window):
    if curve_kind == "global":
        return DEFAULT_OUTPUT_PATH
    return Path(f"./figures/fig_global_entropy_local_window_{float(window):g}.pdf")


def signal_subdir(curve_kind, reverse_time):
    if curve_kind == "global":
        return "global_S_rev" if reverse_time else "global_S"
    return "window_S_rev" if reverse_time else "window_S"


def signal_filename(curve_kind, lamda):
    prefix = "global_S" if curve_kind == "global" else "window_S"
    return f"{prefix}{float(lamda):.11f}"


def direction_metadata_filename(reverse_time):
    return f"metadata_{'backward' if reverse_time else 'forward'}.pkl"


def format_lambda_label(lamda):
    return f"$\\lambda$ = {float(lamda):.2e}"


def resolve_results_base_for_direction(
    curve_kind,
    reverse_time,
    window=None,
    allow_missing=False,
):
    expected_subdir = signal_subdir(curve_kind, reverse_time)
    expected_metadata = direction_metadata_filename(reverse_time)
    candidates = (
        GLOBAL_RESULTS_BASE_CANDIDATES
        if curve_kind == "global"
        else LOCAL_RESULTS_BASE_CANDIDATES
    )

    for candidate in candidates:
        if not candidate.exists():
            continue
        if curve_kind == "global":
            valid = all(
                (candidate / motif_name / expected_subdir).exists()
                and (
                    (candidate / motif_name / expected_metadata).exists()
                    or (candidate / motif_name / "metadata.pkl").exists()
                )
                for motif_name in MOTIF_NAMES
            )
        else:
            valid = all(
                (candidate / motif_name / "metadata.pkl").exists()
                and (
                    candidate
                    / motif_name
                    / expected_subdir
                    / f"{float(window):g}"
                ).exists()
                for motif_name in MOTIF_NAMES
            )
        if valid:
            return candidate

    if allow_missing:
        return None

    raise FileNotFoundError(
        f"Could not find {'backward' if reverse_time else 'forward'} {curve_kind} motif results. "
        "Checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def load_pickle(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_metadata(motif_name, results_base, curve_kind="global", reverse_time=False):
    motif_dir = results_base / motif_name
    if curve_kind == "local":
        metadata_path = motif_dir / "metadata.pkl"
        if metadata_path.exists():
            return load_pickle(metadata_path)
        raise FileNotFoundError(f"Could not find metadata for {motif_name} in {motif_dir}")

    candidates = (
        motif_dir / direction_metadata_filename(reverse_time),
        motif_dir / "metadata.pkl",
    )
    for path in candidates:
        if path.exists():
            return load_pickle(path)
    raise FileNotFoundError(f"Could not find metadata for {motif_name} in {motif_dir}")


def metadata_lambdas(metadata, curve_kind="global", reverse_time=False):
    if curve_kind == "local":
        if "lambdas" in metadata:
            return np.asarray(metadata["lambdas"], dtype=float)
        raise KeyError("Metadata does not contain local lambdas.")

    direction_key = "backward_lambdas" if reverse_time else "forward_lambdas"
    if direction_key in metadata:
        return np.asarray(metadata[direction_key], dtype=float)

    if "lambdas" in metadata:
        metadata_reverse_time = metadata.get("reverse_time")
        if metadata_reverse_time is None or bool(metadata_reverse_time) == bool(reverse_time):
            return np.asarray(metadata["lambdas"], dtype=float)

    raise KeyError(
        f"Metadata does not contain {'backward' if reverse_time else 'forward'} lambdas."
    )


def load_networks(results_base, curve_kind, reverse_time=False):
    networks = {}

    for motif_name in MOTIF_NAMES:
        metadata = load_metadata(
            motif_name,
            results_base=results_base,
            curve_kind=curve_kind,
            reverse_time=reverse_time,
        )
        networks[motif_name] = load_pickle(Path(metadata["network_path"]))
    return networks


def compute_interval_matrices(network, intervals):
    return [
        network.compute_static_adjacency_matrix(start_time=start, end_time=end).toarray()
        for start, end in intervals
    ]


def limit_payload_path(motif_name, results_base, curve_kind, window=None):
    if curve_kind == "global":
        return (
            results_base
            / motif_name
            / "global_limit_selected"
            / "global_limit.pkl"
        )
    return (
        results_base
        / motif_name
        / "window_limit_selected"
        / f"{float(window):g}"
        / "window_limit.pkl"
    )


def compute_component_log_sum(network, end_time):
    adjacency = network.compute_static_adjacency_matrix(
        start_time=float(GLOBAL_LIMIT_START_TIME),
        end_time=float(end_time),
    ).tocsr()

    n_components, labels = connected_components(
        adjacency,
        directed=False,
        return_labels=True,
    )
    component_sizes = np.bincount(labels, minlength=n_components).astype(float)
    weights = component_sizes / float(network.num_nodes)
    return float(np.sum(weights * np.log(component_sizes)))


def compute_global_limit_payload(network, motif_name):
    t_samples = np.asarray(network.times[:-1], dtype=float)
    k_samples = np.arange(len(t_samples), dtype=int)
    values = np.empty(len(t_samples), dtype=float)

    for idx, end_time in enumerate(t_samples):
        values[idx] = compute_component_log_sum(
            network=network,
            end_time=float(end_time),
        )
        if (idx + 1) % 250 == 0 or idx + 1 == len(t_samples):
            print(
                f"{motif_name} global limit: "
                f"{idx + 1}/{len(t_samples)} samples"
            )

    time_limit_array = (
        np.column_stack((t_samples, values))
        if len(t_samples) > 0
        else np.empty((0, 2), dtype=float)
    )

    return {
        "motif_name": motif_name,
        "start_time": float(GLOBAL_LIMIT_START_TIME),
        "k_samples": k_samples,
        "t_samples": t_samples,
        "component_log_sums": values,
        "signal": values,
        "signal_array": values,
        "time_component_log_sums": time_limit_array,
        "statistic": (
            "sum((size / N) * log(size)) over connected components of the "
            "aggregated graph on [0, t]"
        ),
    }


def compute_window_component_log_sum(network, start_time, window_seconds):
    adjacency = network.compute_static_adjacency_matrix(
        start_time=float(start_time),
        end_time=float(start_time) + float(window_seconds),
    ).tocsr()

    n_components, labels = connected_components(
        adjacency,
        directed=False,
        return_labels=True,
    )
    component_sizes = np.bincount(labels, minlength=n_components).astype(float)
    weights = component_sizes / float(network.num_nodes)
    return float(np.sum(weights * np.log(component_sizes)))


def compute_window_limit_payload(network, motif_name, window):
    k_samples = np.flatnonzero(
        np.asarray(network.times, dtype=float) < network.times[-1] - float(window)
    ).astype(int)
    t_samples = np.asarray(network.times[k_samples], dtype=float)
    values = np.empty(len(t_samples), dtype=float)

    for idx, start_time in enumerate(t_samples):
        values[idx] = compute_window_component_log_sum(
            network=network,
            start_time=float(start_time),
            window_seconds=float(window),
        )
        if (idx + 1) % 250 == 0 or idx + 1 == len(t_samples):
            print(
                f"{motif_name} limit, window={float(window):g}: "
                f"{idx + 1}/{len(t_samples)} samples"
            )

    time_limit_array = (
        np.column_stack((t_samples, values))
        if len(t_samples) > 0
        else np.empty((0, 2), dtype=float)
    )

    return {
        "motif_name": motif_name,
        "window": float(window),
        "window_seconds": float(window),
        "k_samples": k_samples,
        "t_samples": t_samples,
        "component_log_sums": values,
        "signal": values,
        "signal_array": values,
        "time_component_log_sums": time_limit_array,
        "statistic": "sum((size / N) * log(size)) over connected components",
    }


def load_or_compute_limit_payload(motif_name, network, results_base, curve_kind, window=None):
    payload_path = limit_payload_path(
        motif_name,
        results_base=results_base,
        curve_kind=curve_kind,
        window=window,
    )
    if payload_path.exists():
        return load_pickle(payload_path)

    if curve_kind == "global":
        payload = compute_global_limit_payload(
            network=network,
            motif_name=motif_name,
        )
    else:
        payload = compute_window_limit_payload(
            network=network,
            motif_name=motif_name,
            window=float(window),
        )
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    with payload_path.open("wb") as handle:
        pickle.dump(payload, handle)
    return payload


def extract_limit_array(payload):
    if "time_component_log_sums" in payload:
        return np.asarray(payload["time_component_log_sums"], dtype=float)

    return np.column_stack(
        (
            np.asarray(payload["t_samples"], dtype=float),
            np.asarray(payload["signal_array"], dtype=float),
        )
    )


def extract_signal_array(payload, lamda):
    if "signal_array" in payload:
        return np.asarray(payload["signal_array"], dtype=float)

    signal = payload["signal"]
    if isinstance(signal, dict):
        return np.asarray(signal[f"{float(lamda):.11f}"], dtype=float)

    return np.asarray(signal, dtype=float)


def load_entropy_curves(
    motif_name,
    results_base,
    curve_kind="global",
    window=None,
    reverse_time=False,
):
    metadata = load_metadata(
        motif_name,
        results_base=results_base,
        curve_kind=curve_kind,
        reverse_time=reverse_time,
    )
    lambdas = np.sort(
        metadata_lambdas(
            metadata,
            curve_kind=curve_kind,
            reverse_time=reverse_time,
        )
    )
    curve_dir = results_base / motif_name / signal_subdir(curve_kind, reverse_time)
    if curve_kind == "local":
        curve_dir = curve_dir / f"{float(window):g}"

    curves = []
    for lamda in lambdas:
        filepath = curve_dir / signal_filename(curve_kind, lamda)
        data = load_pickle(filepath)
        curves.append(
            [
                np.asarray(data["t_samples"], dtype=float),
                extract_signal_array(data, lamda=lamda),
            ]
        )
    return curves


def select_curves(curves, curve_indices):
    if curve_indices is None:
        return curves
    if len(curves) == 0:
        return []
    if max(curve_indices, default=-1) >= len(curves):
        return curves
    return [curves[idx] for idx in curve_indices]


def select_single_curve(curves, preferred_index):
    if len(curves) == 0:
        return None
    if preferred_index is None:
        return curves[-1]
    safe_index = min(max(int(preferred_index), 0), len(curves) - 1)
    return curves[safe_index]


def select_values(values, selected_indices):
    if selected_indices is None:
        return list(values)
    if len(values) == 0:
        return []
    if max(selected_indices, default=-1) >= len(values):
        return list(values)
    return [values[idx] for idx in selected_indices]


def make_inset_cmap():
    cmap = colormaps["inferno"].copy()
    cmap.set_bad(color="white")
    return cmap


def plot_network_panel(
    ax,
    forward_curves,
    matrices,
    panel_title,
    forward_colors,
    time_intervals,
    inset_positions,
    inset_cmap,
    backward_curve=None,
    limit_curve=None,
):
    for color, curve in zip(forward_colors, forward_curves):
        ax.plot(curve[0], curve[1], color=color, alpha=1)

    if backward_curve is not None:
        ax.plot(backward_curve[0], backward_curve[1], **BACKWARD_CURVE_STYLE)

    if limit_curve is not None:
        ax.plot(limit_curve[:, 0], limit_curve[:, 1], **LIMIT_STYLE)

    ax.set_xlim(-5, 310)
    ax.set_ylim(-2, 5)
    ax.set_xlabel("t [s]")
    ax.set_title(panel_title, loc="left", fontsize=14)
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
        inset_ax.set_title(f"{start} ≤ t < {end}", fontsize=8)
        for spine in inset_ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_edgecolor("black")


def main():
    args = parse_args()
    forward_results_base = resolve_results_base_for_direction(
        curve_kind=args.curve_kind,
        reverse_time=False,
        window=args.window,
    )
    backward_results_base = resolve_results_base_for_direction(
        curve_kind=args.curve_kind,
        reverse_time=True,
        window=args.window,
        allow_missing=True,
    )
    networks = load_networks(
        results_base=forward_results_base,
        curve_kind=args.curve_kind,
        reverse_time=False,
    )
    panel_specs = [
        {"key": "merge_merge", "title": "(A)", "backward_index": 7},
        {"key": "merge_split", "title": "(B)", "backward_index": 7},
        {"key": "split_merge", "title": "(C)", "backward_index": 7},
    ]
    inset_positions = [0.06, 0.37, 0.68]
    time_intervals = [(0, 100), (100, 200), (200, 300)]

    interval_matrices = {
        key: compute_interval_matrices(network, time_intervals)
        for key, network in networks.items()
    }
    forward_entropy = {
        key: load_entropy_curves(
            key,
            results_base=forward_results_base,
            curve_kind=args.curve_kind,
            window=args.window,
            reverse_time=False,
        )
        for key in networks
    }
    backward_entropy = (
        {
            key: load_entropy_curves(
                key,
                results_base=backward_results_base,
                curve_kind=args.curve_kind,
                window=args.window,
                reverse_time=True,
            )
            for key in networks
        }
        if backward_results_base is not None
        else {}
    )
    limit_curves = {
        key: extract_limit_array(
            load_or_compute_limit_payload(
                motif_name=key,
                network=networks[key],
                results_base=forward_results_base,
                curve_kind=args.curve_kind,
                window=args.window,
            )
        )
        for key in networks
    }

    reference_motif = panel_specs[0]["key"]
    forward_reference_metadata = load_metadata(
        reference_motif,
        results_base=forward_results_base,
        curve_kind=args.curve_kind,
        reverse_time=False,
    )
    forward_lambdas = np.sort(
        metadata_lambdas(
            forward_reference_metadata,
            curve_kind=args.curve_kind,
            reverse_time=False,
        )
    )
    selected_forward_lambdas = select_values(forward_lambdas, DEFAULT_CURVE_INDICES)
    forward_colors = auxiliary_functions.generate_plasma_colors(
        len(selected_forward_lambdas)
    )

    fig = plt.figure(figsize=(13, 5.4))
    gs = fig.add_gridspec(1, 3)
    axes = np.atleast_1d(gs.subplots(sharey=True))
    inset_cmap = make_inset_cmap()

    for ax, spec in zip(axes, panel_specs):
        key = spec["key"]
        selected_curves = select_curves(
            forward_entropy[key],
            curve_indices=DEFAULT_CURVE_INDICES,
        )
        backward_curve = select_single_curve(
            backward_entropy.get(key, []),
            preferred_index=spec.get("backward_index"),
        )
        plot_network_panel(
            ax=ax,
            forward_curves=selected_curves,
            matrices=interval_matrices[key],
            panel_title=spec["title"],
            forward_colors=forward_colors,
            time_intervals=time_intervals,
            inset_positions=inset_positions,
            inset_cmap=inset_cmap,
            backward_curve=backward_curve,
            limit_curve=limit_curves.get(key),
        )

    axes[0].set_ylabel("Entropy")
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    legend_handles = [
        Line2D([0], [0], color=color, linewidth=1.8, label=format_lambda_label(lamda))
        for color, lamda in zip(forward_colors, selected_forward_lambdas)
    ]
    if backward_results_base is not None:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=BACKWARD_CURVE_STYLE["color"],
                linewidth=BACKWARD_CURVE_STYLE["linewidth"],
                label="Backward Entropy",
            )
        )
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color=LIMIT_STYLE["color"],
            linestyle=LIMIT_STYLE["linestyle"],
            linewidth=LIMIT_STYLE["linewidth"],
            label="Limit Statistic",
        )
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(legend_handles),
        fontsize="small",
    )

    output_path = args.output or default_output_path(args.curve_kind, args.window)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
    if "agg" not in plt.get_backend().lower():
        plt.show()


if __name__ == "__main__":
    main()
