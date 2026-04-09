from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.sparse.csgraph import connected_components

import auxiliary_functions


FORWARD_RESULTS_BASE = Path("gridsearch_results/motifs_f")
BACKWARD_RESULTS_BASE = Path("gridsearch_results/motifs_b")
MOTIF_NAMES = ("merge_merge", "merge_split", "split_merge")
WINDOW = 5
OUTPUT_PATH = Path("./figures/fig_local_entropy_3.pdf")
LIMIT_STYLE = {
    "color": "black",
    "linestyle": "--",
    "linewidth": 1.8,
    "zorder": 5,
}
BACKWARD_CURVE_STYLE = {
    "color": "red",
    "linewidth": 1.8,
    "zorder": 6,
}


def format_lambda_label(lamda):
    return f"$\\lambda$ = {float(lamda):.2e}"


def load_pickle(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def validate_results_base(results_base, reverse_time):
    expected_subdir = "window_S_rev" if reverse_time else "window_S"
    for motif_name in MOTIF_NAMES:
        motif_dir = results_base / motif_name
        metadata_path = motif_dir / "metadata.pkl"
        window_dir = motif_dir / expected_subdir / f"{float(WINDOW):g}"
        if not metadata_path.exists() or not window_dir.exists():
            direction = "backward" if reverse_time else "forward"
            raise FileNotFoundError(
                f"Missing {direction} local-entropy inputs under {results_base} "
                f"for motif {motif_name}."
            )


def load_networks():
    networks = {}
    for motif_name in MOTIF_NAMES:
        metadata = load_pickle(FORWARD_RESULTS_BASE / motif_name / "metadata.pkl")
        networks[motif_name] = load_pickle(Path(metadata["network_path"]))
    return networks


def compute_interval_matrices(network, intervals):
    return [
        network.compute_static_adjacency_matrix(start_time=start, end_time=end).toarray()
        for start, end in intervals
    ]


def extract_signal_array(payload, lamda):
    if "signal_array" in payload:
        return np.asarray(payload["signal_array"], dtype=float)

    signal = payload["signal"]
    if isinstance(signal, dict):
        return np.asarray(signal[f"{float(lamda):.11f}"], dtype=float)

    return np.asarray(signal, dtype=float)


def load_entropy_curves(motif_name, results_base, window=WINDOW, reverse_time=False):
    metadata = load_pickle(results_base / motif_name / "metadata.pkl")
    lambdas = np.sort(np.asarray(metadata["lambdas"], dtype=float))
    signal_subdir = "window_S_rev" if reverse_time else "window_S"
    curves = []
    for lamda in lambdas:
        curve_path = (
            results_base
            / motif_name
            / signal_subdir
            / f"{float(window):g}"
            / f"window_S{lamda:.11f}"
        )
        data = load_pickle(curve_path)
        curves.append(
            {
                "t_samples": np.asarray(data["t_samples"], dtype=float),
                "signal": extract_signal_array(data, lamda=lamda),
                "lambda": float(lamda),
            }
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


def select_single_curve(curves, preferred_index=None):
    if len(curves) == 0:
        return None
    if preferred_index is None:
        return curves[-1]
    safe_index = min(max(int(preferred_index), 0), len(curves) - 1)
    return curves[safe_index]


def limit_payload_path(motif_name, window=WINDOW):
    return (
        FORWARD_RESULTS_BASE
        / motif_name
        / "window_limit_selected"
        / f"{float(window):g}"
        / "window_limit.pkl"
    )


def compute_component_log_sum(network, start_time, window_seconds):
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


def compute_window_limit_payload(network, motif_name, window=WINDOW):
    k_samples = np.flatnonzero(network.times < network.times[-1] - float(window)).astype(int)
    t_samples = np.asarray(network.times[k_samples], dtype=float)
    values = np.empty(len(t_samples), dtype=float)

    for idx, start_time in enumerate(t_samples):
        values[idx] = compute_component_log_sum(
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


def load_or_compute_limit_payload(motif_name, network, window=WINDOW):
    payload_path = limit_payload_path(motif_name, window=window)
    if payload_path.exists():
        return load_pickle(payload_path)

    payload = compute_window_limit_payload(network, motif_name=motif_name, window=window)
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


def make_inset_cmap():
    cmap = colormaps["inferno"].copy()
    cmap.set_bad(color="white")
    return cmap


def plot_network_panel(
    ax,
    forward_curves,
    backward_curve,
    matrices,
    panel_title,
    lambda_colors,
    time_intervals,
    inset_positions,
    inset_cmap,
    limit_curve=None,
):
    for curve in forward_curves:
        ax.plot(
            curve["t_samples"],
            curve["signal"],
            color=lambda_colors[curve["lambda"]],
            alpha=1,
        )

    if backward_curve is not None:
        ax.plot(
            backward_curve["t_samples"],
            backward_curve["signal"],
            **BACKWARD_CURVE_STYLE,
        )

    if limit_curve is not None:
        ax.plot(limit_curve[:, 0], limit_curve[:, 1], **LIMIT_STYLE)

    ax.set_xlim(-5, 310)
    ax.set_xlabel("t [s]")
    ax.set_title(panel_title, loc="left", fontsize=14)

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
    validate_results_base(FORWARD_RESULTS_BASE, reverse_time=False)
    validate_results_base(BACKWARD_RESULTS_BASE, reverse_time=True)
    networks = load_networks()
    panel_specs = [
        {
            "key": "merge_merge",
            "title": "(A)",
            "curve_indices": [0],
            "backward_index": None,
            "y_label": "Entropy",
        },
        {
            "key": "merge_split",
            "title": "(B)",
            "curve_indices": None,
            "backward_index": None,
            "y_label": "Entropy",
        },
        {
            "key": "split_merge",
            "title": "(C)",
            "curve_indices": None,
            "backward_index": None,
            "y_label": "Entropy / Limit Statistic",
        },
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
            results_base=FORWARD_RESULTS_BASE,
            window=WINDOW,
            reverse_time=False,
        )
        for key in networks
    }
    backward_entropy = {
        key: load_entropy_curves(
            key,
            results_base=BACKWARD_RESULTS_BASE,
            window=WINDOW,
            reverse_time=True,
        )
        for key in networks
    }
    limit_curves = {
        "split_merge": extract_limit_array(
            load_or_compute_limit_payload(
                motif_name="split_merge",
                network=networks["split_merge"],
                window=WINDOW,
            )
        )
    }

    reference_key = next(
        (key for key, curves in forward_entropy.items() if len(curves) > 0),
        None,
    )
    all_lambdas = (
        [curve["lambda"] for curve in forward_entropy[reference_key]]
        if reference_key is not None
        else []
    )
    lambda_colors = {
        lamda: color
        for lamda, color in zip(
            all_lambdas,
            auxiliary_functions.generate_plasma_colors(len(all_lambdas)),
        )
    }
    legend_entry_count = len(all_lambdas) + 1 + (1 if limit_curves else 0)
    fig_width = max(12.0, 4.0 * len(panel_specs), 1.2 * legend_entry_count + 4.0)

    fig = plt.figure(figsize=(fig_width, 5.2))
    gs = fig.add_gridspec(1, 3)
    axes = np.atleast_1d(gs.subplots())
    inset_cmap = make_inset_cmap()

    for ax, spec in zip(axes, panel_specs):
        key = spec["key"]
        curve_indices = spec["curve_indices"]
        selected_curves = select_curves(forward_entropy[key], curve_indices)
        backward_candidates = select_curves(
            backward_entropy[key],
            [spec["backward_index"]] if spec["backward_index"] is not None else None,
        )
        backward_curve = select_single_curve(backward_candidates)
        plot_network_panel(
            ax=ax,
            forward_curves=selected_curves,
            backward_curve=backward_curve,
            matrices=interval_matrices[key],
            panel_title=spec["title"],
            lambda_colors=lambda_colors,
            time_intervals=time_intervals,
            inset_positions=inset_positions,
            inset_cmap=inset_cmap,
            limit_curve=limit_curves.get(key),
        )
        ax.set_ylabel(spec["y_label"])

    legend_handles = [
        Line2D([0], [0], color=lambda_colors[lamda], linewidth=1.8, label=format_lambda_label(lamda))
        for lamda in all_lambdas
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color=BACKWARD_CURVE_STYLE["color"],
            linewidth=BACKWARD_CURVE_STYLE["linewidth"],
            label="Backward Entropy",
        )
    )
    if limit_curves:
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
        fontsize="x-small",
    )

    fig.subplots_adjust(left=0.07, right=0.995, top=0.92, bottom=0.17, wspace=0.18)
    fig.savefig(OUTPUT_PATH, format="pdf", dpi=300, bbox_inches="tight")
    if "agg" not in plt.get_backend().lower():
        plt.show()


if __name__ == "__main__":
    main()
