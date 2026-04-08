from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.sparse.csgraph import connected_components

import auxiliary_functions


RESULTS_BASE_CANDIDATES = (
    Path("gridsearch_results/motifs_global"),
    Path("gridsearch_results/motifs_global_f"),
    Path("gridsearch_results/motifs_global_b"),
)
OUTPUT_PATH = Path("./figures/fig_global_entropy.pdf")
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


def signal_subdir(reverse_time):
    return "global_S_rev" if reverse_time else "global_S"


def signal_filename(lamda):
    return f"global_S{float(lamda):.11f}"


def direction_metadata_filename(reverse_time):
    return f"metadata_{'backward' if reverse_time else 'forward'}.pkl"


def resolve_results_base_for_direction(reverse_time, allow_missing=False):
    expected_subdir = signal_subdir(reverse_time)
    expected_metadata = direction_metadata_filename(reverse_time)

    for candidate in RESULTS_BASE_CANDIDATES:
        if not candidate.exists():
            continue
        if all(
            (candidate / motif_name / expected_subdir).exists()
            and (
                (candidate / motif_name / expected_metadata).exists()
                or (candidate / motif_name / "metadata.pkl").exists()
            )
            for motif_name in MOTIF_NAMES
        ):
            return candidate

    if allow_missing:
        return None

    raise FileNotFoundError(
        f"Could not find {'backward' if reverse_time else 'forward'} global motif results. "
        "Checked: "
        + ", ".join(str(candidate) for candidate in RESULTS_BASE_CANDIDATES)
    )


FORWARD_RESULTS_BASE = resolve_results_base_for_direction(reverse_time=False)
BACKWARD_RESULTS_BASE = resolve_results_base_for_direction(
    reverse_time=True,
    allow_missing=True,
)


def load_pickle(path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_metadata(motif_name, results_base, reverse_time=False):
    motif_dir = results_base / motif_name
    candidates = (
        motif_dir / direction_metadata_filename(reverse_time),
        motif_dir / "metadata.pkl",
    )
    for path in candidates:
        if path.exists():
            return load_pickle(path)
    raise FileNotFoundError(f"Could not find metadata for {motif_name} in {motif_dir}")


def metadata_lambdas(metadata, reverse_time=False):
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


def load_networks():
    networks = {}
    results_base = FORWARD_RESULTS_BASE if FORWARD_RESULTS_BASE is not None else BACKWARD_RESULTS_BASE
    reverse_time = False if FORWARD_RESULTS_BASE is not None else True

    for motif_name in MOTIF_NAMES:
        metadata = load_metadata(
            motif_name,
            results_base=results_base,
            reverse_time=reverse_time,
        )
        networks[motif_name] = load_pickle(Path(metadata["network_path"]))
    return networks


def compute_interval_matrices(network, intervals):
    return [
        network.compute_static_adjacency_matrix(start_time=start, end_time=end).toarray()
        for start, end in intervals
    ]


def limit_payload_path(motif_name):
    return (
        FORWARD_RESULTS_BASE
        / motif_name
        / "global_limit_selected"
        / "global_limit.pkl"
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


def load_or_compute_limit_payload(motif_name, network):
    payload_path = limit_payload_path(motif_name)
    if payload_path.exists():
        return load_pickle(payload_path)

    payload = compute_global_limit_payload(
        network=network,
        motif_name=motif_name,
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


def load_entropy_curves(motif_name, results_base, reverse_time=False):
    metadata = load_metadata(
        motif_name,
        results_base=results_base,
        reverse_time=reverse_time,
    )
    lambdas = np.sort(metadata_lambdas(metadata, reverse_time=reverse_time))
    curve_dir = results_base / motif_name / signal_subdir(reverse_time)

    curves = []
    for lamda in lambdas:
        filepath = curve_dir / signal_filename(lamda)
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
    networks = load_networks()
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
            results_base=FORWARD_RESULTS_BASE,
            reverse_time=False,
        )
        for key in networks
    }
    backward_entropy = (
        {
            key: load_entropy_curves(
                key,
                results_base=BACKWARD_RESULTS_BASE,
                reverse_time=True,
            )
            for key in networks
        }
        if BACKWARD_RESULTS_BASE is not None
        else {}
    )
    limit_curves = {
        key: extract_limit_array(
            load_or_compute_limit_payload(
                motif_name=key,
                network=networks[key],
            )
        )
        for key in networks
    }

    fig = plt.figure(figsize=(12, 4))
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
        colors = auxiliary_functions.generate_plasma_colors(len(selected_curves))
        plot_network_panel(
            ax=ax,
            forward_curves=selected_curves,
            matrices=interval_matrices[key],
            panel_title=spec["title"],
            forward_colors=colors,
            time_intervals=time_intervals,
            inset_positions=inset_positions,
            inset_cmap=inset_cmap,
            backward_curve=backward_curve,
            limit_curve=limit_curves.get(key),
        )

    axes[0].set_ylabel("Entropy")
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, format="pdf", dpi=300, bbox_inches="tight")
    if "agg" not in plt.get_backend().lower():
        plt.show()


if __name__ == "__main__":
    main()
