from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.sparse.csgraph import connected_components

import auxiliary_functions
from signal_generation import compute_signals_for_lambdas_prepared, prepare_signal_sample


RESULTS_BASE_CANDIDATES = (
    Path("gridsearch_results/motifs_f"),
    Path("gridsearch_results/motifs"),
    Path("gridsearch_results/motifs_run1"),
    Path("gridsearch_results/motifs_b"),
)
DEFAULT_WINDOWS = [1.0, 5.0, 10.0]
DEFAULT_OUTPUT_DIR = Path("./figures")
MOTIF_NAMES = ("merge_merge", "merge_split", "split_merge")
DEFAULT_CURVE_INDICES = [5, 6, 7, 8]
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot local motif entropy for one or more window lengths. Missing "
            "window signals are computed on demand."
        )
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        type=float,
        default=DEFAULT_WINDOWS,
        help="Window lengths to render in seconds.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the output PDFs will be written.",
    )
    return parser.parse_args()


def window_key(window: float) -> str:
    return f"{float(window):g}"


def format_lambda_label(lamda: float) -> str:
    return f"$\\lambda$ = {float(lamda):.2e}"


def entropy_subdir(reverse_time: bool) -> str:
    return "window_S_rev" if reverse_time else "window_S"


def signal_filename(lamda: float) -> str:
    return f"window_S{float(lamda):.11f}"


def output_path_for_window(window: float, output_dir: Path, total_windows: int) -> Path:
    if total_windows == 1:
        return output_dir / "fig_local_entropy.pdf"
    return output_dir / f"fig_local_entropy_window_{window_key(window)}.pdf"


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_metadata(motif_name: str, results_base: Path) -> dict:
    metadata_path = results_base / motif_name / "metadata.pkl"
    return load_pickle(metadata_path)


def resolve_results_base_for_direction(
    reverse_time: bool,
    allow_missing: bool = False,
) -> Path | None:
    expected_subdir = entropy_subdir(reverse_time)

    for candidate in RESULTS_BASE_CANDIDATES:
        if not candidate.exists():
            continue
        if all(
            (candidate / motif_name / "metadata.pkl").exists()
            and (candidate / motif_name / expected_subdir).exists()
            for motif_name in MOTIF_NAMES
        ):
            return candidate

    if allow_missing:
        return None

    raise FileNotFoundError(
        f"Could not find {'backward' if reverse_time else 'forward'} motif results. "
        "Checked: "
        + ", ".join(str(candidate) for candidate in RESULTS_BASE_CANDIDATES)
    )


def load_networks(results_base: Path) -> dict[str, object]:
    networks = {}
    for motif_name in MOTIF_NAMES:
        metadata = load_metadata(motif_name, results_base=results_base)
        networks[motif_name] = load_pickle(Path(metadata["network_path"]))
    return networks


def compute_interval_matrices(network, intervals):
    return [
        network.compute_static_adjacency_matrix(start_time=start, end_time=end).toarray()
        for start, end in intervals
    ]


def extract_signal_array(payload: dict, lamda: float) -> np.ndarray:
    if "signal_array" in payload:
        return np.asarray(payload["signal_array"], dtype=float)

    signal = payload["signal"]
    if isinstance(signal, dict):
        return np.asarray(signal[f"{float(lamda):.11f}"], dtype=float)

    return np.asarray(signal, dtype=float)


def signal_payload_path(
    motif_name: str,
    results_base: Path,
    window: float,
    lamda: float,
    reverse_time: bool = False,
) -> Path:
    return (
        results_base
        / motif_name
        / entropy_subdir(reverse_time)
        / window_key(window)
        / signal_filename(lamda)
    )


def save_window_signal_payload(signal_result: dict, motif_output_dir: Path) -> None:
    lamda = float(signal_result["lamda"])
    window = float(signal_result["window"])
    lamda_key = f"{lamda:.11f}"
    signal_array = np.asarray(signal_result["signal"], dtype=float)
    reverse_time = bool(signal_result.get("reverse_time", False))

    payload = {
        "lamda": lamda_key,
        "lamda_float": lamda,
        "window": window,
        "k_samples": np.asarray(signal_result["k_samples"], dtype=int),
        "t_samples": np.asarray(signal_result["t_samples"], dtype=float),
        "signal": signal_array,
        "signal_array": signal_array,
        "reverse_time": reverse_time,
        "direction": signal_result.get(
            "direction",
            "backward" if reverse_time else "forward",
        ),
    }

    outdir = motif_output_dir / entropy_subdir(reverse_time) / window_key(window)
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / f"window_S{lamda_key}").open("wb") as handle:
        pickle.dump(payload, handle)


def ensure_entropy_curves(
    motif_name: str,
    results_base: Path,
    window: float,
    reverse_time: bool = False,
    network=None,
) -> np.ndarray:
    metadata = load_metadata(motif_name, results_base=results_base)
    lambdas = np.sort(np.asarray(metadata["lambdas"], dtype=float))
    missing_lambdas = [
        lamda
        for lamda in lambdas
        if not signal_payload_path(
            motif_name=motif_name,
            results_base=results_base,
            window=window,
            lamda=float(lamda),
            reverse_time=reverse_time,
        ).exists()
    ]

    if len(missing_lambdas) == 0:
        return lambdas

    if network is None:
        network = load_pickle(Path(metadata["network_path"]))

    print(
        f"Computing {'backward' if reverse_time else 'forward'} local entropy for "
        f"{motif_name}, window={window_key(window)}"
    )
    sample_fraction = float(metadata.get("sample_fraction", 1.0))
    window_backend = metadata.get("window_backend", "segment_tree")
    p0 = np.ones(network.num_nodes, dtype=float) / float(network.num_nodes)
    prepared = prepare_signal_sample(
        net=network,
        windows=[float(window)],
        sample_fraction=sample_fraction,
        p0=p0,
        reverse_time=reverse_time,
    )
    signals_by_lambda = compute_signals_for_lambdas_prepared(
        prepared=prepared,
        lambdas=lambdas,
        use_linear_approx=False,
        lin_t_s=10,
        window_backend=window_backend,
    )
    motif_output_dir = results_base / motif_name
    for window_results in signals_by_lambda.values():
        save_window_signal_payload(
            signal_result=window_results[float(window)],
            motif_output_dir=motif_output_dir,
        )

    return lambdas


def load_entropy_curves(
    motif_name: str,
    results_base: Path,
    window: float,
    reverse_time: bool = False,
    network=None,
) -> list[list[np.ndarray]]:
    lambdas = ensure_entropy_curves(
        motif_name=motif_name,
        results_base=results_base,
        window=window,
        reverse_time=reverse_time,
        network=network,
    )

    curves = []
    for lamda in lambdas:
        filepath = signal_payload_path(
            motif_name=motif_name,
            results_base=results_base,
            window=window,
            lamda=float(lamda),
            reverse_time=reverse_time,
        )
        data = load_pickle(filepath)
        curves.append(
            [
                np.asarray(data["t_samples"], dtype=float),
                extract_signal_array(data, lamda=lamda),
            ]
        )
    return curves


def limit_payload_path(motif_name: str, results_base: Path, window: float) -> Path:
    return (
        results_base
        / motif_name
        / "window_limit_selected"
        / window_key(window)
        / "window_limit.pkl"
    )


def compute_component_log_sum(network, start_time: float, window_seconds: float) -> float:
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


def compute_window_limit_payload(network, motif_name: str, window: float) -> dict:
    k_samples = np.flatnonzero(
        np.asarray(network.times, dtype=float) < network.times[-1] - float(window)
    ).astype(int)
    start_times = np.asarray(network.times[k_samples], dtype=float)
    t_samples = start_times + 0.5 * float(window)
    values = np.empty(len(start_times), dtype=float)

    for idx, start_time in enumerate(start_times):
        values[idx] = compute_component_log_sum(
            network=network,
            start_time=float(start_time),
            window_seconds=float(window),
        )
        if (idx + 1) % 250 == 0 or idx + 1 == len(start_times):
            print(
                f"{motif_name} limit, window={window_key(window)}: "
                f"{idx + 1}/{len(start_times)} samples"
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


def load_or_compute_limit_payload(
    motif_name: str,
    network,
    results_base: Path,
    window: float,
) -> dict:
    payload_path = limit_payload_path(
        motif_name=motif_name,
        results_base=results_base,
        window=window,
    )
    if payload_path.exists():
        return load_pickle(payload_path)

    payload = compute_window_limit_payload(
        network=network,
        motif_name=motif_name,
        window=window,
    )
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    with payload_path.open("wb") as handle:
        pickle.dump(payload, handle)
    return payload


def extract_limit_array(payload: dict) -> np.ndarray:
    if "time_component_log_sums" in payload:
        return np.asarray(payload["time_component_log_sums"], dtype=float)

    return np.column_stack(
        (
            np.asarray(payload["t_samples"], dtype=float),
            np.asarray(payload["signal_array"], dtype=float),
        )
    )


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
):
    for color, curve in zip(forward_colors, forward_curves):
        ax.plot(curve[0], curve[1], color=color, alpha=1)

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


def render_window_figure(
    window: float,
    forward_results_base: Path,
    output_path: Path,
) -> None:
    networks = load_networks(results_base=forward_results_base)
    panel_specs = [
        {"key": "merge_merge", "title": "(A)"},
        {"key": "merge_split", "title": "(B)"},
        {"key": "split_merge", "title": "(C)"},
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
            window=window,
            reverse_time=False,
            network=networks[key],
        )
        for key in networks
    }

    reference_motif = panel_specs[0]["key"]
    forward_reference_metadata = load_metadata(
        reference_motif,
        results_base=forward_results_base,
    )
    forward_lambdas = np.sort(
        np.asarray(forward_reference_metadata["lambdas"], dtype=float)
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
        plot_network_panel(
            ax=ax,
            forward_curves=selected_curves,
            matrices=interval_matrices[key],
            panel_title=spec["title"],
            forward_colors=forward_colors,
            time_intervals=time_intervals,
            inset_positions=inset_positions,
            inset_cmap=inset_cmap,
        )

    axes[0].set_ylabel("Entropy")
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    legend_handles = [
        Line2D([0], [0], color=color, linewidth=1.8, label=format_lambda_label(lamda))
        for color, lamda in zip(forward_colors, selected_forward_lambdas)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(legend_handles),
        fontsize="small",
    )
    fig.suptitle(f"{window_key(window)} s window", fontsize=14, y=0.98)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.89, bottom=0.17, wspace=0.18)
    fig.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
    if "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    windows = list(dict.fromkeys(float(window) for window in args.windows))

    forward_results_base = resolve_results_base_for_direction(reverse_time=False)

    for window in windows:
        render_window_figure(
            window=window,
            forward_results_base=forward_results_base,
            output_path=output_path_for_window(
                window=window,
                output_dir=args.output_dir,
                total_windows=len(windows),
            ),
        )


if __name__ == "__main__":
    main()
