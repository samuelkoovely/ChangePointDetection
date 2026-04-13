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


NETWORK_PATH = Path("./data/split_merge.pkl")
SIGNAL_RESULTS_CANDIDATES = (
    Path("./gridsearch_results/motifs_f/split_merge"),
    Path("./gridsearch_results/motifs/split_merge"),
    Path("./gridsearch_results/motifs_run1/split_merge"),
)
LIMIT_RESULTS_CANDIDATES = (
    Path("./gridsearch_results/split_merge_limit"),
)
DEFAULT_WINDOWS = [1.0, 5.0, 10.0]
DEFAULT_LAMBDAS = np.asarray([0.1, 1.0, 10.0], dtype=float)
OUTPUT_PATH = Path("./figures/fig_gridsearch_split_merge_limit.pdf")
LIMIT_STYLE = {
    "color": "black",
    "linestyle": "--",
    "linewidth": 1.8,
    "zorder": 5,
}
TIME_INTERVALS = [(0, 100), (100, 200), (200, 300)]
INSET_POSITIONS = [0.06, 0.37, 0.68]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot local split-merge entropy curves against the connected-component "
            "upper bound for one or more window lengths."
        )
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
        "--windows",
        nargs="+",
        type=float,
        default=DEFAULT_WINDOWS,
        help="Window lengths to plot in seconds.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Path to the output PDF.",
    )
    return parser.parse_args()


def load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_split_merge_network(signal_metadata: dict | None = None):
    network_path = NETWORK_PATH
    if signal_metadata is not None and signal_metadata.get("network_path") is not None:
        network_path = Path(signal_metadata["network_path"])
    return load_pickle(Path(network_path))


def compute_interval_matrices(network, intervals):
    return [
        network.compute_static_adjacency_matrix(start_time=start, end_time=end).toarray()
        for start, end in intervals
    ]


def window_key(window: float) -> str:
    """
    Format a window value for directory names.
    """

    return f"{float(window):g}"


def resolve_signal_base(requested_base: Path | None = None) -> Path:
    """
    Find the forward split-merge entropy results directory.
    """

    if requested_base is not None:
        metadata_path = requested_base / "metadata.pkl"
        if metadata_path.exists() and (requested_base / "window_S").exists():
            return requested_base
        raise FileNotFoundError(
            f"Explicit signal base {requested_base} is missing metadata.pkl or window_S."
        )

    for candidate in SIGNAL_RESULTS_CANDIDATES:
        metadata_path = candidate / "metadata.pkl"
        if metadata_path.exists() and (candidate / "window_S").exists():
            return candidate

    raise FileNotFoundError(
        "Could not find forward split-merge entropy results. Checked: "
        + ", ".join(str(path) for path in SIGNAL_RESULTS_CANDIDATES)
    )


def resolve_limit_base(signal_base: Path, requested_base: Path | None = None) -> Path:
    """
    Find the directory containing saved limit curves.
    """

    if requested_base is not None:
        if (requested_base / "window_limit_selected").exists():
            return requested_base
        raise FileNotFoundError(
            f"Explicit limit base {requested_base} is missing window_limit_selected."
        )

    candidates = [*LIMIT_RESULTS_CANDIDATES, signal_base]

    for candidate in candidates:
        if (candidate / "window_limit_selected").exists():
            return candidate

    raise FileNotFoundError(
        "Could not find split-merge limit curves. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def load_metadata(base: Path) -> dict | None:
    """
    Load plotting metadata if it exists.
    """

    metadata_path = base / "metadata.pkl"
    if not metadata_path.exists():
        return None

    return load_pickle(metadata_path)


def resolve_signal_subdir(signal_base: Path, signal_metadata: dict | None) -> str:
    """
    Resolve the local-entropy subdirectory name.
    """

    preferred = "window_S_rev" if signal_metadata and signal_metadata.get("reverse_time") else "window_S"
    if (signal_base / preferred).exists():
        return preferred

    for candidate in ("window_S", "window_S_selected", "window_S_rev"):
        if (signal_base / candidate).exists():
            return candidate

    raise FileNotFoundError(f"Could not find signal subdirectory in {signal_base}")


def get_windows(
    signal_metadata: dict | None,
    requested_windows: list[float] | None = None,
) -> list[float]:
    """
    Return the window lengths to plot.
    """

    if requested_windows is not None:
        return list(dict.fromkeys(float(window) for window in requested_windows))

    if signal_metadata is None:
        return list(DEFAULT_WINDOWS)

    if "windows" in signal_metadata:
        return [float(window) for window in signal_metadata["windows"]]

    if "windows_seconds" in signal_metadata:
        return [float(window) for window in signal_metadata["windows_seconds"]]

    return list(DEFAULT_WINDOWS)


def save_window_signal_payload(signal_result: dict, signal_base: Path) -> None:
    lamda = float(signal_result["lamda"])
    window = float(signal_result["window"])
    lamda_key = f"{lamda:.11f}"
    signal_array = np.asarray(signal_result["signal"], dtype=float)
    reverse_time = bool(signal_result.get("reverse_time", False))
    subdir = "window_S_rev" if reverse_time else "window_S"

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

    outdir = signal_base / subdir / window_key(window)
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / f"window_S{lamda_key}").open("wb") as handle:
        pickle.dump(payload, handle)


def get_selected_lambdas(signal_metadata: dict | None) -> np.ndarray:
    """
    Return the lambda values shown in the figure.
    """

    if signal_metadata is None or "lambdas" not in signal_metadata:
        return np.asarray(DEFAULT_LAMBDAS, dtype=float)

    return np.sort(np.asarray(signal_metadata["lambdas"], dtype=float))


def ensure_signal_payloads(
    signal_base: Path,
    signal_metadata: dict | None,
    signal_subdir: str,
    windows: list[float],
    lambdas: np.ndarray,
) -> None:
    missing_pairs = [
        (float(window), float(lamda))
        for window in windows
        for lamda in lambdas
        if not (
            signal_base
            / signal_subdir
            / window_key(window)
            / f"window_S{float(lamda):.11f}"
        ).exists()
    ]

    if len(missing_pairs) == 0:
        return

    print(
        "Computing missing split-merge local entropy signals for windows "
        + ", ".join(window_key(window) for window in windows)
    )
    network = load_split_merge_network(signal_metadata)
    sample_fraction = (
        float(signal_metadata.get("sample_fraction", 1.0))
        if signal_metadata is not None
        else 1.0
    )
    window_backend = (
        signal_metadata.get("window_backend", "segment_tree")
        if signal_metadata is not None
        else "segment_tree"
    )
    reverse_time = bool(signal_metadata.get("reverse_time", False)) if signal_metadata else False
    p0 = np.ones(network.num_nodes, dtype=float) / float(network.num_nodes)
    prepared = prepare_signal_sample(
        net=network,
        windows=windows,
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
    for window_results in signals_by_lambda.values():
        for window in windows:
            save_window_signal_payload(
                signal_result=window_results[float(window)],
                signal_base=signal_base,
            )


def load_signal_payload(
    window: float,
    lamda: float,
    signal_base: Path,
    signal_subdir: str,
) -> dict:
    """
    Load one saved split-merge entropy payload.
    """

    lamda_key = f"{float(lamda):.11f}"
    signal_path = (
        signal_base
        / signal_subdir
        / window_key(window)
        / f"window_S{lamda_key}"
    )

    if not signal_path.exists():
        raise FileNotFoundError(
            f"Missing signal file {signal_path}. Run compute_entropy_motifs.py first."
        )

    return load_pickle(signal_path)


def limit_payload_path(window: float, limit_base: Path) -> Path:
    return (
        limit_base
        / "window_limit_selected"
        / window_key(window)
        / "window_limit.pkl"
    )


def compute_component_log_sum(network, start_time: float, window: float) -> float:
    adjacency = network.compute_static_adjacency_matrix(
        start_time=float(start_time),
        end_time=float(start_time) + float(window),
    ).tocsr()

    n_components, labels = connected_components(
        adjacency,
        directed=False,
        return_labels=True,
    )
    component_sizes = np.bincount(labels, minlength=n_components).astype(float)
    weights = component_sizes / float(network.num_nodes)
    return float(np.sum(weights * np.log(component_sizes)))


def compute_window_limit_payload(network, window: float) -> dict:
    k_samples = np.flatnonzero(
        np.asarray(network.times, dtype=float) < network.times[-1] - float(window)
    ).astype(int)
    t_samples = np.asarray(network.times[k_samples], dtype=float)
    values = np.empty(len(t_samples), dtype=float)

    for idx, start_time in enumerate(t_samples):
        values[idx] = compute_component_log_sum(
            network=network,
            start_time=float(start_time),
            window=float(window),
        )
        if (idx + 1) % 250 == 0 or idx + 1 == len(t_samples):
            print(
                f"split_merge limit, window={window_key(window)}: "
                f"{idx + 1}/{len(t_samples)} samples"
            )

    time_limit_array = (
        np.column_stack((t_samples, values))
        if len(t_samples) > 0
        else np.empty((0, 2), dtype=float)
    )

    return {
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


def load_or_compute_limit_payload(window: float, limit_base: Path, network) -> dict:
    """
    Load one saved split-merge limit payload.
    """

    signal_path = limit_payload_path(window=window, limit_base=limit_base)

    if signal_path.exists():
        return load_pickle(signal_path)

    payload = compute_window_limit_payload(network=network, window=window)
    signal_path.parent.mkdir(parents=True, exist_ok=True)
    with signal_path.open("wb") as handle:
        pickle.dump(payload, handle)
    return payload


def extract_signal_array(payload: dict, lamda: float) -> np.ndarray:
    """
    Extract the entropy signal from either supported payload layout.
    """

    if "signal_array" in payload:
        return np.asarray(payload["signal_array"], dtype=float)

    lamda_key = f"{float(lamda):.11f}"
    signal = payload["signal"]
    if isinstance(signal, dict):
        return np.asarray(signal[lamda_key], dtype=float)

    return np.asarray(signal, dtype=float)


def extract_limit_array(payload: dict) -> np.ndarray:
    """
    Extract the connected-component limit curve.
    """

    if "time_component_log_sums" in payload:
        return np.asarray(payload["time_component_log_sums"], dtype=float)

    return np.column_stack(
        (
            np.asarray(payload["t_samples"], dtype=float),
            np.asarray(payload["signal_array"], dtype=float),
        )
    )


def window_title(window: float) -> str:
    """
    Format a human-readable window label.
    """

    seconds = float(window)
    if seconds.is_integer():
        return f"{int(seconds)} s window"
    return f"{seconds:g} s window"


def format_lambda_label(lamda: float) -> str:
    """
    Format lambda values consistently in scientific notation for the legend.
    """

    return f"$\\lambda$ = {lamda:.2e}"


def make_inset_cmap():
    cmap = colormaps["inferno"].copy()
    cmap.set_bad(color="white")
    return cmap


def plot_window_panel(
    ax,
    window: float,
    lambdas: np.ndarray,
    colors,
    signal_base: Path,
    signal_subdir: str,
    limit_base: Path,
    network,
    matrices,
    inset_cmap,
):
    for color, lamda in zip(colors, lambdas):
        payload = load_signal_payload(
            window=window,
            lamda=lamda,
            signal_base=signal_base,
            signal_subdir=signal_subdir,
        )
        t_samples = np.asarray(payload["t_samples"], dtype=float)
        signal = extract_signal_array(payload, lamda=lamda)
        ax.plot(
            t_samples,
            signal,
            color=color,
            alpha=0.85,
        )

    limit_payload = load_or_compute_limit_payload(
        window=window,
        limit_base=limit_base,
        network=network,
    )
    limit_curve = extract_limit_array(limit_payload)
    ax.plot(limit_curve[:, 0], limit_curve[:, 1], **LIMIT_STYLE)

    ax.set_xlim(-5, 310)
    ax.set_ylim(-2, 5)
    ax.set_xlabel("t [s]")
    ax.set_title(window_title(window))
    ax.set_box_aspect(1)

    for matrix, pos, (start, end) in zip(matrices, INSET_POSITIONS, TIME_INTERVALS):
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


def main() -> None:
    args = parse_args()
    signal_base = resolve_signal_base(requested_base=args.signal_base)
    signal_metadata = load_metadata(signal_base)
    signal_subdir = resolve_signal_subdir(signal_base, signal_metadata)
    limit_base = resolve_limit_base(signal_base, requested_base=args.limit_base)

    windows = get_windows(
        signal_metadata,
        requested_windows=args.windows,
    )
    selected_lambdas = get_selected_lambdas(signal_metadata)
    ensure_signal_payloads(
        signal_base=signal_base,
        signal_metadata=signal_metadata,
        signal_subdir=signal_subdir,
        windows=windows,
        lambdas=selected_lambdas,
    )
    network = load_split_merge_network(signal_metadata)
    colors = auxiliary_functions.generate_plasma_colors(len(selected_lambdas))
    interval_matrices = compute_interval_matrices(network, TIME_INTERVALS)
    inset_cmap = make_inset_cmap()
    legend_entry_count = len(selected_lambdas) + 1
    fig_width = max(13.0, 4.0 * len(windows), 1.2 * legend_entry_count + 4.0)

    fig = plt.figure(figsize=(fig_width, 5.4))
    gs = fig.add_gridspec(1, len(windows))
    axes = np.atleast_1d(gs.subplots(sharey=True))

    for ax, window in zip(axes, windows):
        plot_window_panel(
            ax=ax,
            window=window,
            lambdas=selected_lambdas,
            colors=colors,
            signal_base=signal_base,
            signal_subdir=signal_subdir,
            limit_base=limit_base,
            network=network,
            matrices=interval_matrices,
            inset_cmap=inset_cmap,
        )

    axes[0].set_ylabel("Entropy")
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    legend_handles = [
        Line2D([0], [0], color=color, linewidth=1.8, label=format_lambda_label(lamda))
        for color, lamda in zip(colors, selected_lambdas)
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color=LIMIT_STYLE["color"],
            linestyle=LIMIT_STYLE["linestyle"],
            linewidth=LIMIT_STYLE["linewidth"],
            label="Upper bound",
        )
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(legend_handles),
        fontsize="small",
    )

    fig.subplots_adjust(left=0.07, right=0.995, top=0.92, bottom=0.17, wspace=0.18)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format="pdf", dpi=300, bbox_inches="tight")
    if "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
