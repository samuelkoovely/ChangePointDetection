from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import auxiliary_functions


SIGNAL_RESULTS_CANDIDATES = (
    Path("./gridsearch_results/motifs_f/split_merge"),
    Path("./gridsearch_results/motifs/split_merge"),
    Path("./gridsearch_results/motifs_run1/split_merge"),
)
LIMIT_RESULTS_CANDIDATES = (
    Path("./gridsearch_results/split_merge_limit"),
)
DEFAULT_WINDOWS = [1.0, 3.0, 5.0]
DEFAULT_LAMBDAS = np.asarray([0.1, 1.0, 10.0], dtype=float)
OUTPUT_PATH = Path("./figures/fig_gridsearch_split_merge_limit.pdf")


def load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def window_key(window: float) -> str:
    """
    Format a window value for directory names.
    """

    return f"{float(window):g}"


def resolve_signal_base() -> Path:
    """
    Find the forward split-merge entropy results directory.
    """

    for candidate in SIGNAL_RESULTS_CANDIDATES:
        metadata_path = candidate / "metadata.pkl"
        if metadata_path.exists() and (candidate / "window_S").exists():
            return candidate

    raise FileNotFoundError(
        "Could not find forward split-merge entropy results. Checked: "
        + ", ".join(str(path) for path in SIGNAL_RESULTS_CANDIDATES)
    )


def resolve_limit_base(signal_base: Path) -> Path:
    """
    Find the directory containing saved limit curves.
    """

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


def get_windows(signal_metadata: dict | None) -> list[float]:
    """
    Return the window lengths to plot.
    """

    if signal_metadata is None:
        return list(DEFAULT_WINDOWS)

    if "windows" in signal_metadata:
        return [float(window) for window in signal_metadata["windows"]]

    if "windows_seconds" in signal_metadata:
        return [float(window) for window in signal_metadata["windows_seconds"]]

    return list(DEFAULT_WINDOWS)


def get_selected_lambdas(signal_metadata: dict | None) -> np.ndarray:
    """
    Return the lambda values shown in the figure.
    """

    if signal_metadata is None or "lambdas" not in signal_metadata:
        return np.asarray(DEFAULT_LAMBDAS, dtype=float)

    return np.sort(np.asarray(signal_metadata["lambdas"], dtype=float))


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


def load_limit_payload(window: float, limit_base: Path) -> dict:
    """
    Load one saved split-merge limit payload.
    """

    signal_path = (
        limit_base
        / "window_limit_selected"
        / window_key(window)
        / "window_limit.pkl"
    )

    if not signal_path.exists():
        raise FileNotFoundError(
            f"Missing limit file {signal_path}. Run split_merge_limit.py first."
        )

    return load_pickle(signal_path)


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


def main() -> None:
    signal_base = resolve_signal_base()
    signal_metadata = load_metadata(signal_base)
    signal_subdir = resolve_signal_subdir(signal_base, signal_metadata)
    limit_base = resolve_limit_base(signal_base)

    windows = get_windows(signal_metadata)
    selected_lambdas = get_selected_lambdas(signal_metadata)
    colors = auxiliary_functions.generate_plasma_colors(len(selected_lambdas))

    fig_width = max(10.0, 3.7 * len(windows))
    fig, axes = plt.subplots(1, len(windows), figsize=(fig_width, 4), sharey=False)
    if len(windows) == 1:
        axes = [axes]

    for panel_idx, (ax, window) in enumerate(zip(axes, windows)):
        for color, lamda in zip(colors, selected_lambdas):
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
                label=f"$\\lambda$ = {lamda:.5g}",
            )

        limit_payload = load_limit_payload(window=window, limit_base=limit_base)
        limit_curve = extract_limit_array(limit_payload)
        ax.plot(
            limit_curve[:, 0],
            limit_curve[:, 1],
            color="black",
            linestyle="--",
            linewidth=1.8,
            label="Limit Statistic" if panel_idx == 0 else None,
        )

        ax.set_title(window_title(window))
        ax.set_xlabel("Time (s)")

    axes[0].set_ylabel("Local Entropy / Limit Statistic")

    legend_handles = list(axes[0].lines)
    legend_labels = [line.get_label() for line in legend_handles]
    axes[0].legend(legend_handles, legend_labels, loc="best", fontsize="small")

    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, format="pdf", dpi=300, bbox_inches="tight")
    if "agg" not in plt.get_backend().lower():
        plt.show()


if __name__ == "__main__":
    main()
