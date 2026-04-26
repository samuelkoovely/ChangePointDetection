from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import auxiliary_functions


SIGNAL_OUTPUT_BASE = Path("./gridsearch_results/primaryschool_day1")
LIMIT_OUTPUT_BASE = Path("./gridsearch_results/primaryschool_day1_limit")
FIGURES_DIR = Path("./figures")
OUTPUT_FIGURE = FIGURES_DIR / "fig_gridsearch_primaryschool_limit.pdf"
DEFAULT_WINDOWS_SECONDS = [120.0, 1800.0, 3600.0]
DEFAULT_LAMBDAS = np.logspace(-5, 0, 10)
SELECTED_LAMBDAS = DEFAULT_LAMBDAS[::2]


def load_metadata(base: Path) -> dict | None:
    """
    Load plotting metadata if it exists.
    """

    metadata_path = base / "metadata.pkl"
    if not metadata_path.exists():
        return None

    with open(metadata_path, "rb") as handle:
        return pickle.load(handle)


def get_windows_seconds(signal_metadata: dict | None) -> list[float]:
    """
    Return the window lengths to plot, in seconds.
    """

    if signal_metadata is None:
        return list(DEFAULT_WINDOWS_SECONDS)
    return [float(window) for window in signal_metadata["windows_seconds"]]


def get_selected_lambdas(signal_metadata: dict | None) -> np.ndarray:
    """
    Return the subset of lambdas shown in the figure.
    """

    if signal_metadata is None:
        return np.asarray(SELECTED_LAMBDAS, dtype=float)

    all_lambdas = np.asarray(signal_metadata["lambdas"], dtype=float)
    return all_lambdas[::2]


def load_signal_payload(
    window_seconds: float,
    lamda: float,
    base: Path = SIGNAL_OUTPUT_BASE,
) -> dict:
    """
    Load one saved primary-school entropy payload.
    """

    lamda_key = f"{float(lamda):.11f}"
    signal_path = base / "window_S_selected" / str(int(window_seconds)) / f"window_S{lamda_key}"

    if not signal_path.exists():
        raise FileNotFoundError(
            f"Missing signal file {signal_path}. Run primary_school_compute.py first."
        )

    with open(signal_path, "rb") as handle:
        return pickle.load(handle)


def load_limit_payload(
    window_seconds: float,
    base: Path = LIMIT_OUTPUT_BASE,
) -> dict:
    """
    Load one saved primary-school limit payload.
    """

    signal_path = base / "window_limit_selected" / str(int(window_seconds)) / "window_limit.pkl"

    if not signal_path.exists():
        raise FileNotFoundError(
            f"Missing limit file {signal_path}. Run primary_school_limit.py first."
        )

    with open(signal_path, "rb") as handle:
        return pickle.load(handle)


def extract_signal_array(payload: dict, lamda: float) -> np.ndarray:
    """
    Extract the entropy signal from either the new or compatibility payload layout.
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


def window_title(window_seconds: float) -> str:
    """
    Format a human-readable window label.
    """

    minutes = float(window_seconds) / 60.0
    if minutes.is_integer():
        return f"{int(minutes)} min window"
    return f"{minutes:g} min window"


def format_lambda_label(lamda: float) -> str:
    """
    Format lambda values consistently in scientific notation for the legend.
    """

    return f"$\\lambda$ = {lamda:.2e}"


def main() -> None:
    signal_metadata = load_metadata(SIGNAL_OUTPUT_BASE)
    windows_seconds = get_windows_seconds(signal_metadata)
    selected_lambdas = get_selected_lambdas(signal_metadata)
    colors = auxiliary_functions.generate_plasma_colors(len(selected_lambdas))

    fig, axes = plt.subplots(1, len(windows_seconds), figsize=(14, 5.5), sharey=False)
    if len(windows_seconds) == 1:
        axes = [axes]

    for panel_idx, (ax, window_seconds) in enumerate(zip(axes, windows_seconds)):
        for color, lamda in zip(colors, selected_lambdas):
            payload = load_signal_payload(window_seconds=window_seconds, lamda=lamda)
            t_samples = np.asarray(payload["t_samples"], dtype=float) / 3600.0
            signal = extract_signal_array(payload, lamda=lamda)
            ax.plot(
                t_samples,
                signal,
                color=color,
                alpha=0.8,
                label=format_lambda_label(lamda),
            )

        limit_payload = load_limit_payload(window_seconds=window_seconds)
        limit_curve = extract_limit_array(limit_payload)

        ax.plot(
            limit_curve[:, 0] / 3600.0,
            limit_curve[:, 1],
            color="black",
            linestyle="--",
            linewidth=1.8,
            label="Upper Bound" if panel_idx == 0 else None,
        )

        ax.set_title(window_title(window_seconds))
        ax.set_xlabel("Time (hours)")
        ax.set_box_aspect(1)

    axes[0].set_ylabel("Entropy")

    legend_handles = list(axes[0].lines)
    legend_labels = [line.get_label() for line in legend_handles]
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.02, 0.96, 0.08),
        mode="expand",
        ncol=len(legend_handles),
        fontsize="medium",
        frameon=False,
        borderaxespad=0.0,
    )

    fig.tight_layout(rect=(0, 0.12, 1, 1))
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIGURE, format="pdf", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
