from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import auxiliary_functions


OUTPUT_BASE = Path("./gridsearch_results/primaryschool_day1")
FIGURES_DIR = Path("./figures")
OUTPUT_FIGURE = FIGURES_DIR / "fig_gridsearch_primaryschool.pdf"
DEFAULT_WINDOWS_SECONDS = [120.0, 1800.0, 3600.0]
DEFAULT_LAMBDAS = np.logspace(-5, 0, 10)
SELECTED_LAMBDAS = DEFAULT_LAMBDAS[::2]


def load_metadata(base: Path = OUTPUT_BASE) -> dict | None:
    """
    Load metadata produced by `primary_school_compute.py` if available.
    """
    metadata_path = base / "metadata.pkl"
    if not metadata_path.exists():
        return None

    with open(metadata_path, "rb") as handle:
        return pickle.load(handle)


def get_windows_seconds(metadata: dict | None) -> list[float]:
    """
    Return the window lengths to plot, in seconds.
    """
    if metadata is None:
        return list(DEFAULT_WINDOWS_SECONDS)
    return [float(window) for window in metadata["windows_seconds"]]


def get_selected_lambdas(metadata: dict | None) -> np.ndarray:
    """
    Return the subset of lambdas shown in the figure.
    """
    if metadata is None:
        return np.asarray(SELECTED_LAMBDAS, dtype=float)

    all_lambdas = np.asarray(metadata["lambdas"], dtype=float)
    return all_lambdas[::2]


def load_signal_payload(
    window_seconds: float,
    lamda: float,
    base: Path = OUTPUT_BASE,
) -> dict:
    """
    Load one saved primary-school signal payload.
    """
    lamda_key = f"{float(lamda):.11f}"
    signal_path = base / "window_S_selected" / str(int(window_seconds)) / f"window_S{lamda_key}"

    if not signal_path.exists():
        raise FileNotFoundError(
            f"Missing signal file {signal_path}. Run primary_school_compute.py first."
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


def window_title(window_seconds: float) -> str:
    """
    Format a human-readable window label.
    """
    minutes = float(window_seconds) / 60.0
    if minutes.is_integer():
        return f"{int(minutes)} min window"
    return f"{minutes:g} min window"


def main() -> None:
    metadata = load_metadata()
    windows_seconds = get_windows_seconds(metadata)
    selected_lambdas = get_selected_lambdas(metadata)
    colors = auxiliary_functions.generate_plasma_colors(len(selected_lambdas))

    fig, axes = plt.subplots(1, len(windows_seconds), figsize=(11, 4), sharey=False)
    if len(windows_seconds) == 1:
        axes = [axes]

    for ax, window_seconds in zip(axes, windows_seconds):
        for color, lamda in zip(colors, selected_lambdas):
            payload = load_signal_payload(window_seconds=window_seconds, lamda=lamda)
            t_samples = np.asarray(payload["t_samples"], dtype=float) / 3600.0
            signal = extract_signal_array(payload, lamda=lamda)
            ax.plot(
                t_samples,
                signal,
                color=color,
                alpha=0.8,
                label=f"$\\lambda$ = {lamda:.5g}",
            )

        ax.set_title(window_title(window_seconds))
        ax.set_xlabel("Time (hours)")

    axes[0].set_ylabel("Window Entropy")
    axes[0].legend(loc="best", fontsize="small")

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FIGURE, format="pdf", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()
