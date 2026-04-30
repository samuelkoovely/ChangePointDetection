from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from primary_school_ruptures_defaults import (
    build_primary_school_ruptures_results_path,
)


BASE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = BASE_DIR / "figures"
DEFAULT_RESULTS_PATH = build_primary_school_ruptures_results_path(BASE_DIR)
DEFAULT_OUTPUT_PATH = FIGURES_DIR / "fig_ruptures_primary_school.pdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the primary-school entropy signal with the detected ruptures "
            "change points for each tested penalty term."
        )
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="Path to the saved ruptures_results.pkl file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output figure path.",
    )
    parser.add_argument(
        "--ncols",
        type=int,
        default=2,
        help="Number of subplot columns in the multipanel layout.",
    )
    return parser.parse_args()


def resolve_existing_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    candidates = [path, BASE_DIR / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return BASE_DIR / path


def load_results(results_path: str | Path) -> dict:
    resolved_path = resolve_existing_path(results_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Missing results file: {resolved_path}")

    with open(resolved_path, "rb") as handle:
        return pickle.load(handle)


def compute_layout(n_panels: int, ncols: int) -> tuple[int, int]:
    if n_panels <= 0:
        raise ValueError("At least one panel is required.")
    if ncols <= 0:
        raise ValueError("ncols must be strictly positive.")

    ncols = min(ncols, n_panels)
    nrows = math.ceil(n_panels / ncols)
    return nrows, ncols


def panel_title(result: dict) -> str:
    return (
        f"pen = {float(result['penalty']):.3g}\n"
        f"{int(result['num_change_points'])} change points"
    )


def main() -> None:
    args = parse_args()
    results = load_results(args.results_path)

    penalty_results = list(results["lambda_results"])
    if len(penalty_results) == 0:
        raise ValueError("No penalty results found in the saved ruptures output.")

    signal = np.asarray(results["signal_array"], dtype=float)
    t_hours = np.asarray(results["t_samples"], dtype=float) / 3600.0

    if signal.ndim != 1:
        raise ValueError(f"Expected a 1D signal, got shape {signal.shape}.")
    if len(signal) != len(t_hours):
        raise ValueError("signal_array and t_samples must have the same length.")

    nrows, ncols = compute_layout(len(penalty_results), args.ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.2 * ncols, 2.8 * nrows),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_1d(axes).ravel()

    signal_name = str(results.get("signal_name", "Entropy"))
    signal_units = results.get("signal_units")
    signal_color = "C0"
    cp_color = "red"

    for ax, result in zip(axes, penalty_results):
        ax.plot(t_hours, signal, color=signal_color, linewidth=1.5)

        cp_times = np.asarray(result["change_point_t_hours"], dtype=float)
        if cp_times.size > 0:
            for cp_time in cp_times:
                ax.axvline(
                    cp_time,
                    color=cp_color,
                    linestyle="--",
                    linewidth=1.1,
                    alpha=0.9,
                )
            cp_indices = np.asarray(result["change_point_indices"], dtype=int)
            ax.scatter(
                cp_times,
                signal[cp_indices],
                color=cp_color,
                s=14,
                zorder=3,
            )

        ax.set_title(panel_title(result))
        ax.set_xlabel("Time (hours)")
        ax.set_box_aspect(1)

    for ax in axes[len(penalty_results):]:
        ax.set_visible(False)

    for ax in axes[::ncols]:
        if ax.get_visible():
            ylabel = signal_name
            if signal_units:
                ylabel = f"{ylabel} ({signal_units})"
            ax.set_ylabel(ylabel)

    legend_handles = [
        Line2D([0], [0], color=signal_color, linewidth=1.5, label=f"{signal_name} signal"),
        Line2D(
            [0],
            [0],
            color=cp_color,
            linewidth=1.1,
            linestyle="--",
            label="Detected change point",
        ),
    ]

    lamda = float(results["lamda"])
    window_minutes = float(results["window_minutes"])
    direction = results.get("direction", "forward")
    fig.suptitle(
        f"Primary-school day 1 ({direction})\n"
        f"{signal_name}\n"
        f"$\\lambda$ = {lamda:.2e}, window = {window_minutes:g} min",
        y=0.995,
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        frameon=False,
    )

    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format="pdf", dpi=300, bbox_inches="tight")
    print(args.output)
    plt.close(fig)


if __name__ == "__main__":
    main()
