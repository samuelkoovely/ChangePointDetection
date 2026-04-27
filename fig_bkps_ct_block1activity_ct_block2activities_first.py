from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from fig_bkps_ct_block_activity_common import (
    BASE_DIR,
    get_best_signal_metadata,
    get_signal_path,
    get_signals_outdir,
    load_pickle,
    resolve_dataset_path,
    resolve_existing_path,
)


DEFAULT_BLOCK1_RESULTS_PATH = (
    BASE_DIR / "gridsearch_results/ct_block1activity/gridsearch_results.pkl"
)
DEFAULT_BLOCK2_RESULTS_PATH = (
    BASE_DIR / "gridsearch_results/ct_block2activities/gridsearch_results.pkl"
)
DEFAULT_BLOCK1_DATASET_PATH = Path("data/ct_block1activity.pkl")
DEFAULT_BLOCK2_DATASET_PATH = Path("data/ct_block2activities.pkl")
DEFAULT_OUTPUT_PATH = (
    BASE_DIR / "figures" / "fig_bkps_ct_block1activity_ct_block2activities_first.pdf"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the first continuous-time block-1 sample and the first "
            "continuous-time block-2 sample using each experiment's best "
            "lambda/window pair."
        )
    )
    parser.add_argument(
        "--block1-results-path",
        type=Path,
        default=None,
        help="Optional grid-search summary pickle for ct_block1activity.",
    )
    parser.add_argument(
        "--block2-results-path",
        type=Path,
        default=None,
        help="Optional grid-search summary pickle for ct_block2activities.",
    )
    parser.add_argument(
        "--block1-dataset-path",
        type=Path,
        default=None,
        help="Optional dataset pickle override for ct_block1activity.",
    )
    parser.add_argument(
        "--block2-dataset-path",
        type=Path,
        default=None,
        help="Optional dataset pickle override for ct_block2activities.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Optional file path to save the rendered figure.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=250,
        help="Raster resolution used when saving the figure.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively after rendering.",
    )
    return parser.parse_args()


def resolve_results_path(
    requested_path: Path | None,
    default_path: Path,
    label: str,
) -> Path:
    if requested_path is not None:
        resolved_path = resolve_existing_path(requested_path)
        if resolved_path.exists():
            return resolved_path
        raise FileNotFoundError(f"Could not find {label} results at {resolved_path}.")

    if default_path.exists():
        return default_path
    raise FileNotFoundError(f"Could not find {label} results at {default_path}.")


def load_panel_data(
    *,
    results_path: Path,
    default_dataset_path: Path,
    dataset_path_override: Path | None,
) -> dict[str, Any]:
    results = load_pickle(results_path)
    dataset_path = resolve_dataset_path(
        results,
        default_dataset_path=default_dataset_path,
        dataset_path_override=dataset_path_override,
    )
    dataset = load_pickle(dataset_path)

    if len(dataset) == 0:
        raise ValueError(f"Dataset {dataset_path} is empty.")

    best_lamda, best_window, predicted_change_points, sample_names = get_best_signal_metadata(results)
    if len(predicted_change_points) == 0:
        raise ValueError(f"No predicted change points stored in {results_path}.")

    sample_index = 0
    sample_name = sample_names[sample_index] or f"sample_{sample_index}"
    signal = load_pickle(
        get_signal_path(
            signals_outdir=get_signals_outdir(results, results_path),
            sample_name=sample_name,
            lamda=best_lamda,
            window=best_window,
            reverse_time=bool(results.get("reverse_time", False)),
        )
    )

    signal_values = np.asarray(signal["signal"], dtype=float)
    x_values = np.asarray(
        signal.get("t_samples", signal.get("k_samples", np.arange(len(signal_values)))),
        dtype=float,
    )

    return {
        "x_values": x_values,
        "signal_values": signal_values,
        "true_change_points": [float(bkp) for bkp in dataset[sample_index]["bkps"]],
        "predicted_change_points": [float(bkp) for bkp in predicted_change_points[sample_index]],
    }


def draw_panel(ax: plt.Axes, panel_data: dict[str, Any], panel_label: str) -> None:
    x_values = panel_data["x_values"]
    signal_values = panel_data["signal_values"]

    ax.plot(x_values, signal_values, color="C0")
    ax.set_title(panel_label, loc="left")
    ax.set_ylabel("signal")

    if signal_values.size == 0:
        ax.text(
            0.5,
            0.5,
            "Empty signal",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    ymin = float(np.min(signal_values))
    ymax = float(np.max(signal_values))
    if np.isclose(ymin, ymax):
        pad = max(abs(ymin) * 0.05, 1e-6)
        ymin -= pad
        ymax += pad

    for bkp in panel_data["true_change_points"]:
        ax.vlines(float(bkp), ymin=ymin, ymax=ymax, color="black")
    for pred_cp in panel_data["predicted_change_points"]:
        ax.vlines(float(pred_cp), ymin=ymin, ymax=ymax, color="red", linestyles="dashed")


def main() -> None:
    args = parse_args()

    block1_results_path = resolve_results_path(
        args.block1_results_path,
        DEFAULT_BLOCK1_RESULTS_PATH,
        "ct_block1activity",
    )
    block2_results_path = resolve_results_path(
        args.block2_results_path,
        DEFAULT_BLOCK2_RESULTS_PATH,
        "ct_block2activities",
    )

    block1_panel = load_panel_data(
        results_path=block1_results_path,
        default_dataset_path=DEFAULT_BLOCK1_DATASET_PATH,
        dataset_path_override=args.block1_dataset_path,
    )
    block2_panel = load_panel_data(
        results_path=block2_results_path,
        default_dataset_path=DEFAULT_BLOCK2_DATASET_PATH,
        dataset_path_override=args.block2_dataset_path,
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 5.4), sharex=False)
    axes = np.atleast_1d(axes)

    draw_panel(axes[0], block1_panel, "(A)")
    draw_panel(axes[1], block2_panel, "(B)")
    axes[-1].set_xlabel("time")

    legend_handles = [
        Line2D([0], [0], color="C0", linewidth=1.5, label="Signal"),
        Line2D([0], [0], color="black", linewidth=1.5, label="Change point"),
        Line2D(
            [0],
            [0],
            color="red",
            linewidth=1.5,
            linestyle="dashed",
            label="Predicted change point",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output_path, dpi=int(args.dpi), bbox_inches="tight")
        print(args.output_path)

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
