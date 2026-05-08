from __future__ import annotations

import argparse
import os
import string
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import fig_ct_examples as ct_examples
from ct_examples_common import (
    DEFAULT_DATA_DIR,
    DEFAULT_FIGURE_DIR,
    SPECS,
    generate_sparse_sample,
    get_specs,
    load_pickle,
    sample_path,
    signal_path,
)

plt.style.use(Path(__file__).with_name("paper.mplstyle"))


DEFAULT_OUTPUT = DEFAULT_FIGURE_DIR / "fig_ct_examples_combined.pdf"
ACTIVE_STYLE = {
    "color": "tab:red",
    "linewidth": 1.4,
}
ENTROPY_STYLE = {
    "color": "tab:blue",
    "linewidth": 1.4,
    "alpha": 0.95,
}
BREAKPOINT_STYLE = {
    "color": "black",
    "linestyle": "--",
    "linewidth": 1.2,
    "alpha": 0.85,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine the sparse block-activity example plots into one figure "
            "with one row per dataset and one column per entropy window."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[spec.key for spec in SPECS],
        choices=[spec.key for spec in SPECS],
        help="Subset of sparse figures to include.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing sparse samples and signal bundles.",
    )
    parser.add_argument(
        "--signal-grid-base",
        type=Path,
        default=None,
        help=(
            "Optional base directory containing per-dataset signal-grid results "
            "folders named after the dataset keys."
        ),
    )
    parser.add_argument(
        "--dataset-results",
        nargs="+",
        default=None,
        help=(
            "Optional per-dataset signal-grid directories as "
            "'dataset=/path/to/results_dir'."
        ),
    )
    parser.add_argument(
        "--selection",
        nargs="+",
        default=None,
        help=(
            "Optional per-dataset selection as 'dataset:lambda' or "
            "'dataset:lambda:window'. "
            "When provided, that dataset row is built from the saved signal grid "
            "instead of the single-bundle pickle."
        ),
    )
    parser.add_argument(
        "--selected-windows",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Optional window list to use together with 'dataset:lambda' "
            "selection entries. If omitted, all windows available in the "
            "signal-grid metadata are loaded for that lambda."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to the combined output PDF.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution used when saving rasterized plot elements.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure after saving it.",
    )
    return parser.parse_args()


def format_window_title(window: float) -> str:
    seconds = float(window)
    if seconds.is_integer():
        return rf"$\Delta = {int(seconds)}$ s"
    return rf"$\Delta = {seconds:g}$ s"


def load_sample_for_results_dir(
    spec,
    results_dir: Path,
    data_dir: Path,
) -> dict[str, Any]:
    metadata = load_pickle(results_dir / "metadata.pkl")

    candidate_paths = []
    sample_path_raw = metadata.get("sample_path")
    if sample_path_raw is not None:
        candidate_paths.append(Path(sample_path_raw))
    candidate_paths.append(sample_path(spec, data_dir))

    seen_candidates = set()
    for candidate in candidate_paths:
        candidate = Path(candidate)
        if candidate in seen_candidates:
            continue
        seen_candidates.add(candidate)
        if candidate.exists():
            return load_pickle(candidate)

    return generate_sparse_sample(spec.key)


def load_inputs_for_spec(
    spec,
    args: argparse.Namespace,
    dataset_results: dict[str, Path],
    selections: dict[str, tuple[float, float | None]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if spec.key in selections:
        lamda, window = selections[spec.key]
        results_dir = ct_examples.resolve_results_dir(
            spec_key=spec.key,
            dataset_results=dataset_results,
            signal_grid_base=args.signal_grid_base,
        )
        sample = load_sample_for_results_dir(
            spec=spec,
            results_dir=results_dir,
            data_dir=args.data_dir,
        )
        selected_windows = (
            [float(window)]
            if window is not None
            else (
                [float(value) for value in args.selected_windows]
                if args.selected_windows is not None
                else None
            )
        )
        signal_bundle = ct_examples.build_selected_signal_bundle(
            results_dir=results_dir,
            lamda=lamda,
            windows=selected_windows,
        )
        return sample, signal_bundle

    signal_input_path = signal_path(spec, args.data_dir)
    if not signal_input_path.exists():
        raise FileNotFoundError(
            f"Missing sparse signal bundle {signal_input_path}. "
            "Run compute_ct_examples_signals.py first, or use --selection with "
            "signal-grid results."
        )

    sample_input_path = sample_path(spec, args.data_dir)
    sample = (
        load_pickle(sample_input_path)
        if sample_input_path.exists()
        else generate_sparse_sample(spec.key)
    )
    signal_bundle = load_pickle(signal_input_path)
    return sample, signal_bundle


def panel_label(index: int) -> str:
    if index < len(string.ascii_uppercase):
        return f"({string.ascii_uppercase[index]})"
    return f"({index + 1})"


def entropy_limits_for_windows(
    signal_bundle: dict[str, Any],
    windows: list[float],
) -> tuple[float, float]:
    values = []
    for window in windows:
        payload = signal_bundle["signals_by_window"][float(window)]
        values.append(np.asarray(payload["signal"], dtype=float))
    return ct_examples.pad_limits(np.concatenate(values))


def main() -> None:
    args = parse_args()
    dataset_results = ct_examples.parse_dataset_results(args.dataset_results)
    selections = ct_examples.parse_selection_entries(args.selection)
    specs = get_specs(args.datasets)

    loaded_rows = []
    for spec in specs:
        sample, signal_bundle = load_inputs_for_spec(
            spec=spec,
            args=args,
            dataset_results=dataset_results,
            selections=selections,
        )
        windows = [float(window) for window in signal_bundle["windows"]]
        loaded_rows.append(
            {
                "spec": spec,
                "sample": sample,
                "signal_bundle": signal_bundle,
                "windows": windows,
            }
        )

    n_rows = len(loaded_rows)
    n_cols = max(len(row["windows"]) for row in loaded_rows)
    fig_width = max(12.0, 4.2 * n_cols)
    fig_height = max(6.5, 3.8 * n_rows)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )

    inset_cmap = ct_examples.make_inset_cmap()
    panel_index = 0

    for row_index, row in enumerate(loaded_rows):
        spec = row["spec"]
        sample = row["sample"]
        signal_bundle = row["signal_bundle"]
        windows = row["windows"]
        net = sample["tnet"]
        breakpoint = float(sample["bkps"][0])
        t_min = float(net.times[0])
        t_max = float(net.times[-1])
        active_times, active_counts = ct_examples.compute_active_event_signal(sample)
        active_ylim = ct_examples.pad_limits(active_counts)
        entropy_ylim = entropy_limits_for_windows(signal_bundle, windows)
        inset_intervals = ct_examples.build_inset_intervals(
            t_min=t_min,
            t_max=t_max,
            breakpoint=breakpoint,
            two_interval_split=True,
        )
        interval_matrices = ct_examples.compute_interval_matrices(net, inset_intervals)
        inset_positions = [0.14, 0.62]

        for col_index, axis in enumerate(axes[row_index]):
            if col_index >= len(windows):
                axis.set_visible(False)
                continue

            window = float(windows[col_index])
            signal_payload = signal_bundle["signals_by_window"][window]
            entropy_times = np.asarray(signal_payload["t_samples"], dtype=float)
            entropy_values = np.asarray(signal_payload["signal"], dtype=float)

            axis.step(
                active_times,
                active_counts,
                where="post",
                **ACTIVE_STYLE,
            )
            axis.axvline(breakpoint, **BREAKPOINT_STYLE)
            axis.set_xlim(t_min, t_max)
            axis.set_ylim(*active_ylim)
            axis.tick_params(axis="y", labelcolor=ACTIVE_STYLE["color"])
            axis.set_title(
                f"{panel_label(panel_index)} {format_window_title(window)}",
                loc="left",
                fontsize=12,
            )
            axis.set_box_aspect(1)

            if row_index == n_rows - 1:
                axis.set_xlabel("t [s]")

            if col_index == 0:
                axis.set_ylabel("# Active Links", color=ACTIVE_STYLE["color"])
            else:
                axis.tick_params(axis="y", labelleft=False)

            entropy_axis = axis.twinx()
            entropy_axis.plot(entropy_times, entropy_values, **ENTROPY_STYLE)
            entropy_axis.set_ylim(*entropy_ylim)
            entropy_axis.tick_params(axis="y", labelcolor=ENTROPY_STYLE["color"])
            if col_index == len(windows) - 1:
                entropy_axis.set_ylabel("Local entropy", color=ENTROPY_STYLE["color"])
            else:
                entropy_axis.tick_params(axis="y", labelright=False)

            ct_examples.draw_matrix_insets(
                host_ax=axis,
                matrices=interval_matrices,
                intervals=inset_intervals,
                inset_positions=inset_positions,
                inset_cmap=inset_cmap,
            )
            panel_index += 1

    legend_handles = [
        Line2D([0], [0], label="# Active Links", drawstyle="steps-post", **ACTIVE_STYLE),
        Line2D([0], [0], label="Local entropy", **ENTROPY_STYLE),
        Line2D([0], [0], label="Breakpoint", **BREAKPOINT_STYLE),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(legend_handles),
        frameon=False,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.94,
        top=0.95,
        bottom=0.12,
        wspace=0.28,
        hspace=0.4,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=int(args.dpi), bbox_inches="tight")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
