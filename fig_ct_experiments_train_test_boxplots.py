from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from fig_snapshot_experiments_boxplots import (
    METHOD_COLORS,
    extract_metric_arrays,
    load_pickle,
    style_axes,
)
from fig_snapshot_experiments_train_test_boxplots import (
    SPLIT_ORDER,
    _compute_log_floor,
    _draw_boxplot,
    extract_test_metric_arrays,
)


BASE_DIR = Path(__file__).resolve().parent
ENTROPY_COLOR = METHOD_COLORS["Entropy"]
DEFAULT_OUTPUT = BASE_DIR / "figures" / "fig_ct_experiments_train_test_boxplots.pdf"

CT_EXPERIMENT_RESULT_CANDIDATES = {
    "CT Block1": {
        "train": (
            BASE_DIR / "gridsearch_results/ct_block1activity/gridsearch_results.pkl",
            BASE_DIR / "gridsearch_results/ct_block1/gridsearch_results.pkl",
        ),
        "test": (
            BASE_DIR / "gridsearch_results/ct_block1activity/test_set_results.pkl",
            BASE_DIR / "gridsearch_results/ct_block1/test_set_results.pkl",
        ),
    },
    "CT Block2": {
        "train": (
            BASE_DIR / "gridsearch_results/ct_block2activities/gridsearch_results.pkl",
            BASE_DIR / "gridsearch_results/ct_block2/gridsearch_results.pkl",
        ),
        "test": (
            BASE_DIR / "gridsearch_results/ct_block2activities/test_set_results.pkl",
            BASE_DIR / "gridsearch_results/ct_block2/test_set_results.pkl",
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot paired train/test boxplots of per-sample Hausdorff performance "
            "for the continuous-time block-activity experiments."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output image path. Use --show to display interactively as well.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Figure DPI when saving to disk.",
    )
    parser.add_argument(
        "--showfliers",
        action="store_true",
        help="Show boxplot fliers.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively after rendering.",
    )
    return parser.parse_args()


def resolve_summary_path(candidates: tuple[Path, ...], *, split_name: str) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    candidate_text = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Could not find a {split_name} summary in any of: {candidate_text}"
    )


def load_all_split_metric_arrays() -> dict[str, dict[str, dict[str, np.ndarray]]]:
    metrics_by_experiment: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for experiment_name, split_candidates in CT_EXPERIMENT_RESULT_CANDIDATES.items():
        train_summary_path = resolve_summary_path(
            split_candidates["train"],
            split_name=f"{experiment_name} training",
        )
        test_summary_path = resolve_summary_path(
            split_candidates["test"],
            split_name=f"{experiment_name} test",
        )
        train_summary = load_pickle(train_summary_path)
        test_summary = load_pickle(test_summary_path)
        metrics_by_experiment[experiment_name] = {
            "Train": extract_metric_arrays(train_summary),
            "Test": extract_test_metric_arrays(test_summary),
        }
    return metrics_by_experiment


def draw_grouped_boxplots(
    ax: plt.Axes,
    metrics_by_experiment: dict[str, dict[str, dict[str, np.ndarray]]],
    showfliers: bool,
) -> None:
    experiment_names = list(metrics_by_experiment.keys())

    centers = np.arange(len(experiment_names), dtype=float) * 2.8
    split_offsets = {"Train": -0.14, "Test": 0.14}
    width = 0.22

    style_axes(ax)
    finite_values: list[float] = []
    has_infinite_values = False
    for experiment_name in experiment_names:
        for split_name in SPLIT_ORDER:
            values = metrics_by_experiment[experiment_name][split_name]["hausdorff"]
            finite_values.extend(values[np.isfinite(values)].tolist())
            if np.any(~np.isfinite(values)):
                has_infinite_values = True

    finite_max = max(finite_values) if finite_values else 1.0
    hausdorff_cap = 1.1 * finite_max if finite_max > 0 else 1.0
    log_floor = _compute_log_floor(
        {
            experiment_name: {"Entropy": splits}
            for experiment_name, splits in metrics_by_experiment.items()
        }
    )

    for experiment_idx, experiment_name in enumerate(experiment_names):
        center = centers[experiment_idx]
        for split_name in SPLIT_ORDER:
            position = center + split_offsets[split_name]
            values = metrics_by_experiment[experiment_name][split_name]["hausdorff"]
            _draw_boxplot(
                ax,
                values=values,
                position=position,
                width=width,
                method_color=ENTROPY_COLOR,
                split_name=split_name,
                hausdorff_cap=hausdorff_cap,
                log_floor=log_floor,
                showfliers=showfliers,
            )

    ax.set_yscale("log")
    ax.set_xticks(centers)
    ax.set_xticklabels(experiment_names)
    ax.set_xlim(centers[0] - 0.8, centers[-1] + 0.8)
    ax.set_ylim(
        bottom=log_floor / 1.15,
        top=hausdorff_cap * 1.12 if has_infinite_values else None,
    )
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)


def build_figure(
    metrics_by_experiment: dict[str, dict[str, dict[str, np.ndarray]]],
    showfliers: bool,
) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(7.4, 4.1))
    draw_grouped_boxplots(
        ax=ax,
        metrics_by_experiment=metrics_by_experiment,
        showfliers=showfliers,
    )
    ax.set_ylabel("Hausdorff distance (log scale)")
    ax.set_title("Train and test Hausdorff performance across CT experiments")

    legend_handles = [
        Patch(facecolor=ENTROPY_COLOR, edgecolor=ENTROPY_COLOR, label="Entropy"),
        Patch(facecolor="#808080", edgecolor="#505050", alpha=0.88, label="Train"),
        Patch(facecolor="white", edgecolor="#505050", hatch="///", label="Test"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.90))
    return fig


def main() -> None:
    args = parse_args()
    metrics_by_experiment = load_all_split_metric_arrays()
    fig = build_figure(
        metrics_by_experiment=metrics_by_experiment,
        showfliers=args.showfliers,
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=int(args.dpi), bbox_inches="tight")
        print(args.output)

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
