from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

plt.style.use(Path(__file__).with_name("paper.mplstyle"))

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    BASE_DIR / "figures" / "fig_snapshot_experiments_train_test_split_violins.pdf"
)
METHOD_COLORS = {
    "Entropy": "#1f77b4",
    "Frobenius": "#ff7f0e",
    "LAD": "#2ca02c",
}
EXPERIMENT_ORDER = (
    "Block2 snapshots",
    "Block1 snapshots",
    "Multi-bkps snapshots",
)
EXPERIMENT_DISPLAY_NAMES = {
    "Block2 snapshots": "Benchmark1",
    "Block1 snapshots": "Benchmark2",
    "Multi-bkps snapshots": "Benchmark3",
}
SNAPSHOT_RESULT_PATHS = {
    "Block2 snapshots": {
        "Entropy": {
            "train": BASE_DIR
            / "gridsearch_results/block2activities_snapshots/gridsearch_results.pkl",
            "test": BASE_DIR
            / "gridsearch_results/block2activities_snapshots/test_set_results.pkl",
        },
        "Frobenius": {
            "train": BASE_DIR
            / "gridsearch_results/block2activities_snapshots_frobenius/gridsearch_results.pkl",
            "test": BASE_DIR
            / "gridsearch_results/block2activities_snapshots_frobenius/test_set_results.pkl",
        },
        "LAD": {
            "train": BASE_DIR
            / "gridsearch_results/block2activities_snapshots_laplacians/gridsearch_results.pkl",
            "test": BASE_DIR
            / "gridsearch_results/block2activities_snapshots_laplacians/test_set_results.pkl",
        },
    },
    "Block1 snapshots": {
        "Entropy": {
            "train": BASE_DIR
            / "gridsearch_results/block1activity_snapshots/gridsearch_results.pkl",
            "test": BASE_DIR
            / "gridsearch_results/block1activity_snapshots/test_set_results.pkl",
        },
        "Frobenius": {
            "train": BASE_DIR
            / "gridsearch_results/block1activity_snapshots_frobenius/gridsearch_results.pkl",
            "test": BASE_DIR
            / "gridsearch_results/block1activity_snapshots_frobenius/test_set_results.pkl",
        },
        "LAD": {
            "train": BASE_DIR
            / "gridsearch_results/block1activity_snapshots_laplacians/gridsearch_results.pkl",
            "test": BASE_DIR
            / "gridsearch_results/block1activity_snapshots_laplacians/test_set_results.pkl",
        },
    },
    "Multi-bkps snapshots": {
        "Entropy": {
            "train": BASE_DIR
            / "gridsearch_results/multibkps_block2activities_snapshots/gridsearch_results.pkl",
            "test": BASE_DIR
            / "gridsearch_results/multibkps_block2activities_snapshots/test_set_results.pkl",
        },
        "Frobenius": {
            "train": BASE_DIR
            / "gridsearch_results/multibkps_block2activities_snapshots_frobenius/gridsearch_results.pkl",
            "test": BASE_DIR
            / "gridsearch_results/multibkps_block2activities_snapshots_frobenius/test_set_results.pkl",
        },
        "LAD": {
            "train": BASE_DIR
            / "gridsearch_results/multibkps_block2activities_snapshots_laplacians/gridsearch_results.pkl",
            "test": BASE_DIR
            / "gridsearch_results/multibkps_block2activities_snapshots_laplacians/test_set_results.pkl",
        },
    },
}
SPLIT_ORDER = ("Train", "Test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot paired train/test split violins of per-sample Hausdorff "
            "performance for the snapshot experiments."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output PDF path. Use --show to display interactively as well.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Figure DPI when saving to disk.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively after rendering.",
    )
    return parser.parse_args()


def _lighten_color(
    color: str,
    mix_with_white: float = 0.58,
) -> tuple[float, float, float]:
    base = np.asarray(to_rgb(color), dtype=float)
    return tuple(base + (1.0 - base) * mix_with_white)


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)


def _find_matching_float_key(mapping: dict[Any, Any], target: float, name: str) -> Any:
    for key in mapping:
        if np.isclose(float(key), float(target)):
            return key
    raise KeyError(f"Could not find {name}={target} in saved results.")


def _find_matching_int_key(mapping: dict[Any, Any], target: int, name: str) -> Any:
    for key in mapping:
        if int(key) == int(target):
            return key
    raise KeyError(f"Could not find {name}={target} in saved results.")


def _extract_entropy_metric_arrays(summary: dict[str, Any]) -> dict[str, np.ndarray]:
    best_lamda = float(summary["best_lamda"])
    best_window = float(summary["best_window"])
    best_penalty = summary.get("best_penalty", summary.get("penalty"))
    if best_penalty is not None:
        best_penalty = float(best_penalty)

    best_result = None
    for lambda_result in summary.get("lambda_results", []):
        if not np.isclose(float(lambda_result["lamda"]), best_lamda):
            continue
        result_penalty = lambda_result.get("penalty")
        if best_penalty is not None:
            if result_penalty is None or not np.isclose(
                float(result_penalty),
                best_penalty,
            ):
                continue
        best_result = lambda_result
        break

    if best_result is None:
        raise ValueError("Could not locate the best entropy training result.")

    window_key = _find_matching_float_key(
        best_result["per_window_f1_scores"],
        best_window,
        "window",
    )
    return {
        "f1": np.asarray(best_result["per_window_f1_scores"][window_key], dtype=float),
        "hausdorff": np.asarray(
            best_result["per_window_hausdorff"][window_key],
            dtype=float,
        ),
    }


def _extract_frobenius_metric_arrays(summary: dict[str, Any]) -> dict[str, np.ndarray]:
    best_window_length = int(summary["best_window_length"])
    best_penalty = summary.get("best_penalty", summary.get("penalty"))
    if best_penalty is not None:
        best_penalty = float(best_penalty)

    best_result = None
    results_by_penalty = summary.get("results_by_penalty")
    if isinstance(results_by_penalty, dict) and best_penalty is not None:
        penalty_key = _find_matching_float_key(
            results_by_penalty,
            best_penalty,
            "penalty",
        )
        best_result = results_by_penalty[penalty_key]

    if best_result is None:
        for penalty_result in summary.get("penalty_results", []):
            result_penalty = penalty_result.get("penalty")
            if best_penalty is not None:
                if result_penalty is None or not np.isclose(
                    float(result_penalty),
                    best_penalty,
                ):
                    continue
            best_result = penalty_result
            break

    if best_result is None:
        best_result = summary.get("window_results")

    if best_result is None:
        raise ValueError("Could not locate the best Frobenius training result.")

    window_key = _find_matching_int_key(
        best_result["per_window_f1_scores"],
        best_window_length,
        "window_length",
    )
    return {
        "f1": np.asarray(best_result["per_window_f1_scores"][window_key], dtype=float),
        "hausdorff": np.asarray(
            best_result["per_window_hausdorff"][window_key],
            dtype=float,
        ),
    }


def _extract_lad_metric_arrays(summary: dict[str, Any]) -> dict[str, np.ndarray]:
    best_n_eigen = int(summary["best_n_eigen"])
    best_window_length = int(summary["best_window_length"])

    best_result = None
    results_by_n_eigen = summary.get("results_by_n_eigen")
    if isinstance(results_by_n_eigen, dict):
        for key, result in results_by_n_eigen.items():
            if int(key) == best_n_eigen:
                best_result = result
                break

    if best_result is None:
        for n_eigen_result in summary.get("n_eigen_results", []):
            if int(n_eigen_result["n_eigen"]) == best_n_eigen:
                best_result = n_eigen_result
                break

    if best_result is None:
        raise ValueError("Could not locate the best LAD training result.")

    window_key = _find_matching_int_key(
        best_result["per_window_f1_scores"],
        best_window_length,
        "window_length",
    )
    return {
        "f1": np.asarray(best_result["per_window_f1_scores"][window_key], dtype=float),
        "hausdorff": np.asarray(
            best_result["per_window_hausdorff"][window_key],
            dtype=float,
        ),
    }


def extract_metric_arrays(summary: dict[str, Any]) -> dict[str, np.ndarray]:
    if "lambda_results" in summary:
        return _extract_entropy_metric_arrays(summary)
    if "window_results" in summary or "penalty_results" in summary:
        return _extract_frobenius_metric_arrays(summary)
    if "n_eigen_results" in summary or "results_by_n_eigen" in summary:
        return _extract_lad_metric_arrays(summary)
    raise ValueError("Unsupported training summary format.")


def extract_test_metric_arrays(summary: dict[str, Any]) -> dict[str, np.ndarray]:
    per_sample_results = summary.get("per_sample_results")
    if not per_sample_results:
        raise ValueError("Test summary does not contain per-sample results.")
    return {
        "f1": np.asarray([entry["f1"] for entry in per_sample_results], dtype=float),
        "hausdorff": np.asarray(
            [entry["hausdorff"] for entry in per_sample_results],
            dtype=float,
        ),
    }


def load_all_split_metric_arrays() -> dict[str, dict[str, dict[str, dict[str, np.ndarray]]]]:
    metrics_by_experiment: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]] = {}
    for experiment_name, result_paths_by_method in SNAPSHOT_RESULT_PATHS.items():
        method_metrics: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        for method_name, result_paths in result_paths_by_method.items():
            train_summary = load_pickle(result_paths["train"])
            test_summary = load_pickle(result_paths["test"])
            method_metrics[method_name] = {
                "Train": extract_metric_arrays(train_summary),
                "Test": extract_test_metric_arrays(test_summary),
            }
        metrics_by_experiment[experiment_name] = method_metrics
    return metrics_by_experiment


def _compute_log_floor(
    metrics_by_experiment: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
) -> float:
    positive_values: list[float] = []
    for methods in metrics_by_experiment.values():
        for splits in methods.values():
            for split_metrics in splits.values():
                values = np.asarray(split_metrics["hausdorff"], dtype=float)
                mask = np.isfinite(values) & (values > 0)
                positive_values.extend(values[mask].tolist())
    if not positive_values:
        return 1e-3
    return max(min(positive_values) * 0.5, 1e-6)


def _prepare_boxplot_values(
    values: np.ndarray,
    hausdorff_cap: float,
    log_floor: float,
) -> np.ndarray:
    display_values = np.asarray(values, dtype=float).copy()
    display_values[np.isposinf(display_values)] = hausdorff_cap
    display_values[np.isneginf(display_values)] = log_floor
    display_values[display_values <= 0] = log_floor
    display_values = display_values[~np.isnan(display_values)]
    if display_values.size == 0:
        return np.asarray([log_floor], dtype=float)
    return display_values


def _draw_boxplot(
    ax: plt.Axes,
    *,
    values: np.ndarray,
    position: float,
    width: float,
    method_color: str,
    split_name: str,
    hausdorff_cap: float,
    log_floor: float,
    showfliers: bool,
) -> None:
    display_values = _prepare_boxplot_values(values, hausdorff_cap, log_floor)
    is_train = split_name == "Train"
    facecolor = method_color if is_train else "white"
    hatch = None if is_train else "///"
    alpha = 0.9 if is_train else 1.0

    artists = ax.boxplot(
        [display_values],
        positions=[position],
        widths=width,
        patch_artist=True,
        showfliers=showfliers,
    )
    box = artists["boxes"][0]
    box.set_facecolor(facecolor)
    box.set_edgecolor(method_color)
    box.set_linewidth(1.1)
    box.set_alpha(alpha)
    if hatch is not None:
        box.set_hatch(hatch)

    for median in artists["medians"]:
        median.set_color("#2f2f2f")
        median.set_linewidth(1.4)
    for whisker in artists["whiskers"]:
        whisker.set_color(method_color)
        whisker.set_linewidth(1.0)
    for cap in artists["caps"]:
        cap.set_color(method_color)
        cap.set_linewidth(1.0)
    for flier in artists["fliers"]:
        flier.set_markeredgecolor(method_color)
        flier.set_markerfacecolor(method_color)
        flier.set_alpha(0.5)


def _select_experiments(
    metrics_by_experiment: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
) -> dict[str, dict[str, dict[str, dict[str, np.ndarray]]]]:
    filtered_names = [
        experiment_name
        for experiment_name in EXPERIMENT_ORDER
        if experiment_name in metrics_by_experiment
    ]
    filtered_names.extend(
        experiment_name
        for experiment_name in metrics_by_experiment
        if experiment_name not in filtered_names
    )
    filtered = {
        experiment_name: metrics_by_experiment[experiment_name]
        for experiment_name in filtered_names
    }
    if not filtered:
        raise ValueError("No experiments available to plot.")
    return filtered


def _compute_hausdorff_cap(
    metrics_by_experiment: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
) -> tuple[float, bool]:
    finite_values: list[float] = []
    has_infinite_values = False
    for methods in metrics_by_experiment.values():
        for splits in methods.values():
            for split_metrics in splits.values():
                values = np.asarray(split_metrics["hausdorff"], dtype=float)
                finite_values.extend(values[np.isfinite(values)].tolist())
                if np.any(~np.isfinite(values)):
                    has_infinite_values = True

    finite_max = max(finite_values) if finite_values else 1.0
    hausdorff_cap = 1.1 * finite_max if finite_max > 0 else 1.0
    return hausdorff_cap, has_infinite_values


def _clip_violin_half(body: PolyCollection, *, position: float, side: str) -> None:
    vertices = body.get_paths()[0].vertices
    if side == "left":
        vertices[:, 0] = np.minimum(vertices[:, 0], position)
        return
    if side == "right":
        vertices[:, 0] = np.maximum(vertices[:, 0], position)
        return
    raise ValueError(f"Unsupported violin side: {side}")


def _draw_half_violin(
    ax: plt.Axes,
    *,
    values: np.ndarray,
    original_values: np.ndarray,
    position: float,
    width: float,
    side: str,
    facecolor: tuple[float, float, float] | str,
    edgecolor: str,
    hausdorff_cap: float,
    log_floor: float,
) -> None:
    display_values = _prepare_boxplot_values(values, hausdorff_cap, log_floor)
    violin = ax.violinplot(
        [display_values],
        positions=[position],
        widths=width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    body = violin["bodies"][0]
    body.set_facecolor(facecolor)
    body.set_edgecolor(edgecolor)
    body.set_alpha(0.9)
    body.set_linewidth(1.1)
    _clip_violin_half(body, position=position, side=side)

    median = float(np.median(display_values))
    median_span = width * 0.22
    median_margin = width * 0.03
    if side == "left":
        ax.hlines(
            median,
            position - median_span,
            position - median_margin,
            color="#2f2f2f",
            linewidth=1.6,
            zorder=4,
        )
        text_x = position - width * 0.18
    else:
        ax.hlines(
            median,
            position + median_margin,
            position + median_span,
            color="#2f2f2f",
            linewidth=1.6,
            zorder=4,
        )
        text_x = position + width * 0.18

    if np.any(np.isposinf(original_values)):
        ax.text(
            text_x,
            hausdorff_cap,
            r"$\infty$",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#404040",
        )


def draw_grouped_split_violins(
    ax: plt.Axes,
    metrics_by_experiment: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
) -> None:
    experiment_names = list(metrics_by_experiment.keys())
    experiment_display_names = [
        EXPERIMENT_DISPLAY_NAMES.get(experiment_name, experiment_name)
        for experiment_name in experiment_names
    ]
    method_names = list(METHOD_COLORS.keys())

    centers = np.arange(len(experiment_names), dtype=float) * 3.8
    method_offsets = np.linspace(-0.95, 0.95, len(method_names))
    width = 0.50

    style_axes(ax)
    hausdorff_cap, has_infinite_values = _compute_hausdorff_cap(metrics_by_experiment)
    log_floor = _compute_log_floor(metrics_by_experiment)

    for experiment_idx, experiment_name in enumerate(experiment_names):
        center = centers[experiment_idx]
        for method_idx, method_name in enumerate(method_names):
            position = center + method_offsets[method_idx]
            method_color = METHOD_COLORS[method_name]
            splits = metrics_by_experiment[experiment_name][method_name]

            train_values = np.asarray(splits["Train"]["hausdorff"], dtype=float)
            test_values = np.asarray(splits["Test"]["hausdorff"], dtype=float)
            combined_display_values = np.concatenate(
                [
                    _prepare_boxplot_values(train_values, hausdorff_cap, log_floor),
                    _prepare_boxplot_values(test_values, hausdorff_cap, log_floor),
                ]
            )
            ax.vlines(
                position,
                float(np.min(combined_display_values)),
                float(np.max(combined_display_values)),
                color="#565656",
                linewidth=0.9,
                alpha=0.85,
                zorder=3,
            )

            _draw_half_violin(
                ax,
                values=train_values,
                original_values=train_values,
                position=position,
                width=width,
                side="left",
                facecolor=method_color,
                edgecolor=method_color,
                hausdorff_cap=hausdorff_cap,
                log_floor=log_floor,
            )
            _draw_half_violin(
                ax,
                values=test_values,
                original_values=test_values,
                position=position,
                width=width,
                side="right",
                facecolor=_lighten_color(method_color),
                edgecolor=method_color,
                hausdorff_cap=hausdorff_cap,
                log_floor=log_floor,
            )

    #ax.set_yscale("log")
    ax.set_xticks(centers)
    ax.set_xticklabels(experiment_display_names)
    ax.set_xlim(centers[0] - 1.55, centers[-1] + 1.55)
    ax.set_ylim(
        bottom=log_floor / 1.15,
        top=hausdorff_cap * 1.12 if has_infinite_values else None,
    )
    ax.tick_params(
        axis="x",
        labelsize=11,
        labeltop=True,
        labelbottom=False,
        top=False,
        bottom=False,
        length=0,
        pad=8,
    )
    ax.tick_params(axis="y", labelsize=11)


def build_figure(
    metrics_by_experiment: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]],
) -> plt.Figure:
    fig_width = 4.6 + 3.0 * len(metrics_by_experiment)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 4.3))
    draw_grouped_split_violins(ax=ax, metrics_by_experiment=metrics_by_experiment)
    ax.set_ylabel("Hausdorff distance")

    legend_handles = [
        Patch(
            facecolor=METHOD_COLORS[method_name],
            edgecolor=METHOD_COLORS[method_name],
            label=method_name,
        )
        for method_name in METHOD_COLORS
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0.02, 0.08, 0.98, 0.94))
    return fig


def main() -> None:
    args = parse_args()
    metrics_by_experiment = _select_experiments(load_all_split_metric_arrays())
    fig = build_figure(metrics_by_experiment=metrics_by_experiment)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
        print(args.output)

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
