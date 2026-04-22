from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


BASE_DIR = Path(__file__).resolve().parent

EXPERIMENT_RESULT_PATHS = {
    "Block1 snapshots": {
        "Entropy": {
            "train": (
                BASE_DIR
                / "gridsearch_results/block1activity_snapshots/gridsearch_results.pkl"
            ),
            "test": (
                BASE_DIR
                / "gridsearch_results/block1activity_snapshots/test_set_results.pkl"
            ),
        },
        "Frobenius": {
            "train": (
                BASE_DIR
                / "gridsearch_results/block1activity_snapshots_frobenius/gridsearch_results.pkl"
            ),
            "test": (
                BASE_DIR
                / "gridsearch_results/block1activity_snapshots_frobenius/test_set_results.pkl"
            ),
        },
        "Laplacian": {
            "train": (
                BASE_DIR
                / "gridsearch_results/block1activity_snapshots_laplacians/gridsearch_results.pkl"
            ),
            "test": (
                BASE_DIR
                / "gridsearch_results/block1activity_snapshots_laplacians/test_set_results.pkl"
            ),
        },
    },
    "Block2 snapshots": {
        "Entropy": {
            "train": (
                BASE_DIR
                / "gridsearch_results/block2activities_snapshots/gridsearch_results.pkl"
            ),
            "test": (
                BASE_DIR
                / "gridsearch_results/block2activities_snapshots/test_set_results.pkl"
            ),
        },
        "Frobenius": {
            "train": (
                BASE_DIR
                / "gridsearch_results/block2activities_snapshots_frobenius/gridsearch_results.pkl"
            ),
            "test": (
                BASE_DIR
                / "gridsearch_results/block2activities_snapshots_frobenius/test_set_results.pkl"
            ),
        },
        "Laplacian": {
            "train": (
                BASE_DIR
                / "gridsearch_results/block2activities_snapshots_laplacians/gridsearch_results.pkl"
            ),
            "test": (
                BASE_DIR
                / "gridsearch_results/block2activities_snapshots_laplacians/test_set_results.pkl"
            ),
        },
    },
    "Multi-bkps snapshots": {
        "Entropy": {
            "train": (
                BASE_DIR
                / "gridsearch_results/multibkps_block2activities_snapshots/gridsearch_results.pkl"
            ),
            "test": (
                BASE_DIR
                / "gridsearch_results/multibkps_block2activities_snapshots/test_set_results.pkl"
            ),
        },
        "Frobenius": {
            "train": (
                BASE_DIR
                / "gridsearch_results/multibkps_block2activities_snapshots_frobenius/gridsearch_results.pkl"
            ),
            "test": (
                BASE_DIR
                / "gridsearch_results/multibkps_block2activities_snapshots_frobenius/test_set_results.pkl"
            ),
        },
        "Laplacian": {
            "train": (
                BASE_DIR
                / "gridsearch_results/multibkps_block2activities_snapshots_laplacians/gridsearch_results.pkl"
            ),
            "test": (
                BASE_DIR
                / "gridsearch_results/multibkps_block2activities_snapshots_laplacians/test_set_results.pkl"
            ),
        },
    },
}

METHOD_COLORS = {
    "Entropy": "#4C78A8",
    "Frobenius": "#54A24B",
    "Laplacian": "#B279A2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot boxplots of per-sample training performance for the three "
            "snapshot experiments."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path. If omitted, show the figure.",
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
        "--test-value",
        type=float,
        default=None,
        help=(
            "Optional override Hausdorff value to use for all test-set markers. "
            "If omitted, load the saved test-set summaries."
        ),
    )
    return parser.parse_args()


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _match_float_key(keys: list[float], target: float) -> float:
    for key in keys:
        if np.isclose(float(key), float(target)):
            return key
    raise KeyError(f"Could not match key for target={target}. Available keys: {keys}")


def _match_optional_numeric_key(keys: list[Any], target: float | int | None) -> Any:
    if target is None:
        if None in keys:
            return None
        if len(keys) == 1:
            return keys[0]
        raise KeyError(f"Could not match None target. Available keys: {keys}")

    for key in keys:
        if key is None:
            continue
        if np.isclose(float(key), float(target)):
            return key

    raise KeyError(f"Could not match key for target={target}. Available keys: {keys}")


def _matches_optional_numeric(value: Any, target: float | int | None) -> bool:
    if target is None:
        return value is None
    if value is None:
        return False
    return bool(np.isclose(float(value), float(target)))


def _extract_windowed_metric_arrays(
    result: dict[str, Any],
    window_target: float | int,
) -> dict[str, np.ndarray]:
    per_window_f1_scores = result.get("per_window_f1_scores")
    per_window_hausdorff = result.get("per_window_hausdorff")
    if per_window_f1_scores is None or per_window_hausdorff is None:
        raise ValueError("Result is missing per-window metric arrays.")

    window_key = _match_optional_numeric_key(
        list(per_window_f1_scores.keys()),
        window_target,
    )
    return {
        "f1": np.asarray(per_window_f1_scores[window_key], dtype=float),
        "hausdorff": np.asarray(per_window_hausdorff[window_key], dtype=float),
    }


def _select_penalty_result(
    container: Any,
    summary: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(container, dict):
        raise ValueError("Expected a dictionary when selecting a penalty result.")

    if "per_window_f1_scores" in container and "per_window_hausdorff" in container:
        return container

    best_penalty = summary.get("best_penalty", summary.get("penalty"))
    penalty_key = _match_optional_numeric_key(list(container.keys()), best_penalty)
    penalty_result = container[penalty_key]
    if not isinstance(penalty_result, dict):
        raise ValueError("Penalty result is not a dictionary.")
    return penalty_result


def _find_result_in_list(
    results: list[dict[str, Any]],
    *,
    parameter_name: str,
    parameter_value: float | int,
    penalty_value: float | int | None,
) -> dict[str, Any]:
    for result in results:
        if not _matches_optional_numeric(result.get(parameter_name), parameter_value):
            continue
        if ("penalty" in result) or (penalty_value is not None):
            if not _matches_optional_numeric(result.get("penalty"), penalty_value):
                continue
        return result

    raise ValueError(
        f"Could not find result for {parameter_name}={parameter_value}, "
        f"penalty={penalty_value}."
    )


def extract_metric_arrays(summary: dict[str, Any]) -> dict[str, np.ndarray]:
    if "lambda_results" in summary or "results_by_lambda" in summary:
        best_lamda = summary.get("best_lamda")
        best_window = summary.get("best_window")
        if best_lamda is None or best_window is None:
            best_index = summary.get("best_index")
            if best_index is None:
                raise ValueError("Missing best lambda/window information.")
            best_lamda = float(summary["lambdas"][best_index[0]])
            best_window = float(summary["windows"][best_index[1]])

        best_penalty = summary.get("best_penalty", summary.get("penalty"))
        results_by_lambda = summary.get("results_by_lambda")
        if isinstance(results_by_lambda, dict):
            lambda_key = _match_float_key(list(results_by_lambda.keys()), float(best_lamda))
            best_result = _select_penalty_result(results_by_lambda[lambda_key], summary)
        else:
            best_result = _find_result_in_list(
                summary["lambda_results"],
                parameter_name="lamda",
                parameter_value=float(best_lamda),
                penalty_value=best_penalty,
            )

        return _extract_windowed_metric_arrays(best_result, float(best_window))

    window_results = summary.get("window_results")
    if window_results is not None or "results_by_penalty" in summary or "penalty_results" in summary:
        best_window_length = summary.get("best_window_length")
        if best_window_length is None:
            best_index = summary.get("best_index")
            if best_index is None:
                raise ValueError("Missing best window_length information.")
            if isinstance(best_index, tuple):
                best_window_length = int(summary["window_lengths"][best_index[0]])
            else:
                best_window_length = int(summary["window_lengths"][best_index])

        if window_results is not None:
            best_result = window_results
        elif isinstance(summary.get("results_by_penalty"), dict):
            best_result = _select_penalty_result(summary["results_by_penalty"], summary)
        else:
            best_penalty = summary.get("best_penalty", summary.get("penalty"))
            best_result = _find_result_in_list(
                summary["penalty_results"],
                parameter_name="penalty",
                parameter_value=best_penalty,
                penalty_value=best_penalty,
            )

        return _extract_windowed_metric_arrays(best_result, int(best_window_length))

    if "results_by_n_eigen" in summary or "n_eigen_results" in summary:
        best_n_eigen = summary.get("best_n_eigen")
        best_window_length = summary.get("best_window_length")
        if best_n_eigen is None or best_window_length is None:
            best_index = summary.get("best_index")
            if best_index is None:
                raise ValueError("Missing best n_eigen/window_length information.")
            best_n_eigen = int(summary["n_eigens"][best_index[0]])
            best_window_length = int(summary["window_lengths"][best_index[1]])

        results_by_n_eigen = summary.get("results_by_n_eigen")
        if isinstance(results_by_n_eigen, dict):
            n_eigen_key = _match_optional_numeric_key(
                list(results_by_n_eigen.keys()),
                int(best_n_eigen),
            )
            best_result = _select_penalty_result(results_by_n_eigen[n_eigen_key], summary)
        else:
            best_result = _find_result_in_list(
                summary["n_eigen_results"],
                parameter_name="n_eigen",
                parameter_value=int(best_n_eigen),
                penalty_value=summary.get("best_penalty", summary.get("penalty")),
            )

        return _extract_windowed_metric_arrays(best_result, int(best_window_length))

    raise ValueError("Unsupported summary format.")


def load_all_metric_arrays() -> dict[str, dict[str, dict[str, np.ndarray]]]:
    metrics_by_experiment: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for experiment_name, method_paths in EXPERIMENT_RESULT_PATHS.items():
        metrics_by_experiment[experiment_name] = {}
        for method_name, result_paths in method_paths.items():
            summary = load_pickle(result_paths["train"])
            metrics_by_experiment[experiment_name][method_name] = extract_metric_arrays(
                summary
            )
    return metrics_by_experiment


def build_test_values(test_value: float | None) -> dict[str, dict[str, float]]:
    if test_value is not None:
        return {
            experiment_name: {
                method_name: float(test_value)
                for method_name in method_paths
            }
            for experiment_name, method_paths in EXPERIMENT_RESULT_PATHS.items()
        }

    test_values: dict[str, dict[str, float]] = {}
    for experiment_name, method_paths in EXPERIMENT_RESULT_PATHS.items():
        test_values[experiment_name] = {}
        for method_name, result_paths in method_paths.items():
            summary = load_pickle(result_paths["test"])
            if "test_mean_hausdorff" not in summary:
                raise KeyError(
                    f"Missing 'test_mean_hausdorff' in test summary: {result_paths['test']}"
                )
            test_values[experiment_name][method_name] = float(
                summary["test_mean_hausdorff"]
            )
    return test_values


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("#4a4a4a")
        spine.set_linewidth(1.0)


def draw_grouped_boxplots(
    ax: plt.Axes,
    metrics_by_experiment: dict[str, dict[str, dict[str, np.ndarray]]],
    test_values: dict[str, dict[str, float]],
    showfliers: bool,
) -> None:
    experiment_names = list(metrics_by_experiment.keys())
    method_names = list(METHOD_COLORS.keys())

    centers = np.arange(len(experiment_names), dtype=float) * 2.8
    offsets = np.linspace(-0.42, 0.42, len(method_names))
    width = 0.10

    style_axes(ax)
    has_infinite_values = False
    finite_values = []
    for experiment_name in experiment_names:
        for method_name in method_names:
            values = metrics_by_experiment[experiment_name][method_name]["hausdorff"]
            finite_values.extend(values[np.isfinite(values)].tolist())
            test_value = float(test_values[experiment_name][method_name])
            if np.isfinite(test_value):
                finite_values.append(test_value)
            if np.any(~np.isfinite(values)) or (not np.isfinite(test_value)):
                has_infinite_values = True
    finite_max = max(finite_values) if finite_values else 1.0
    hausdorff_cap = 1.1 * finite_max if finite_max > 0 else 1.0

    for experiment_idx, experiment_name in enumerate(experiment_names):
        center = centers[experiment_idx]
        for method_idx, method_name in enumerate(method_names):
            values = metrics_by_experiment[experiment_name][method_name]["hausdorff"]
            test_value = float(test_values[experiment_name][method_name])
            position = center + offsets[method_idx]
            display_values = values
            if np.any(~np.isfinite(values)):
                display_values = np.where(
                    np.isposinf(values),
                    hausdorff_cap,
                    values,
                )
            display_test_value = (
                hausdorff_cap if np.isposinf(test_value) else test_value
            )
            boxplot = ax.boxplot(
                [display_values],
                positions=[position],
                widths=width,
                patch_artist=True,
                showfliers=showfliers,
                whis=1.5,
                medianprops={"color": "#404040", "linewidth": 1.5},
                whiskerprops={"color": "#505050", "linewidth": 1.0},
                capprops={"color": "#505050", "linewidth": 1.0},
                boxprops={"edgecolor": "#505050", "linewidth": 1.0},
            )
            boxplot["boxes"][0].set_facecolor(METHOD_COLORS[method_name])
            boxplot["boxes"][0].set_alpha(0.9)
            ax.scatter(
                position,
                display_test_value,
                s=34,
                marker="o",
                facecolors="white",
                edgecolors=METHOD_COLORS[method_name],
                linewidths=1.4,
                zorder=4,
            )
            if np.any(np.isposinf(values)) or np.isposinf(test_value):
                ax.text(
                    position,
                    hausdorff_cap,
                    r"$\infty$",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color="#404040",
                )

    ax.set_xticks(centers)
    ax.set_xticklabels(experiment_names)
    ax.set_xlim(centers[0] - 0.85, centers[-1] + 0.85)
    if has_infinite_values:
        ax.set_ylim(top=hausdorff_cap * 1.12)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)


def build_figure(
    metrics_by_experiment: dict[str, dict[str, dict[str, np.ndarray]]],
    test_values: dict[str, dict[str, float]],
    showfliers: bool,
) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(10.2, 3.9))
    draw_grouped_boxplots(
        ax=ax,
        metrics_by_experiment=metrics_by_experiment,
        test_values=test_values,
        showfliers=showfliers,
    )
    ax.set_ylabel("Hausdorff distance")
    ax.set_title("Hausdorff performance across snapshot experiments", fontsize=13)

    legend_handles = [
        Patch(facecolor=METHOD_COLORS[method_name], edgecolor="#505050", label=method_name)
        for method_name in METHOD_COLORS
    ]
    legend_handles.append(
        Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="#404040",
            markeredgewidth=1.4,
            label="Test value",
        )
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=False,
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.90))
    return fig


def main() -> None:
    args = parse_args()
    metrics_by_experiment = load_all_metric_arrays()
    test_values = build_test_values(args.test_value)
    fig = build_figure(
        metrics_by_experiment=metrics_by_experiment,
        test_values=test_values,
        showfliers=args.showfliers,
    )

    if args.output is None:
        plt.show()
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    


if __name__ == "__main__":
    main()
