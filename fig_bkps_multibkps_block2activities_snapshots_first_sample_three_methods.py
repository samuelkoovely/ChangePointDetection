from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from gridsearch_block2activities_snapshots_laplacian_similarity import (
    get_signal_result_filename,
)


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "figures" / f"{Path(__file__).stem}.pdf"
DATASET_PATH = BASE_DIR / "data/multibkps_block2activities_snapshots.pkl"
SAMPLE_INDEX = 0

ENTROPY_RESULTS_PATH = (
    BASE_DIR
    / "gridsearch_results/multibkps_block2activities_snapshots/gridsearch_results.pkl"
)
FROBENIUS_RESULTS_PATH = (
    BASE_DIR
    / "gridsearch_results/multibkps_block2activities_snapshots_frobenius/gridsearch_results.pkl"
)
LAD_RESULTS_PATH = (
    BASE_DIR
    / "gridsearch_results/multibkps_block2activities_snapshots_laplacians/gridsearch_results.pkl"
)


@dataclass
class PanelSignal:
    y_label: str
    signal_values: np.ndarray
    x_values: np.ndarray
    true_change_points: list[float]
    predicted_change_points: list[float]


def resolve_existing_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    candidates = [path, BASE_DIR / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return BASE_DIR / path


def find_matching_float_key(mapping: dict, target: float, name: str) -> float:
    for key in mapping:
        if np.isclose(float(key), float(target)):
            return float(key)
    raise ValueError(f"Could not find {name}={target} in saved results.")


def find_matching_int_key(mapping: dict, target: int, name: str) -> int:
    for key in mapping:
        if int(key) == int(target):
            return int(key)
    raise ValueError(f"Could not find {name}={target} in saved results.")


def load_dataset_entry(sample_index: int) -> dict:
    with open(DATASET_PATH, "rb") as handle:
        dataset = pickle.load(handle)

    if sample_index >= len(dataset):
        raise IndexError(
            f"Requested sample_index={sample_index}, but dataset has only {len(dataset)} samples."
        )

    return dataset[sample_index]


def load_entropy_panel(sample_index: int) -> PanelSignal:
    with open(ENTROPY_RESULTS_PATH, "rb") as handle:
        results = pickle.load(handle)

    best_lamda = results.get("best_lamda")
    best_window = results.get("best_window")
    best_penalty = results.get("best_penalty", results.get("penalty"))
    if best_lamda is None or best_window is None:
        best_index = results.get("best_index")
        if best_index is None:
            raise ValueError("No best lambda/window found in entropy grid-search results.")
        best_lamda = float(results["lambdas"][best_index[0]])
        best_window = float(results["windows"][best_index[1]])
        penalties = results.get("penalties")
        if best_penalty is None and penalties is not None and len(best_index) > 2:
            best_penalty = float(penalties[best_index[2]])

    best_result = None
    results_by_lambda = results.get("results_by_lambda")
    if isinstance(results_by_lambda, dict) and len(results_by_lambda) > 0:
        lambda_key = find_matching_float_key(
            results_by_lambda,
            float(best_lamda),
            "lambda",
        )
        lambda_result = results_by_lambda[lambda_key]
        if isinstance(lambda_result, dict) and best_penalty is not None:
            penalty_key = find_matching_float_key(
                lambda_result,
                float(best_penalty),
                "penalty",
            )
            best_result = lambda_result[penalty_key]
        else:
            best_result = lambda_result

    if best_result is None:
        for lambda_result in results["lambda_results"]:
            lambda_matches = np.isclose(float(lambda_result["lamda"]), float(best_lamda))
            result_penalty = lambda_result.get("penalty")
            penalty_matches = (
                best_penalty is None
                or (
                    result_penalty is not None
                    and np.isclose(float(result_penalty), float(best_penalty))
                )
            )
            if lambda_matches and penalty_matches:
                best_result = lambda_result
                break

    if best_result is None:
        raise ValueError("Could not find the selected entropy result bundle.")

    predicted_change_points_by_window = best_result["predicted_change_points"]
    window_key = find_matching_float_key(
        predicted_change_points_by_window,
        float(best_window),
        "window",
    )
    predicted_change_points = predicted_change_points_by_window[window_key]
    sample_names = best_result.get("sample_names")
    if sample_names is None:
        sample_names = [f"sample_{i}" for i in range(len(predicted_change_points))]

    sample_name = (
        sample_names[sample_index]
        if sample_names[sample_index] is not None
        else f"sample_{sample_index}"
    )
    signals_outdir = results.get("signals_outdir")
    if signals_outdir is None:
        signals_outdir = ENTROPY_RESULTS_PATH.parent / "signals"
    else:
        signals_outdir = resolve_existing_path(signals_outdir)
    signal_path = (
        signals_outdir
        / sample_name
        / f"signal_lamda_{float(best_lamda):.11f}_window_{float(best_window):g}.pkl"
    )

    with open(signal_path, "rb") as handle:
        signal = pickle.load(handle)

    entry = load_dataset_entry(sample_index)
    signal_values = np.asarray(signal["signal"], dtype=float)
    x_values = np.asarray(signal.get("k_samples", np.arange(len(signal_values))), dtype=float)
    return PanelSignal(
        y_label="Entropy",
        signal_values=signal_values,
        x_values=x_values,
        true_change_points=[float(bkp) for bkp in entry["bkps"]],
        predicted_change_points=[float(cp) for cp in predicted_change_points[sample_index]],
    )


def load_frobenius_panel(sample_index: int) -> PanelSignal:
    with open(FROBENIUS_RESULTS_PATH, "rb") as handle:
        results = pickle.load(handle)

    best_window_length = results.get("best_window_length")
    best_penalty = results.get("best_penalty", results.get("penalty"))
    if best_window_length is None:
        best_index = results.get("best_index")
        if best_index is None:
            raise ValueError("No best window_length found in Frobenius grid-search results.")
        best_window_length = int(results["window_lengths"][best_index[0]])
        penalties = results.get("penalties")
        if best_penalty is None and penalties is not None and len(best_index) > 1:
            best_penalty = float(penalties[best_index[1]])

    best_result = None
    results_by_penalty = results.get("results_by_penalty")
    if isinstance(results_by_penalty, dict) and best_penalty is not None:
        penalty_key = find_matching_float_key(
            results_by_penalty,
            float(best_penalty),
            "penalty",
        )
        best_result = results_by_penalty[penalty_key]

    if best_result is None:
        penalty_results = results.get("penalty_results", [])
        for penalty_result in penalty_results:
            result_penalty = penalty_result.get("penalty")
            penalty_matches = (
                best_penalty is None
                or (
                    result_penalty is not None
                    and np.isclose(float(result_penalty), float(best_penalty))
                )
            )
            if penalty_matches:
                best_result = penalty_result
                break

    if best_result is None:
        best_result = results.get("window_results")

    if best_result is None:
        raise ValueError("Could not find the selected Frobenius result bundle.")

    predicted_change_points_by_window = best_result["predicted_change_points"]
    window_key = find_matching_int_key(
        predicted_change_points_by_window,
        int(best_window_length),
        "window_length",
    )
    predicted_change_points = predicted_change_points_by_window[window_key]
    sample_names = best_result.get("sample_names")
    if sample_names is None:
        sample_names = [f"sample_{i}" for i in range(len(predicted_change_points))]

    sample_name = (
        sample_names[sample_index]
        if sample_names[sample_index] is not None
        else f"sample_{sample_index}"
    )
    signals_outdir = results.get("signals_outdir")
    if signals_outdir is None:
        signals_outdir = FROBENIUS_RESULTS_PATH.parent / "signals"
    else:
        signals_outdir = resolve_existing_path(signals_outdir)

    signal_path_candidates = [
        signals_outdir
        / sample_name
        / f"signal_window_length_{int(best_window_length)}.pkl",
        signals_outdir
        / sample_name
        / f"signal_window_{float(best_window_length):g}.pkl",
    ]
    for candidate in signal_path_candidates:
        if candidate.exists():
            signal_path = candidate
            break
    else:
        raise FileNotFoundError(signal_path_candidates[0])

    with open(signal_path, "rb") as handle:
        signal = pickle.load(handle)

    entry = load_dataset_entry(sample_index)
    signal_values = np.asarray(signal["signal"], dtype=float)
    x_values = np.asarray(
        signal.get("snapshot_indices", np.arange(len(signal_values))),
        dtype=float,
    )
    return PanelSignal(
        y_label="Frobenius Score",
        signal_values=signal_values,
        x_values=x_values,
        true_change_points=[float(bkp) for bkp in entry["bkps"]],
        predicted_change_points=[float(cp) for cp in predicted_change_points[sample_index]],
    )


def load_lad_panel(sample_index: int) -> PanelSignal:
    with open(LAD_RESULTS_PATH, "rb") as handle:
        results = pickle.load(handle)

    best_n_eigen = results.get("best_n_eigen")
    best_window_length = results.get("best_window_length")
    if best_n_eigen is None or best_window_length is None:
        best_index = results.get("best_index")
        if best_index is None:
            raise ValueError("No best LAD parameter pair found in grid-search results.")
        best_n_eigen = int(results["n_eigens"][best_index[0]])
        best_window_length = int(results["window_lengths"][best_index[1]])

    results_by_n_eigen = results.get("results_by_n_eigen")
    if results_by_n_eigen is not None:
        best_result = results_by_n_eigen[int(best_n_eigen)]
    else:
        best_result = results["n_eigen_results"][results["best_index"][0]]

    predicted_change_points_by_window = best_result["predicted_change_points"]
    predicted_change_points = predicted_change_points_by_window[int(best_window_length)]
    sample_names = best_result.get("sample_names")
    if sample_names is None:
        sample_names = [f"sample_{i}" for i in range(len(predicted_change_points))]

    sample_name = (
        sample_names[sample_index]
        if sample_names[sample_index] is not None
        else f"sample_{sample_index}"
    )
    signals_outdir = results.get("signals_outdir")
    if signals_outdir is None:
        signals_outdir = LAD_RESULTS_PATH.parent / "signals"
    else:
        signals_outdir = resolve_existing_path(signals_outdir)

    top = bool(results.get("top", True))
    difference = bool(results.get("difference", True))
    best_second_window_length = results.get("best_second_window_length")
    if best_second_window_length is not None:
        best_second_window_length = int(best_second_window_length)

    signal_path = (
        signals_outdir
        / sample_name
        / get_signal_result_filename(
            n_eigen=int(best_n_eigen),
            window_length=int(best_window_length),
            top=top,
            difference=difference,
            second_window_length=best_second_window_length,
        )
    )

    with open(signal_path, "rb") as handle:
        signal = pickle.load(handle)

    entry = load_dataset_entry(sample_index)
    signal_values = np.asarray(signal["signal"], dtype=float)
    x_values = np.asarray(
        signal.get("snapshot_indices", np.arange(len(signal_values))),
        dtype=float,
    )
    return PanelSignal(
        y_label="LAD score",
        signal_values=signal_values,
        x_values=x_values,
        true_change_points=[float(bkp) for bkp in entry["bkps"]],
        predicted_change_points=[float(cp) for cp in predicted_change_points[sample_index]],
    )


def style_axis(ax: plt.Axes, y_label: str) -> None:
    ax.set_ylabel(y_label)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.85", linewidth=0.8)


def draw_panel(ax: plt.Axes, panel: PanelSignal) -> None:
    ax.plot(panel.x_values, panel.signal_values, color="C0", linewidth=1.5)

    if panel.signal_values.size == 0:
        style_axis(ax, panel.y_label)
        return

    ymin = float(np.min(panel.signal_values))
    ymax = float(np.max(panel.signal_values))
    if np.isclose(ymin, ymax):
        pad = max(abs(ymin) * 0.05, 1e-6)
        ymin -= pad
        ymax += pad

    for breakpoint in panel.true_change_points:
        ax.vlines(breakpoint, ymin=ymin, ymax=ymax, color="black", linewidth=1.2)
    for predicted_cp in panel.predicted_change_points:
        ax.vlines(
            predicted_cp,
            ymin=ymin,
            ymax=ymax,
            color="red",
            linestyle="dashed",
            linewidth=1.2,
        )

    style_axis(ax, panel.y_label)


def main() -> None:
    panels = [
        load_entropy_panel(SAMPLE_INDEX),
        load_frobenius_panel(SAMPLE_INDEX),
        load_lad_panel(SAMPLE_INDEX),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(14, 7.5), sharex=False)
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

    for ax, panel in zip(np.atleast_1d(axes), panels):
        draw_panel(ax, panel)

    axes[-1].set_xlabel("snapshot index")
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, format="pdf", dpi=300, bbox_inches="tight")
    print(OUTPUT_PATH)
    plt.close(fig)


if __name__ == "__main__":
    main()
