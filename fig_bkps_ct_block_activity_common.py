from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def resolve_existing_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    candidates = [path, BASE_DIR / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return BASE_DIR / path


def find_matching_float_key(mapping: dict[Any, Any], target: float, name: str) -> float:
    for key in mapping:
        if np.isclose(float(key), float(target)):
            return float(key)
    raise ValueError(f"Could not find {name}={target} in saved results.")


def get_best_signal_metadata(
    results: dict[str, Any],
) -> tuple[float, float, list[list[float]], list[str]]:
    best_lamda = results.get("best_lamda")
    best_window = results.get("best_window")

    if best_lamda is None or best_window is None:
        best_index = results.get("best_index")
        if best_index is None:
            raise ValueError("No best lambda/window found in grid-search results.")
        best_lamda = float(results["lambdas"][best_index[0]])
        best_window = float(results["windows"][best_index[1]])

    best_result = None
    results_by_lambda = results.get("results_by_lambda")
    if isinstance(results_by_lambda, dict) and len(results_by_lambda) > 0:
        lambda_key = find_matching_float_key(results_by_lambda, float(best_lamda), "lambda")
        best_result = results_by_lambda[lambda_key]

    if best_result is None:
        for lambda_result in results["lambda_results"]:
            if np.isclose(float(lambda_result["lamda"]), float(best_lamda)):
                best_result = lambda_result
                break

    if best_result is None:
        raise ValueError("Could not find the best lambda/window combination in results.")

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

    return float(best_lamda), float(best_window), predicted_change_points, sample_names


def get_signals_outdir(results: dict[str, Any], results_path: Path) -> Path:
    signals_outdir = results.get("signals_outdir") or results.get("signals_dir")
    if signals_outdir is None:
        return results_path.parent / "signals"
    return resolve_existing_path(signals_outdir)


def resolve_dataset_path(
    results: dict[str, Any],
    default_dataset_path: Path,
    dataset_path_override: Path | None = None,
) -> Path:
    if dataset_path_override is not None:
        return resolve_existing_path(dataset_path_override)

    data_path = results.get("data_path")
    if data_path is not None:
        return resolve_existing_path(data_path)
    return resolve_existing_path(default_dataset_path)


def get_signal_path(
    signals_outdir: Path,
    sample_name: str,
    lamda: float,
    window: float,
    reverse_time: bool = False,
) -> Path:
    direction_suffix = "_rev" if reverse_time else ""
    filename = (
        f"signal_lamda_{float(lamda):.11f}"
        f"_window_{float(window):g}{direction_suffix}.pkl"
    )
    return signals_outdir / sample_name / filename


def parse_args(default_results_path: Path, default_dataset_path: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the best continuous-time entropy signal per sample with "
            "ground-truth and predicted breakpoints."
        )
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=default_results_path,
        help="Path to the grid-search summary pickle.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help=(
            "Optional dataset pickle override. If omitted, uses the data path "
            "stored in the results summary when available."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional file path to save the rendered figure.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Maximum number of samples to plot. Defaults to all available samples.",
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
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional explicit figure title.",
    )
    return parser.parse_args()


def default_output_path(results_path: Path) -> Path:
    stem = results_path.parent.name or results_path.stem
    return BASE_DIR / "figures" / f"fig_bkps_{stem}.pdf"


def plot_best_signals(
    *,
    results_path: Path,
    dataset_path: Path,
    output_path: Path | None,
    num_samples: int,
    dpi: int,
    show: bool,
    title: str | None,
) -> None:
    results = load_pickle(results_path)
    dataset = load_pickle(dataset_path)

    best_lamda, best_window, predicted_change_points, sample_names = get_best_signal_metadata(results)
    signals_outdir = get_signals_outdir(results, results_path)

    available_samples = min(len(dataset), len(predicted_change_points))
    if num_samples is None or int(num_samples) <= 0:
        n_samples = available_samples
    else:
        n_samples = min(int(num_samples), available_samples)
    if n_samples <= 0:
        raise ValueError("No samples available to plot.")
    fig_height = max(2.2 * n_samples, 4.0)
    fig, axes = plt.subplots(n_samples, 1, figsize=(14, fig_height), sharex=False)
    axes = np.atleast_1d(axes)

    for sample_idx in range(n_samples):
        entry = dataset[sample_idx]
        sample_name = sample_names[sample_idx] or f"sample_{sample_idx}"
        signal = load_pickle(
            get_signal_path(
                signals_outdir=signals_outdir,
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

        ax = axes[sample_idx]
        ax.plot(x_values, signal_values, label=sample_name)
        if signal_values.size > 0:
            ymin = float(np.min(signal_values))
            ymax = float(np.max(signal_values))
            if np.isclose(ymin, ymax):
                pad = max(abs(ymin) * 0.05, 1e-6)
                ymin -= pad
                ymax += pad

            for bkp in entry["bkps"]:
                ax.vlines(
                    float(bkp),
                    ymin=ymin,
                    ymax=ymax,
                    color="black",
                )
            for pred_cp in predicted_change_points[sample_idx]:
                ax.vlines(
                    float(pred_cp),
                    ymin=ymin,
                    ymax=ymax,
                    linestyles="dashed",
                    color="red",
                )
        else:
            ax.text(
                0.5,
                0.5,
                "Empty signal",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        ax.legend(loc="upper right")
        ax.set_ylabel(f"s{sample_idx}")

    if title is None:
        title = f"{results.get('dataset', results_path.parent.name)}\nlambda={best_lamda:.5g}, window={best_window:g}"
    axes[0].set_title(title)
    axes[-1].set_xlabel("time")
    fig.tight_layout()

    resolved_output_path = output_path
    if resolved_output_path is None and not show:
        resolved_output_path = default_output_path(results_path)

    if resolved_output_path is not None:
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(resolved_output_path, dpi=int(dpi), bbox_inches="tight")
        print(resolved_output_path)

    if show:
        plt.show()
    else:
        plt.close(fig)


def main(default_results_path: Path, default_dataset_path: Path) -> None:
    args = parse_args(default_results_path, default_dataset_path)
    results_path = resolve_existing_path(args.results_path)
    dataset_path = resolve_dataset_path(
        load_pickle(results_path),
        default_dataset_path=default_dataset_path,
        dataset_path_override=args.dataset_path,
    )
    plot_best_signals(
        results_path=results_path,
        dataset_path=dataset_path,
        output_path=args.output_path,
        num_samples=args.num_samples,
        dpi=args.dpi,
        show=bool(args.show),
        title=args.title,
    )
