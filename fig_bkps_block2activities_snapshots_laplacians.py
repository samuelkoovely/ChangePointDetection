import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from gridsearch_block2activities_snapshots_laplacian_similarity import (
    get_signal_result_filename,
)


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "figures" / f"{Path(__file__).stem}.pdf"
RESULTS_PATH = (
    BASE_DIR / "gridsearch_results/block2activities_snapshots_laplacians/gridsearch_results.pkl"
)
DATASET_PATH = BASE_DIR / "data/block2activities_train_snapshots.pkl"


def resolve_existing_path(path_str: str | Path, base_dir: Path) -> Path:
    path = Path(path_str)
    candidates = [path, base_dir / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return base_dir / path


def get_best_result(results: dict) -> tuple[int, int, dict]:
    best_n_eigen = results.get("best_n_eigen")
    best_window_length = results.get("best_window_length")
    if best_n_eigen is None or best_window_length is None:
        best_index = results.get("best_index")
        if best_index is None:
            raise ValueError("No best (n_eigen, window_length) pair found.")
        best_n_eigen = int(results["n_eigens"][best_index[0]])
        best_window_length = int(results["window_lengths"][best_index[1]])

    results_by_n_eigen = results.get("results_by_n_eigen")
    if results_by_n_eigen is not None:
        best_result = results_by_n_eigen[int(best_n_eigen)]
    else:
        best_result = results["n_eigen_results"][results["best_index"][0]]

    return int(best_n_eigen), int(best_window_length), best_result


def get_signal_result_path(
    signals_outdir: Path,
    sample_dir: str,
    n_eigen: int,
    window_length: int,
    top: bool,
    difference: bool,
    second_window_length: int | None,
) -> Path:
    return (
        signals_outdir
        / sample_dir
        / get_signal_result_filename(
            n_eigen=n_eigen,
            window_length=window_length,
            top=top,
            difference=difference,
            second_window_length=second_window_length,
        )
    )


with open(RESULTS_PATH, "rb") as handle:
    results = pickle.load(handle)

with open(DATASET_PATH, "rb") as handle:
    dataset = pickle.load(handle)

best_n_eigen, best_window_length, best_result = get_best_result(results)
signals_outdir = results.get("signals_outdir")
if signals_outdir is None:
    signals_outdir = RESULTS_PATH.parent / "signals"
else:
    signals_outdir = resolve_existing_path(signals_outdir, BASE_DIR)

predicted_change_points = best_result["predicted_change_points"][int(best_window_length)]
sample_names = best_result.get("sample_names")
if sample_names is None:
    sample_names = [f"sample_{i}" for i in range(len(predicted_change_points))]

top = bool(results.get("top", True))
difference = bool(results.get("difference", True))
best_second_window_length = results.get("best_second_window_length")
if best_second_window_length is not None:
    best_second_window_length = int(best_second_window_length)

n_samples = min(len(dataset), len(predicted_change_points))
if n_samples <= 0:
    raise ValueError("No samples available to plot.")

fig, axes = plt.subplots(n_samples, 1, figsize=(14, max(2.2 * n_samples, 4)), sharex=False)
axes = np.atleast_1d(axes)
legend_handles = [
    Line2D([0], [0], color="C0", linewidth=1.5, label="Signal"),
    Line2D([0], [0], color="black", linewidth=1.5, label="Change point"),
    Line2D([0], [0], color="red", linewidth=1.5, linestyle="dashed", label="Predicted change point"),
]

for sample in range(n_samples):
    entry = dataset[sample]
    sample_name = sample_names[sample] if sample_names[sample] is not None else f"sample_{sample}"

    signal_path = get_signal_result_path(
        signals_outdir=signals_outdir,
        sample_dir=sample_name,
        n_eigen=int(best_n_eigen),
        window_length=int(best_window_length),
        top=top,
        difference=difference,
        second_window_length=best_second_window_length,
    )
    with open(signal_path, "rb") as handle:
        signal = pickle.load(handle)

    signal_values = np.asarray(signal["signal"])
    x_values = np.asarray(signal.get("snapshot_indices", np.arange(len(signal_values))))

    ax = axes[sample]
    ax.plot(x_values, signal_values, color="C0")
    if signal_values.size > 0:
        ymin = np.min(signal_values)
        ymax = np.max(signal_values)
        if np.isclose(ymin, ymax):
            pad = max(abs(float(ymin)) * 0.05, 1e-6)
            ymin -= pad
            ymax += pad
        for bkp in entry["bkps"]:
            ax.vlines(bkp, ymin=ymin, ymax=ymax)
        for pred_cp in predicted_change_points[sample]:
            ax.vlines(
                pred_cp,
                ymin=ymin,
                ymax=ymax,
                linestyles="dashed",
                color="red",
            )
    ax.set_ylabel(f"s{sample}")

axes[0].set_title(
    "Block2 snapshot LAD\n"
    f"n_eigen={best_n_eigen}, window_length={best_window_length}, "
    f"second_window_length={best_second_window_length}, difference={difference}"
)
axes[-1].set_xlabel("time")
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
