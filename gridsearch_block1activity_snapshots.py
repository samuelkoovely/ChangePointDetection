from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np

from gridsearch_score_snapshots import CPSample, extract_true_change_points, grid_search


DATASET_PATH = Path("data/block1activity_train_snapshots.pkl")
OUTDIR = Path("gridsearch_results/block1activity_snapshots")
SIGNALS_OUTDIR = OUTDIR / "signals"


def load_training_samples(dataset_path: Path) -> tuple[list[CPSample], float]:
    with open(dataset_path, "rb") as handle:
        dataset = pickle.load(handle)

    aggregation_window = float(dataset[0]["aggregation_window"])
    training_samples = []
    for sample_index, entry in enumerate(dataset):
        true_change_points, n_bkps = extract_true_change_points(entry)
        training_samples.append(
            CPSample(
                data=entry["tnet"],
                true_change_points=true_change_points,
                n_bkps=n_bkps,
                name=f"sample_{sample_index}",
            )
        )

    return training_samples, aggregation_window


def main() -> None:
    training_samples, aggregation_window = load_training_samples(DATASET_PATH)
    lambdas = np.logspace(-5, 0, 10)
    windows = [aggregation_window / 2]
    margin = 1.0
    n_jobs = 6

    summary = grid_search(
        samples=training_samples,
        lambdas=lambdas,
        windows=windows,
        margin=margin,
        n_jobs=n_jobs,
        outdir=OUTDIR,
        # Snapshot labels are stored as indices, so using the full signal avoids
        # having to remap ground-truth indices onto a subsampled entropy signal.
        sample_fraction=1.0,
        kernel="linear",
        save_signals=True,
        signals_outdir=SIGNALS_OUTDIR,
        selection_metric="hausdorff",
    )

    print("Number of samples:", len(training_samples))
    print("Score array shape:", summary["score_array"].shape)
    print("Selection metric:", summary["selection_metric"])
    print("Best lamda:", summary["best_lamda"])
    print("Best window:", summary["best_window"])
    print("Best selected score:", summary["best_score"])
    print("Best mean F1:", summary["best_f1"])
    print("Best mean Hausdorff:", summary["best_hausdorff"])
    print("Total runtime:", summary["elapsed_seconds"])
    print("Signal generation phase runtime:", summary["signal_generation_phase_seconds"])
    print(
        "Detection and metrics phase runtime:",
        summary["detection_metrics_phase_seconds"],
    )
    print("Signals saved:", summary["save_signals"])
    print("Signals output directory:", summary["signals_outdir"])


if __name__ == "__main__":
    main()
