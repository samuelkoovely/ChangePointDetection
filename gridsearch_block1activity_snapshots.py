import numpy as np
import pickle
from gridsearch_score_snapshots import CPSample, grid_search

with open("data/block1activity_snapshots.pkl", "rb") as f:
        dataset = pickle.load(f)

aggregation_window = dataset[0]['aggregation_window'] 
lambdas = np.logspace(-5, 0, 10)
windows = [aggregation_window / 2]
margin = 1.0
n_jobs = 6

training_samples = [
    CPSample(
        data=entry["tnet"],
        true_change_points=[float(entry["bkp"])],
        n_bkps=1,
        name=f"sample_{i}",
    )
    for i, entry in enumerate(dataset)
]

# Generated entropy signals can optionally be saved while running the
# grid-search under:
#     signals_outdir / sample_name / signal_lamda_xxx_window_y.pkl
summary = grid_search(
    samples=training_samples,
    lambdas=lambdas,
    windows=windows,
    margin=margin,
    n_jobs=n_jobs,
    outdir="./gridsearch_results/block1activity_snapshots",
    # Snapshot labels are stored as indices, so using the full signal avoids
    # having to remap ground-truth indices onto a subsampled entropy signal.
    sample_fraction=1.0,
    kernel="linear",
    save_signals=True,
    signals_outdir="./gridsearch_results/block1activity_snapshots/signals",
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
print("Detection and metrics phase runtime:", summary["detection_metrics_phase_seconds"])
print("Signals saved:", summary["save_signals"])
print("Signals output directory:", summary["signals_outdir"])
