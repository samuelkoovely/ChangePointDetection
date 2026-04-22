import numpy as np
import pickle
from gridsearch_score import CPSample, extract_true_change_points, grid_search

with open("data/block2activities_train.pkl", "rb") as f:
    dataset = pickle.load(f)

lambdas = np.logspace(-5, 0, 10)
windows = [2.0]
margin = 5.0
n_jobs = 6

training_samples = []
for i, entry in enumerate(dataset):
    true_change_points, n_bkps = extract_true_change_points(entry)
    training_samples.append(
        CPSample(
            data=entry["tnet"],
            true_change_points=true_change_points,
            n_bkps=n_bkps,
            name=f"sample_{i}",
        )
    )

# Generated entropy signals can optionally be saved while running the
# grid-search under:
#     signals_outdir / sample_name / signal_lamda_xxx_window_y.pkl
summary = grid_search(
    samples=training_samples,
    lambdas=lambdas,
    windows=windows,
    margin=margin,
    n_jobs=n_jobs,
    outdir="./gridsearch_results/block2activities",
    sample_fraction=0.0001,
    kernel="linear",
    save_signals=True,
    signals_outdir="./gridsearch_results/block2activities/signals",
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
