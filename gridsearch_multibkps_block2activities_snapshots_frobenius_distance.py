import numpy as np
import pickle

from gridsearch_block2activities_snapshots_frobenius_distance import (
    CPSample,
    grid_search_frobenius_distance,
)
from gridsearch_score_snapshots import extract_true_change_points

# `ruptures` requires strictly positive penalties.
PENALTIES = np.linspace(0.01, 2.0, 10)


with open("data/multibkps_block2activities_snapshots.pkl", "rb") as f:
    dataset = pickle.load(f)

first_net = dataset[0]["tnet"]
num_snapshots = max(1, len(first_net.times) - 1)
max_window_length = max(1, min(5, num_snapshots - 1))
window_lengths = list(range(1, max_window_length + 1))
margin = 1.0
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

summary = grid_search_frobenius_distance(
    samples=training_samples,
    window_lengths=window_lengths,
    margin=margin,
    n_jobs=n_jobs,
    outdir="./gridsearch_results/multibkps_block2activities_snapshots_frobenius",
    kernel="linear",
    save_signals=True,
    signals_outdir="./gridsearch_results/multibkps_block2activities_snapshots_frobenius/signals",
    selection_metric="hausdorff",
    stopping_rule="penalty",
    penalties=PENALTIES,
)

print("Number of samples:", len(training_samples))
print("Score array shape:", summary["score_array"].shape)
print("Backend used:", summary["backend_used"])
print("Selection metric:", summary["selection_metric"])
print("Stopping rule:", summary["stopping_rule"])
print("Penalties:", summary["penalties"])
print("Best window_length:", summary["best_window_length"])
print("Best penalty:", summary["best_penalty"])
print("Best selected score:", summary["best_score"])
print("Best mean F1:", summary["best_f1"])
print("Best mean Hausdorff:", summary["best_hausdorff"])
print("Total runtime:", summary["elapsed_seconds"])
print("Signal generation phase runtime:", summary["signal_generation_phase_seconds"])
print("Detection and metrics phase runtime:", summary["detection_metrics_phase_seconds"])
print("Signals saved:", summary["save_signals"])
print("Signals output directory:", summary["signals_outdir"])
