import numpy as np
import pickle

from gridsearch_block2activities_snapshots_laplacian_similarity import (
    CPSample,
    grid_search_laplacian_similarity,
)
from gridsearch_score_snapshots import extract_true_change_points


PENALTY = 0.5


with open("data/multibkps_block2activities_snapshots.pkl", "rb") as f:
    dataset = pickle.load(f)

aggregation_window = int(dataset[0]["aggregation_window"])
first_net = dataset[0]["tnet"]
window_lengths = [max(1, aggregation_window // 2)]
max_n_eigen = max(1, int(first_net.num_nodes) - 1)
n_eigens = [k for k in [3, 4, 6, 8, 10] if k <= max_n_eigen]
if not n_eigens:
    raise ValueError("No valid n_eigen >= 3 is available for this dataset.")

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

summary = grid_search_laplacian_similarity(
    samples=training_samples,
    window_lengths=window_lengths,
    n_eigens=n_eigens,
    margin=margin,
    n_jobs=n_jobs,
    outdir="./gridsearch_results/multibkps_block2activities_snapshots_laplacians",
    kernel="linear",
    normalize=False,
    save_signals=True,
    signals_outdir="./gridsearch_results/multibkps_block2activities_snapshots_laplacians/signals",
    selection_metric="f1",
    stopping_rule="penalty",
    penalty=PENALTY,
)

print("Number of samples:", len(training_samples))
print("Score array shape:", summary["score_array"].shape)
print("Backend used:", summary["backend_used"])
print("Selection metric:", summary["selection_metric"])
print("Stopping rule:", summary["stopping_rule"])
print("Penalty:", summary["penalty"])
print("Best n_eigen:", summary["best_n_eigen"])
print("Best window_length:", summary["best_window_length"])
print("Best selected score:", summary["best_score"])
print("Best mean F1:", summary["best_f1"])
print("Best mean Hausdorff:", summary["best_hausdorff"])
print("Total runtime:", summary["elapsed_seconds"])
print("Signal generation phase runtime:", summary["signal_generation_phase_seconds"])
print("Detection and metrics phase runtime:", summary["detection_metrics_phase_seconds"])
print("Signals saved:", summary["save_signals"])
print("Signals output directory:", summary["signals_outdir"])
