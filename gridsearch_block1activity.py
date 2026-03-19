import numpy as np
import pickle
from gridsearch_F1score import CPSample, grid_search_f1
import math

with open("block1activity.pkl", "rb") as f:
        dataset = pickle.load(f)

lambdas = np.logspace(-1, 1, 2)
windows = [1, 5]
margin = 5.0
n_jobs = 4

training_samples = [
    CPSample(
        data=entry["tnet"],
        true_change_points=[float(entry["t_bkp"])],
        n_bkps=1,
        name=f"sample_{i}",
    )
    for i, entry in enumerate(dataset)
]

summary = grid_search_f1(
        samples=training_samples,
        lambdas=lambdas,
        windows=windows,
        margin=margin,
        n_jobs=n_jobs,
        outdir="./gridsearch_results/block1activity_fast",
        sample_fraction=0.001,
        kernel="linear",
        save_signals=True,
        signals_outdir="./gridsearch_results/block1activity_fast/signals",
        signal_dir_order="lambda_window",
        selection_metric="f1",
    )

print("Number of samples:", len(training_samples))
print("Score array shape:", summary["score_array"].shape)
print("Best lamda:", summary["best_lamda"])
print("Best window:", summary["best_window"])
print("Best mean F1:", summary["best_score"])
print("Hausdorff at best F1 params:", summary["hausdorff_array"][summary["best_index"]] if summary["best_index"] is not None else math.nan)
print("Total runtime:", summary["elapsed_seconds"])
print("Total signal generation time:", summary["total_signal_generation_seconds"])
print("Total change-point detection time:", summary["total_change_point_detection_seconds"])
print("Total metrics computation time:", summary["total_metrics_seconds"])
if summary["best_index"] is not None:
    print("Signal generation time at best params:", summary["signal_generation_time_array"][summary["best_index"]])
    print("Change-point detection time at best params:", summary["change_point_detection_time_array"][summary["best_index"]])
    print("Metrics computation time at best params:", summary["metrics_time_array"][summary["best_index"]])