from __future__ import annotations

import pickle

from gridsearch_block2activities_snapshots_laplacian_similarity import (
    CPSample,
    grid_search_laplacian_similarity,
    laplacian_spectrum_similarity,
)
from gridsearch_score_snapshots import extract_true_change_points


OUTDIR = "./gridsearch_results/block1activity_snapshots_laplacians"
SIGNALS_OUTDIR = f"{OUTDIR}/signals"
SECOND_WINDOW_SCALE = 2.0


def main() -> None:
    with open("data/block1activity_train_snapshots.pkl", "rb") as handle:
        dataset = pickle.load(handle)

    first_net = dataset[0]["tnet"]
    num_laplacians = max(1, len(first_net.times) - 1)
    max_window_length = max(1, min(8, num_laplacians - 1))
    max_n_eigen = max(1, int(first_net.num_nodes) - 1)

    window_lengths = list(range(1, max_window_length + 1))
    n_eigens = [k for k in [3, 4, 6, 8, 10] if k <= max_n_eigen]
    if not n_eigens:
        raise ValueError("No valid n_eigen is available for this dataset.")

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

    demo_window_length = min(5, max_window_length)
    demo_n_eigen = n_eigens[min(2, len(n_eigens) - 1)]
    demo_signal, demo_indices = laplacian_spectrum_similarity(
        first_net,
        window_length=demo_window_length,
        n_eigen=demo_n_eigen,
        normalize=False,
        top=True,
        difference=True,
        second_window_scale=SECOND_WINDOW_SCALE,
    )
    print("Demo signal length:", len(demo_signal))
    print("Demo index range:", demo_indices[:3], "...", demo_indices[-3:])

    summary = grid_search_laplacian_similarity(
        samples=training_samples,
        window_lengths=window_lengths,
        n_eigens=n_eigens,
        margin=margin,
        n_jobs=n_jobs,
        outdir=OUTDIR,
        normalize=False,
        save_signals=True,
        signals_outdir=SIGNALS_OUTDIR,
        selection_metric="hausdorff",
        top=True,
        difference=True,
        second_window_scale=SECOND_WINDOW_SCALE,
    )

    print("Number of samples:", len(training_samples))
    print("Score array shape:", summary["score_array"].shape)
    print("Backend used:", summary["backend_used"])
    print("Selection metric:", summary["selection_metric"])
    print("Ranking rule:", summary["ranking_rule"])
    print("Top singular values:", summary["top"])
    print("Difference signal:", summary["difference"])
    print("Second-window scale:", summary["second_window_scale"])
    print("Best n_eigen:", summary["best_n_eigen"])
    print("Best window_length:", summary["best_window_length"])
    print("Best second_window_length:", summary["best_second_window_length"])
    print("Best selected score:", summary["best_score"])
    print("Best mean F1:", summary["best_f1"])
    print("Best mean Hausdorff:", summary["best_hausdorff"])
    print("Total runtime:", summary["elapsed_seconds"])
    print(
        "Signal generation phase runtime:",
        summary["signal_generation_phase_seconds"],
    )
    print(
        "Detection and metrics phase runtime:",
        summary["detection_metrics_phase_seconds"],
    )
    print("Signals saved:", summary["save_signals"])
    print("Signals output directory:", summary["signals_outdir"])


if __name__ == "__main__":
    main()
