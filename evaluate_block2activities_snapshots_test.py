from __future__ import annotations

from evaluate_snapshot_test_set_common import (
    print_test_summary,
    run_entropy_test_evaluation,
)


def main() -> None:
    summary = run_entropy_test_evaluation(
        training_results_path="gridsearch_results/block2activities_snapshots/gridsearch_results.pkl",
        test_dataset_path="data/block2activities_test_snapshots.pkl",
        outdir="gridsearch_results/block2activities_snapshots",
    )
    print_test_summary(summary)


if __name__ == "__main__":
    main()
