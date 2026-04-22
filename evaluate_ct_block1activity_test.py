from __future__ import annotations

from evaluate_ct_test_set_common import (
    print_test_summary,
    run_ct_entropy_test_evaluation,
)


def main() -> None:
    summary = run_ct_entropy_test_evaluation(
        training_results_path="gridsearch_results/ct_block1activity/gridsearch_results.pkl",
        test_dataset_path="data/ct_block1activity_test.pkl",
        outdir="gridsearch_results/ct_block1activity",
    )
    print_test_summary(summary)


if __name__ == "__main__":
    main()
