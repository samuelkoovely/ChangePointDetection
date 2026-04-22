from __future__ import annotations

from pathlib import Path

from latex_table_ct_hausdorff_common import main


if __name__ == "__main__":
    main(
        description=(
            "Build a LaTeX table with Hausdorff scores for the "
            "ct_block1activity continuous-time training and test sets."
        ),
        default_training_summary_paths=(
            Path("gridsearch_results/ct_block1activity/gridsearch_results.pkl"),
            Path("gridsearch_results/ct_block1/gridsearch_results.pkl"),
        ),
        default_test_summary_paths=(
            Path("gridsearch_results/ct_block1activity/test_set_results.pkl"),
            Path("gridsearch_results/ct_block1/test_set_results.pkl"),
        ),
        default_caption=(
            "Hausdorff scores for the ct_block1activity continuous-time dataset. "
            "Training and test values are loaded from the saved grid-search "
            "and test-set summaries."
        ),
        default_label="tab:ct-block1activity-hausdorff",
    )
