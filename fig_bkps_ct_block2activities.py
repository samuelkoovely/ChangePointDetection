from __future__ import annotations

from pathlib import Path

from fig_bkps_ct_block_activity_common import main


if __name__ == "__main__":
    main(
        default_results_path=Path("gridsearch_results/ct_block2activities/gridsearch_results.pkl"),
        default_dataset_path=Path("data/ct_block2activities.pkl"),
    )
