from __future__ import annotations

from pathlib import Path

from gridsearch_ct_block_activity_common import CTGridSearchSpec, main


SPEC = CTGridSearchSpec(
    dataset_key="ct_block2activities",
    dataset_path=Path("data/ct_block2activities.pkl"),
    output_dir=Path("gridsearch_results/ct_block2activities"),
    sample_fraction=0.0001,
)


if __name__ == "__main__":
    main(SPEC)
