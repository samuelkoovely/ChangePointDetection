from __future__ import annotations

from pathlib import Path

from gridsearch_ct_block_activity_common import CTGridSearchSpec, main


SPEC = CTGridSearchSpec(
    dataset_key="ct_block1activity",
    dataset_path=Path("data/ct_block1activity.pkl"),
    output_dir=Path("gridsearch_results/ct_block1activity"),
    sample_fraction=0.1,
    default_lambdas=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0),
    default_windows=(0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
)


if __name__ == "__main__":
    main(SPEC)
