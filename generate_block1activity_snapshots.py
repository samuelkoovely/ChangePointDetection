from __future__ import annotations

from pathlib import Path

from generate_snapshot_dataset_common import (
    AGGREGATION_WINDOW,
    SNAPSHOTS_TO_SKIP_AT_START,
    convert_dataset_files,
)


DATASET_SPECS = (
    (
        Path("data/block1activity_train.pkl"),
        (Path("data/block1activity_train_snapshots.pkl"),),
    ),
    (
        Path("data/block1activity_test.pkl"),
        (Path("data/block1activity_test_snapshots.pkl"),),
    ),
)


def main() -> None:
    convert_dataset_files(
        dataset_specs=DATASET_SPECS,
        aggregation_window=AGGREGATION_WINDOW,
        snapshots_to_skip_at_start=SNAPSHOTS_TO_SKIP_AT_START,
    )


if __name__ == "__main__":
    main()
