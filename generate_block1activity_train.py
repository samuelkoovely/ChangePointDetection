from __future__ import annotations

from generate_block1activity import (
    TRAIN_OUTPUT_PATH,
    generate_training_dataset,
    write_dataset,
)


def main() -> None:
    training_dataset = generate_training_dataset()
    write_dataset(training_dataset, TRAIN_OUTPUT_PATH)


if __name__ == "__main__":
    main()
