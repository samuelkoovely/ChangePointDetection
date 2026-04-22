from __future__ import annotations

from generate_block1activity import (
    TEST_OUTPUT_PATH,
    generate_test_dataset,
    write_dataset,
)


def main() -> None:
    test_dataset = generate_test_dataset()
    write_dataset(test_dataset, TEST_OUTPUT_PATH)


if __name__ == "__main__":
    main()
