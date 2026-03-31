from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path


DEFAULT_SUMMARY_PATHS = {
    "Entropy": Path("gridsearch_results/block1activity_snapshots/gridsearch_results.pkl"),
    "Frobenius": Path(
        "gridsearch_results/block1activity_snapshots_frobenius/gridsearch_results.pkl"
    ),
    "Laplacian": Path(
        "gridsearch_results/block1activity_snapshots_laplacians/gridsearch_results.pkl"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a LaTeX table with Hausdorff scores for the "
            "block1activity snapshot training set."
        )
    )
    parser.add_argument(
        "--test-score",
        type=float,
        default=0.5,
        help="Placeholder Hausdorff score to use for the test set.",
    )
    parser.add_argument(
        "--caption",
        default=(
            "Hausdorff scores for the block1activity snapshot dataset. "
            "Training values are loaded from the saved grid-search summaries; "
            "test values are placeholders."
        ),
        help="LaTeX table caption.",
    )
    parser.add_argument(
        "--label",
        default="tab:block1activity-snapshots-hausdorff",
        help="LaTeX table label.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output .tex file. If omitted, print to stdout.",
    )
    return parser.parse_args()


def load_best_hausdorff(summary_path: Path) -> float:
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")

    with open(summary_path, "rb") as handle:
        summary = pickle.load(handle)

    if "best_hausdorff" not in summary:
        raise KeyError(
            f"Summary file does not contain 'best_hausdorff': {summary_path}"
        )

    return float(summary["best_hausdorff"])


def format_score(value: float) -> str:
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return r"$\infty$"
    return f"{value:.3f}"


def build_latex_table(
    training_scores: dict[str, float],
    test_score: float,
    caption: str,
    label: str,
) -> str:
    method_names = list(training_scores.keys())
    header = " & ".join(["Split", *method_names]) + r" \\"
    train_row = " & ".join(
        ["Training Hausdorff", *[format_score(training_scores[name]) for name in method_names]]
    ) + r" \\"
    test_row = " & ".join(
        ["Test Hausdorff", *[format_score(test_score) for _ in method_names]]
    ) + r" \\"

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\begin{tabular}{l" + "c" * len(method_names) + "}",
        r"\hline",
        header,
        r"\hline",
        train_row,
        test_row,
        r"\hline",
        r"\end{tabular}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    training_scores = {
        method_name: load_best_hausdorff(summary_path)
        for method_name, summary_path in DEFAULT_SUMMARY_PATHS.items()
    }
    latex_table = build_latex_table(
        training_scores=training_scores,
        test_score=args.test_score,
        caption=args.caption,
        label=args.label,
    )

    if args.output is None:
        print(latex_table)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(latex_table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
