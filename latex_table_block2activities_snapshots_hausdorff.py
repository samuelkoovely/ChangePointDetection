from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path


DEFAULT_SUMMARY_PATHS = {
    "Entropy": Path("gridsearch_results/block2activities_snapshots/gridsearch_results.pkl"),
    "Frobenius": Path(
        "gridsearch_results/block2activities_snapshots_frobenius/gridsearch_results.pkl"
    ),
    "LAD": Path(
        "gridsearch_results/block2activities_snapshots_laplacians/gridsearch_results.pkl"
    ),
}

DEFAULT_TEST_SUMMARY_PATHS = {
    "Entropy": Path("gridsearch_results/block2activities_snapshots/test_set_results.pkl"),
    "Frobenius": Path(
        "gridsearch_results/block2activities_snapshots_frobenius/test_set_results.pkl"
    ),
    "LAD": Path(
        "gridsearch_results/block2activities_snapshots_laplacians/test_set_results.pkl"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a LaTeX table with Hausdorff scores for the "
            "block2activities snapshot training and test sets."
        )
    )
    parser.add_argument(
        "--test-score",
        type=float,
        default=None,
        help=(
            "Optional override Hausdorff score to use for all test-set methods. "
            "If omitted, load the saved test-set results."
        ),
    )
    parser.add_argument(
        "--caption",
        default=(
            "Hausdorff scores for the block2activities snapshot dataset. "
            "Training and test values are loaded from the saved grid-search "
            "and test-set summaries."
        ),
        help="LaTeX table caption.",
    )
    parser.add_argument(
        "--label",
        default="tab:block2activities-snapshots-hausdorff",
        help="LaTeX table label.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output .tex file. If omitted, print to stdout.",
    )
    return parser.parse_args()


def load_metric(summary_path: Path, key: str) -> float:
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")

    with open(summary_path, "rb") as handle:
        summary = pickle.load(handle)

    if key not in summary:
        raise KeyError(f"Summary file does not contain '{key}': {summary_path}")

    return float(summary[key])


def format_score(value: float) -> str:
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return r"$\infty$"
    return f"{value:.3f}"


def build_latex_table(
    training_scores: dict[str, float],
    test_scores: dict[str, float],
    caption: str,
    label: str,
) -> str:
    method_names = list(training_scores.keys())
    header = " & ".join(["Split", *method_names]) + r" \\"
    train_row = " & ".join(
        ["Training Hausdorff", *[format_score(training_scores[name]) for name in method_names]]
    ) + r" \\"
    test_row = " & ".join(
        ["Test Hausdorff", *[format_score(test_scores[name]) for name in method_names]]
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
        method_name: load_metric(summary_path, "best_hausdorff")
        for method_name, summary_path in DEFAULT_SUMMARY_PATHS.items()
    }
    if args.test_score is None:
        test_scores = {
            method_name: load_metric(summary_path, "test_mean_hausdorff")
            for method_name, summary_path in DEFAULT_TEST_SUMMARY_PATHS.items()
        }
    else:
        test_scores = {
            method_name: float(args.test_score)
            for method_name in training_scores
        }

    latex_table = build_latex_table(
        training_scores=training_scores,
        test_scores=test_scores,
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
