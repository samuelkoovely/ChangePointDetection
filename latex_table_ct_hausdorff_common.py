from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path
from typing import Sequence


def resolve_existing_summary_path(candidates: Sequence[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def parse_args(
    *,
    description: str,
    default_training_summary_paths: Sequence[Path],
    default_test_summary_paths: Sequence[Path],
    default_caption: str,
    default_label: str,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--training-summary",
        type=Path,
        default=None,
        help=(
            "Optional override for the training grid-search summary pickle. "
            "If omitted, the first existing default candidate is used."
        ),
    )
    parser.add_argument(
        "--test-summary",
        type=Path,
        default=None,
        help=(
            "Optional override for the test-set evaluation summary pickle. "
            "If omitted, the first existing default candidate is used."
        ),
    )
    parser.add_argument(
        "--test-score",
        type=float,
        default=None,
        help=(
            "Optional override Hausdorff score to use for the test set. "
            "If omitted, load the saved test-set results."
        ),
    )
    parser.add_argument(
        "--caption",
        default=default_caption,
        help="LaTeX table caption.",
    )
    parser.add_argument(
        "--label",
        default=default_label,
        help="LaTeX table label.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output .tex file. If omitted, print to stdout.",
    )
    parser.set_defaults(
        default_training_summary=resolve_existing_summary_path(default_training_summary_paths),
        default_test_summary=resolve_existing_summary_path(default_test_summary_paths),
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
    *,
    training_score: float,
    test_score: float,
    caption: str,
    label: str,
) -> str:
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\begin{tabular}{lc}",
        r"\hline",
        r"Split & Entropy \\",
        r"\hline",
        f"Training Hausdorff & {format_score(training_score)} \\\\",
        f"Test Hausdorff & {format_score(test_score)} \\\\",
        r"\hline",
        r"\end{tabular}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main(
    *,
    description: str,
    default_training_summary_paths: Sequence[Path],
    default_test_summary_paths: Sequence[Path],
    default_caption: str,
    default_label: str,
) -> None:
    args = parse_args(
        description=description,
        default_training_summary_paths=default_training_summary_paths,
        default_test_summary_paths=default_test_summary_paths,
        default_caption=default_caption,
        default_label=default_label,
    )

    training_summary = args.training_summary or args.default_training_summary
    test_summary = args.test_summary or args.default_test_summary

    training_score = load_metric(training_summary, "best_hausdorff")
    if args.test_score is None:
        test_score = load_metric(test_summary, "test_mean_hausdorff")
    else:
        test_score = float(args.test_score)

    latex_table = build_latex_table(
        training_score=training_score,
        test_score=test_score,
        caption=args.caption,
        label=args.label,
    )

    if args.output is None:
        print(latex_table)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(latex_table + "\n", encoding="utf-8")
