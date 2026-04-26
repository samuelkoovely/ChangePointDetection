from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path


DEFAULT_SUMMARY_PATHS = {
    "Entropy": Path(
        "gridsearch_results/multibkps_block2activities_snapshots/gridsearch_results.pkl"
    ),
    "Frobenius": Path(
        "gridsearch_results/multibkps_block2activities_snapshots_frobenius/gridsearch_results.pkl"
    ),
    "LAD": Path(
        "gridsearch_results/multibkps_block2activities_snapshots_laplacians/gridsearch_results.pkl"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a LaTeX table with F1 and Hausdorff scores for the "
            "multibkps block2activities snapshot training set."
        )
    )
    parser.add_argument(
        "--test-f1",
        type=float,
        default=0.5,
        help="Placeholder F1 score to use for the test set.",
    )
    parser.add_argument(
        "--test-hausdorff",
        type=float,
        default=0.5,
        help="Placeholder Hausdorff score to use for the test set.",
    )
    parser.add_argument(
        "--caption",
        default=(
            "F1 and Hausdorff scores for the multibkps block2activities "
            "snapshot dataset. Training values are loaded from the saved "
            "grid-search summaries; test values are placeholders."
        ),
        help="LaTeX table caption.",
    )
    parser.add_argument(
        "--label",
        default="tab:multibkps-block2activities-snapshots-metrics",
        help="LaTeX table label.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output .tex file. If omitted, print to stdout.",
    )
    return parser.parse_args()


def load_summary_metrics(summary_path: Path) -> dict[str, float]:
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")

    with open(summary_path, "rb") as handle:
        summary = pickle.load(handle)

    missing_keys = [key for key in ("best_f1", "best_hausdorff") if key not in summary]
    if missing_keys:
        raise KeyError(
            f"Summary file is missing keys {missing_keys}: {summary_path}"
        )

    return {
        "best_f1": float(summary["best_f1"]),
        "best_hausdorff": float(summary["best_hausdorff"]),
    }


def format_score(value: float) -> str:
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return r"$\infty$"
    return f"{value:.3f}"


def build_latex_table(
    training_metrics: dict[str, dict[str, float]],
    test_f1: float,
    test_hausdorff: float,
    caption: str,
    label: str,
) -> str:
    method_names = list(training_metrics.keys())
    header = " & ".join(["Metric", *method_names]) + r" \\"
    rows = [
        (
            "Training F1",
            [format_score(training_metrics[name]["best_f1"]) for name in method_names],
        ),
        (
            "Training Hausdorff",
            [
                format_score(training_metrics[name]["best_hausdorff"])
                for name in method_names
            ],
        ),
        ("Test F1", [format_score(test_f1) for _ in method_names]),
        (
            "Test Hausdorff",
            [format_score(test_hausdorff) for _ in method_names],
        ),
    ]

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\begin{tabular}{l" + "c" * len(method_names) + "}",
        r"\hline",
        header,
        r"\hline",
    ]
    for label_text, values in rows:
        lines.append(" & ".join([label_text, *values]) + r" \\")

    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    training_metrics = {
        method_name: load_summary_metrics(summary_path)
        for method_name, summary_path in DEFAULT_SUMMARY_PATHS.items()
    }
    latex_table = build_latex_table(
        training_metrics=training_metrics,
        test_f1=args.test_f1,
        test_hausdorff=args.test_hausdorff,
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
