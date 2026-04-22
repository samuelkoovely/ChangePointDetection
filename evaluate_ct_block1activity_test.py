from __future__ import annotations

from evaluate_ct_test_set_common import (
    build_ct_test_evaluation_parser,
    print_test_summary,
    recommended_parallel_jobs,
    run_ct_entropy_test_evaluation,
)


def main() -> None:
    parser = build_ct_test_evaluation_parser(
        default_training_results_path="gridsearch_results/ct_block1activity/gridsearch_results.pkl",
        default_test_dataset_path="data/ct_block1activity_test.pkl",
        default_outdir="gridsearch_results/ct_block1activity",
        description="Evaluate the best continuous-time entropy model on the ct_block1activity test set.",
    )
    args = parser.parse_args()
    if args.recommend_max_jobs:
        print(recommended_parallel_jobs())
        return
    summary = run_ct_entropy_test_evaluation(
        training_results_path=args.training_results_path,
        test_dataset_path=args.test_dataset_path,
        outdir=args.outdir,
        save_signals=bool(args.save_signals),
        results_filename=str(args.results_filename),
        test_signals_dirname=str(args.test_signals_dirname),
        n_jobs=int(args.n_jobs),
        backend=str(args.backend),
        verbose=int(args.verbose),
    )
    print_test_summary(summary)


if __name__ == "__main__":
    main()
