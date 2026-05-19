import argparse
from pathlib import Path

import pandas as pd

from claas_stone_detection.data.synthetic_io import (
    metadata_to_stone_events,
    read_synthetic_dataset,
    read_synthetic_metadata,
)
from claas_stone_detection.pipelines.baseline_pipeline import (
    RandomForestEvaluationConfig,
    build_labeled_feature_table,
    parse_threshold_sweep,
    print_labeled_dataset_summary,
    print_overall_threshold_sweep,
    run_grouped_random_forest_evaluation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Random Forest baseline on generated synthetic data."
    )
    parser.add_argument(
        "--synthetic-dir",
        type=Path,
        default=Path("synthetic_data"),
        help="Directory containing synthetic_run_*.csv and metadata.csv.",
    )
    parser.add_argument("--window-s", type=float, default=0.5)
    parser.add_argument("--hop-s", type=float, default=0.1)
    parser.add_argument("--positive-horizon-s", type=float, default=1.0)
    parser.add_argument("--post-event-exclusion-s", type=float, default=1.0)
    parser.add_argument("--score-threshold", type=float, default=0.10)
    parser.add_argument(
        "--threshold-sweep",
        type=str,
        default="0.05,0.10,0.15,0.20,0.30,0.50",
    )
    parser.add_argument("--max-early-s", type=float, default=1.0)
    parser.add_argument("--refractory-s", type=float, default=1.0)
    parser.add_argument("--consensus-k", type=int, default=1)
    parser.add_argument("--consensus-n", type=int, default=1)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional limit for quick synthetic smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    threshold_sweep = parse_threshold_sweep(args.threshold_sweep)

    print("\nCLAAS synthetic Random Forest baseline")
    print("=" * 40)
    print(f"Synthetic directory: {args.synthetic_dir}")
    print(f"Window: {args.window_s}s")
    print(f"Hop: {args.hop_s}s")
    print(f"Positive horizon: {args.positive_horizon_s}s")
    print(f"Consensus: {args.consensus_k} of {args.consensus_n}")

    dataset = read_synthetic_dataset(
        synthetic_dir=args.synthetic_dir,
        max_runs=args.max_runs,
    )
    metadata = read_synthetic_metadata(
        synthetic_dir=args.synthetic_dir,
        allowed_runs=set(dataset.keys()),
    )
    events_by_run = metadata_to_stone_events(metadata)

    labeled_table, evaluated_duration_by_run = build_labeled_feature_table(
        dataset=dataset,
        events_by_run=events_by_run,
        window_s=args.window_s,
        hop_s=args.hop_s,
        positive_horizon_s=args.positive_horizon_s,
        post_event_exclusion_s=args.post_event_exclusion_s,
        window_region="header-on",
    )

    if labeled_table.empty:
        raise RuntimeError("No synthetic feature windows were generated.")

    print_labeled_dataset_summary(
        dataset=dataset,
        labeled_table=labeled_table,
        events_by_run=events_by_run,
    )

    config = RandomForestEvaluationConfig(
        n_splits=args.n_splits,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
        score_threshold=args.score_threshold,
        max_early_s=args.max_early_s,
        refractory_s=args.refractory_s,
        consensus_k=args.consensus_k,
        consensus_n=args.consensus_n,
    )

    predictions = run_grouped_random_forest_evaluation(
        labeled_table=labeled_table,
        events_by_run=events_by_run,
        evaluated_duration_by_run=evaluated_duration_by_run,
        config=config,
        threshold_sweep=threshold_sweep,
        label="Grouped synthetic cross-validation",
    )

    prediction_table = pd.concat(predictions, ignore_index=True)

    print_overall_threshold_sweep(
        prediction_table=prediction_table,
        events_by_run=events_by_run,
        evaluated_duration_by_run=evaluated_duration_by_run,
        config=config,
        threshold_sweep=threshold_sweep,
        title="Overall synthetic grouped cross-validation threshold sweep",
    )


if __name__ == "__main__":
    main()
