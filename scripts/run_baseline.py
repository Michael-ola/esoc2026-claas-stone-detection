import argparse
from pathlib import Path

import pandas as pd

from claas_stone_detection.data.io import read_dataset
from claas_stone_detection.pipelines.baseline_pipeline import (
    RandomForestEvaluationConfig,
    build_labeled_feature_table,
    detect_reference_events_from_voltage,
    parse_threshold_sweep,
    print_labeled_dataset_summary,
    print_overall_threshold_sweep,
    run_grouped_random_forest_evaluation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Random Forest early stone-detection baseline."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--raster", type=float, default=0.001)
    parser.add_argument("--window-s", type=float, default=0.5)
    parser.add_argument("--hop-s", type=float, default=0.1)
    parser.add_argument("--positive-horizon-s", type=float, default=1.0)
    parser.add_argument(
        "--window-region",
        type=str,
        choices=["header-on", "extended"],
        default="header-on",
        help=(
            "Region used to create model input windows. "
            "'header-on' uses only active header operation. "
            "'extended' also includes the post-header grace region."
        ),
    )
    parser.add_argument("--post-event-exclusion-s", type=float, default=1.0)
    parser.add_argument("--event-threshold-quantile", type=float, default=0.999)
    parser.add_argument(
        "--min-event-ratio",
        type=float,
        default=1.0,
        help=(
            "Minimum peak-to-threshold ratio for keeping reference voltage events. "
            "Use values such as 1.10 to evaluate only stronger reference events."
        ),
    )
    parser.add_argument("--score-threshold", type=float, default=0.10)
    parser.add_argument(
        "--threshold-sweep",
        type=str,
        default="0.05,0.10,0.15,0.20,0.30,0.50",
    )
    parser.add_argument("--max-early-s", type=float, default=2.0)
    parser.add_argument("--refractory-s", type=float, default=1.0)
    parser.add_argument("--consensus-k", type=int, default=1)
    parser.add_argument("--consensus-n", type=int, default=1)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    threshold_sweep = parse_threshold_sweep(args.threshold_sweep)

    print("\nCLAAS Random Forest baseline")
    print("=" * 32)
    print(f"Data directory: {args.data_dir}")
    print(f"Raster: {args.raster}")
    print(f"Window: {args.window_s}s")
    print(f"Hop: {args.hop_s}s")
    print(f"Consensus: {args.consensus_k} of {args.consensus_n}")
    print(f"Window region: {args.window_region}")

    dataset = read_dataset(args.data_dir, raster=args.raster)
    events_by_run = detect_reference_events_from_voltage(
        dataset=dataset,
        event_threshold_quantile=args.event_threshold_quantile,
        min_event_ratio=args.min_event_ratio,
    )

    labeled_table, evaluated_duration_by_run = build_labeled_feature_table(
        dataset=dataset,
        events_by_run=events_by_run,
        window_s=args.window_s,
        hop_s=args.hop_s,
        positive_horizon_s=args.positive_horizon_s,
        post_event_exclusion_s=args.post_event_exclusion_s,
        window_region=args.window_region,
    )

    if labeled_table.empty:
        raise RuntimeError("No feature windows were generated.")

    print_labeled_dataset_summary(
        dataset=dataset,
        labeled_table=labeled_table,
        events_by_run=events_by_run,
        min_event_ratio=args.min_event_ratio,
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
        label="Grouped cross-validation",
    )

    prediction_table = pd.concat(predictions, ignore_index=True)

    print_overall_threshold_sweep(
        prediction_table=prediction_table,
        events_by_run=events_by_run,
        evaluated_duration_by_run=evaluated_duration_by_run,
        config=config,
        threshold_sweep=threshold_sweep,
        title="Overall grouped cross-validation threshold sweep",
    )


if __name__ == "__main__":
    main()
