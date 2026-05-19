import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold

from claas_stone_detection.data.io import read_dataset
from claas_stone_detection.evaluation.metrics import evaluate_predictions
from claas_stone_detection.models.baseline import train_random_forest_baseline
from claas_stone_detection.reference.episodes import extract_header_on_episodes
from claas_stone_detection.reference.events import StoneEvent, detect_voltage_events
from claas_stone_detection.reference.labels import IGNORE_LABEL, label_feature_table
from claas_stone_detection.streaming.features import (
    add_temporal_delta_features,
    make_feature_table,
)
from claas_stone_detection.streaming.windowing import make_sliding_windows


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

    (
        feature_table,
        events_by_run,
        evaluated_duration_by_run,
    ) = build_feature_and_reference_tables(
        dataset=dataset,
        window_s=args.window_s,
        hop_s=args.hop_s,
        event_threshold_quantile=args.event_threshold_quantile,
        min_event_ratio=args.min_event_ratio,
        window_region=args.window_region,
    )

    if feature_table.empty:
        raise RuntimeError("No feature windows were generated.")

    feature_table = add_temporal_delta_features(feature_table)

    labeled_table = label_feature_table(
        feature_table=feature_table,
        events_by_run=events_by_run,
        positive_horizon_s=args.positive_horizon_s,
        post_event_exclusion_s=args.post_event_exclusion_s,
    )

    print("\nDataset summary")
    print("-" * 15)
    print(f"Runs: {len(dataset)}")
    print(f"Windows: {len(labeled_table)}")
    print(f"Reference events: {sum(len(events) for events in events_by_run.values())}")
    print(f"Min event ratio: {args.min_event_ratio}")
    print("Label counts:")
    print(labeled_table["label"].value_counts().sort_index().to_string())

    predictions = run_grouped_cross_validation(
        labeled_table=labeled_table,
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
        events_by_run=events_by_run,
        evaluated_duration_by_run=evaluated_duration_by_run,
        threshold_sweep=threshold_sweep,
    )

    if not predictions:
        raise RuntimeError("No cross-validation predictions were produced.")

    prediction_table = pd.concat(predictions, ignore_index=True)
    reference_events = events_to_dataframe(events_by_run)
    total_duration_s = sum(evaluated_duration_by_run.values())

    print("\nOverall grouped cross-validation threshold sweep")
    print("-" * 50)

    for threshold in threshold_sweep:
        overall = evaluate_predictions(
            prediction_table=prediction_table,
            reference_events=reference_events,
            evaluated_duration_s=total_duration_s,
            threshold=threshold,
            max_early_s=args.max_early_s,
            refractory_s=args.refractory_s,
            consensus_k=args.consensus_k,
            consensus_n=args.consensus_n,
        )

        print(
            f"threshold={threshold:.2f} | "
            f"TPR={overall.true_positive_rate:.3f} | "
            f"detected={overall.n_detected_events}/{overall.n_reference_events} | "
            f"false/hour={overall.false_detections_per_hour:.3f} | "
            "sec/false="
            f"{format_optional_float(overall.mean_seconds_between_false_detections)} | "
            f"advance={format_optional_float(overall.average_advance_time_s)}"
        )


def build_feature_and_reference_tables(
    dataset: dict[str, pd.DataFrame],
    window_s: float,
    hop_s: float,
    event_threshold_quantile: float,
    min_event_ratio: float,
    window_region: str,
) -> tuple[pd.DataFrame, dict[str, list[StoneEvent]], dict[str, float]]:
    feature_tables: list[pd.DataFrame] = []
    events_by_run: dict[str, list[StoneEvent]] = {}
    evaluated_duration_by_run: dict[str, float] = {}

    for run_name, df in dataset.items():
        episodes = extract_header_on_episodes(df)
        events = detect_voltage_events(
            df=df,
            episodes=episodes,
            threshold_quantile=event_threshold_quantile,
            run_name=run_name,
        )
        events_by_run[run_name] = filter_events_by_ratio(
            events=events,
            min_event_ratio=min_event_ratio,
        )

        windows = []
        for episode in episodes:
            window_end_time = get_window_end_time(
                episode_end_time=episode.end_time,
                episode_extended_end_time=episode.extended_end_time,
                window_region=window_region,
            )

            windows.extend(
                make_sliding_windows(
                    df=df,
                    window_s=window_s,
                    hop_s=hop_s,
                    run_name=run_name,
                    start_time=episode.start_time,
                    end_time=window_end_time,
                )
            )

        evaluated_duration_by_run[run_name] = sum(
            get_evaluated_duration_s(
                episode_duration_s=episode.duration_s,
                episode_extended_duration_s=episode.extended_duration_s,
                window_region=window_region,
            )
            for episode in episodes
        )

        if windows:
            feature_tables.append(make_feature_table(df=df, windows=windows))

    if not feature_tables:
        return pd.DataFrame(), events_by_run, evaluated_duration_by_run

    return (
        pd.concat(feature_tables, ignore_index=True),
        events_by_run,
        evaluated_duration_by_run,
    )


def get_window_end_time(
    episode_end_time: float,
    episode_extended_end_time: float,
    window_region: str,
) -> float:
    if window_region == "header-on":
        return episode_end_time

    if window_region == "extended":
        return episode_extended_end_time

    raise ValueError("window_region must be 'header-on' or 'extended'.")


def get_evaluated_duration_s(
    episode_duration_s: float,
    episode_extended_duration_s: float,
    window_region: str,
) -> float:
    if window_region == "header-on":
        return episode_duration_s

    if window_region == "extended":
        return episode_extended_duration_s

    raise ValueError("window_region must be 'header-on' or 'extended'.")


def run_grouped_cross_validation(
    labeled_table: pd.DataFrame,
    n_splits: int,
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
    random_state: int,
    score_threshold: float,
    max_early_s: float,
    refractory_s: float,
    consensus_k: int,
    consensus_n: int,
    events_by_run: dict[str, list[StoneEvent]],
    evaluated_duration_by_run: dict[str, float],
    threshold_sweep: list[float],
) -> list[pd.DataFrame]:
    run_names = labeled_table["run_name"]
    unique_runs = sorted(run_names.unique().tolist())

    if len(unique_runs) < 2:
        raise ValueError("At least two runs are required for grouped validation.")

    actual_splits = min(n_splits, len(unique_runs))
    splitter = GroupKFold(n_splits=actual_splits)
    predictions: list[pd.DataFrame] = []

    print("\nGrouped cross-validation")
    print("-" * 24)

    for fold_index, (train_index, test_index) in enumerate(
        splitter.split(labeled_table, groups=run_names),
        start=1,
    ):
        train_table = labeled_table.iloc[train_index].copy()
        test_table = labeled_table.iloc[test_index].copy()
        validation_runs = sorted(test_table["run_name"].unique().tolist())

        baseline = train_random_forest_baseline(
            labeled_table=train_table,
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
        )

        test_table["score"] = baseline.predict_proba(test_table)
        evaluation_table = test_table[test_table["label"] != IGNORE_LABEL].copy()

        validation_events = {
            run_name: events_by_run.get(run_name, []) for run_name in validation_runs
        }
        reference_events = events_to_dataframe(validation_events)
        validation_duration_s = sum(
            evaluated_duration_by_run.get(run_name, 0.0)
            for run_name in validation_runs
        )

        fold_result = evaluate_predictions(
            prediction_table=evaluation_table,
            reference_events=reference_events,
            evaluated_duration_s=validation_duration_s,
            threshold=score_threshold,
            max_early_s=max_early_s,
            refractory_s=refractory_s,
            consensus_k=consensus_k,
            consensus_n=consensus_n,
        )

        print(
            f"Fold {fold_index}: "
            f"validation_runs={validation_runs} | "
            f"TPR={fold_result.true_positive_rate:.3f} | "
            f"false/hour={fold_result.false_detections_per_hour:.3f} | "
            f"advance={format_optional_float(fold_result.average_advance_time_s)}"
        )

        sweep_summary = []
        for threshold in threshold_sweep:
            sweep_result = evaluate_predictions(
                prediction_table=evaluation_table,
                reference_events=reference_events,
                evaluated_duration_s=validation_duration_s,
                threshold=threshold,
                max_early_s=max_early_s,
                refractory_s=refractory_s,
                consensus_k=consensus_k,
                consensus_n=consensus_n,
            )
            sweep_summary.append(
                f"{threshold:.2f}:TPR={sweep_result.true_positive_rate:.2f},"
                f"FA/h={sweep_result.false_detections_per_hour:.1f},"
            f"sec/FA={format_optional_float(sweep_result.mean_seconds_between_false_detections)}"
            )

        print("  sweep: " + " | ".join(sweep_summary))
        predictions.append(evaluation_table)

    return predictions


def filter_events_by_ratio(
    events: list[StoneEvent],
    min_event_ratio: float,
) -> list[StoneEvent]:
    if min_event_ratio < 0:
        raise ValueError("min_event_ratio cannot be negative.")

    return [
        event
        for event in events
        if event.peak_to_threshold_ratio >= min_event_ratio
    ]


def events_to_dataframe(events_by_run: dict[str, list[StoneEvent]]) -> pd.DataFrame:
    rows = []

    for run_name, events in events_by_run.items():
        for event in events:
            rows.append(
                {
                    "run_name": run_name,
                    "peak_time": event.peak_time,
                    "peak_voltage": event.peak_voltage,
                    "threshold": event.threshold,
                    "peak_to_threshold_ratio": event.peak_to_threshold_ratio,
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "run_name",
            "peak_time",
            "peak_voltage",
            "threshold",
            "peak_to_threshold_ratio",
        ],
    )


def parse_threshold_sweep(raw_thresholds: str) -> list[float]:
    thresholds = [float(value.strip()) for value in raw_thresholds.split(",")]

    if not thresholds:
        raise ValueError("At least one threshold must be provided.")

    for threshold in thresholds:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("All threshold sweep values must be between 0 and 1.")

    return thresholds


def format_optional_float(value: float | None) -> str:
    if value is None:
        return "n/a"

    return f"{value:.3f}s"


if __name__ == "__main__":
    main()
