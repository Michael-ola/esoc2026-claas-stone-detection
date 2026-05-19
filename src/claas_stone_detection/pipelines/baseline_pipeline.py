from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import GroupKFold

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


@dataclass(frozen=True)
class RandomForestEvaluationConfig:
    """Configuration for grouped Random Forest baseline evaluation."""

    n_splits: int = 5
    n_estimators: int = 100
    max_depth: int | None = None
    min_samples_leaf: int = 1
    random_state: int = 42
    score_threshold: float = 0.10
    max_early_s: float = 2.0
    refractory_s: float = 1.0
    consensus_k: int = 1
    consensus_n: int = 1


def detect_reference_events_from_voltage(
    dataset: dict[str, pd.DataFrame],
    event_threshold_quantile: float,
    min_event_ratio: float,
) -> dict[str, list[StoneEvent]]:
    """Detect and optionally filter voltage-derived reference events."""
    events_by_run: dict[str, list[StoneEvent]] = {}

    for run_name, frame in dataset.items():
        episodes = extract_header_on_episodes(frame)
        events = detect_voltage_events(
            df=frame,
            episodes=episodes,
            threshold_quantile=event_threshold_quantile,
            run_name=run_name,
        )
        events_by_run[run_name] = filter_events_by_ratio(
            events=events,
            min_event_ratio=min_event_ratio,
        )

    return events_by_run


def build_labeled_feature_table(
    dataset: dict[str, pd.DataFrame],
    events_by_run: dict[str, list[StoneEvent]],
    window_s: float,
    hop_s: float,
    positive_horizon_s: float,
    post_event_exclusion_s: float,
    window_region: str = "header-on",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Build labeled window features from a dataset and reference events.

    This is the shared Task 2 feature/label pipeline. It is used by the real
    MF4 baseline and by Bonus 1 synthetic datasets.
    """
    feature_table, evaluated_duration_by_run = build_feature_table_from_dataset(
        dataset=dataset,
        window_s=window_s,
        hop_s=hop_s,
        window_region=window_region,
    )

    if feature_table.empty:
        return feature_table, evaluated_duration_by_run

    feature_table = add_temporal_delta_features(feature_table)

    labeled_table = label_feature_table(
        feature_table=feature_table,
        events_by_run=events_by_run,
        positive_horizon_s=positive_horizon_s,
        post_event_exclusion_s=post_event_exclusion_s,
    )

    return labeled_table, evaluated_duration_by_run


def build_feature_table_from_dataset(
    dataset: dict[str, pd.DataFrame],
    window_s: float,
    hop_s: float,
    window_region: str = "header-on",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Create live-style window features for all runs in a dataset."""
    feature_tables: list[pd.DataFrame] = []
    evaluated_duration_by_run: dict[str, float] = {}

    for run_name, frame in dataset.items():
        episodes = extract_header_on_episodes(frame)
        windows = []

        for episode in episodes:
            window_end_time = get_window_end_time(
                episode_end_time=episode.end_time,
                episode_extended_end_time=episode.extended_end_time,
                window_region=window_region,
            )

            windows.extend(
                make_sliding_windows(
                    df=frame,
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
            feature_tables.append(make_feature_table(df=frame, windows=windows))

    if not feature_tables:
        return pd.DataFrame(), evaluated_duration_by_run

    return pd.concat(feature_tables, ignore_index=True), evaluated_duration_by_run


def run_grouped_random_forest_evaluation(
    labeled_table: pd.DataFrame,
    events_by_run: dict[str, list[StoneEvent]],
    evaluated_duration_by_run: dict[str, float],
    config: RandomForestEvaluationConfig,
    threshold_sweep: list[float],
    label: str = "Grouped cross-validation",
) -> list[pd.DataFrame]:
    """Run GroupKFold Random Forest evaluation by complete measurement run."""
    run_names = labeled_table["run_name"]
    unique_runs = sorted(run_names.unique().tolist())

    if len(unique_runs) < 2:
        raise ValueError("At least two runs are required for grouped validation.")

    actual_splits = min(config.n_splits, len(unique_runs))
    splitter = GroupKFold(n_splits=actual_splits)
    predictions: list[pd.DataFrame] = []

    print(f"\n{label}")
    print("-" * len(label))

    for fold_index, (train_index, test_index) in enumerate(
        splitter.split(labeled_table, groups=run_names),
        start=1,
    ):
        train_table = labeled_table.iloc[train_index].copy()
        test_table = labeled_table.iloc[test_index].copy()
        validation_runs = sorted(test_table["run_name"].unique().tolist())

        baseline = train_random_forest_baseline(
            labeled_table=train_table,
            n_estimators=config.n_estimators,
            random_state=config.random_state,
            max_depth=config.max_depth,
            min_samples_leaf=config.min_samples_leaf,
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
            threshold=config.score_threshold,
            max_early_s=config.max_early_s,
            refractory_s=config.refractory_s,
            consensus_k=config.consensus_k,
            consensus_n=config.consensus_n,
        )

        print(
            f"Fold {fold_index}: "
            f"runs={len(validation_runs)} | "
            f"TPR={fold_result.true_positive_rate:.3f} | "
            f"false/hour={fold_result.false_detections_per_hour:.3f} | "
            "sec/false="
            f"{format_optional_float(fold_result.mean_seconds_between_false_detections)} | "
            f"advance={format_optional_float(fold_result.average_advance_time_s)}"
        )

        print_threshold_sweep_summary(
            prediction_table=evaluation_table,
            reference_events=reference_events,
            evaluated_duration_s=validation_duration_s,
            config=config,
            threshold_sweep=threshold_sweep,
        )

        predictions.append(evaluation_table)

    return predictions


def print_overall_threshold_sweep(
    prediction_table: pd.DataFrame,
    events_by_run: dict[str, list[StoneEvent]],
    evaluated_duration_by_run: dict[str, float],
    config: RandomForestEvaluationConfig,
    threshold_sweep: list[float],
    title: str,
) -> None:
    """Print overall threshold sweep over all held-out predictions."""
    reference_events = events_to_dataframe(events_by_run)
    total_duration_s = sum(evaluated_duration_by_run.values())

    print(f"\n{title}")
    print("-" * len(title))

    for threshold in threshold_sweep:
        overall = evaluate_predictions(
            prediction_table=prediction_table,
            reference_events=reference_events,
            evaluated_duration_s=total_duration_s,
            threshold=threshold,
            max_early_s=config.max_early_s,
            refractory_s=config.refractory_s,
            consensus_k=config.consensus_k,
            consensus_n=config.consensus_n,
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


def print_threshold_sweep_summary(
    prediction_table: pd.DataFrame,
    reference_events: pd.DataFrame,
    evaluated_duration_s: float,
    config: RandomForestEvaluationConfig,
    threshold_sweep: list[float],
) -> None:
    """Print compact threshold sweep for one validation fold."""
    sweep_summary = []

    for threshold in threshold_sweep:
        sweep_result = evaluate_predictions(
            prediction_table=prediction_table,
            reference_events=reference_events,
            evaluated_duration_s=evaluated_duration_s,
            threshold=threshold,
            max_early_s=config.max_early_s,
            refractory_s=config.refractory_s,
            consensus_k=config.consensus_k,
            consensus_n=config.consensus_n,
        )
        sweep_summary.append(
            f"{threshold:.2f}:TPR={sweep_result.true_positive_rate:.2f},"
            f"FA/h={sweep_result.false_detections_per_hour:.1f},"
            "sec/FA="
            f"{format_optional_float(sweep_result.mean_seconds_between_false_detections)}"
        )

    print("  sweep: " + " | ".join(sweep_summary))


def print_labeled_dataset_summary(
    dataset: dict[str, pd.DataFrame],
    labeled_table: pd.DataFrame,
    events_by_run: dict[str, list[StoneEvent]],
    min_event_ratio: float | None = None,
) -> None:
    """Print a compact summary of runs, windows, events, and labels."""
    print("\nDataset summary")
    print("-" * 15)
    print(f"Runs: {len(dataset)}")
    print(f"Windows: {len(labeled_table)}")
    print(f"Reference events: {sum(len(events) for events in events_by_run.values())}")

    if min_event_ratio is not None:
        print(f"Min event ratio: {min_event_ratio}")

    print("Label counts:")
    print(labeled_table["label"].value_counts().sort_index().to_string())


def filter_events_by_ratio(
    events: list[StoneEvent],
    min_event_ratio: float,
) -> list[StoneEvent]:
    """Filter reference events by peak-to-threshold ratio."""
    if min_event_ratio < 0:
        raise ValueError("min_event_ratio cannot be negative.")

    return [
        event
        for event in events
        if event.peak_to_threshold_ratio >= min_event_ratio
    ]


def events_to_dataframe(events_by_run: dict[str, list[StoneEvent]]) -> pd.DataFrame:
    """Convert event dictionaries to the evaluation DataFrame format."""
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
    """Parse comma-separated threshold values."""
    thresholds = [float(value.strip()) for value in raw_thresholds.split(",")]

    if not thresholds:
        raise ValueError("At least one threshold must be provided.")

    for threshold in thresholds:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("All threshold sweep values must be between 0 and 1.")

    return thresholds


def get_window_end_time(
    episode_end_time: float,
    episode_extended_end_time: float,
    window_region: str,
) -> float:
    """Return window end time according to the selected region."""
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
    """Return evaluated duration according to the selected region."""
    if window_region == "header-on":
        return episode_duration_s

    if window_region == "extended":
        return episode_extended_duration_s

    raise ValueError("window_region must be 'header-on' or 'extended'.")


def format_optional_float(value: float | None) -> str:
    """Format optional seconds values."""
    if value is None:
        return "n/a"

    return f"{value:.3f}s"
