from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DetectionEvent:
    """One model alarm in a live signal."""

    run_name: str
    detection_time: float
    score: float


@dataclass(frozen=True)
class EventMatch:
    """Match between one reference event and one model detection."""

    run_name: str
    event_peak_time: float
    detection_time: float | None
    advance_time_s: float | None

    @property
    def detected(self) -> bool:
        """Whether this reference event was detected."""
        return self.detection_time is not None


@dataclass(frozen=True)
class EvaluationResult:
    """Summary metrics for early stone detection."""

    n_reference_events: int
    n_detected_events: int
    n_false_detections: int
    true_positive_rate: float
    false_detections_per_hour: float
    mean_seconds_between_false_detections: float | None
    average_advance_time_s: float | None


def prediction_table_to_detections(
    prediction_table: pd.DataFrame,
    threshold: float = 0.5,
    score_column: str = "score",
    consensus_k: int = 1,
    consensus_n: int = 1,
) -> list[DetectionEvent]:
    """Convert window-level prediction scores into model alarm events.

    A detection is emitted only when at least consensus_k of the latest
    consensus_n windows in the same run exceed the score threshold. The default
    consensus setting, 1-of-1, keeps the standard single-window alarm behavior.
    """
    required_columns = {"run_name", "detection_time", score_column}
    missing_columns = required_columns.difference(prediction_table.columns)

    if missing_columns:
        raise ValueError(f"Missing prediction table columns: {missing_columns}")

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    _validate_consensus(consensus_k=consensus_k, consensus_n=consensus_n)

    alarm_table = apply_consensus_alarm_filter(
        prediction_table=prediction_table,
        threshold=threshold,
        score_column=score_column,
        consensus_k=consensus_k,
        consensus_n=consensus_n,
    )

    detections: list[DetectionEvent] = []

    for row in alarm_table.itertuples(index=False):
        detections.append(
            DetectionEvent(
                run_name=str(row.run_name),
                detection_time=float(row.detection_time),
                score=float(getattr(row, score_column)),
            )
        )

    return detections


def apply_consensus_alarm_filter(
    prediction_table: pd.DataFrame,
    threshold: float,
    score_column: str = "score",
    consensus_k: int = 1,
    consensus_n: int = 1,
) -> pd.DataFrame:
    """Return rows that satisfy a k-of-n past-window alarm rule.

    The consensus window is causal: it uses the current row and previous rows
    within the same run, never future windows.
    """
    required_columns = {"run_name", "detection_time", score_column}
    missing_columns = required_columns.difference(prediction_table.columns)

    if missing_columns:
        raise ValueError(f"Missing prediction table columns: {missing_columns}")

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    _validate_consensus(consensus_k=consensus_k, consensus_n=consensus_n)

    sorted_table = prediction_table.sort_values(["run_name", "detection_time"]).copy()
    sorted_table["_above_threshold"] = sorted_table[score_column] >= threshold

    rolling_hits = (
        sorted_table.groupby("run_name")["_above_threshold"]
        .rolling(window=consensus_n, min_periods=consensus_k)
        .sum()
        .reset_index(level=0, drop=True)
    )

    sorted_table["_consensus_alarm"] = rolling_hits >= consensus_k
    alarm_table = sorted_table[sorted_table["_consensus_alarm"]].copy()

    return alarm_table.drop(columns=["_above_threshold", "_consensus_alarm"])


def suppress_repeated_detections(
    detections: list[DetectionEvent],
    refractory_s: float = 1.0,
) -> list[DetectionEvent]:
    """Suppress repeated alarms that occur within a refractory period."""
    if refractory_s < 0:
        raise ValueError("refractory_s cannot be negative.")

    sorted_detections = sorted(
        detections,
        key=lambda detection: (detection.run_name, detection.detection_time),
    )

    kept: list[DetectionEvent] = []
    last_kept_by_run: dict[str, DetectionEvent] = {}

    for detection in sorted_detections:
        previous = last_kept_by_run.get(detection.run_name)

        if (
            previous is None
            or detection.detection_time - previous.detection_time > refractory_s
        ):
            kept.append(detection)
            last_kept_by_run[detection.run_name] = detection

    return kept


def match_detections_to_events(
    reference_events: pd.DataFrame,
    detections: list[DetectionEvent],
    max_early_s: float = 2.0,
) -> tuple[list[EventMatch], list[DetectionEvent]]:
    """Match model detections to reference events efficiently.

    A detection matches an event if it occurs before the event peak and within
    the allowed early-detection horizon. Each detection can match at most one
    reference event.
    """
    required_columns = {"run_name", "peak_time"}
    missing_columns = required_columns.difference(reference_events.columns)

    if missing_columns:
        raise ValueError(f"Missing reference event columns: {missing_columns}")

    if max_early_s <= 0:
        raise ValueError("max_early_s must be positive.")

    detections_by_run = _group_detections_by_run(detections)
    used_detection_ids: set[int] = set()
    matches: list[EventMatch] = []

    sorted_events = reference_events.sort_values(["run_name", "peak_time"])

    for event in sorted_events.itertuples(index=False):
        run_name = str(event.run_name)
        event_peak_time = float(event.peak_time)
        run_detections = detections_by_run.get(run_name)

        if run_detections is None:
            matches.append(
                EventMatch(
                    run_name=run_name,
                    event_peak_time=event_peak_time,
                    detection_time=None,
                    advance_time_s=None,
                )
            )
            continue

        detection_times = run_detections["times"]
        detection_indices = run_detections["indices"]
        lower_bound = event_peak_time - max_early_s

        start = int(np.searchsorted(detection_times, lower_bound, side="right"))
        stop = int(np.searchsorted(detection_times, event_peak_time, side="left"))

        candidate_global_indices = [
            detection_indices[index]
            for index in range(start, stop)
            if detection_indices[index] not in used_detection_ids
        ]

        if not candidate_global_indices:
            matches.append(
                EventMatch(
                    run_name=run_name,
                    event_peak_time=event_peak_time,
                    detection_time=None,
                    advance_time_s=None,
                )
            )
            continue

        best_global_index = max(
            candidate_global_indices,
            key=lambda index: detections[index].detection_time,
        )
        best_detection = detections[best_global_index]
        used_detection_ids.add(best_global_index)

        matches.append(
            EventMatch(
                run_name=run_name,
                event_peak_time=event_peak_time,
                detection_time=best_detection.detection_time,
                advance_time_s=event_peak_time - best_detection.detection_time,
            )
        )

    false_detections = [
        detection
        for index, detection in enumerate(detections)
        if index not in used_detection_ids
    ]

    return matches, false_detections


def summarize_evaluation(
    matches: list[EventMatch],
    false_detections: list[DetectionEvent],
    evaluated_duration_s: float,
) -> EvaluationResult:
    """Summarize early detection performance."""
    if evaluated_duration_s <= 0:
        raise ValueError("evaluated_duration_s must be positive.")

    n_reference_events = len(matches)
    n_detected_events = sum(match.detected for match in matches)
    n_false_detections = len(false_detections)

    true_positive_rate = (
        n_detected_events / n_reference_events if n_reference_events > 0 else 0.0
    )

    false_detections_per_hour = n_false_detections / (evaluated_duration_s / 3600.0)
    mean_seconds_between_false_detections = (
        evaluated_duration_s / n_false_detections
        if n_false_detections > 0
        else None
    )

    advance_times = [
        match.advance_time_s for match in matches if match.advance_time_s is not None
    ]
    average_advance_time_s = float(np.mean(advance_times)) if advance_times else None

    return EvaluationResult(
        n_reference_events=n_reference_events,
        n_detected_events=n_detected_events,
        n_false_detections=n_false_detections,
        true_positive_rate=true_positive_rate,
        false_detections_per_hour=false_detections_per_hour,
        mean_seconds_between_false_detections=mean_seconds_between_false_detections,
        average_advance_time_s=average_advance_time_s,
    )


def evaluate_predictions(
    prediction_table: pd.DataFrame,
    reference_events: pd.DataFrame,
    evaluated_duration_s: float,
    threshold: float = 0.5,
    score_column: str = "score",
    max_early_s: float = 2.0,
    refractory_s: float = 1.0,
    consensus_k: int = 1,
    consensus_n: int = 1,
) -> EvaluationResult:
    """Evaluate window-level prediction scores against reference events."""
    detections = prediction_table_to_detections(
        prediction_table=prediction_table,
        threshold=threshold,
        score_column=score_column,
        consensus_k=consensus_k,
        consensus_n=consensus_n,
    )
    detections = suppress_repeated_detections(
        detections=detections,
        refractory_s=refractory_s,
    )
    matches, false_detections = match_detections_to_events(
        reference_events=reference_events,
        detections=detections,
        max_early_s=max_early_s,
    )

    return summarize_evaluation(
        matches=matches,
        false_detections=false_detections,
        evaluated_duration_s=evaluated_duration_s,
    )


def _group_detections_by_run(
    detections: list[DetectionEvent],
) -> dict[str, dict[str, np.ndarray]]:
    """Group detections by run and keep sorted times plus original indices."""
    grouped: dict[str, list[tuple[int, DetectionEvent]]] = {}

    for index, detection in enumerate(detections):
        grouped.setdefault(detection.run_name, []).append((index, detection))

    result: dict[str, dict[str, np.ndarray]] = {}

    for run_name, indexed_detections in grouped.items():
        indexed_detections = sorted(
            indexed_detections,
            key=lambda item: item[1].detection_time,
        )
        result[run_name] = {
            "times": np.array(
                [detection.detection_time for _, detection in indexed_detections],
                dtype=float,
            ),
            "indices": np.array(
                [index for index, _ in indexed_detections],
                dtype=int,
            ),
        }

    return result


def _validate_consensus(consensus_k: int, consensus_n: int) -> None:
    if consensus_k <= 0:
        raise ValueError("consensus_k must be positive.")

    if consensus_n <= 0:
        raise ValueError("consensus_n must be positive.")

    if consensus_k > consensus_n:
        raise ValueError("consensus_k cannot be greater than consensus_n.")
