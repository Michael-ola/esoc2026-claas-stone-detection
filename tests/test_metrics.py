import pandas as pd
import pytest

from claas_stone_detection.evaluation.metrics import (
    DetectionEvent,
    evaluate_predictions,
    match_detections_to_events,
    prediction_table_to_detections,
    summarize_evaluation,
    suppress_repeated_detections,
)


def make_prediction_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run_name": ["run_a", "run_a", "run_a"],
            "detection_time": [8.0, 9.2, 12.0],
            "score": [0.2, 0.8, 0.9],
        }
    )


def make_reference_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run_name": ["run_a"],
            "peak_time": [10.0],
        }
    )


def test_prediction_table_to_detections_applies_threshold() -> None:
    detections = prediction_table_to_detections(
        prediction_table=make_prediction_table(),
        threshold=0.5,
    )

    assert len(detections) == 2
    assert detections[0].detection_time == 9.2
    assert detections[0].score == 0.8


def test_prediction_table_to_detections_rejects_missing_columns() -> None:
    table = pd.DataFrame({"score": [0.5]})

    with pytest.raises(ValueError, match="Missing prediction table columns"):
        prediction_table_to_detections(table)


def test_prediction_table_to_detections_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
        prediction_table_to_detections(make_prediction_table(), threshold=1.2)


def test_suppress_repeated_detections_keeps_first_alarm_in_refractory_period() -> None:
    detections = [
        DetectionEvent(run_name="run_a", detection_time=1.0, score=0.8),
        DetectionEvent(run_name="run_a", detection_time=1.5, score=0.9),
        DetectionEvent(run_name="run_a", detection_time=3.0, score=0.7),
    ]

    result = suppress_repeated_detections(detections, refractory_s=1.0)

    assert [d.detection_time for d in result] == [1.0, 3.0]


def test_match_detections_to_events_matches_pre_event_detection() -> None:
    detections = [
        DetectionEvent(run_name="run_a", detection_time=9.2, score=0.8),
        DetectionEvent(run_name="run_a", detection_time=12.0, score=0.9),
    ]

    matches, false_detections = match_detections_to_events(
        reference_events=make_reference_events(),
        detections=detections,
        max_early_s=2.0,
    )

    assert len(matches) == 1
    assert matches[0].detected
    assert matches[0].advance_time_s == pytest.approx(0.8)
    assert [d.detection_time for d in false_detections] == [12.0]


def test_match_detections_to_events_uses_latest_valid_detection() -> None:
    detections = [
        DetectionEvent(run_name="run_a", detection_time=8.5, score=0.7),
        DetectionEvent(run_name="run_a", detection_time=9.4, score=0.8),
        DetectionEvent(run_name="run_a", detection_time=9.8, score=0.9),
    ]

    matches, false_detections = match_detections_to_events(
        reference_events=make_reference_events(),
        detections=detections,
        max_early_s=2.0,
    )

    assert matches[0].detection_time == pytest.approx(9.8)
    assert matches[0].advance_time_s == pytest.approx(0.2)
    assert [d.detection_time for d in false_detections] == [8.5, 9.4]


def test_match_detections_to_events_marks_missed_event() -> None:
    detections = [DetectionEvent(run_name="run_a", detection_time=6.0, score=0.8)]

    matches, false_detections = match_detections_to_events(
        reference_events=make_reference_events(),
        detections=detections,
        max_early_s=2.0,
    )

    assert len(matches) == 1
    assert not matches[0].detected
    assert false_detections == detections


def test_match_detections_to_events_respects_run_names() -> None:
    detections = [DetectionEvent(run_name="run_b", detection_time=9.2, score=0.8)]

    matches, false_detections = match_detections_to_events(
        reference_events=make_reference_events(),
        detections=detections,
        max_early_s=2.0,
    )

    assert len(matches) == 1
    assert not matches[0].detected
    assert false_detections == detections


def test_summarize_evaluation_computes_metrics() -> None:
    detections = [DetectionEvent(run_name="run_a", detection_time=9.2, score=0.8)]
    matches, false_detections = match_detections_to_events(
        reference_events=make_reference_events(),
        detections=detections,
        max_early_s=2.0,
    )

    result = summarize_evaluation(
        matches=matches,
        false_detections=false_detections,
        evaluated_duration_s=3600.0,
    )

    assert result.n_reference_events == 1
    assert result.n_detected_events == 1
    assert result.n_false_detections == 0
    assert result.true_positive_rate == pytest.approx(1.0)
    assert result.false_detections_per_hour == pytest.approx(0.0)
    assert result.average_advance_time_s == pytest.approx(0.8)


def test_evaluate_predictions_end_to_end() -> None:
    result = evaluate_predictions(
        prediction_table=make_prediction_table(),
        reference_events=make_reference_events(),
        evaluated_duration_s=3600.0,
        threshold=0.5,
        max_early_s=2.0,
        refractory_s=0.5,
    )

    assert result.n_reference_events == 1
    assert result.n_detected_events == 1
    assert result.n_false_detections == 1
    assert result.true_positive_rate == pytest.approx(1.0)
