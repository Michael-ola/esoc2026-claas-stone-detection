import pandas as pd
import pytest

from claas_stone_detection.reference.events import StoneEvent
from claas_stone_detection.reference.labels import (
    IGNORE_LABEL,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    label_feature_table,
    label_window,
    label_windows,
)
from claas_stone_detection.streaming.windowing import SignalWindow


def make_window(
    run_name: str,
    start_time: float,
    end_time: float,
) -> SignalWindow:
    return SignalWindow(
        run_name=run_name,
        start_time=start_time,
        end_time=end_time,
        start_index=0,
        end_index=0,
    )


def make_event(run_name: str, peak_time: float) -> StoneEvent:
    return StoneEvent(
        run_name=run_name,
        start_time=peak_time - 0.05,
        peak_time=peak_time,
        end_time=peak_time + 0.05,
        peak_voltage=100.0,
        threshold=80.0,
        episode_start_time=0.0,
        episode_end_time=peak_time,
    )


def test_label_window_marks_pre_event_window_positive() -> None:
    window = make_window(run_name="run_a", start_time=8.5, end_time=9.0)
    events = [make_event(run_name="run_a", peak_time=10.0)]

    label = label_window(
        window=window,
        events=events,
        positive_horizon_s=2.0,
        post_event_exclusion_s=1.0,
    )

    assert label.label == POSITIVE_LABEL
    assert label.event_peak_time == 10.0
    assert label.time_to_event_s == pytest.approx(1.0)


def test_label_window_marks_far_window_negative() -> None:
    window = make_window(run_name="run_a", start_time=5.0, end_time=6.0)
    events = [make_event(run_name="run_a", peak_time=10.0)]

    label = label_window(
        window=window,
        events=events,
        positive_horizon_s=2.0,
        post_event_exclusion_s=1.0,
    )

    assert label.label == NEGATIVE_LABEL
    assert label.event_peak_time is None
    assert label.time_to_event_s is None


def test_label_window_ignores_post_event_window() -> None:
    window = make_window(run_name="run_a", start_time=10.2, end_time=10.5)
    events = [make_event(run_name="run_a", peak_time=10.0)]

    label = label_window(
        window=window,
        events=events,
        positive_horizon_s=2.0,
        post_event_exclusion_s=1.0,
    )

    assert label.label == IGNORE_LABEL
    assert label.event_peak_time == 10.0
    assert label.time_to_event_s == pytest.approx(-0.5)
    assert label.is_ignored


def test_label_window_prioritizes_post_event_exclusion_over_positive_horizon() -> None:
    window = make_window(run_name="run_a", start_time=10.5, end_time=10.7)
    events = [
        make_event(run_name="run_a", peak_time=10.0),
        make_event(run_name="run_a", peak_time=11.5),
    ]

    label = label_window(
        window=window,
        events=events,
        positive_horizon_s=2.0,
        post_event_exclusion_s=1.0,
    )

    assert label.label == IGNORE_LABEL
    assert label.event_peak_time == 10.0
    assert label.time_to_event_s == pytest.approx(-0.7)


def test_label_window_uses_matching_run_only() -> None:
    window = make_window(run_name="run_a", start_time=9.0, end_time=9.5)
    events = [make_event(run_name="run_b", peak_time=10.0)]

    label = label_window(
        window=window,
        events=events,
        positive_horizon_s=2.0,
        post_event_exclusion_s=1.0,
    )

    assert label.label == NEGATIVE_LABEL


def test_label_windows_returns_label_table() -> None:
    windows = [
        make_window(run_name="run_a", start_time=8.5, end_time=9.0),
        make_window(run_name="run_a", start_time=5.0, end_time=6.0),
    ]
    events = [make_event(run_name="run_a", peak_time=10.0)]

    labels = label_windows(
        windows=windows,
        events=events,
        positive_horizon_s=2.0,
        post_event_exclusion_s=1.0,
    )

    assert labels["label"].tolist() == [POSITIVE_LABEL, NEGATIVE_LABEL]
    assert labels["run_name"].tolist() == ["run_a", "run_a"]


def test_label_feature_table_attaches_labels() -> None:
    feature_table = pd.DataFrame(
        {
            "run_name": ["run_a", "run_a", "run_b"],
            "window_start": [8.5, 5.0, 8.5],
            "window_end": [9.0, 6.0, 9.0],
            "detection_time": [9.0, 6.0, 9.0],
            "audio_rms": [0.1, 0.2, 0.3],
        }
    )
    events_by_run = {
        "run_a": [make_event(run_name="run_a", peak_time=10.0)],
        "run_b": [make_event(run_name="run_b", peak_time=10.0)],
    }

    labeled = label_feature_table(
        feature_table=feature_table,
        events_by_run=events_by_run,
        positive_horizon_s=2.0,
        post_event_exclusion_s=1.0,
    )

    assert labeled["label"].tolist() == [
        POSITIVE_LABEL,
        NEGATIVE_LABEL,
        POSITIVE_LABEL,
    ]
    assert "audio_rms" in labeled.columns
    assert labeled.loc[0, "time_to_event_s"] == pytest.approx(1.0)


def test_label_feature_table_rejects_missing_required_columns() -> None:
    feature_table = pd.DataFrame({"audio_rms": [0.1]})

    with pytest.raises(ValueError, match="Missing required feature table columns"):
        label_feature_table(feature_table=feature_table, events_by_run={})


def test_label_window_rejects_invalid_horizon() -> None:
    window = make_window(run_name="run_a", start_time=0.0, end_time=1.0)

    with pytest.raises(ValueError, match="positive_horizon_s must be positive"):
        label_window(window=window, events=[], positive_horizon_s=0.0)


def test_label_window_rejects_invalid_post_event_exclusion() -> None:
    window = make_window(run_name="run_a", start_time=0.0, end_time=1.0)

    with pytest.raises(ValueError, match="post_event_exclusion_s cannot be negative"):
        label_window(window=window, events=[], post_event_exclusion_s=-1.0)
