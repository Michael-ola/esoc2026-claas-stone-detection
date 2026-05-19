import pandas as pd
import pytest

from claas_stone_detection.pipelines.baseline_pipeline import (
    events_to_dataframe,
    filter_events_by_ratio,
    get_evaluated_duration_s,
    get_window_end_time,
    parse_threshold_sweep,
)
from claas_stone_detection.reference.events import StoneEvent


def make_event(threshold: float) -> StoneEvent:
    return StoneEvent(
        run_name="run_a",
        start_time=1.0,
        peak_time=1.5,
        end_time=2.0,
        peak_voltage=10.0,
        threshold=threshold,
        episode_start_time=0.0,
        episode_end_time=3.0,
    )


def test_parse_threshold_sweep_parses_valid_values() -> None:
    assert parse_threshold_sweep("0.05,0.10,0.50") == [0.05, 0.10, 0.50]


def test_parse_threshold_sweep_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        parse_threshold_sweep("0.1,1.5")


def test_filter_events_by_ratio_keeps_strong_events() -> None:
    weak = make_event(threshold=10.0)
    strong = make_event(threshold=5.0)

    result = filter_events_by_ratio([weak, strong], min_event_ratio=1.5)

    assert result == [strong]


def test_filter_events_by_ratio_rejects_negative_ratio() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        filter_events_by_ratio([], min_event_ratio=-1.0)


def test_events_to_dataframe_uses_expected_columns() -> None:
    frame = events_to_dataframe({"run_a": [make_event(threshold=5.0)]})

    assert isinstance(frame, pd.DataFrame)
    assert frame.columns.tolist() == [
        "run_name",
        "peak_time",
        "peak_voltage",
        "threshold",
        "peak_to_threshold_ratio",
    ]
    assert frame.loc[0, "peak_to_threshold_ratio"] == 2.0


def test_window_region_helpers_support_header_on_and_extended() -> None:
    assert get_window_end_time(10.0, 12.0, "header-on") == 10.0
    assert get_window_end_time(10.0, 12.0, "extended") == 12.0
    assert get_evaluated_duration_s(5.0, 7.0, "header-on") == 5.0
    assert get_evaluated_duration_s(5.0, 7.0, "extended") == 7.0


def test_window_region_helpers_reject_invalid_region() -> None:
    with pytest.raises(ValueError, match="window_region"):
        get_window_end_time(10.0, 12.0, "bad")
