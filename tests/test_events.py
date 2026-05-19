import math

import pandas as pd
import pytest

from claas_stone_detection.reference.episodes import Episode
from claas_stone_detection.reference.events import (
    detect_voltage_events,
    detect_voltage_events_in_dataset,
    group_candidate_regions,
)


def make_event_dataframe(
    voltage: list[float],
    header_on: list[bool] | None = None,
) -> pd.DataFrame:
    if header_on is None:
        header_on = [True] * len(voltage)

    return pd.DataFrame(
        {
            "VoltageSignal": voltage,
            "HeaderOn": header_on,
        },
        index=pd.Index([float(i) for i in range(len(voltage))], name="time_s"),
    )


def test_group_candidate_regions_groups_nearby_times() -> None:
    candidate_times = pd.Index([1.0, 1.1, 1.2, 3.0, 3.1]).to_numpy()

    groups = group_candidate_regions(candidate_times, min_gap_s=0.5)

    assert groups == [(1.0, 1.2), (3.0, 3.1)]


def test_group_candidate_regions_returns_empty_list_for_no_candidates() -> None:
    groups = group_candidate_regions(
        candidate_times=pd.Index([]).to_numpy(),
        min_gap_s=0.5,
    )

    assert groups == []


def test_detect_voltage_events_finds_event_inside_episode() -> None:
    df = make_event_dataframe(
        voltage=[10, 10, 10, 100, 120, 100, 10, 10],
    )
    episodes = [
        Episode(
            start_time=0.0,
            end_time=7.0,
            extended_end_time=7.0,
        )
    ]

    events = detect_voltage_events(
        df,
        episodes=episodes,
        threshold_quantile=0.7,
        min_gap_s=1.1,
        min_duration_s=0.0,
    )

    assert len(events) == 1
    assert events[0].run_name == ""
    assert events[0].start_time == 3.0
    assert events[0].peak_time == 4.0
    assert events[0].end_time == 5.0
    assert events[0].peak_voltage == 120.0
    assert events[0].source == "VoltageSignal"


def test_detect_voltage_events_uses_episode_window_only() -> None:
    df = make_event_dataframe(
        voltage=[10, 10, 500, 10, 10, 20, 25, 20],
    )
    episodes = [
        Episode(
            start_time=5.0,
            end_time=7.0,
            extended_end_time=7.0,
        )
    ]

    events = detect_voltage_events(
        df,
        episodes=episodes,
        threshold_quantile=0.5,
        min_gap_s=1.1,
        min_duration_s=0.0,
    )

    assert len(events) == 1
    assert events[0].peak_time == 6.0
    assert events[0].peak_voltage == 25.0


def test_detect_voltage_events_filters_short_events() -> None:
    df = make_event_dataframe(
        voltage=[10, 10, 100, 10, 10],
    )
    episodes = [
        Episode(
            start_time=0.0,
            end_time=4.0,
            extended_end_time=4.0,
        )
    ]

    events = detect_voltage_events(
        df,
        episodes=episodes,
        threshold_quantile=0.7,
        min_gap_s=0.5,
        min_duration_s=0.01,
    )

    assert events == []


def test_detect_voltage_events_stores_threshold_and_ratio() -> None:
    df = make_event_dataframe(
        voltage=[10, 10, 10, 100, 120, 100, 10, 10],
    )
    episodes = [
        Episode(
            start_time=0.0,
            end_time=7.0,
            extended_end_time=7.0,
        )
    ]

    events = detect_voltage_events(
        df,
        episodes=episodes,
        threshold_quantile=0.7,
        min_gap_s=1.1,
        min_duration_s=0.0,
    )

    assert len(events) == 1
    assert events[0].threshold > 0
    assert events[0].peak_to_threshold_ratio == pytest.approx(
        events[0].peak_voltage / events[0].threshold
    )


def test_stone_event_ratio_is_infinite_when_threshold_is_zero() -> None:
    df = make_event_dataframe(
        voltage=[0, 0, 0, 10, 12, 10, 0, 0],
    )
    episodes = [
        Episode(
            start_time=0.0,
            end_time=7.0,
            extended_end_time=7.0,
        )
    ]

    events = detect_voltage_events(
        df,
        episodes=episodes,
        threshold_quantile=0.1,
        min_gap_s=1.1,
        min_duration_s=0.0,
    )

    assert len(events) == 1
    assert events[0].threshold == 0.0
    assert math.isinf(events[0].peak_to_threshold_ratio)


def test_detect_voltage_events_rejects_missing_voltage_column() -> None:
    df = pd.DataFrame(
        {"HeaderOn": [True, True]},
        index=pd.Index([0.0, 1.0], name="time_s"),
    )

    with pytest.raises(ValueError, match="Missing voltage column"):
        detect_voltage_events(df)


def test_detect_voltage_events_in_dataset_returns_events_per_run() -> None:
    dataset = {
        "run_a": make_event_dataframe([10, 10, 100, 120, 100, 10]),
        "run_b": make_event_dataframe([5, 5, 50, 70, 50, 5]),
    }

    events = detect_voltage_events_in_dataset(
        dataset,
        threshold_quantile=0.7,
        min_gap_s=1.1,
        min_duration_s=0.0,
    )

    assert set(events.keys()) == {"run_a", "run_b"}
    assert len(events["run_a"]) == 1
    assert len(events["run_b"]) == 1
    assert events["run_a"][0].run_name == "run_a"
    assert events["run_b"][0].run_name == "run_b"