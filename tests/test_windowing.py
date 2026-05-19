import pandas as pd
import pytest

from claas_stone_detection.windowing import (
    infer_sample_rate_hz_from_index,
    make_sliding_windows,
    slice_window,
)


def make_time_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {"Sensor1": [0, 1, 2, 3, 4]},
        index=pd.Index([0.0, 0.5, 1.0, 1.5, 2.0], name="time_s"),
    )


def test_make_sliding_windows_creates_expected_windows() -> None:
    df = make_time_dataframe()

    windows = make_sliding_windows(
        df=df,
        window_s=1.0,
        hop_s=0.5,
        run_name="run_a",
    )

    assert len(windows) == 3

    assert windows[0].run_name == "run_a"
    assert windows[0].start_time == 0.0
    assert windows[0].end_time == 1.0
    assert windows[0].detection_time == 1.0
    assert windows[0].start_index == 0
    assert windows[0].end_index == 2

    assert windows[1].start_time == 0.5
    assert windows[1].end_time == 1.5
    assert windows[1].start_index == 1
    assert windows[1].end_index == 3

    assert windows[2].start_time == 1.0
    assert windows[2].end_time == 2.0
    assert windows[2].start_index == 2
    assert windows[2].end_index == 4


def test_make_sliding_windows_respects_start_and_end_time() -> None:
    df = make_time_dataframe()

    windows = make_sliding_windows(
        df=df,
        window_s=0.5,
        hop_s=0.5,
        start_time=0.5,
        end_time=1.5,
    )

    assert [(w.start_time, w.end_time) for w in windows] == [
        (0.5, 1.0),
        (1.0, 1.5),
    ]

    assert [(w.start_index, w.end_index) for w in windows] == [
        (1, 2),
        (2, 3),
    ]


def test_make_sliding_windows_returns_empty_for_empty_dataframe() -> None:
    df = pd.DataFrame(index=pd.Index([], name="time_s"))

    windows = make_sliding_windows(df=df, window_s=1.0, hop_s=0.5)

    assert windows == []


def test_make_sliding_windows_rejects_invalid_window_size() -> None:
    df = make_time_dataframe()

    with pytest.raises(ValueError, match="window_s must be positive"):
        make_sliding_windows(df=df, window_s=0.0, hop_s=0.5)


def test_make_sliding_windows_rejects_invalid_hop_size() -> None:
    df = make_time_dataframe()

    with pytest.raises(ValueError, match="hop_s must be positive"):
        make_sliding_windows(df=df, window_s=1.0, hop_s=0.0)


def test_slice_window_uses_index_positions() -> None:
    df = make_time_dataframe()
    window = make_sliding_windows(df=df, window_s=1.0, hop_s=0.5)[1]

    result = slice_window(df, window)

    assert result.index.min() == 0.5
    assert result.index.max() == 1.0
    assert result["Sensor1"].tolist() == [1, 2]


def test_infer_sample_rate_hz_from_index() -> None:
    df = make_time_dataframe()

    sample_rate = infer_sample_rate_hz_from_index(df.index)

    assert sample_rate == pytest.approx(2.0)
