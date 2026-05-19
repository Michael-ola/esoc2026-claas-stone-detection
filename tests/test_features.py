import numpy as np
import pandas as pd
import pytest

from claas_stone_detection.streaming.features import (
    extract_frequency_features,
    extract_window_features,
    infer_sample_rate_hz,
    make_feature_table,
    zero_crossing_rate,
)
from claas_stone_detection.streaming.windowing import make_sliding_windows


def make_feature_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sensor1": [1.0, -1.0, 1.0, -1.0, 1.0],
            "VehicleSpeed": [2.0, 2.0, 3.0, 3.0, 4.0],
            "CutLength": [9.0, 9.0, 9.0, 9.0, 9.0],
        },
        index=pd.Index([0.0, 0.25, 0.5, 0.75, 1.0], name="time_s"),
    )

def test_infer_sample_rate_hz_from_time_index() -> None:
    df = make_feature_dataframe()

    sample_rate = infer_sample_rate_hz(df.index)

    assert sample_rate == pytest.approx(4.0)

def test_zero_crossing_rate_counts_crossings_per_second() -> None:
    df = make_feature_dataframe()

    rate = zero_crossing_rate(
        signal=df["Sensor1"].to_numpy(),
        duration_s=1.0,
    )

    assert rate == pytest.approx(4.0)

def test_extract_window_features_returns_audio_and_context_features() -> None:
    df = make_feature_dataframe()

    features = extract_window_features(df)

    assert features["sample_rate_hz"] == pytest.approx(4.0)
    assert features["audio_rms"] == pytest.approx(1.0)
    assert features["audio_peak_abs"] == pytest.approx(1.0)
    assert features["audio_zero_crossing_rate"] == pytest.approx(4.0)
    assert features["VehicleSpeed_mean"] == pytest.approx(2.8)
    assert features["CutLength_mean"] == pytest.approx(9.0)
    assert "spectral_centroid_hz" in features
    assert "high_frequency_energy_ratio" in features

def test_extract_window_features_rejects_empty_window() -> None:
    df = pd.DataFrame(
        {"Sensor1": []},
        index=pd.Index([], name="time_s"),
    )

    with pytest.raises(ValueError, match="empty window"):
        extract_window_features(df)

def test_extract_window_features_rejects_missing_audio_column() -> None:
    df = pd.DataFrame(
        {"OtherColumn": [1.0, 2.0]},
        index=pd.Index([0.0, 1.0], name="time_s"),
    )

    with pytest.raises(ValueError, match="Missing audio column"):
        extract_window_features(df)

def test_make_feature_table_adds_window_metadata() -> None:
    df = make_feature_dataframe()
    windows = make_sliding_windows(
        df=df,
        window_s=0.5,
        hop_s=0.5,
        run_name="run_a",
    )

    table = make_feature_table(df=df, windows=windows)

    assert len(table) == 2
    assert table["run_name"].tolist() == ["run_a", "run_a"]
    assert table["window_start"].tolist() == [0.0, 0.5]
    assert table["window_end"].tolist() == [0.5, 1.0]
    assert table["start_index"].tolist() == [0, 2]
    assert table["end_index"].tolist() == [2, 4]
    assert "audio_rms" in table.columns

def test_extract_frequency_features_handles_invalid_sample_rate() -> None:
    features = extract_frequency_features(
        audio=np.array([1.0, 2.0, 3.0]),
        sample_rate_hz=0.0,
        frequency_bands_hz=((0.0, 500.0),),
    )

    assert features["band_energy_0_500_hz"] == 0.0
    assert features["spectral_centroid_hz"] == 0.0
    assert features["high_frequency_energy_ratio"] == 0.0

def test_extract_frequency_features_handles_bands_above_nyquist() -> None:
    features = extract_frequency_features(
        audio=np.array([1.0, -1.0, 1.0, -1.0]),
        sample_rate_hz=4.0,
        frequency_bands_hz=((1000.0, 2000.0),),
    )

    assert features["band_energy_1000_2000_hz"] == 0.0
    assert "spectral_centroid_hz" in features

def test_add_temporal_delta_features_adds_run_local_deltas() -> None:
    from claas_stone_detection.streaming.features import add_temporal_delta_features

    table = pd.DataFrame(
        {
            "run_name": ["run_a", "run_a", "run_b", "run_b"],
            "detection_time": [0.0, 0.5, 0.0, 0.5],
            "window_start": [0.0, 0.5, 0.0, 0.5],
            "window_end": [0.5, 1.0, 0.5, 1.0],
            "audio_rms": [1.0, 1.5, 10.0, 12.0],
            "high_frequency_energy_ratio": [0.2, 0.5, 0.1, 0.4],
        }
    )

    result = add_temporal_delta_features(table, lag=1)

    assert result["audio_rms_delta_1"].tolist() == [0.0, 0.5, 0.0, 2.0]
    assert result["high_frequency_energy_ratio_delta_1"].tolist() == pytest.approx(
        [0.0, 0.3, 0.0, 0.3]
    )

def test_add_temporal_delta_features_preserves_original_row_order() -> None:
    from claas_stone_detection.streaming.features import add_temporal_delta_features

    table = pd.DataFrame(
        {
            "run_name": ["run_a", "run_a", "run_a"],
            "detection_time": [1.0, 0.0, 0.5],
            "audio_rms": [3.0, 1.0, 2.0],
        },
        index=[10, 20, 30],
    )

    result = add_temporal_delta_features(table, lag=1)

    assert result.index.tolist() == [10, 20, 30]
    assert result.loc[20, "audio_rms_delta_1"] == 0.0
    assert result.loc[30, "audio_rms_delta_1"] == 1.0
    assert result.loc[10, "audio_rms_delta_1"] == 1.0

def test_add_temporal_delta_features_rejects_missing_group_column() -> None:
    from claas_stone_detection.streaming.features import add_temporal_delta_features

    table = pd.DataFrame({"detection_time": [0.0], "audio_rms": [1.0]})

    with pytest.raises(ValueError, match="Missing group column"):
        add_temporal_delta_features(table)

def test_add_temporal_delta_features_rejects_missing_order_column() -> None:
    from claas_stone_detection.streaming.features import add_temporal_delta_features

    table = pd.DataFrame({"run_name": ["run_a"], "audio_rms": [1.0]})

    with pytest.raises(ValueError, match="Missing order column"):
        add_temporal_delta_features(table)

def test_add_temporal_delta_features_rejects_invalid_lag() -> None:
    from claas_stone_detection.streaming.features import add_temporal_delta_features

    table = pd.DataFrame(
        {
            "run_name": ["run_a"],
            "detection_time": [0.0],
            "audio_rms": [1.0],
        }
    )

    with pytest.raises(ValueError, match="lag must be positive"):
        add_temporal_delta_features(table, lag=0)
