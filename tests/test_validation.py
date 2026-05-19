import pandas as pd
import pytest

from claas_stone_detection.core.validation import validate_measurement_dataframe


def make_valid_measurement_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sensor1": [0.1, 0.2, 0.3],
            "VehicleSpeed": [1.0, 1.1, 1.2],
            "CutLength": [9.0, 9.0, 9.0],
            "VoltageSignal": [100.0, 101.0, 100.5],
            "Status": [b"On", b"On", b"Off"],
            "HeaderOn": [True, True, False],
        },
        index=pd.Index([0.0, 0.1, 0.2], name="time_s"),
    )


def test_validate_measurement_dataframe_accepts_valid_dataframe() -> None:
    df = make_valid_measurement_dataframe()

    validate_measurement_dataframe(df)


def test_validate_measurement_dataframe_rejects_empty_dataframe() -> None:
    df = pd.DataFrame()

    with pytest.raises(ValueError, match="empty"):
        validate_measurement_dataframe(df)


def test_validate_measurement_dataframe_rejects_wrong_index_name() -> None:
    df = make_valid_measurement_dataframe()
    df.index.name = "timestamp"

    with pytest.raises(ValueError, match="time_s"):
        validate_measurement_dataframe(df)


def test_validate_measurement_dataframe_rejects_unsorted_time_index() -> None:
    df = make_valid_measurement_dataframe()
    df = df.reindex([0.0, 0.2, 0.1])

    with pytest.raises(ValueError, match="monotonically increasing"):
        validate_measurement_dataframe(df)


def test_validate_measurement_dataframe_rejects_missing_required_channel() -> None:
    df = make_valid_measurement_dataframe()
    df = df.drop(columns=["VoltageSignal"])

    with pytest.raises(ValueError, match="Missing required channels"):
        validate_measurement_dataframe(df)