import pandas as pd
import pytest

from claas_stone_detection.core.preprocessing import normalize_status_column


def test_normalize_status_column_converts_bytes_values() -> None:
    df = pd.DataFrame(
        {"Status": [b"On", b"Off", b"On"]},
        index=pd.Index([0.0, 0.1, 0.2], name="time_s"),
    )

    result = normalize_status_column(df)

    assert result["HeaderOn"].tolist() == [True, False, True]


def test_normalize_status_column_converts_string_values() -> None:
    df = pd.DataFrame(
        {"Status": ["On", "Off", "true", "false", "1", "0"]},
        index=pd.Index([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], name="time_s"),
    )

    result = normalize_status_column(df)

    assert result["HeaderOn"].tolist() == [True, False, True, False, True, False]


def test_normalize_status_column_converts_numeric_values() -> None:
    df = pd.DataFrame(
        {"Status": [1, 0, 1.0, 0.0]},
        index=pd.Index([0.0, 0.1, 0.2, 0.3], name="time_s"),
    )

    result = normalize_status_column(df)

    assert result["HeaderOn"].tolist() == [True, False, True, False]


def test_normalize_status_column_preserves_raw_status_column() -> None:
    df = pd.DataFrame(
        {"Status": [b"On", b"Off"]},
        index=pd.Index([0.0, 0.1], name="time_s"),
    )

    result = normalize_status_column(df)

    assert "Status" in result.columns
    assert "HeaderOn" in result.columns
    assert result["Status"].tolist() == [b"On", b"Off"]


def test_normalize_status_column_does_not_mutate_input_dataframe() -> None:
    df = pd.DataFrame(
        {"Status": [b"On", b"Off"]},
        index=pd.Index([0.0, 0.1], name="time_s"),
    )

    result = normalize_status_column(df)

    assert "HeaderOn" not in df.columns
    assert "HeaderOn" in result.columns


def test_normalize_status_column_rejects_missing_status_column() -> None:
    df = pd.DataFrame(
        {"OtherColumn": [1, 0]},
        index=pd.Index([0.0, 0.1], name="time_s"),
    )

    with pytest.raises(ValueError, match="Missing status column"):
        normalize_status_column(df)


def test_normalize_status_column_rejects_unsupported_value() -> None:
    df = pd.DataFrame(
        {"Status": [object()]},
        index=pd.Index([0.0], name="time_s"),
    )

    with pytest.raises(ValueError, match="Unsupported status value"):
        normalize_status_column(df)