import numpy as np
import pandas as pd
import pytest

from claas_stone_detection.models.baseline import (
    NON_FEATURE_COLUMNS,
    get_feature_columns,
    prepare_training_table,
    train_random_forest_baseline,
)
from claas_stone_detection.reference.labels import IGNORE_LABEL


def make_labeled_feature_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run_name": ["run_a", "run_a", "run_a", "run_b", "run_b"],
            "window_start": [0.0, 1.0, 2.0, 0.0, 1.0],
            "window_end": [0.5, 1.5, 2.5, 0.5, 1.5],
            "detection_time": [0.5, 1.5, 2.5, 0.5, 1.5],
            "start_index": [0, 10, 20, 0, 10],
            "end_index": [10, 20, 30, 10, 20],
            "audio_rms": [0.1, 0.2, 1.0, 0.15, 1.2],
            "audio_peak_abs": [0.2, 0.3, 2.0, 0.25, 2.2],
            "VehicleSpeed_mean": [2.0, 2.1, 2.2, 2.0, 2.3],
            "event_peak_time": [None, None, 3.0, None, 2.0],
            "time_to_event_s": [None, None, 0.5, None, 0.5],
            "label": [0, 0, 1, IGNORE_LABEL, 1],
        }
    )


def test_get_feature_columns_selects_only_numeric_model_features() -> None:
    table = make_labeled_feature_table()

    feature_columns = get_feature_columns(table)

    assert "audio_rms" in feature_columns
    assert "audio_peak_abs" in feature_columns
    assert "VehicleSpeed_mean" in feature_columns

    for column in NON_FEATURE_COLUMNS:
        assert column not in feature_columns


def test_get_feature_columns_rejects_missing_label_column() -> None:
    table = make_labeled_feature_table().drop(columns=["label"])

    with pytest.raises(ValueError, match="Missing label column"):
        get_feature_columns(table)


def test_prepare_training_table_removes_ignored_rows() -> None:
    table = make_labeled_feature_table()

    training_table = prepare_training_table(table)

    assert IGNORE_LABEL not in training_table["label"].tolist()
    assert training_table["label"].tolist() == [0, 0, 1, 1]


def test_prepare_training_table_rejects_no_trainable_rows() -> None:
    table = pd.DataFrame(
        {
            "audio_rms": [0.1, 0.2],
            "label": [IGNORE_LABEL, IGNORE_LABEL],
        }
    )

    with pytest.raises(ValueError, match="No trainable rows"):
        prepare_training_table(table)


def test_prepare_training_table_requires_two_classes() -> None:
    table = pd.DataFrame(
        {
            "audio_rms": [0.1, 0.2],
            "label": [0, 0],
        }
    )

    with pytest.raises(ValueError, match="both negative and positive labels"):
        prepare_training_table(table)


def test_train_random_forest_baseline_returns_predictive_model() -> None:
    table = make_labeled_feature_table()

    baseline = train_random_forest_baseline(
        labeled_table=table,
        n_estimators=10,
        random_state=42,
    )

    probabilities = baseline.predict_proba(table)
    predictions = baseline.predict(table, threshold=0.5)

    assert len(probabilities) == len(table)
    assert len(predictions) == len(table)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert set(predictions.tolist()).issubset({0, 1})
    assert "audio_rms" in baseline.feature_columns


def test_baseline_predict_rejects_missing_feature_columns() -> None:
    table = make_labeled_feature_table()
    baseline = train_random_forest_baseline(
        labeled_table=table,
        n_estimators=10,
        random_state=42,
    )

    with pytest.raises(ValueError, match="Missing feature columns"):
        baseline.predict_proba(table.drop(columns=["audio_rms"]))


def test_baseline_predict_rejects_invalid_threshold() -> None:
    table = make_labeled_feature_table()
    baseline = train_random_forest_baseline(
        labeled_table=table,
        n_estimators=10,
        random_state=42,
    )

    with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
        baseline.predict(table, threshold=1.5)
