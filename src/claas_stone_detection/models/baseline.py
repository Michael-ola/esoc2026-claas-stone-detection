from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from claas_stone_detection.reference.labels import IGNORE_LABEL

NON_FEATURE_COLUMNS = {
    "run_name",
    "window_start",
    "window_end",
    "detection_time",
    "start_index",
    "end_index",
    "label",
    "event_peak_time",
    "time_to_event_s",
}


@dataclass(frozen=True)
class BaselineModel:
    """Trained Random Forest baseline and the feature columns used by it."""

    model: RandomForestClassifier
    feature_columns: list[str]

    def predict_proba(self, feature_table: pd.DataFrame) -> np.ndarray:
        """Predict positive-class probabilities for a feature table."""
        missing_columns = set(self.feature_columns).difference(feature_table.columns)

        if missing_columns:
            raise ValueError(f"Missing feature columns: {missing_columns}")

        return self.model.predict_proba(feature_table[self.feature_columns])[:, 1]

    def predict(
        self,
        feature_table: pd.DataFrame,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Predict binary labels using a probability threshold."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1.")

        probabilities = self.predict_proba(feature_table)
        return (probabilities >= threshold).astype(int)


def get_feature_columns(
    table: pd.DataFrame,
    label_column: str = "label",
) -> list[str]:
    """Return numeric feature columns suitable for model training."""
    if label_column not in table.columns:
        raise ValueError(f"Missing label column: {label_column}")

    numeric_columns = table.select_dtypes(include=["number"]).columns.tolist()

    return [
        column
        for column in numeric_columns
        if column not in NON_FEATURE_COLUMNS and column != label_column
    ]


def prepare_training_table(
    labeled_table: pd.DataFrame,
    label_column: str = "label",
) -> pd.DataFrame:
    """Remove ignored rows and validate that the training table is usable."""
    if label_column not in labeled_table.columns:
        raise ValueError(f"Missing label column: {label_column}")

    training_table = labeled_table[labeled_table[label_column] != IGNORE_LABEL].copy()

    if training_table.empty:
        raise ValueError("No trainable rows remain after removing ignored labels.")

    labels = sorted(training_table[label_column].unique().tolist())

    if labels != [0, 1]:
        raise ValueError(
            "Training data must contain both negative and positive labels after "
            "removing ignored rows."
        )

    return training_table


def train_random_forest_baseline(
    labeled_table: pd.DataFrame,
    label_column: str = "label",
    n_estimators: int = 100,
    random_state: int = 42,
    class_weight: str | dict[int, float] | None = "balanced",
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    max_features: str | int | float | None = "sqrt",
) -> BaselineModel:
    """Train a configurable Random Forest baseline on a labeled feature table.

    The tree-depth and leaf-size parameters make it possible to compare a
    higher-capacity baseline with smaller edge-aware forests for later embedded
    deployment analysis.
    """
    training_table = prepare_training_table(
        labeled_table=labeled_table,
        label_column=label_column,
    )
    feature_columns = get_feature_columns(training_table, label_column=label_column)

    if not feature_columns:
        raise ValueError("No numeric feature columns available for training.")

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight=class_weight,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
    )

    model.fit(training_table[feature_columns], training_table[label_column])

    return BaselineModel(model=model, feature_columns=feature_columns)
