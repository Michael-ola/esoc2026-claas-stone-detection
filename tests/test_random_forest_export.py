import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from claas_stone_detection.edge.random_forest_export import (
    estimate_export_size_bytes,
    export_random_forest_for_mcu,
    write_c_header,
    write_deployment_note,
    write_export_json,
    write_feature_list,
)


def make_fitted_forest() -> RandomForestClassifier:
    x = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.2],
            [1.0, 1.0],
            [1.2, 1.1],
        ]
    )
    y = np.array([0, 0, 1, 1])

    model = RandomForestClassifier(
        n_estimators=2,
        max_depth=2,
        random_state=42,
    )
    model.fit(x, y)
    return model


def test_export_random_forest_for_mcu_creates_trees() -> None:
    model = make_fitted_forest()

    export = export_random_forest_for_mcu(
        model=model,
        feature_names=["feature_a", "feature_b"],
    )

    assert export.n_classes == 2
    assert export.class_labels == [0, 1]
    assert export.feature_names == ["feature_a", "feature_b"]
    assert len(export.trees) == 2
    assert export.estimated_model_bytes > 0


def test_export_random_forest_for_mcu_rejects_empty_features() -> None:
    model = make_fitted_forest()

    with pytest.raises(ValueError, match="feature_names cannot be empty"):
        export_random_forest_for_mcu(model=model, feature_names=[])


def test_estimate_export_size_bytes_is_positive() -> None:
    model = make_fitted_forest()
    export = export_random_forest_for_mcu(
        model=model,
        feature_names=["feature_a", "feature_b"],
    )

    assert estimate_export_size_bytes(export.trees) > 0


def test_write_export_files(tmp_path: Path) -> None:
    model = make_fitted_forest()
    export = export_random_forest_for_mcu(
        model=model,
        feature_names=["feature_a", "feature_b"],
    )

    json_path = tmp_path / "model.json"
    header_path = tmp_path / "model.h"
    feature_path = tmp_path / "feature_order.txt"
    note_path = tmp_path / "README.md"

    write_export_json(export, json_path)
    write_c_header(export, header_path)
    write_feature_list(export.feature_names, feature_path)
    write_deployment_note(export, note_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert payload["n_classes"] == 2
    assert "claas_rf_predict_proba" in header_path.read_text(encoding="utf-8")
    assert feature_path.read_text(encoding="utf-8").splitlines() == [
        "feature_a",
        "feature_b",
    ]
    assert "2 MB fit check" in note_path.read_text(encoding="utf-8")
