from dataclasses import FrozenInstanceError

import pytest

from claas_stone_detection.schema import DEFAULT_SCHEMA


def test_default_schema_contains_expected_raw_channels() -> None:
    assert DEFAULT_SCHEMA.sensor == "Sensor1"
    assert DEFAULT_SCHEMA.vehicle_speed == "VehicleSpeed"
    assert DEFAULT_SCHEMA.cut_length == "CutLength"
    assert DEFAULT_SCHEMA.voltage == "VoltageSignal"
    assert DEFAULT_SCHEMA.status == "Status"


def test_default_schema_contains_expected_derived_columns() -> None:
    assert DEFAULT_SCHEMA.header_on == "HeaderOn"


def test_required_channels_excludes_derived_columns() -> None:
    assert DEFAULT_SCHEMA.required_channels == (
        "Sensor1",
        "VehicleSpeed",
        "CutLength",
        "VoltageSignal",
        "Status",
    )

    assert DEFAULT_SCHEMA.header_on not in DEFAULT_SCHEMA.required_channels


def test_default_schema_is_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        DEFAULT_SCHEMA.sensor = "DifferentSensor"