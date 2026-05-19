from pathlib import Path

import pandas as pd
import pytest

from claas_stone_detection.data.synthetic_io import (
    metadata_to_stone_events,
    read_synthetic_dataset,
    read_synthetic_metadata,
)


def write_synthetic_run(path: Path, run_name: str) -> None:
    frame = pd.DataFrame(
        {
            "time_s": [0.0, 0.1, 0.2],
            "Sensor1": [0.0, 1.0, 0.0],
            "VehicleSpeed": [2.0, 2.1, 2.2],
            "CutLength": [10.0, 10.0, 10.0],
            "VoltageSignal": [35.0, 100.0, 35.0],
            "Status": ["On", "On", "On"],
            "HeaderOn": [True, True, True],
        }
    )
    frame.to_csv(path / f"{run_name}.csv", index=False)


def test_read_synthetic_dataset_loads_csv_runs(tmp_path: Path) -> None:
    write_synthetic_run(tmp_path, "synthetic_run_000")
    write_synthetic_run(tmp_path, "synthetic_run_001")

    dataset = read_synthetic_dataset(tmp_path)

    assert sorted(dataset.keys()) == ["synthetic_run_000", "synthetic_run_001"]
    assert dataset["synthetic_run_000"].index.name == "time_s"
    assert dataset["synthetic_run_000"].index.tolist() == [0.0, 0.1, 0.2]
    assert dataset["synthetic_run_000"]["HeaderOn"].dtype == bool


def test_read_synthetic_dataset_respects_max_runs(tmp_path: Path) -> None:
    write_synthetic_run(tmp_path, "synthetic_run_000")
    write_synthetic_run(tmp_path, "synthetic_run_001")
    write_synthetic_run(tmp_path, "synthetic_run_002")

    dataset = read_synthetic_dataset(tmp_path, max_runs=2)

    assert sorted(dataset.keys()) == ["synthetic_run_000", "synthetic_run_001"]


def test_read_synthetic_dataset_rejects_missing_directory(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="Synthetic directory not found"):
        read_synthetic_dataset(missing_path)


def test_read_synthetic_dataset_rejects_invalid_max_runs(tmp_path: Path) -> None:
    write_synthetic_run(tmp_path, "synthetic_run_000")

    with pytest.raises(ValueError, match="max_runs must be positive"):
        read_synthetic_dataset(tmp_path, max_runs=0)


def test_read_synthetic_metadata_loads_and_filters_runs(tmp_path: Path) -> None:
    metadata = pd.DataFrame(
        {
            "run_name": ["synthetic_run_000", "synthetic_run_001"],
            "event_time": [1.0, 2.0],
            "amplitude": [5.0, 6.0],
            "width_s": [0.02, 0.03],
            "vehicle_speed": [3.0, 3.1],
            "cut_length": [10.0, 11.0],
            "run_file": ["synthetic_run_000.csv", "synthetic_run_001.csv"],
        }
    )
    metadata.to_csv(tmp_path / "metadata.csv", index=False)

    result = read_synthetic_metadata(
        tmp_path,
        allowed_runs={"synthetic_run_001"},
    )

    assert result["run_name"].tolist() == ["synthetic_run_001"]


def test_read_synthetic_metadata_rejects_missing_metadata(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Synthetic metadata not found"):
        read_synthetic_metadata(tmp_path)


def test_metadata_to_stone_events_converts_rows() -> None:
    metadata = pd.DataFrame(
        {
            "run_name": ["synthetic_run_000", "synthetic_run_000"],
            "event_time": [1.5, 3.0],
            "amplitude": [4.2, 5.5],
        }
    )

    events_by_run = metadata_to_stone_events(metadata)

    assert sorted(events_by_run.keys()) == ["synthetic_run_000"]
    assert len(events_by_run["synthetic_run_000"]) == 2
    assert events_by_run["synthetic_run_000"][0].peak_time == 1.5
    assert events_by_run["synthetic_run_000"][0].source == "synthetic_metadata"


def test_metadata_to_stone_events_rejects_missing_columns() -> None:
    metadata = pd.DataFrame({"run_name": ["synthetic_run_000"]})

    with pytest.raises(ValueError, match="Missing synthetic metadata columns"):
        metadata_to_stone_events(metadata)
