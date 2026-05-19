from pathlib import Path

import pytest

from claas_stone_detection.data.io import list_mf4_files, read_dataset, read_mf4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def test_list_mf4_files_returns_sorted_mf4_files(tmp_path: Path) -> None:
    file_b = tmp_path / "b.mf4"
    file_a = tmp_path / "a.mf4"
    text_file = tmp_path / "notes.txt"

    file_b.touch()
    file_a.touch()
    text_file.touch()

    result = list_mf4_files(tmp_path)

    assert result == [file_a, file_b]


def test_list_mf4_files_rejects_missing_directory(tmp_path: Path) -> None:
    missing_dir = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="Data directory not found"):
        list_mf4_files(missing_dir)


def test_list_mf4_files_rejects_file_instead_of_directory(tmp_path: Path) -> None:
    file_path = tmp_path / "not_a_directory.mf4"
    file_path.touch()

    with pytest.raises(NotADirectoryError, match="Expected a directory"):
        list_mf4_files(file_path)


def test_read_mf4_rejects_missing_file(tmp_path: Path) -> None:
    missing_file = tmp_path / "missing.mf4"

    with pytest.raises(FileNotFoundError, match="File not found"):
        read_mf4(missing_file)


def test_read_mf4_rejects_non_mf4_file(tmp_path: Path) -> None:
    text_file = tmp_path / "notes.txt"
    text_file.touch()

    with pytest.raises(ValueError, match="Expected an .mf4 file"):
        read_mf4(text_file)


def test_read_dataset_rejects_directory_without_mf4_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No .mf4 files found"):
        read_dataset(tmp_path)


@pytest.mark.skipif(
    not DATA_DIR.exists() or not list(DATA_DIR.glob("*.mf4")),
    reason="No real MF4 files available in data directory.",
)
def test_read_first_real_mf4_file_returns_expected_dataframe() -> None:
    files = list_mf4_files(DATA_DIR)

    df = read_mf4(files[0], raster=0.001)

    assert not df.empty
    assert df.index.name == "time_s"

    expected_columns = {
        "Sensor1",
        "VehicleSpeed",
        "CutLength",
        "VoltageSignal",
        "Status",
        "HeaderOn",
    }
    assert expected_columns.issubset(df.columns)


@pytest.mark.skipif(
    not DATA_DIR.exists() or not list(DATA_DIR.glob("*.mf4")),
    reason="No real MF4 files available in data directory.",
)
def test_read_dataset_returns_all_available_real_mf4_files() -> None:
    files = list_mf4_files(DATA_DIR)

    dataset = read_dataset(DATA_DIR, raster=0.001)

    assert len(dataset) == len(files)
    assert all(not df.empty for df in dataset.values())