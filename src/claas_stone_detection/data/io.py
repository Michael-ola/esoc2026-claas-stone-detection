from pathlib import Path

import pandas as pd
from asammdf import MDF

from claas_stone_detection.core.preprocessing import normalize_status_column
from claas_stone_detection.core.schema import DEFAULT_SCHEMA, ChannelSchema
from claas_stone_detection.core.validation import validate_measurement_dataframe


def read_mf4(
    file_path: str | Path,
    schema: ChannelSchema = DEFAULT_SCHEMA,
    raster: float | None = None,
) -> pd.DataFrame:
    """Read one CLAAS MF4 measurement file into a validated DataFrame.

    Parameters
    ----------
    file_path:
        Path to the MF4 file.
    schema:
        Expected channel names.
    raster:
        Optional sampling interval in seconds. If provided, channels are
        resampled to a common time base.

    Returns
    -------
    pandas.DataFrame
        Time-indexed DataFrame containing the required sensor channels.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() != ".mf4":
        raise ValueError(f"Expected an .mf4 file, got: {path.name}")

    with MDF(path) as mdf:
        df = mdf.to_dataframe(
            channels=list(schema.required_channels),
            raster=raster,
            time_from_zero=True,
        )

    df = df.sort_index()
    df.index.name = "time_s"
    df = normalize_status_column(df, schema=schema)

    validate_measurement_dataframe(df, schema=schema)

    return df


def list_mf4_files(data_dir: str | Path) -> list[Path]:
    """Return sorted MF4 files from a data directory."""
    path = Path(data_dir)

    if not path.exists():
        raise FileNotFoundError(f"Data directory not found: {path}")

    if not path.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {path}")

    return sorted(path.glob("*.mf4"))

def read_dataset(
    data_dir: str | Path,
    schema: ChannelSchema = DEFAULT_SCHEMA,
    raster: float | None = None,
) -> dict[str, pd.DataFrame]:
    """Read all MF4 files in a directory.

    Parameters
    ----------
    data_dir:
        Directory containing MF4 files.
    schema:
        Expected channel names.
    raster:
        Optional sampling interval in seconds. If provided, channels are
        resampled to a common time base.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Mapping from file stem to validated measurement DataFrame.
    """
    files = list_mf4_files(data_dir)

    if not files:
        raise FileNotFoundError(f"No .mf4 files found in: {data_dir}")

    return {
        file.stem: read_mf4(file, schema=schema, raster=raster)
        for file in files
    }