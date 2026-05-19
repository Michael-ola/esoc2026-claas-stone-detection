from pathlib import Path

import pandas as pd

from claas_stone_detection.reference.events import StoneEvent


def read_synthetic_dataset(
    synthetic_dir: str | Path,
    max_runs: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Read generated synthetic CSV runs from disk.

    The returned structure matches the real-data Task 1 interface:

        dict[str, pandas.DataFrame]

    This allows synthetic data to reuse the same Task 2 windowing, feature,
    labeling, model, and evaluation pipeline.
    """
    synthetic_path = Path(synthetic_dir)

    if not synthetic_path.exists():
        raise FileNotFoundError(f"Synthetic directory not found: {synthetic_path}")

    csv_paths = sorted(synthetic_path.glob("synthetic_run_*.csv"))

    if max_runs is not None:
        if max_runs <= 0:
            raise ValueError("max_runs must be positive when provided.")

        csv_paths = csv_paths[:max_runs]

    if not csv_paths:
        raise FileNotFoundError(
            f"No synthetic_run_*.csv files found in {synthetic_path}"
        )

    dataset: dict[str, pd.DataFrame] = {}

    for csv_path in csv_paths:
        run_name = csv_path.stem
        frame = pd.read_csv(csv_path, index_col="time_s")
        frame.index = frame.index.astype(float)

        if "HeaderOn" in frame.columns:
            frame["HeaderOn"] = frame["HeaderOn"].astype(bool)

        dataset[run_name] = frame

    return dataset


def read_synthetic_metadata(
    synthetic_dir: str | Path,
    allowed_runs: set[str] | None = None,
) -> pd.DataFrame:
    """Read synthetic event metadata produced by the generator script."""
    synthetic_path = Path(synthetic_dir)
    metadata_path = synthetic_path / "metadata.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Synthetic metadata not found: {metadata_path}")

    metadata = pd.read_csv(metadata_path)

    if allowed_runs is None or metadata.empty:
        return metadata

    return metadata[metadata["run_name"].isin(allowed_runs)].copy()


def metadata_to_stone_events(
    metadata: pd.DataFrame,
) -> dict[str, list[StoneEvent]]:
    """Convert synthetic event metadata into StoneEvent references.

    Synthetic metadata contains the injected event time directly. To reuse the
    existing Task 2 labeling and evaluation code, each injected event is mapped
    into the same StoneEvent dataclass used by voltage-derived real references.
    """
    required_columns = {"run_name", "event_time", "amplitude"}
    missing_columns = required_columns.difference(metadata.columns)

    if missing_columns:
        raise ValueError(f"Missing synthetic metadata columns: {missing_columns}")

    events_by_run: dict[str, list[StoneEvent]] = {}

    for row in metadata.itertuples(index=False):
        run_name = str(row.run_name)
        event_time = float(row.event_time)
        amplitude = float(row.amplitude)

        events_by_run.setdefault(run_name, []).append(
            StoneEvent(
                run_name=run_name,
                start_time=event_time,
                peak_time=event_time,
                end_time=event_time,
                peak_voltage=amplitude,
                threshold=1.0,
                episode_start_time=event_time,
                episode_end_time=event_time,
                source="synthetic_metadata",
            )
        )

    return events_by_run
