import pandas as pd

from claas_stone_detection.core.schema import DEFAULT_SCHEMA, ChannelSchema


def validate_measurement_dataframe(
    df: pd.DataFrame,
    schema: ChannelSchema = DEFAULT_SCHEMA,
) -> None:
    """Validate the structure of a loaded CLAAS measurement DataFrame.

    Parameters
    ----------
    df:
        Loaded measurement data.
    schema:
        Expected channel schema.

    Raises
    ------
    ValueError
        If the DataFrame is empty, has an invalid time index, or is missing
        required channels.
    """
    if df.empty:
        raise ValueError("Measurement DataFrame is empty.")

    if df.index.name != "time_s":
        raise ValueError("Expected DataFrame index name to be 'time_s'.")

    if not df.index.is_monotonic_increasing:
        raise ValueError("Expected time index to be monotonically increasing.")

    missing_channels = [
        channel for channel in schema.required_channels if channel not in df.columns
    ]
    if missing_channels:
        raise ValueError(f"Missing required channels: {missing_channels}")