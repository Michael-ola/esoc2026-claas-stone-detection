import pandas as pd

from claas_stone_detection.core.schema import DEFAULT_SCHEMA, ChannelSchema


def normalize_status_column(
    df: pd.DataFrame,
    schema: ChannelSchema = DEFAULT_SCHEMA,
) -> pd.DataFrame:
    """Add a boolean header-on column derived from the raw Status channel.

    The README describes Status as a boolean channel, but the MF4 files may
    expose it as values such as b'On', b'Off', 'On', 'Off', 1, or 0.
    This function preserves the raw Status column and adds a normalized
    boolean column named HeaderOn.
    """
    if schema.status not in df.columns:
        raise ValueError(f"Missing status column: {schema.status}")

    result = df.copy()

    def to_header_on(value: object) -> bool:
        if isinstance(value, bytes):
            value = value.decode(errors="ignore")

        if isinstance(value, str):
            return value.strip().lower() in {"on", "1", "true", "yes"}

        if isinstance(value, bool):
            return value

        if isinstance(value, (int, float)):
            return bool(value)

        raise ValueError(f"Unsupported status value: {value!r}")

    result[schema.header_on] = result[schema.status].map(to_header_on)

    return result