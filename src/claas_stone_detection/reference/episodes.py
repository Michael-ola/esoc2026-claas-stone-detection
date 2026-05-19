from dataclasses import dataclass

import numpy as np
import pandas as pd

from claas_stone_detection.core.schema import DEFAULT_SCHEMA, ChannelSchema


@dataclass(frozen=True)
class Episode:
    """A continuous header-on operating period in one measurement run."""

    start_time: float
    end_time: float
    extended_end_time: float

    @property
    def duration_s(self) -> float:
        """Duration of the original header-on episode in seconds."""
        return self.end_time - self.start_time

    @property
    def extended_duration_s(self) -> float:
        """Duration including the post-episode grace period."""
        return self.extended_end_time - self.start_time


def extract_header_on_episodes(
    df: pd.DataFrame,
    schema: ChannelSchema = DEFAULT_SCHEMA,
    min_duration_s: float = 0.1,
    grace_s: float = 2.0,
) -> list[Episode]:
    """Extract continuous header-on episodes from a measurement DataFrame.

    Parameters
    ----------
    df:
        Measurement DataFrame containing the derived HeaderOn column.
    schema:
        Channel schema containing the HeaderOn column name.
    min_duration_s:
        Minimum original HeaderOn duration required to keep an episode.
    grace_s:
        Extra time added after the HeaderOn period ends. This helps associate
        voltage peaks that occur immediately after automatic shutdown.

    Returns
    -------
    list[Episode]
        Extracted header-on episodes.
    """
    if schema.header_on not in df.columns:
        raise ValueError(f"Missing header-on column: {schema.header_on}")

    if df.empty:
        return []

    header_on = df[schema.header_on].astype(bool)
    times = df.index.to_numpy(dtype=float)
    max_time = float(times[-1])

    state = header_on.astype(int)
    previous_state = state.shift(fill_value=0)
    transitions = state - previous_state

    starts = df.index[transitions == 1].to_numpy(dtype=float)
    ends = df.index[transitions == -1].to_numpy(dtype=float)

    if bool(header_on.iloc[0]):
        starts = np.insert(starts, 0, times[0])

    if bool(header_on.iloc[-1]):
        ends = np.append(ends, times[-1])

    episodes: list[Episode] = []

    for start_time, end_time in zip(starts, ends):
        start_time = float(start_time)
        end_time = float(end_time)

        if end_time - start_time < min_duration_s:
            continue

        extended_end_time = min(end_time + grace_s, max_time)

        episodes.append(
            Episode(
                start_time=start_time,
                end_time=end_time,
                extended_end_time=extended_end_time,
            )
        )

    return episodes