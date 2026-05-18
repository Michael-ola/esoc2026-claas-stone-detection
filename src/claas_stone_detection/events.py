from dataclasses import dataclass

import numpy as np
import pandas as pd

from claas_stone_detection.episodes import Episode, extract_header_on_episodes
from claas_stone_detection.schema import DEFAULT_SCHEMA, ChannelSchema


@dataclass(frozen=True)
class StoneEvent:
    """Reference stone event inferred from a voltage signal peak."""

    run_name: str
    start_time: float
    peak_time: float
    end_time: float
    peak_voltage: float
    threshold: float
    episode_start_time: float
    episode_end_time: float
    source: str = "VoltageSignal"

    @property
    def duration_s(self) -> float:
        """Duration of the above-threshold voltage region."""
        return self.end_time - self.start_time

    @property
    def peak_to_threshold_ratio(self) -> float:
        """Peak voltage divided by the local detection threshold."""
        if self.threshold == 0:
            return float("inf")
        return self.peak_voltage / self.threshold


def group_candidate_regions(
    candidate_times: np.ndarray,
    min_gap_s: float,
) -> list[tuple[float, float]]:
    """Group candidate timestamps into contiguous event regions.

    Candidate timestamps separated by at most `min_gap_s` are treated as part
    of the same event region.
    """
    if len(candidate_times) == 0:
        return []

    groups: list[tuple[float, float]] = []
    start_time = float(candidate_times[0])
    previous_time = float(candidate_times[0])

    for time_s in candidate_times[1:]:
        time_s = float(time_s)

        if time_s - previous_time > min_gap_s:
            groups.append((start_time, previous_time))
            start_time = time_s

        previous_time = time_s

    groups.append((start_time, previous_time))

    return groups


def detect_voltage_events(
    df: pd.DataFrame,
    episodes: list[Episode] | None = None,
    schema: ChannelSchema = DEFAULT_SCHEMA,
    threshold_quantile: float = 0.999,
    min_gap_s: float = 0.5,
    min_duration_s: float = 0.01,
    run_name: str = "",
) -> list[StoneEvent]:
    """Detect voltage reference events within header-on episodes.

    The voltage threshold is computed separately for each extended episode
    window. This adapts the detection to local operating conditions.

    Parameters
    ----------
    df:
        Measurement DataFrame containing VoltageSignal and HeaderOn columns.
    episodes:
        Optional precomputed header-on episodes. If not provided, episodes are
        extracted using the default episode extraction settings.
    schema:
        Channel schema.
    threshold_quantile:
        Episode-level voltage quantile used as the adaptive threshold.
    min_gap_s:
        Minimum time gap separating voltage event regions.
    min_duration_s:
        Minimum above-threshold region duration required to keep an event.
    run_name:
        Name of the measurement run. Stored in each returned StoneEvent.

    Returns
    -------
    list[StoneEvent]
        Detected voltage reference events.
    """
    if schema.voltage not in df.columns:
        raise ValueError(f"Missing voltage column: {schema.voltage}")

    if episodes is None:
        episodes = extract_header_on_episodes(df, schema=schema)

    events: list[StoneEvent] = []

    for episode in episodes:
        episode_df = df.loc[episode.start_time : episode.extended_end_time]

        if episode_df.empty:
            continue

        voltage = episode_df[schema.voltage]
        threshold = float(voltage.quantile(threshold_quantile))

        candidate_df = episode_df[episode_df[schema.voltage] > threshold]
        candidate_times = candidate_df.index.to_numpy(dtype=float)

        groups = group_candidate_regions(
            candidate_times=candidate_times,
            min_gap_s=min_gap_s,
        )

        for start_time, end_time in groups:
            duration_s = end_time - start_time
            if duration_s < min_duration_s:
                continue

            group_df = df.loc[start_time:end_time]
            peak_time = float(group_df[schema.voltage].idxmax())
            peak_voltage = float(group_df.loc[peak_time, schema.voltage])

            events.append(
                StoneEvent(
                    run_name=run_name,
                    start_time=start_time,
                    peak_time=peak_time,
                    end_time=end_time,
                    peak_voltage=peak_voltage,
                    threshold=threshold,
                    episode_start_time=episode.start_time,
                    episode_end_time=episode.end_time,
                    source=schema.voltage,
                )
            )

    return events


def detect_voltage_events_in_dataset(
    dataset: dict[str, pd.DataFrame],
    schema: ChannelSchema = DEFAULT_SCHEMA,
    threshold_quantile: float = 0.999,
    min_gap_s: float = 0.5,
    min_duration_s: float = 0.01,
) -> dict[str, list[StoneEvent]]:
    """Detect voltage reference events for all runs in a dataset."""
    return {
        run_name: detect_voltage_events(
            df=df,
            schema=schema,
            threshold_quantile=threshold_quantile,
            min_gap_s=min_gap_s,
            min_duration_s=min_duration_s,
            run_name=run_name,
        )
        for run_name, df in dataset.items()
    }