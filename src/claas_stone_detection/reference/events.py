from dataclasses import dataclass

import numpy as np
import pandas as pd

from claas_stone_detection.core.schema import DEFAULT_SCHEMA, ChannelSchema
from claas_stone_detection.reference.episodes import Episode, extract_header_on_episodes


@dataclass(frozen=True)
class StoneEvent:
    """Reference voltage event inferred from the metal detector signal."""

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
        """Duration of the above-threshold event region in seconds."""
        return self.end_time - self.start_time

    @property
    def peak_to_threshold_ratio(self) -> float:
        """Ratio between the event peak and the local detection threshold."""
        if self.threshold == 0.0:
            return float("inf")

        return self.peak_voltage / self.threshold


def group_candidate_regions(
    candidate_times: np.ndarray,
    min_gap_s: float,
) -> list[tuple[float, float]]:
    """Group candidate timestamps separated by less than min_gap_s."""
    if len(candidate_times) == 0:
        return []

    if min_gap_s < 0:
        raise ValueError("min_gap_s cannot be negative.")

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
    threshold_quantile: float = 0.999,
    min_gap_s: float = 0.5,
    min_duration_s: float = 0.01,
    run_name: str = "",
    schema: ChannelSchema = DEFAULT_SCHEMA,
    deduplicate: bool = True,
    duplicate_tolerance_s: float = 0.001,
) -> list[StoneEvent]:
    """Detect candidate reference events from the voltage signal.

    Detection is performed inside each extended header-on episode using an
    episode-local quantile threshold. Events are deduplicated by peak time
    because overlapping or extended episode windows can otherwise identify the
    same physical voltage peak multiple times.
    """
    if schema.voltage not in df.columns:
        raise ValueError(f"Missing voltage column: {schema.voltage}")

    if not 0.0 < threshold_quantile < 1.0:
        raise ValueError("threshold_quantile must be between 0 and 1.")

    if min_duration_s < 0:
        raise ValueError("min_duration_s cannot be negative.")

    if duplicate_tolerance_s <= 0:
        raise ValueError("duplicate_tolerance_s must be positive.")

    if episodes is None:
        episodes = extract_header_on_episodes(df)

    events: list[StoneEvent] = []

    for episode in episodes:
        episode_df = df.loc[episode.start_time : episode.extended_end_time]

        if episode_df.empty:
            continue

        threshold = float(episode_df[schema.voltage].quantile(threshold_quantile))
        candidate_df = episode_df[episode_df[schema.voltage] > threshold]
        candidate_times = candidate_df.index.to_numpy(dtype=float)

        groups = group_candidate_regions(
            candidate_times=candidate_times,
            min_gap_s=min_gap_s,
        )

        for start_time, end_time in groups:
            if end_time - start_time < min_duration_s:
                continue

            group_df = episode_df.loc[start_time:end_time]

            if group_df.empty:
                continue

            peak_time = float(group_df[schema.voltage].idxmax())
            peak_voltage = float(group_df.loc[peak_time, schema.voltage])

            events.append(
                StoneEvent(
                    run_name=run_name,
                    start_time=float(start_time),
                    peak_time=peak_time,
                    end_time=float(end_time),
                    peak_voltage=peak_voltage,
                    threshold=threshold,
                    episode_start_time=episode.start_time,
                    episode_end_time=episode.end_time,
                    source=schema.voltage,
                )
            )

    if not deduplicate:
        return events

    return deduplicate_stone_events(
        events=events,
        duplicate_tolerance_s=duplicate_tolerance_s,
    )


def detect_voltage_events_in_dataset(
    dataset: dict[str, pd.DataFrame],
    threshold_quantile: float = 0.999,
    min_gap_s: float = 0.5,
    min_duration_s: float = 0.01,
    schema: ChannelSchema = DEFAULT_SCHEMA,
) -> dict[str, list[StoneEvent]]:
    """Detect reference voltage events for each run in a dataset."""
    events_by_run: dict[str, list[StoneEvent]] = {}

    for run_name, df in dataset.items():
        episodes = extract_header_on_episodes(df)
        events_by_run[run_name] = detect_voltage_events(
            df=df,
            episodes=episodes,
            threshold_quantile=threshold_quantile,
            min_gap_s=min_gap_s,
            min_duration_s=min_duration_s,
            run_name=run_name,
            schema=schema,
        )

    return events_by_run


def deduplicate_stone_events(
    events: list[StoneEvent],
    duplicate_tolerance_s: float = 0.001,
) -> list[StoneEvent]:
    """Remove duplicate event detections with nearly identical peak times.

    When the same voltage peak is found through multiple episode windows, the
    preferred event is the one whose peak is closest to the end of its associated
    header-on episode. This keeps shutdown-trigger-like reference events aligned
    with the episode most likely to have caused them. Ties are resolved by the
    larger peak-to-threshold ratio.
    """
    if duplicate_tolerance_s <= 0:
        raise ValueError("duplicate_tolerance_s must be positive.")

    if not events:
        return []

    grouped: dict[tuple[str, int], list[StoneEvent]] = {}

    for event in events:
        peak_key = int(round(event.peak_time / duplicate_tolerance_s))
        grouped.setdefault((event.run_name, peak_key), []).append(event)

    deduplicated: list[StoneEvent] = []

    for duplicates in grouped.values():
        best_event = min(
            duplicates,
            key=lambda event: (
                abs(event.peak_time - event.episode_end_time),
                -event.peak_to_threshold_ratio,
                -event.peak_voltage,
            ),
        )
        deduplicated.append(best_event)

    return sorted(
        deduplicated,
        key=lambda event: (event.run_name, event.peak_time),
    )
