from dataclasses import dataclass

import pandas as pd

from claas_stone_detection.events import StoneEvent
from claas_stone_detection.windowing import SignalWindow

IGNORE_LABEL = -1
NEGATIVE_LABEL = 0
POSITIVE_LABEL = 1


@dataclass(frozen=True)
class WindowLabel:
    """Label assigned to one live-style signal window."""

    run_name: str
    window_start: float
    window_end: float
    detection_time: float
    label: int
    event_peak_time: float | None
    time_to_event_s: float | None

    @property
    def is_ignored(self) -> bool:
        """Whether this label should be excluded from model training."""
        return self.label == IGNORE_LABEL


def label_window(
    window: SignalWindow,
    events: list[StoneEvent],
    positive_horizon_s: float = 2.0,
    post_event_exclusion_s: float = 1.0,
) -> WindowLabel:
    """Assign an early-detection label to one live-style window.

    Label priority is intentionally conservative:

    1. Windows shortly after a reference event are ignored first, because they
       may contain shutdown or post-impact transients.
    2. Remaining windows shortly before a reference event are positive.
    3. All other windows are negative.

    This prevents contaminated post-event windows from becoming positive labels
    when events occur close together.
    """
    if positive_horizon_s <= 0:
        raise ValueError("positive_horizon_s must be positive.")

    if post_event_exclusion_s < 0:
        raise ValueError("post_event_exclusion_s cannot be negative.")

    run_events = sorted(
        (event for event in events if event.run_name == window.run_name),
        key=lambda event: event.peak_time,
    )

    detection_time = window.detection_time

    for event in run_events:
        time_since_event_s = detection_time - event.peak_time

        if 0.0 <= time_since_event_s <= post_event_exclusion_s:
            return WindowLabel(
                run_name=window.run_name,
                window_start=window.start_time,
                window_end=window.end_time,
                detection_time=detection_time,
                label=IGNORE_LABEL,
                event_peak_time=event.peak_time,
                time_to_event_s=-time_since_event_s,
            )

    for event in run_events:
        time_to_event_s = event.peak_time - detection_time

        if 0.0 < time_to_event_s <= positive_horizon_s:
            return WindowLabel(
                run_name=window.run_name,
                window_start=window.start_time,
                window_end=window.end_time,
                detection_time=detection_time,
                label=POSITIVE_LABEL,
                event_peak_time=event.peak_time,
                time_to_event_s=time_to_event_s,
            )

    return WindowLabel(
        run_name=window.run_name,
        window_start=window.start_time,
        window_end=window.end_time,
        detection_time=detection_time,
        label=NEGATIVE_LABEL,
        event_peak_time=None,
        time_to_event_s=None,
    )


def label_windows(
    windows: list[SignalWindow],
    events: list[StoneEvent],
    positive_horizon_s: float = 2.0,
    post_event_exclusion_s: float = 1.0,
) -> pd.DataFrame:
    """Create a label table for a list of live-style windows."""
    labels = [
        label_window(
            window=window,
            events=events,
            positive_horizon_s=positive_horizon_s,
            post_event_exclusion_s=post_event_exclusion_s,
        )
        for window in windows
    ]

    return pd.DataFrame(
        [
            {
                "run_name": label.run_name,
                "window_start": label.window_start,
                "window_end": label.window_end,
                "detection_time": label.detection_time,
                "label": label.label,
                "event_peak_time": label.event_peak_time,
                "time_to_event_s": label.time_to_event_s,
            }
            for label in labels
        ]
    )


def label_feature_table(
    feature_table: pd.DataFrame,
    events_by_run: dict[str, list[StoneEvent]],
    positive_horizon_s: float = 2.0,
    post_event_exclusion_s: float = 1.0,
) -> pd.DataFrame:
    """Attach early-detection labels to a feature table.

    The input table must contain run_name, window_start, window_end, and
    detection_time columns.
    """
    required_columns = {"run_name", "window_start", "window_end", "detection_time"}
    missing_columns = required_columns.difference(feature_table.columns)

    if missing_columns:
        raise ValueError(f"Missing required feature table columns: {missing_columns}")

    labeled_table = feature_table.copy()
    labels: list[int] = []
    event_peak_times: list[float | None] = []
    time_to_events: list[float | None] = []

    for row in labeled_table.itertuples(index=False):
        run_name = str(row.run_name)
        detection_time = float(row.detection_time)
        window_start = float(row.window_start)
        window_end = float(row.window_end)

        pseudo_window = SignalWindow(
            run_name=run_name,
            start_time=window_start,
            end_time=window_end,
            start_index=0,
            end_index=0,
        )

        label = label_window(
            window=pseudo_window,
            events=events_by_run.get(run_name, []),
            positive_horizon_s=positive_horizon_s,
            post_event_exclusion_s=post_event_exclusion_s,
        )

        if label.detection_time != detection_time:
            raise RuntimeError("Unexpected detection time mismatch while labeling.")

        labels.append(label.label)
        event_peak_times.append(label.event_peak_time)
        time_to_events.append(label.time_to_event_s)

    labeled_table["label"] = labels
    labeled_table["event_peak_time"] = event_peak_times
    labeled_table["time_to_event_s"] = time_to_events

    return labeled_table
