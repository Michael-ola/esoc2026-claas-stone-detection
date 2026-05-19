from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SignalWindow:
    """A live-style time window used for streaming inference.

    The time fields are used for event matching and evaluation. The index
    fields are used for fast slicing with pandas iloc.

    end_index is exclusive, following normal Python slicing convention.
    """

    run_name: str
    start_time: float
    end_time: float
    start_index: int
    end_index: int

    @property
    def duration_s(self) -> float:
        """Window duration in seconds."""
        return self.end_time - self.start_time

    @property
    def detection_time(self) -> float:
        """Time at which this window can produce a live detection."""
        return self.end_time


def make_sliding_windows(
    df: pd.DataFrame,
    window_s: float,
    hop_s: float,
    run_name: str = "",
    start_time: float | None = None,
    end_time: float | None = None,
) -> list[SignalWindow]:
    """Create live-style sliding windows over a time-indexed DataFrame.

    The detection time of each window is the nominal window end time, because a
    live detector can only act after observing the full window.

    Windows store both timestamps and integer index positions. The index
    positions allow fast slicing with iloc on large high-frequency data.
    """
    if df.empty:
        return []

    if window_s <= 0:
        raise ValueError("window_s must be positive.")

    if hop_s <= 0:
        raise ValueError("hop_s must be positive.")

    index = df.index.to_numpy(dtype=float)
    min_time = float(index[0])
    max_time = float(index[-1])

    requested_start_time = min_time if start_time is None else float(start_time)
    requested_end_time = max_time if end_time is None else float(end_time)

    if requested_start_time < min_time:
        raise ValueError("start_time cannot be earlier than the DataFrame index.")

    if requested_end_time > max_time:
        raise ValueError("end_time cannot be later than the DataFrame index.")

    if requested_start_time + window_s > requested_end_time:
        return []

    start_index = int(np.searchsorted(index, requested_start_time, side="left"))
    sample_rate_hz = infer_sample_rate_hz_from_index(df.index)

    if sample_rate_hz > 0:
        return _make_index_based_windows(
            index=index,
            run_name=run_name,
            window_s=window_s,
            hop_s=hop_s,
            start_index=start_index,
            requested_end_time=requested_end_time,
            sample_rate_hz=sample_rate_hz,
        )

    return _make_time_based_windows(
        index=index,
        run_name=run_name,
        window_s=window_s,
        hop_s=hop_s,
        requested_start_time=requested_start_time,
        requested_end_time=requested_end_time,
    )


def infer_sample_rate_hz_from_index(index: pd.Index) -> float:
    """Infer sampling rate from a numeric time index in seconds."""
    if len(index) < 2:
        return 0.0

    times = index.to_numpy(dtype=float)
    diffs = np.diff(times)
    positive_diffs = diffs[diffs > 0]

    if len(positive_diffs) == 0:
        return 0.0

    median_dt = float(np.median(positive_diffs))

    if median_dt <= 0:
        return 0.0

    return 1.0 / median_dt


def slice_window(df: pd.DataFrame, window: SignalWindow) -> pd.DataFrame:
    """Return the DataFrame slice corresponding to a SignalWindow."""
    return df.iloc[window.start_index : window.end_index]


def _make_index_based_windows(
    index: np.ndarray,
    run_name: str,
    window_s: float,
    hop_s: float,
    start_index: int,
    requested_end_time: float,
    sample_rate_hz: float,
) -> list[SignalWindow]:
    """Create windows using integer sample steps for efficient slicing."""
    window_samples = max(1, int(round(window_s * sample_rate_hz)))
    hop_samples = max(1, int(round(hop_s * sample_rate_hz)))

    windows: list[SignalWindow] = []
    current_start_index = start_index
    epsilon = 1e-12

    while current_start_index < len(index):
        current_start_time = float(index[current_start_index])
        current_end_time = current_start_time + window_s

        if current_end_time > requested_end_time + epsilon:
            break

        current_end_index = current_start_index + window_samples

        if current_end_index > len(index):
            break

        windows.append(
            SignalWindow(
                run_name=run_name,
                start_time=current_start_time,
                end_time=float(current_end_time),
                start_index=current_start_index,
                end_index=current_end_index,
            )
        )

        current_start_index += hop_samples

    return windows


def _make_time_based_windows(
    index: np.ndarray,
    run_name: str,
    window_s: float,
    hop_s: float,
    requested_start_time: float,
    requested_end_time: float,
) -> list[SignalWindow]:
    """Fallback windowing path for indexes where sample rate is unavailable."""
    windows: list[SignalWindow] = []
    current_start_time = requested_start_time
    epsilon = 1e-12

    while current_start_time + window_s <= requested_end_time + epsilon:
        current_end_time = current_start_time + window_s

        current_start_index = int(
            np.searchsorted(index, current_start_time, side="left")
        )
        current_end_index = int(np.searchsorted(index, current_end_time, side="left"))

        if current_end_index <= current_start_index:
            break

        windows.append(
            SignalWindow(
                run_name=run_name,
                start_time=float(current_start_time),
                end_time=float(current_end_time),
                start_index=current_start_index,
                end_index=current_end_index,
            )
        )

        current_start_time += hop_s

    return windows
