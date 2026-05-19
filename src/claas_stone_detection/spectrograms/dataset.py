from dataclasses import dataclass

import numpy as np
import pandas as pd

from claas_stone_detection.core.schema import DEFAULT_SCHEMA, ChannelSchema
from claas_stone_detection.streaming.windowing import SignalWindow


@dataclass(frozen=True)
class SpectrogramConfig:
    """Configuration for log-spectrogram extraction."""

    frame_size: int = 128
    hop_size: int = 32
    eps: float = 1e-8
    normalize: bool = True


@dataclass(frozen=True)
class SpectrogramWindow:
    """One spectrogram sample extracted from a live signal window."""

    run_name: str
    detection_time: float
    spectrogram: np.ndarray
    label: int | None = None


def compute_log_spectrogram(
    audio: np.ndarray,
    config: SpectrogramConfig = SpectrogramConfig(),
) -> np.ndarray:
    """Compute a log-power spectrogram using only NumPy.

    The output shape is:

        frequency_bins x time_frames

    This function is intentionally framework-agnostic so it can feed PyTorch,
    TensorFlow, ONNX, or embedded preprocessing later.
    """
    audio = np.asarray(audio, dtype=float)

    if audio.ndim != 1:
        raise ValueError("audio must be a one-dimensional array.")

    if config.frame_size <= 1:
        raise ValueError("frame_size must be greater than 1.")

    if config.hop_size <= 0:
        raise ValueError("hop_size must be positive.")

    if config.eps <= 0:
        raise ValueError("eps must be positive.")

    if len(audio) == 0:
        return np.empty((config.frame_size // 2 + 1, 0), dtype=float)

    padded_audio = pad_audio_to_frame_size(
        audio=audio,
        frame_size=config.frame_size,
    )
    frames = frame_signal(
        audio=padded_audio,
        frame_size=config.frame_size,
        hop_size=config.hop_size,
    )

    window = np.hanning(config.frame_size)
    windowed_frames = frames * window[None, :]
    spectrum = np.fft.rfft(windowed_frames, axis=1)
    power = np.abs(spectrum) ** 2
    log_power = np.log(power + config.eps).T

    if config.normalize:
        log_power = normalize_spectrogram(log_power)

    return log_power.astype(float)


def extract_window_spectrogram(
    df: pd.DataFrame,
    window: SignalWindow,
    config: SpectrogramConfig = SpectrogramConfig(),
    schema: ChannelSchema = DEFAULT_SCHEMA,
) -> SpectrogramWindow:
    """Extract a log-spectrogram for one SignalWindow."""
    if schema.sensor not in df.columns:
        raise ValueError(f"Missing sensor column: {schema.sensor}")

    audio = df.iloc[window.start_index : window.end_index][schema.sensor].to_numpy()
    spectrogram = compute_log_spectrogram(audio=audio, config=config)

    return SpectrogramWindow(
        run_name=window.run_name,
        detection_time=window.end_time,
        spectrogram=spectrogram,
        label=None,
    )


def extract_labeled_spectrograms(
    df: pd.DataFrame,
    labeled_windows: pd.DataFrame,
    config: SpectrogramConfig = SpectrogramConfig(),
    schema: ChannelSchema = DEFAULT_SCHEMA,
) -> list[SpectrogramWindow]:
    """Extract spectrogram samples from a labeled window table.

    The input is the same labeled table produced by the shared Task 2 baseline
    pipeline. This lets the 2D CNN reuse the existing windowing and labeling
    logic instead of rebuilding it.
    """
    required_columns = {
        "run_name",
        "detection_time",
        "start_index",
        "end_index",
        "label",
    }
    missing_columns = required_columns.difference(labeled_windows.columns)

    if missing_columns:
        raise ValueError(f"Missing labeled window columns: {missing_columns}")

    if schema.sensor not in df.columns:
        raise ValueError(f"Missing sensor column: {schema.sensor}")

    samples: list[SpectrogramWindow] = []

    for row in labeled_windows.itertuples(index=False):
        audio = df.iloc[int(row.start_index) : int(row.end_index)][
            schema.sensor
        ].to_numpy()
        spectrogram = compute_log_spectrogram(audio=audio, config=config)

        samples.append(
            SpectrogramWindow(
                run_name=str(row.run_name),
                detection_time=float(row.detection_time),
                spectrogram=spectrogram,
                label=int(row.label),
            )
        )

    return samples


def pad_audio_to_frame_size(audio: np.ndarray, frame_size: int) -> np.ndarray:
    """Pad short audio clips so at least one FFT frame can be computed."""
    if frame_size <= 1:
        raise ValueError("frame_size must be greater than 1.")

    audio = np.asarray(audio, dtype=float)

    if len(audio) >= frame_size:
        return audio

    return np.pad(audio, (0, frame_size - len(audio)), mode="constant")


def frame_signal(
    audio: np.ndarray,
    frame_size: int,
    hop_size: int,
) -> np.ndarray:
    """Split a one-dimensional signal into overlapping frames."""
    if frame_size <= 1:
        raise ValueError("frame_size must be greater than 1.")

    if hop_size <= 0:
        raise ValueError("hop_size must be positive.")

    audio = np.asarray(audio, dtype=float)

    if len(audio) < frame_size:
        audio = pad_audio_to_frame_size(audio, frame_size=frame_size)

    starts = np.arange(0, len(audio) - frame_size + 1, hop_size)

    if len(starts) == 0:
        starts = np.array([0])

    frames = np.stack([audio[start : start + frame_size] for start in starts])
    return frames


def normalize_spectrogram(spectrogram: np.ndarray) -> np.ndarray:
    """Normalize a spectrogram to zero mean and unit variance."""
    spectrogram = np.asarray(spectrogram, dtype=float)
    std = float(np.std(spectrogram))

    if std == 0.0:
        return np.zeros_like(spectrogram, dtype=float)

    return (spectrogram - float(np.mean(spectrogram))) / std
