from dataclasses import dataclass

import numpy as np
import pandas as pd

from claas_stone_detection.core.schema import DEFAULT_SCHEMA, ChannelSchema
from claas_stone_detection.streaming.windowing import SignalWindow, slice_window


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for fixed window-level feature extraction."""

    frequency_bands_hz: tuple[tuple[float, float], ...] = (
        (0.0, 500.0),
        (500.0, 2000.0),
        (2000.0, 8000.0),
        (8000.0, 16000.0),
    )


DEFAULT_FEATURE_CONFIG = FeatureConfig()


def infer_sample_rate_hz(index: pd.Index) -> float:
    """Infer the sampling rate from a numeric time index in seconds."""
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


def zero_crossing_rate(signal: np.ndarray, duration_s: float) -> float:
    """Compute zero-crossing rate in crossings per second."""
    if len(signal) < 2 or duration_s <= 0:
        return 0.0

    crossings = np.count_nonzero(np.diff(np.signbit(signal)))
    return float(crossings / duration_s)


def extract_window_features(
    window_df: pd.DataFrame,
    schema: ChannelSchema = DEFAULT_SCHEMA,
    config: FeatureConfig = DEFAULT_FEATURE_CONFIG,
) -> dict[str, float]:
    """Extract fixed audio and machine-context features from one window."""
    if window_df.empty:
        raise ValueError("Cannot extract features from an empty window.")

    if schema.sensor not in window_df.columns:
        raise ValueError(f"Missing audio column: {schema.sensor}")

    audio = window_df[schema.sensor].to_numpy(dtype=float)
    duration_s = float(window_df.index.max() - window_df.index.min())
    sample_rate_hz = infer_sample_rate_hz(window_df.index)

    audio_mean = float(np.mean(audio))
    audio_std = float(np.std(audio))
    audio_rms = float(np.sqrt(np.mean(np.square(audio))))
    audio_peak_abs = float(np.max(np.abs(audio)))
    crest_factor = audio_peak_abs / audio_rms if audio_rms > 0 else 0.0

    features: dict[str, float] = {
        "sample_rate_hz": sample_rate_hz,
        "audio_mean": audio_mean,
        "audio_std": audio_std,
        "audio_rms": audio_rms,
        "audio_peak_abs": audio_peak_abs,
        "audio_crest_factor": float(crest_factor),
        "audio_zero_crossing_rate": zero_crossing_rate(audio, duration_s),
    }

    features.update(
        extract_frequency_features(
            audio=audio,
            sample_rate_hz=sample_rate_hz,
            frequency_bands_hz=config.frequency_bands_hz,
        )
    )

    for column in (schema.vehicle_speed, schema.cut_length):
        if column in window_df.columns:
            values = window_df[column].to_numpy(dtype=float)
            features[f"{column}_mean"] = float(np.mean(values))
            features[f"{column}_std"] = float(np.std(values))

    return features


def extract_frequency_features(
    audio: np.ndarray,
    sample_rate_hz: float,
    frequency_bands_hz: tuple[tuple[float, float], ...],
) -> dict[str, float]:
    """Extract compact frequency-domain features from an audio window."""
    features = _zero_frequency_features(frequency_bands_hz)

    if len(audio) < 2 or sample_rate_hz <= 0.0:
        return features

    centered_audio = audio - np.mean(audio)

    if np.allclose(centered_audio, 0.0):
        return features

    window_function = np.hanning(len(centered_audio))
    windowed_audio = centered_audio * window_function

    spectrum = np.fft.rfft(windowed_audio)
    frequencies = np.fft.rfftfreq(len(windowed_audio), d=1.0 / sample_rate_hz)
    power = np.square(np.abs(spectrum))
    total_power = float(np.sum(power))

    if total_power <= 0:
        return features

    centroid = float(np.sum(frequencies * power) / total_power)
    bandwidth = float(
        np.sqrt(np.sum(np.square(frequencies - centroid) * power) / total_power)
    )

    features["spectral_centroid_hz"] = centroid
    features["spectral_bandwidth_hz"] = bandwidth

    high_frequency_power = 0.0

    for low_hz, high_hz in frequency_bands_hz:
        mask = (frequencies >= low_hz) & (frequencies < high_hz)
        band_power = float(np.sum(power[mask]))
        features[_band_feature_name(low_hz, high_hz)] = band_power / total_power

        if low_hz >= 2000.0:
            high_frequency_power += band_power

    features["high_frequency_energy_ratio"] = high_frequency_power / total_power

    return features


def make_feature_table(
    df: pd.DataFrame,
    windows: list[SignalWindow],
    schema: ChannelSchema = DEFAULT_SCHEMA,
    config: FeatureConfig = DEFAULT_FEATURE_CONFIG,
) -> pd.DataFrame:
    """Create a feature table from a list of live-style signal windows."""
    rows: list[dict[str, float | str]] = []

    for window in windows:
        window_df = slice_window(df, window)
        features = extract_window_features(
            window_df=window_df,
            schema=schema,
            config=config,
        )

        rows.append(
            {
                "run_name": window.run_name,
                "window_start": window.start_time,
                "window_end": window.end_time,
                "detection_time": window.detection_time,
                "start_index": window.start_index,
                "end_index": window.end_index,
                **features,
            }
        )

    return pd.DataFrame(rows)


def _zero_frequency_features(
    frequency_bands_hz: tuple[tuple[float, float], ...],
) -> dict[str, float]:
    features: dict[str, float] = {}

    for low_hz, high_hz in frequency_bands_hz:
        features[_band_feature_name(low_hz, high_hz)] = 0.0

    features["spectral_centroid_hz"] = 0.0
    features["spectral_bandwidth_hz"] = 0.0
    features["high_frequency_energy_ratio"] = 0.0

    return features


def _band_feature_name(low_hz: float, high_hz: float) -> str:
    low = int(low_hz)
    high = int(high_hz)
    return f"band_energy_{low}_{high}_hz"
