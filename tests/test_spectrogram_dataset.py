import numpy as np
import pandas as pd
import pytest

from claas_stone_detection.spectrograms.dataset import (
    SpectrogramConfig,
    compute_log_spectrogram,
    extract_labeled_spectrograms,
    frame_signal,
    normalize_spectrogram,
    pad_audio_to_frame_size,
)


def test_compute_log_spectrogram_returns_frequency_by_time_array() -> None:
    audio = np.sin(2.0 * np.pi * 50.0 * np.arange(0.0, 1.0, 0.001))
    config = SpectrogramConfig(frame_size=128, hop_size=64)

    spectrogram = compute_log_spectrogram(audio, config=config)

    assert spectrogram.shape[0] == 65
    assert spectrogram.shape[1] > 0
    assert np.isfinite(spectrogram).all()


def test_compute_log_spectrogram_normalizes_output() -> None:
    audio = np.random.default_rng(123).normal(0.0, 1.0, size=512)
    config = SpectrogramConfig(frame_size=128, hop_size=64, normalize=True)

    spectrogram = compute_log_spectrogram(audio, config=config)

    assert float(np.mean(spectrogram)) == pytest.approx(0.0, abs=1e-12)
    assert float(np.std(spectrogram)) == pytest.approx(1.0)


def test_compute_log_spectrogram_rejects_non_1d_audio() -> None:
    audio = np.zeros((2, 10))

    with pytest.raises(ValueError, match="one-dimensional"):
        compute_log_spectrogram(audio)


def test_pad_audio_to_frame_size_pads_short_audio() -> None:
    audio = np.array([1.0, 2.0])

    result = pad_audio_to_frame_size(audio, frame_size=5)

    assert result.tolist() == [1.0, 2.0, 0.0, 0.0, 0.0]


def test_frame_signal_creates_overlapping_frames() -> None:
    audio = np.arange(8, dtype=float)

    frames = frame_signal(audio, frame_size=4, hop_size=2)

    assert frames.tolist() == [
        [0.0, 1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0, 5.0],
        [4.0, 5.0, 6.0, 7.0],
    ]


def test_normalize_spectrogram_handles_constant_input() -> None:
    spectrogram = np.ones((4, 4))

    result = normalize_spectrogram(spectrogram)

    assert np.allclose(result, 0.0)


def test_extract_labeled_spectrograms_reuses_labeled_window_table() -> None:
    df = pd.DataFrame(
        {
            "Sensor1": np.sin(2.0 * np.pi * 20.0 * np.arange(20) / 100.0),
        }
    )
    labeled_windows = pd.DataFrame(
        {
            "run_name": ["run_a"],
            "detection_time": [0.2],
            "start_index": [0],
            "end_index": [20],
            "label": [1],
        }
    )
    config = SpectrogramConfig(frame_size=8, hop_size=4)

    samples = extract_labeled_spectrograms(
        df=df,
        labeled_windows=labeled_windows,
        config=config,
    )

    assert len(samples) == 1
    assert samples[0].run_name == "run_a"
    assert samples[0].label == 1
    assert samples[0].spectrogram.shape[0] == 5


def test_extract_labeled_spectrograms_rejects_missing_columns() -> None:
    df = pd.DataFrame({"Sensor1": [0.0, 1.0]})
    labeled_windows = pd.DataFrame({"run_name": ["run_a"]})

    with pytest.raises(ValueError, match="Missing labeled window columns"):
        extract_labeled_spectrograms(df, labeled_windows)
