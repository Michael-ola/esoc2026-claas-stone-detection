import importlib.util

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch is not installed.",
)

if TORCH_AVAILABLE:
    import torch

    from claas_stone_detection.models.cnn2d import (
        CNN2DConfig,
        TinySpectrogramCNN,
        count_trainable_parameters,
    )


def test_tiny_spectrogram_cnn_forward_shape() -> None:
    model = TinySpectrogramCNN(
        CNN2DConfig(
            input_channels=1,
            n_classes=2,
            base_channels=8,
            dropout=0.1,
        )
    )
    inputs = torch.randn(4, 1, 65, 12)

    outputs = model(inputs)

    assert outputs.shape == (4, 2)


def test_tiny_spectrogram_cnn_has_trainable_parameters() -> None:
    model = TinySpectrogramCNN(CNN2DConfig(base_channels=8))

    assert count_trainable_parameters(model) > 0


def test_tiny_spectrogram_cnn_rejects_invalid_config() -> None:
    with pytest.raises(ValueError, match="n_classes"):
        TinySpectrogramCNN(CNN2DConfig(n_classes=1))

    with pytest.raises(ValueError, match="dropout"):
        TinySpectrogramCNN(CNN2DConfig(dropout=1.0))
