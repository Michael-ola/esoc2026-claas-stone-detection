from dataclasses import dataclass
from typing import Any

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


@dataclass(frozen=True)
class CNN2DConfig:
    """Configuration for the compact 2D CNN spectrogram classifier."""

    input_channels: int = 1
    n_classes: int = 2
    base_channels: int = 16
    dropout: float = 0.20


def require_torch() -> None:
    """Raise a clear error if PyTorch is unavailable."""
    if torch is None or nn is None:
        raise ImportError(
            "PyTorch is required for the 2D CNN model. "
            "Install torch before running CNN experiments."
        )


class TinySpectrogramCNN(nn.Module if nn is not None else object):
    """Compact 2D CNN for spectrogram-based stone detection.

    Input shape:
        batch x channels x frequency_bins x time_frames

    Output shape:
        batch x n_classes
    """

    def __init__(self, config: CNN2DConfig = CNN2DConfig()) -> None:
        require_torch()
        super().__init__()

        if config.input_channels <= 0:
            raise ValueError("input_channels must be positive.")

        if config.n_classes <= 1:
            raise ValueError("n_classes must be greater than 1.")

        if config.base_channels <= 0:
            raise ValueError("base_channels must be positive.")

        if not 0.0 <= config.dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1.")

        self.config = config

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=config.input_channels,
                out_channels=config.base_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(config.base_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=config.base_channels,
                out_channels=config.base_channels * 2,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(config.base_channels * 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=config.base_channels * 2,
                out_channels=config.base_channels * 4,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(config.base_channels * 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(config.dropout),
            nn.Linear(config.base_channels * 4, config.n_classes),
        )

    def forward(self, inputs: Any) -> Any:
        """Run a forward pass."""
        features = self.features(inputs)
        return self.classifier(features)


def count_trainable_parameters(model: Any) -> int:
    """Count trainable model parameters."""
    require_torch()
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
