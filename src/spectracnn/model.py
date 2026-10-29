"""1D CNN for automatic modulation classification."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .signals import DEFAULT_LENGTH, MODULATIONS


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ModulationCNN(nn.Module):
    """Four convolutional blocks followed by a small classifier head."""

    def __init__(
        self,
        n_classes: int = len(MODULATIONS),
        in_channels: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32, kernel_size=7),
            ConvBlock(32, 64, kernel_size=5),
            ConvBlock(64, 128, kernel_size=3),
            ConvBlock(128, 128, kernel_size=3),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        pooled = self.pool(features).flatten(1)
        return self.classifier(pooled)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def save_torchscript(
    model: nn.Module,
    path: str | Path,
    length: int = DEFAULT_LENGTH,
) -> Path:
    """Trace on CPU and persist a TorchScript module for serving."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    example = torch.randn(1, 2, length)
    with torch.inference_mode():
        traced = torch.jit.trace(model, example, strict=False)
    traced.save(str(path))
    return path


def load_torchscript(path: str | Path) -> torch.jit.ScriptModule:
    module = torch.jit.load(str(path), map_location="cpu")
    module.eval()
    return module


def predict_proba(
    model: nn.Module | torch.jit.ScriptModule,
    X: torch.Tensor,
) -> torch.Tensor:
    with torch.inference_mode():
        logits = model(X)
        return torch.softmax(logits, dim=1)
