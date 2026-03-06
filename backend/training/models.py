"""
Model definitions for ECG Arrhythmia Detection.

IndustryCNN – Deep 1-D Residual Convolutional Neural Network for
binary classification of 10-second 12-lead ECG recordings.

Architecture:
    Conv1D(12 → 64) + BN + ReLU
    4 × ResidualBlock (double conv with skip connection)
    Dropout(0.3)
    Global Average Pooling
    Linear(→ 1)

Input:  (batch, 12, 1000) — channel-first 12-lead ECG
Output: (batch, 1)        — logit for BCEWithLogitsLoss
"""

import torch
import torch.nn as nn

from training.config import MODEL_CFG


class ResidualBlock(nn.Module):
    """1-D residual block: two convolutions with skip connection.

    If in_channels != out_channels, a 1×1 convolution is used for the
    shortcut to match dimensions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(dropout)

        # Shortcut projection
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual
        out = self.relu(out)
        return out


class IndustryCNN(nn.Module):
    """Deep 1-D Residual CNN for 12-lead ECG binary classification.

    Fits comfortably in 4 GB VRAM with batch_size=16.
    """

    def __init__(
        self,
        input_channels: int = MODEL_CFG.input_channels,
        initial_filters: int = MODEL_CFG.initial_filters,
        num_res_blocks: int = MODEL_CFG.num_res_blocks,
        dropout: float = MODEL_CFG.dropout,
        num_classes: int = MODEL_CFG.num_classes,
    ) -> None:
        super().__init__()

        # ── Stem ──
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, initial_filters, kernel_size=15, padding=7),
            nn.BatchNorm1d(initial_filters),
            nn.ReLU(inplace=True),
        )

        # ── Residual blocks ──
        # Channel progression: 64 → 64 → 128 → 128 → 256
        channels = [initial_filters]
        for i in range(num_res_blocks):
            if i % 2 == 1:
                channels.append(channels[-1] * 2)
            else:
                channels.append(channels[-1])

        blocks = []
        for i in range(num_res_blocks):
            blocks.append(
                ResidualBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=7,
                    dropout=dropout,
                )
            )
            # Downsample after each block to reduce sequence length
            blocks.append(nn.MaxPool1d(kernel_size=2, stride=2))

        self.res_blocks = nn.Sequential(*blocks)

        self.dropout = nn.Dropout(dropout)

        # ── Global Average Pooling + Classifier ──
        self.classifier = nn.Linear(channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, 12, 1000) — channel-first

        Returns
        -------
        logits : (batch, 1)
        """
        x = self.stem(x)
        x = self.res_blocks(x)
        x = self.dropout(x)

        # Global Average Pooling over the time dimension
        x = x.mean(dim=-1)  # (batch, C)

        return self.classifier(x)


# ──────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "IndustryCNN": IndustryCNN,
}


def build_model(name: str, **kwargs) -> nn.Module:
    """Instantiate a model by its registered name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](**kwargs)
