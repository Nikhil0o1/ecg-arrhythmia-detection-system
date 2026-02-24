"""
Model definitions for ECG Arrhythmia Detection.

Three architectures for binary classification of 10-second single-lead ECG:
    1. CNN1D          – 1-D Convolutional Neural Network
    2. LSTMClassifier – Bi-directional LSTM
    3. TransformerClassifier – Lightweight Transformer encoder

All models:
    • Accept input of shape (batch_size, 1000, 1)
    • Output a single logit (binary classification via BCEWithLogitsLoss)
"""

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.config import MODEL_CFG


# ══════════════════════════════════════════════════════════════
# 1. 1-D CNN
# ══════════════════════════════════════════════════════════════
class CNN1D(nn.Module):
    """
    Multi-layer 1-D convolutional network with batch-norm, ReLU, max-pool,
    and a final fully-connected classifier head.
    """

    def __init__(
        self,
        input_channels: int = MODEL_CFG.input_channels,
        input_length: int = MODEL_CFG.input_length,
        conv_channels: List[int] = None,
        kernel_sizes: List[int] = None,
        dropout: float = MODEL_CFG.cnn_dropout,
        num_classes: int = MODEL_CFG.num_classes,
    ) -> None:
        super().__init__()
        if conv_channels is None:
            conv_channels = MODEL_CFG.cnn_channels
        if kernel_sizes is None:
            kernel_sizes = MODEL_CFG.cnn_kernel_sizes

        layers: list = []
        in_ch = input_channels
        for out_ch, ks in zip(conv_channels, kernel_sizes):
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=ks, padding=ks // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Dropout(dropout),
            ])
            in_ch = out_ch

        self.feature_extractor = nn.Sequential(*layers)

        # Calculate flattened dimension: each MaxPool halves the length
        feat_len = input_length
        for _ in conv_channels:
            feat_len = feat_len // 2

        self.classifier = nn.Sequential(
            nn.Linear(in_ch * feat_len, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len=1000, channels=1)

        Returns
        -------
        logits : (batch, 1)
        """
        # Conv1d expects (batch, channels, length)
        x = x.permute(0, 2, 1)
        x = self.feature_extractor(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)


# ══════════════════════════════════════════════════════════════
# 2. LSTM Classifier
# ══════════════════════════════════════════════════════════════
class LSTMClassifier(nn.Module):
    """
    Bi-directional LSTM with attention pooling over time-steps.
    """

    def __init__(
        self,
        input_channels: int = MODEL_CFG.input_channels,
        hidden_size: int = MODEL_CFG.lstm_hidden_size,
        num_layers: int = MODEL_CFG.lstm_num_layers,
        dropout: float = MODEL_CFG.lstm_dropout,
        bidirectional: bool = MODEL_CFG.lstm_bidirectional,
        num_classes: int = MODEL_CFG.num_classes,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        direction_factor = 2 if bidirectional else 1
        lstm_out_dim = hidden_size * direction_factor

        # Attention
        self.attention = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Linear(lstm_out_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len=1000, 1)

        Returns
        -------
        logits : (batch, 1)
        """
        output, _ = self.lstm(x)  # (batch, seq, lstm_out_dim)

        # Attention pooling
        attn_weights = self.attention(output)          # (batch, seq, 1)
        attn_weights = F.softmax(attn_weights, dim=1)  # normalise over time
        context = (output * attn_weights).sum(dim=1)   # (batch, lstm_out_dim)

        return self.classifier(context)


# ══════════════════════════════════════════════════════════════
# 3. Lightweight Transformer Classifier
# ══════════════════════════════════════════════════════════════
class _PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    """
    Lightweight Transformer encoder for 1-D ECG time-series.

    Projects the input to d_model, adds positional encoding,
    passes through Transformer encoder layers, then uses CLS-style
    mean pooling before the classification head.
    """

    def __init__(
        self,
        input_channels: int = MODEL_CFG.input_channels,
        d_model: int = MODEL_CFG.transformer_d_model,
        nhead: int = MODEL_CFG.transformer_nhead,
        num_layers: int = MODEL_CFG.transformer_num_layers,
        dim_feedforward: int = MODEL_CFG.transformer_dim_feedforward,
        dropout: float = MODEL_CFG.transformer_dropout,
        num_classes: int = MODEL_CFG.num_classes,
        input_length: int = MODEL_CFG.input_length,
    ) -> None:
        super().__init__()

        self.input_proj = nn.Linear(input_channels, d_model)
        self.pos_encoder = _PositionalEncoding(d_model, max_len=input_length, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len=1000, 1)

        Returns
        -------
        logits : (batch, 1)
        """
        x = self.input_proj(x)            # (batch, seq, d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)   # (batch, seq, d_model)

        # Global average pooling over the time dimension
        x = x.mean(dim=1)                 # (batch, d_model)
        return self.classifier(x)


# ──────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "CNN1D": CNN1D,
    "LSTMClassifier": LSTMClassifier,
    "TransformerClassifier": TransformerClassifier,
}


def build_model(name: str, **kwargs) -> nn.Module:
    """Instantiate a model by its registered name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)
