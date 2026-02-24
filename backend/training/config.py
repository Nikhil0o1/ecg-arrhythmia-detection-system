"""
Configuration module for ECG Arrhythmia Detection pipeline.

Central configuration for hyperparameters, paths, and training settings.
"""

import os
from dataclasses import dataclass, field
from typing import List


# ──────────────────────────────────────────────────────────────
# Path configuration (relative to backend/)
# ──────────────────────────────────────────────────────────────
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_DIR = os.path.join(_BACKEND_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(_BACKEND_DIR, "data", "processed")
SAVED_MODELS_DIR = os.path.join(_BACKEND_DIR, "models", "saved_models")
ARTIFACTS_DIR = os.path.join(_BACKEND_DIR, "evaluation", "artifacts")
REPORTS_DIR = os.path.join(_BACKEND_DIR, "evaluation", "reports")
TRAINING_LOG_PATH = os.path.join(_BACKEND_DIR, "training", "training.log")


@dataclass
class DataConfig:
    """Dataset and preprocessing configuration."""

    # PhysioNet PTB-XL dataset name
    ptbxl_db_name: str = "ptb-xl"
    ptbxl_version: str = "1.0.3"

    # Signal parameters
    sampling_rate: int = 100          # target sampling rate in Hz
    signal_duration_sec: int = 10     # 10-second recordings
    signal_length: int = 1000         # sampling_rate * signal_duration_sec
    n_leads: int = 12
    lead_index: int = 0               # Lead I (index 0)

    # Splitting ratios
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Reproducibility
    seed: int = 42


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    # Optimiser
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # Scheduler – ReduceLROnPlateau
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6

    # Training loop
    epochs: int = 50
    batch_size: int = 64

    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4

    # Reproducibility
    seed: int = 42

    # Device
    device: str = "auto"  # "auto", "cuda", "cpu"


@dataclass
class ModelConfig:
    """Architecture configuration for all three models."""

    input_length: int = 1000   # number of time-steps
    input_channels: int = 1    # single lead
    num_classes: int = 1       # binary (logit)

    # ── 1D CNN ──
    cnn_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    cnn_kernel_sizes: List[int] = field(default_factory=lambda: [7, 5, 3])
    cnn_dropout: float = 0.3

    # ── LSTM ──
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.3
    lstm_bidirectional: bool = True

    # ── Transformer ──
    transformer_d_model: int = 64
    transformer_nhead: int = 4
    transformer_num_layers: int = 2
    transformer_dim_feedforward: int = 128
    transformer_dropout: float = 0.2


# ──────────────────────────────────────────────────────────────
# Convenience singletons
# ──────────────────────────────────────────────────────────────
DATA_CFG = DataConfig()
TRAIN_CFG = TrainingConfig()
MODEL_CFG = ModelConfig()

# Model registry – maps names to class identifiers used in models.py
MODEL_NAMES: List[str] = ["CNN1D", "LSTMClassifier", "TransformerClassifier"]
