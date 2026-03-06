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
DEMO_DIR = os.path.join(_BACKEND_DIR, "demo")
TRAINING_LOG_PATH = os.path.join(_BACKEND_DIR, "training", "training.log")

PTBXL_FOLDER_NAME = (
    "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)
PTBXL_DIR = os.path.join(RAW_DATA_DIR, PTBXL_FOLDER_NAME)


@dataclass
class DataConfig:
    """Dataset and preprocessing configuration."""

    # Signal parameters
    sampling_rate: int = 100          # target sampling rate in Hz
    signal_duration_sec: int = 10     # 10-second recordings
    signal_length: int = 1000         # sampling_rate * signal_duration_sec
    n_leads: int = 12

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

    # Scheduler – ReduceLROnPlateau (monitor val AUC)
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6

    # Training loop
    epochs: int = 50
    batch_size: int = 16

    # Early stopping (monitor val AUC)
    early_stopping_patience: int = 8

    # Reproducibility
    seed: int = 42

    # Device
    device: str = "auto"  # "auto", "cuda", "cpu"

    # DataLoader
    num_workers: int = 2
    pin_memory: bool = True


@dataclass
class ModelConfig:
    """Architecture configuration for IndustryCNN."""

    input_length: int = 1000   # number of time-steps
    input_channels: int = 12   # all 12 leads
    num_classes: int = 1       # binary (logit)

    # ── Residual CNN ──
    initial_filters: int = 64
    num_res_blocks: int = 4
    dropout: float = 0.3


# ──────────────────────────────────────────────────────────────
# Convenience singletons
# ──────────────────────────────────────────────────────────────
DATA_CFG = DataConfig()
TRAIN_CFG = TrainingConfig()
MODEL_CFG = ModelConfig()
