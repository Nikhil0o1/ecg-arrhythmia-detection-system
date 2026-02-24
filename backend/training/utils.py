"""
Utility helpers for the ECG Arrhythmia Detection pipeline.

Provides:
    - Seed setting for full reproducibility
    - Logging setup
    - Device resolution
    - Directory creation helpers
"""

import logging
import os
import random
from typing import Optional

import numpy as np
import torch

from training.config import (
    ARTIFACTS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
    SAVED_MODELS_DIR,
    TRAINING_LOG_PATH,
)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic behaviour (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_device(preference: str = "auto") -> torch.device:
    """Return the appropriate torch device."""
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def setup_logging(log_path: Optional[str] = None) -> logging.Logger:
    """
    Configure the root logger for the pipeline.

    Logs are written to both the console and a file.
    """
    if log_path is None:
        log_path = TRAINING_LOG_PATH

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger("ecg_pipeline")
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers on re-init
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def ensure_directories() -> None:
    """Create all required project directories if they do not exist."""
    for d in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        SAVED_MODELS_DIR,
        ARTIFACTS_DIR,
        REPORTS_DIR,
    ]:
        os.makedirs(d, exist_ok=True)


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for binary labels.

    Parameters
    ----------
    labels : np.ndarray
        1-D array of binary labels (0 or 1).

    Returns
    -------
    torch.Tensor
        Scalar weight for the positive class (used with BCEWithLogitsLoss pos_weight).
    """
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0:
        return torch.tensor(1.0)
    pos_weight = n_neg / n_pos
    return torch.tensor(pos_weight, dtype=torch.float32)
