"""
Data loading module for the ECG Arrhythmia Detection pipeline.

Provides:
    - Stratified train / validation / test splitting
    - PyTorch Dataset and DataLoader construction
"""

import logging
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset

from training.config import DATA_CFG, TRAIN_CFG

logger = logging.getLogger("ecg_pipeline.data_loader")


# ──────────────────────────────────────────────────────────────
# Stratified split
# ──────────────────────────────────────────────────────────────
def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = DATA_CFG.train_ratio,
    val_ratio: float = DATA_CFG.val_ratio,
    test_ratio: float = DATA_CFG.test_ratio,
    seed: int = DATA_CFG.seed,
) -> Tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
]:
    """
    Perform a stratified split into train / val / test sets.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "Split ratios must sum to 1.0"
    )

    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y,
        test_size=val_test_ratio,
        stratify=y,
        random_state=seed,
    )

    # Second split: val vs test
    relative_test = test_ratio / val_test_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=relative_test,
        stratify=y_tmp,
        random_state=seed,
    )

    logger.info(
        "Split sizes → train=%d  val=%d  test=%d",
        len(y_train), len(y_val), len(y_test),
    )
    for name, labels in [("train", y_train), ("val", y_val), ("test", y_test)]:
        pos = labels.sum()
        neg = len(labels) - pos
        logger.info("  %s → NORM=%d  ABNORM=%d  (pos_rate=%.2f%%)",
                     name, neg, pos, 100 * pos / len(labels))

    return X_train, y_train, X_val, y_val, X_test, y_test


# ──────────────────────────────────────────────────────────────
# PyTorch Dataset / DataLoader
# ──────────────────────────────────────────────────────────────
class ECGDataset(Dataset):
    """Simple wrapper around numpy arrays → torch tensors."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        # X: (N, seq_len, 1)   y: (N,)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def build_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = TRAIN_CFG.batch_size,
) -> Dict[str, DataLoader]:
    """
    Build PyTorch DataLoaders for train / val / test.

    Returns
    -------
    dict with keys "train", "val", "test"
    """
    loaders: Dict[str, DataLoader] = {}

    for split, (X, y) in [
        ("train", (X_train, y_train)),
        ("val", (X_val, y_val)),
        ("test", (X_test, y_test)),
    ]:
        dataset = ECGDataset(X, y)
        shuffle = split == "train"
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        logger.info("DataLoader[%s]: %d samples, %d batches",
                     split, len(dataset), len(loaders[split]))

    return loaders
