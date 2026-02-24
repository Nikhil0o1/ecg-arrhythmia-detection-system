"""
Training loop for the ECG Arrhythmia Detection pipeline.

Features:
    - Adam optimiser with weight decay
    - ReduceLROnPlateau scheduler
    - Early stopping
    - Weighted BCEWithLogitsLoss (handles class imbalance)
    - Best-model checkpointing to disk
    - Per-epoch logging of train & validation loss
"""

import logging
import os
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.config import SAVED_MODELS_DIR, TRAIN_CFG

logger = logging.getLogger("ecg_pipeline.train")


# ──────────────────────────────────────────────────────────────
# Early stopping helper
# ──────────────────────────────────────────────────────────────
class EarlyStopping:
    """Monitors validation loss and signals when to stop."""

    def __init__(
        self,
        patience: int = TRAIN_CFG.early_stopping_patience,
        min_delta: float = TRAIN_CFG.early_stopping_min_delta,
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: Optional[float] = None
        self.should_stop = False

    def step(self, val_loss: float) -> None:
        if self.best_loss is None:
            self.best_loss = val_loss
            return
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


# ──────────────────────────────────────────────────────────────
# Single-epoch helpers
# ──────────────────────────────────────────────────────────────
def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch. Returns mean loss."""
    model.train()
    running_loss = 0.0
    n_samples = 0

    for X_batch, y_batch in tqdm(loader, desc="  train", leave=False):
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimiser.zero_grad()
        logits = model(X_batch).squeeze(-1)
        loss = criterion(logits, y_batch)
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        running_loss += loss.item() * len(y_batch)
        n_samples += len(y_batch)

    return running_loss / n_samples


@torch.no_grad()
def _validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run validation. Returns mean loss."""
    model.eval()
    running_loss = 0.0
    n_samples = 0

    for X_batch, y_batch in tqdm(loader, desc="  val  ", leave=False):
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        logits = model(X_batch).squeeze(-1)
        loss = criterion(logits, y_batch)

        running_loss += loss.item() * len(y_batch)
        n_samples += len(y_batch)

    return running_loss / n_samples


# ──────────────────────────────────────────────────────────────
# Main training driver
# ──────────────────────────────────────────────────────────────
def train_model(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    pos_weight: torch.Tensor,
    device: torch.device,
    model_name: str,
    epochs: int = TRAIN_CFG.epochs,
    lr: float = TRAIN_CFG.learning_rate,
    weight_decay: float = TRAIN_CFG.weight_decay,
) -> Dict[str, list]:
    """
    Full training loop for a single model.

    Parameters
    ----------
    model : nn.Module
    loaders : dict with "train" and "val" DataLoaders
    pos_weight : weight tensor for BCEWithLogitsLoss
    device : torch device
    model_name : used for checkpoint filename
    epochs : max epochs
    lr : learning rate
    weight_decay : L2 regularisation

    Returns
    -------
    history : dict with keys "train_loss", "val_loss"
    """
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight.to(device),
    )

    optimiser = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = ReduceLROnPlateau(
        optimiser,
        mode="min",
        factor=TRAIN_CFG.scheduler_factor,
        patience=TRAIN_CFG.scheduler_patience,
        min_lr=TRAIN_CFG.scheduler_min_lr,
        
    )

    early_stopping = EarlyStopping(
        patience=TRAIN_CFG.early_stopping_patience,
        min_delta=TRAIN_CFG.early_stopping_min_delta,
    )

    history: Dict[str, list] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    save_path = os.path.join(SAVED_MODELS_DIR, f"{model_name}_best.pt")
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

    logger.info("═" * 60)
    logger.info("Training %s  |  device=%s  |  epochs=%d  |  lr=%.1e",
                model_name, device, epochs, lr)
    logger.info("═" * 60)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = _train_one_epoch(model, loaders["train"], criterion, optimiser, device)
        val_loss = _validate(model, loaders["val"], criterion, device)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        current_lr = optimiser.param_groups[0]["lr"]

        logger.info(
            "[%s] Epoch %3d/%d  │  train_loss=%.5f  val_loss=%.5f  │  lr=%.2e  │  %.1fs",
            model_name, epoch, epochs, train_loss, val_loss, current_lr, elapsed,
        )

        # Checkpoint best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            logger.info("  ✓ Saved best model (val_loss=%.5f) → %s", val_loss, save_path)

        scheduler.step(val_loss)

        early_stopping.step(val_loss)
        if early_stopping.should_stop:
            logger.info("  ⏹  Early stopping triggered at epoch %d", epoch)
            break

    # Reload best weights
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    logger.info("Loaded best checkpoint for %s (val_loss=%.5f)", model_name, best_val_loss)

    return history
