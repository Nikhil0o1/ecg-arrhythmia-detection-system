"""
Training loop for the ECG Arrhythmia Detection pipeline.

Features:
    - AdamW optimiser with weight decay
    - ReduceLROnPlateau scheduler (monitors validation ROC-AUC)
    - Early stopping on validation ROC-AUC
    - BCEWithLogitsLoss with pos_weight for class imbalance
    - Best-model checkpointing based on highest val ROC-AUC
    - Per-epoch logging
"""

import logging
import os
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from training.config import SAVED_MODELS_DIR, TRAIN_CFG

logger = logging.getLogger("ecg_pipeline.train")


# ──────────────────────────────────────────────────────────────
# Early stopping helper (monitors val AUC — higher is better)
# ──────────────────────────────────────────────────────────────
class EarlyStopping:
    """Monitors validation ROC-AUC and signals when to stop."""

    def __init__(self, patience: int = TRAIN_CFG.early_stopping_patience) -> None:
        self.patience = patience
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def step(self, val_auc: float) -> bool:
        """Returns True if the score improved (new best)."""
        if self.best_score is None or val_auc > self.best_score:
            self.best_score = val_auc
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


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

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimiser.zero_grad()
        logits = model(X_batch).squeeze(-1)
        loss = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        running_loss += loss.item() * len(y_batch)
        n_samples += len(y_batch)

    return running_loss / max(n_samples, 1)


@torch.no_grad()
def _validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """Run validation. Returns (mean_loss, roc_auc)."""
    model.eval()
    running_loss = 0.0
    n_samples = 0
    all_labels = []
    all_probs = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        logits = model(X_batch).squeeze(-1)
        loss = criterion(logits, y_batch)

        running_loss += loss.item() * len(y_batch)
        n_samples += len(y_batch)

        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y_batch.cpu().numpy())

    mean_loss = running_loss / max(n_samples, 1)

    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)

    try:
        val_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        val_auc = 0.0

    return mean_loss, val_auc


# ──────────────────────────────────────────────────────────────
# Main training driver
# ──────────────────────────────────────────────────────────────
def train_model(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    pos_weight: torch.Tensor,
    device: torch.device,
    model_name: str = "IndustryCNN",
    epochs: int = TRAIN_CFG.epochs,
    lr: float = TRAIN_CFG.learning_rate,
    weight_decay: float = TRAIN_CFG.weight_decay,
) -> Dict[str, list]:
    """Full training loop for a single model.

    Checkpoints based on highest validation ROC-AUC.

    Returns
    -------
    history : dict with keys "train_loss", "val_loss", "val_auc"
    """
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    optimiser = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = ReduceLROnPlateau(
        optimiser,
        mode="max",  # monitoring AUC (higher is better)
        factor=TRAIN_CFG.scheduler_factor,
        patience=TRAIN_CFG.scheduler_patience,
        min_lr=TRAIN_CFG.scheduler_min_lr,
    )

    early_stopping = EarlyStopping(patience=TRAIN_CFG.early_stopping_patience)

    history: Dict[str, list] = {"train_loss": [], "val_loss": [], "val_auc": []}
    best_val_auc = 0.0
    save_path = os.path.join(SAVED_MODELS_DIR, f"{model_name}_best.pt")
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

    logger.info("=" * 60)
    logger.info(
        "Training %s | device=%s | epochs=%d | lr=%.1e | batch=%d",
        model_name, device, epochs, lr, TRAIN_CFG.batch_size,
    )
    logger.info("=" * 60)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = _train_one_epoch(
            model, loaders["train"], criterion, optimiser, device
        )
        val_loss, val_auc = _validate(
            model, loaders["val"], criterion, device
        )
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        current_lr = optimiser.param_groups[0]["lr"]

        logger.info(
            "[%s] Epoch %3d/%d | train_loss=%.5f  val_loss=%.5f  val_AUC=%.4f | lr=%.2e | %.1fs",
            model_name, epoch, epochs, train_loss, val_loss, val_auc, current_lr, elapsed,
        )

        # Checkpoint best model (based on val AUC)
        improved = early_stopping.step(val_auc)
        if improved:
            best_val_auc = val_auc
            torch.save(model.state_dict(), save_path)
            logger.info(
                "  -> Saved best model (val_AUC=%.4f) -> %s", val_auc, save_path
            )

        scheduler.step(val_auc)

        if early_stopping.should_stop:
            logger.info("  Early stopping triggered at epoch %d", epoch)
            break

    # Reload best weights
    model.load_state_dict(
        torch.load(save_path, map_location=device, weights_only=True)
    )
    logger.info(
        "Loaded best checkpoint for %s (val_AUC=%.4f)", model_name, best_val_auc
    )

    return history
