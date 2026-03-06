"""
Evaluation module for the ECG Arrhythmia Detection pipeline.

Computes professional metrics and generates diagnostic plots:
    - Accuracy, Precision, Recall, F1, ROC-AUC, Sensitivity, Specificity
    - Confusion matrix heatmap
    - ROC curve
    - Precision-Recall curve
    - Saves test_metrics.json and all PNG artifacts
"""

import json
import logging
import os
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from training.config import ARTIFACTS_DIR

logger = logging.getLogger("ecg_pipeline.evaluate")


# ──────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference and collect predictions.

    Returns
    -------
    y_true  : ground-truth labels  (N,)
    y_prob  : predicted probabilities  (N,)
    y_pred  : predicted binary labels  (N,)
    """
    model.eval()
    all_labels = []
    all_probs = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        logits = model(X_batch).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y_batch.numpy())

    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    y_pred = (y_prob >= 0.5).astype(np.int64)
    return y_true, y_prob, y_pred


# ──────────────────────────────────────────────────────────────
# Metrics computation
# ──────────────────────────────────────────────────────────────
def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute standard binary classification metrics including sensitivity/specificity."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }
    return metrics


# ──────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────
def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
) -> None:
    """Generate and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["NORM", "ABNORM"],
        yticklabels=["NORM", "ABNORM"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — IndustryCNN")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix -> %s", save_path)


def _plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str,
) -> None:
    """Generate and save an ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc_val = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc_val:.4f}")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — IndustryCNN")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved ROC curve -> %s", save_path)


def _plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str,
) -> None:
    """Generate and save a Precision-Recall curve plot."""
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall_vals, precision_vals)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall_vals, precision_vals, color="blue", lw=2, label=f"PR-AUC = {pr_auc:.4f}")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — IndustryCNN")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved precision-recall curve -> %s", save_path)


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    artifacts_dir: str = ARTIFACTS_DIR,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Full evaluation pipeline:
        1. Run inference on the test set.
        2. Compute classification metrics.
        3. Save diagnostic plots and test_metrics.json.

    Returns
    -------
    metrics : dict of metric name -> value
    y_true  : ground-truth labels
    y_prob  : predicted probabilities
    """
    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. Predict
    y_true, y_prob, y_pred = predict(model, loader, device)

    # 2. Metrics
    metrics = compute_metrics(y_true, y_prob, y_pred)
    logger.info(
        "Test Metrics: %s",
        {k: f"{v:.4f}" for k, v in metrics.items()},
    )

    # 3. Plots
    _plot_confusion_matrix(
        y_true, y_pred,
        save_path=os.path.join(artifacts_dir, "confusion_matrix.png"),
    )
    _plot_roc_curve(
        y_true, y_prob,
        save_path=os.path.join(artifacts_dir, "roc_curve.png"),
    )
    _plot_precision_recall_curve(
        y_true, y_prob,
        save_path=os.path.join(artifacts_dir, "pr_curve.png"),
    )

    # 4. Save metrics JSON
    metrics_path = os.path.join(artifacts_dir, "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved test_metrics.json -> %s", metrics_path)

    return metrics, y_true, y_prob
