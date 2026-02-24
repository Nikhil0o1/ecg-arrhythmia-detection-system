"""
Evaluation module for the ECG Arrhythmia Detection pipeline.

Computes metrics and generates publication-quality diagnostic plots:
    - Accuracy, Precision, Recall, F1, ROC-AUC
    - Confusion matrix heatmap
    - ROC curve
    - Precision-Recall curve
    - Saves metrics.json and all PNG artifacts
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
from tqdm import tqdm

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
    """
    Run inference and collect predictions.

    Returns
    -------
    y_true  : ground-truth labels  (N,)
    y_prob  : predicted probabilities  (N,)
    y_pred  : predicted binary labels  (N,)
    """
    model.eval()
    all_labels = []
    all_probs = []

    for X_batch, y_batch in tqdm(loader, desc="  predict", leave=False):
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
    """Compute standard binary classification metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }
    return metrics


# ──────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────
def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    model_name: str,
) -> None:
    """Generate and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
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
    ax.set_title(f"Confusion Matrix – {model_name}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix → %s", save_path)


def _plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str,
    model_name: str,
) -> None:
    """Generate and save an ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc_val = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2,
            label=f"AUC = {roc_auc_val:.4f}")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve – {model_name}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved ROC curve → %s", save_path)


def _plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str,
    model_name: str,
) -> None:
    """Generate and save a Precision-Recall curve plot."""
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall_vals, precision_vals)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall_vals, precision_vals, color="blue", lw=2,
            label=f"PR-AUC = {pr_auc:.4f}")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve – {model_name}")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved precision-recall curve → %s", save_path)


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_name: str,
    artifacts_dir: str = ARTIFACTS_DIR,
) -> Dict[str, float]:
    """
    Full evaluation pipeline for a single model:
        1. Run inference on the test set.
        2. Compute classification metrics.
        3. Save diagnostic plots and metrics JSON.

    Parameters
    ----------
    model : trained nn.Module
    loader : test DataLoader
    device : torch.device
    model_name : identifier used in filenames and plot titles
    artifacts_dir : directory for output artifacts

    Returns
    -------
    metrics : dict of metric name → value
    """
    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. Predict
    y_true, y_prob, y_pred = predict(model, loader, device)

    # 2. Metrics
    metrics = compute_metrics(y_true, y_prob, y_pred)
    logger.info("Metrics for %s: %s", model_name,
                {k: f"{v:.4f}" for k, v in metrics.items()})

    # 3. Plots
    _plot_confusion_matrix(
        y_true, y_pred,
        save_path=os.path.join(artifacts_dir, f"{model_name}_confusion_matrix.png"),
        model_name=model_name,
    )
    _plot_roc_curve(
        y_true, y_prob,
        save_path=os.path.join(artifacts_dir, f"{model_name}_roc_curve.png"),
        model_name=model_name,
    )
    _plot_precision_recall_curve(
        y_true, y_prob,
        save_path=os.path.join(artifacts_dir, f"{model_name}_precision_recall_curve.png"),
        model_name=model_name,
    )

    # 4. Save metrics JSON
    metrics_path = os.path.join(artifacts_dir, f"{model_name}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics JSON → %s", metrics_path)

    return metrics


def save_comparison_report(
    all_metrics: Dict[str, Dict[str, float]],
    artifacts_dir: str = ARTIFACTS_DIR,
) -> None:
    """
    Save a combined comparison of all models' metrics to a single JSON.
    """
    path = os.path.join(artifacts_dir, "comparison_metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info("Saved model comparison report → %s", path)
