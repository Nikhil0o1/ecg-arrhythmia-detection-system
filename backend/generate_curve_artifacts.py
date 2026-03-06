#!/usr/bin/env python
"""
generate_curve_artifacts.py — Generate JSON artifacts for ROC curve,
confusion matrix, and precision-recall curve from the trained IndustryCNN model.

Usage:
    cd backend/
    python generate_curve_artifacts.py
"""

import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from training.config import ARTIFACTS_DIR, PTBXL_DIR, TRAIN_CFG
from training.data_loader import build_dataloaders, patient_wise_split
from training.models import build_model
from training.preprocessing import build_manifest
from training.utils import resolve_device, set_seed


def main() -> None:
    set_seed(TRAIN_CFG.seed)
    device = resolve_device(TRAIN_CFG.device)
    print(f"Device: {device}")

    # ── Load data ──
    print("Building manifest & splits...")
    manifest = build_manifest(PTBXL_DIR)
    train_df, val_df, test_df = patient_wise_split(manifest)
    loaders = build_dataloaders(train_df, val_df, test_df, dataset_dir=PTBXL_DIR)
    test_loader = loaders["test"]

    # ── Load IndustryCNN ──
    print("Loading IndustryCNN...")
    model = build_model("IndustryCNN")
    ckpt_path = os.path.join(_BACKEND_DIR, "models", "saved_models", "IndustryCNN_best.pt")
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # ── Run inference ──
    print("Running inference on test set...")
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            logits = model(X_batch).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y_batch.numpy())

    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)
    y_pred = (y_prob >= 0.5).astype(np.int64)

    print(f"Test samples: {len(y_true)}")
    print(f"Positive rate: {y_true.mean():.4f}")

    # ── ROC curve data ──
    print("Computing ROC curve...")
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    roc_auc_val = float(auc(fpr, tpr))

    # Downsample for frontend performance (keep ~200 points)
    n_roc = len(fpr)
    if n_roc > 200:
        indices = np.linspace(0, n_roc - 1, 200, dtype=int)
        fpr_ds = fpr[indices]
        tpr_ds = tpr[indices]
    else:
        fpr_ds = fpr
        tpr_ds = tpr

    roc_data = {
        "fpr": [round(float(x), 6) for x in fpr_ds],
        "tpr": [round(float(x), 6) for x in tpr_ds],
        "auc": round(roc_auc_val, 6),
        "model_name": "IndustryCNN",
        "n_samples": int(len(y_true)),
    }

    roc_path = os.path.join(ARTIFACTS_DIR, "roc_curve_data.json")
    with open(roc_path, "w") as f:
        json.dump(roc_data, f, indent=2)
    print(f"Saved ROC curve data -> {roc_path}")

    # ── Confusion matrix data ──
    print("Computing confusion matrix...")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    cm_data = {
        "matrix": cm.tolist(),  # [[TN, FP], [FN, TP]]
        "labels": ["Normal", "Arrhythmia"],
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "total": int(len(y_true)),
        "model_name": "IndustryCNN",
    }

    cm_path = os.path.join(ARTIFACTS_DIR, "confusion_matrix_data.json")
    with open(cm_path, "w") as f:
        json.dump(cm_data, f, indent=2)
    print(f"Saved confusion matrix data -> {cm_path}")

    # ── Precision-Recall curve data ──
    print("Computing Precision-Recall curve...")
    prec_vals, rec_vals, _ = precision_recall_curve(y_true, y_prob)
    pr_auc_val = float(auc(rec_vals, prec_vals))

    n_pr = len(prec_vals)
    if n_pr > 200:
        indices = np.linspace(0, n_pr - 1, 200, dtype=int)
        prec_ds = prec_vals[indices]
        rec_ds = rec_vals[indices]
    else:
        prec_ds = prec_vals
        rec_ds = rec_vals

    pr_data = {
        "precision": [round(float(x), 6) for x in prec_ds],
        "recall": [round(float(x), 6) for x in rec_ds],
        "auc": round(pr_auc_val, 6),
        "model_name": "IndustryCNN",
    }

    pr_path = os.path.join(ARTIFACTS_DIR, "pr_curve_data.json")
    with open(pr_path, "w") as f:
        json.dump(pr_data, f, indent=2)
    print(f"Saved Precision-Recall curve data -> {pr_path}")

    print("\nDone! All JSON artifacts generated.")


if __name__ == "__main__":
    main()
