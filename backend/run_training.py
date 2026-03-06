#!/usr/bin/env python
"""
run_training.py — Single entry point for the ECG Arrhythmia Detection pipeline.

Orchestrates:
    1. Verify dataset exists
    2. Build manifest & patient-wise splits
    3. Train IndustryCNN (residual 1-D CNN, 12-lead)
    4. Evaluate on test set
    5. Save metrics & plots
    6. Generate demo samples
    7. Exit cleanly

Usage:
    cd backend/
    python run_training.py
"""

import logging
import os
import sys
import time

import numpy as np

# Ensure the backend directory is on sys.path so `training` is importable
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from training.config import DEMO_DIR, PTBXL_DIR, TRAIN_CFG
from training.data_loader import build_dataloaders, patient_wise_split
from training.evaluate import evaluate_model
from training.models import build_model
from training.preprocessing import build_manifest, load_and_preprocess_record
from training.train import train_model
from training.utils import (
    compute_pos_weight,
    ensure_directories,
    resolve_device,
    set_seed,
    setup_logging,
)


def _verify_dataset(dataset_dir: str) -> None:
    """Verify the PTB-XL dataset exists at the expected path."""
    csv_path = os.path.join(dataset_dir, "ptbxl_database.csv")
    records_dir = os.path.join(dataset_dir, "records100")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"ptbxl_database.csv not found at {csv_path}. "
            f"Ensure PTB-XL is downloaded to {dataset_dir}"
        )
    if not os.path.isdir(records_dir):
        raise FileNotFoundError(
            f"records100/ folder not found at {records_dir}. "
            f"Ensure PTB-XL is fully extracted."
        )


def _generate_demo_samples(
    model,
    test_df,
    device,
    dataset_dir: str,
    demo_dir: str,
    logger: logging.Logger,
) -> None:
    """Find high-confidence positive and negative samples from the test set."""
    import torch

    model.eval()
    os.makedirs(demo_dir, exist_ok=True)

    best_pos_prob = 0.0
    best_pos_signal = None
    best_neg_prob = 1.0
    best_neg_signal = None

    with torch.no_grad():
        for _, row in test_df.iterrows():
            try:
                signal = load_and_preprocess_record(row["filename_lr"], dataset_dir)
            except Exception:
                continue

            # signal: (1000, 12)
            tensor = torch.from_numpy(signal.T).unsqueeze(0).to(device)  # (1, 12, 1000)
            logit = model(tensor).squeeze()
            prob = torch.sigmoid(logit).item()

            if prob > 0.9 and prob > best_pos_prob:
                best_pos_prob = prob
                best_pos_signal = signal.copy()  # (1000, 12)

            if prob < 0.1 and prob < best_neg_prob:
                best_neg_prob = prob
                best_neg_signal = signal.copy()

            # Early exit if we found good examples
            if best_pos_signal is not None and best_neg_signal is not None:
                if best_pos_prob > 0.95 and best_neg_prob < 0.05:
                    break

    if best_pos_signal is not None:
        path = os.path.join(demo_dir, "demo_positive.npy")
        np.save(path, best_pos_signal)
        logger.info("Saved demo positive (prob=%.4f) -> %s", best_pos_prob, path)
    else:
        logger.warning("Could not find a high-confidence positive sample (prob > 0.9)")

    if best_neg_signal is not None:
        path = os.path.join(demo_dir, "demo_negative.npy")
        np.save(path, best_neg_signal)
        logger.info("Saved demo negative (prob=%.4f) -> %s", best_neg_prob, path)
    else:
        logger.warning("Could not find a high-confidence negative sample (prob < 0.1)")


def main() -> None:
    """Run the full pipeline end-to-end."""

    # ── 0. Setup ──────────────────────────────────────────────
    ensure_directories()
    logger = setup_logging()
    set_seed(TRAIN_CFG.seed)
    device = resolve_device(TRAIN_CFG.device)

    logger.info("=" * 70)
    logger.info("ECG Arrhythmia Detection Pipeline — START")
    logger.info("=" * 70)
    logger.info("Device: %s", device)
    pipeline_start = time.time()

    # ── 1. Verify dataset ─────────────────────────────────────
    logger.info("-" * 70)
    logger.info("STEP 1 / 6 : Verify PTB-XL dataset")
    logger.info("-" * 70)
    _verify_dataset(PTBXL_DIR)
    logger.info("Dataset verified at %s", PTBXL_DIR)

    # ── 2. Build manifest & patient-wise split ────────────────
    logger.info("-" * 70)
    logger.info("STEP 2 / 6 : Build manifest & patient-wise split")
    logger.info("-" * 70)
    manifest = build_manifest(PTBXL_DIR)
    train_df, val_df, test_df = patient_wise_split(manifest)

    # Build DataLoaders (lazy loading)
    loaders = build_dataloaders(train_df, val_df, test_df, dataset_dir=PTBXL_DIR)

    # Compute pos_weight from TRAIN split only
    train_labels = train_df["label"].values
    pos_weight = compute_pos_weight(train_labels)
    logger.info("pos_weight (from train): %.4f", pos_weight.item())

    # ── 3. Build & train model ────────────────────────────────
    logger.info("-" * 70)
    logger.info("STEP 3 / 6 : Train IndustryCNN")
    logger.info("-" * 70)
    model = build_model("IndustryCNN")
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %s", f"{n_params:,}")

    # run training, history not needed here
    train_model(
        model=model,
        loaders=loaders,
        pos_weight=pos_weight,
        device=device,
        model_name="IndustryCNN",
    )

    # ── 4. Evaluate on test set ───────────────────────────────
    logger.info("-" * 70)
    logger.info("STEP 4 / 6 : Evaluate on test set")
    logger.info("-" * 70)
    metrics, y_true, y_prob = evaluate_model(
        model=model,
        loader=loaders["test"],
        device=device,
    )

    # ── 5. Generate demo samples ──────────────────────────────
    logger.info("-" * 70)
    logger.info("STEP 5 / 6 : Generate demo samples from test set")
    logger.info("-" * 70)
    _generate_demo_samples(
        model=model,
        test_df=test_df,
        device=device,
        dataset_dir=PTBXL_DIR,
        demo_dir=DEMO_DIR,
        logger=logger,
    )

    # ── 6. Summary ────────────────────────────────────────────
    elapsed = time.time() - pipeline_start
    logger.info("=" * 70)
    logger.info("Pipeline complete in %.1f s", elapsed)
    logger.info("=" * 70)

    header = f"{'Metric':<15} {'Value':>10}"
    logger.info(header)
    logger.info("-" * len(header))
    for k, v in metrics.items():
        logger.info(f"{k:<15} {v:>10.4f}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
