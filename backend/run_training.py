#!/usr/bin/env python
"""
run_training.py – Single entry point for the ECG Arrhythmia Detection pipeline.

Orchestrates:
    1. Dataset download (PTB-XL from PhysioNet)
    2. Preprocessing & caching
    3. Stratified splitting
    4. Training of all three models (CNN1D, LSTMClassifier, TransformerClassifier)
    5. Evaluation on the held-out test set
    6. Saving comparison metrics

Usage:
    cd backend/
    python run_training.py
"""

import logging
import os
import sys
import time

# Ensure the backend directory is on sys.path so `training` is importable
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from training.config import (
    DATA_CFG,
    MODEL_NAMES,
    TRAIN_CFG,
)
from training.data_loader import build_dataloaders, stratified_split
from training.evaluate import evaluate_model, save_comparison_report
from training.models import build_model
from training.preprocessing import (
    download_ptbxl,
    load_and_preprocess,
    load_processed,
    processed_data_exists,
    save_processed,
)
from training.train import train_model
from training.utils import (
    compute_class_weights,
    ensure_directories,
    resolve_device,
    set_seed,
    setup_logging,
)


def main() -> None:
    """Run the full pipeline end-to-end."""

    # ── 0. Setup ──────────────────────────────────────────────
    ensure_directories()
    logger = setup_logging()
    set_seed(TRAIN_CFG.seed)
    device = resolve_device(TRAIN_CFG.device)

    logger.info("=" * 70)
    logger.info("ECG Arrhythmia Detection Pipeline – START")
    logger.info("=" * 70)
    logger.info("Device: %s", device)
    pipeline_start = time.time()

    # ── 1. Download dataset ───────────────────────────────────
    logger.info("─" * 70)
    logger.info("STEP 1 / 5 : Download PTB-XL dataset")
    logger.info("─" * 70)
    dataset_dir = download_ptbxl()

    # ── 2. Preprocess (or load cached) ────────────────────────
    logger.info("─" * 70)
    logger.info("STEP 2 / 5 : Preprocess ECG signals")
    logger.info("─" * 70)
    if processed_data_exists():
        logger.info("Found cached processed data – loading from disk.")
        X, y = load_processed()
    else:
        X, y = load_and_preprocess(dataset_dir)
        save_processed(X, y)

    # ── 3. Split & build loaders ──────────────────────────────
    logger.info("─" * 70)
    logger.info("STEP 3 / 5 : Stratified split & DataLoader construction")
    logger.info("─" * 70)
    X_train, y_train, X_val, y_val, X_test, y_test = stratified_split(X, y)
    loaders = build_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test)

    pos_weight = compute_class_weights(y_train)
    logger.info("Positive class weight (pos_weight): %.4f", pos_weight.item())

    # ── 4. Train all models ───────────────────────────────────
    logger.info("─" * 70)
    logger.info("STEP 4 / 5 : Train models")
    logger.info("─" * 70)

    trained_models = {}
    histories = {}

    for model_name in MODEL_NAMES:
        logger.info("\n>>> Building model: %s", model_name)
        model = build_model(model_name)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info("  Parameters: %s", f"{n_params:,}")

        history = train_model(
            model=model,
            loaders=loaders,
            pos_weight=pos_weight,
            device=device,
            model_name=model_name,
        )
        trained_models[model_name] = model
        histories[model_name] = history

    # ── 5. Evaluate all models ────────────────────────────────
    logger.info("─" * 70)
    logger.info("STEP 5 / 5 : Evaluate on test set")
    logger.info("─" * 70)

    all_metrics = {}
    for model_name, model in trained_models.items():
        logger.info("\n>>> Evaluating: %s", model_name)
        metrics = evaluate_model(
            model=model,
            loader=loaders["test"],
            device=device,
            model_name=model_name,
        )
        all_metrics[model_name] = metrics

    # ── 6. Save comparison report ─────────────────────────────
    save_comparison_report(all_metrics)

    # ── Summary ───────────────────────────────────────────────
    elapsed = time.time() - pipeline_start
    logger.info("=" * 70)
    logger.info("Pipeline complete in %.1f s", elapsed)
    logger.info("=" * 70)

    # Print comparison table
    header = f"{'Model':<25} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}"
    logger.info(header)
    logger.info("-" * len(header))
    for name, m in all_metrics.items():
        logger.info(
            f"{name:<25} {m['accuracy']:>7.4f} {m['precision']:>7.4f} "
            f"{m['recall']:>7.4f} {m['f1_score']:>7.4f} {m['roc_auc']:>7.4f}"
        )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
