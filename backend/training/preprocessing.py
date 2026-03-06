"""
Preprocessing module for the ECG Arrhythmia Detection pipeline.

Responsibilities:
    1. Load PTB-XL metadata and build record manifest.
    2. Provide lazy per-record signal loading (all 12 leads, 100 Hz).
    3. Apply per-lead z-score normalisation.
    4. Convert multi-label diagnoses to binary (NORM vs ABNORM).
"""

import ast
import logging
import os
from typing import Tuple

import numpy as np
import pandas as pd
import wfdb

from training.config import DATA_CFG, PTBXL_DIR

logger = logging.getLogger("ecg_pipeline.preprocessing")


# ──────────────────────────────────────────────────────────────
# 1. Metadata & Labels
# ──────────────────────────────────────────────────────────────
def load_metadata(dataset_dir: str = PTBXL_DIR) -> pd.DataFrame:
    """Load PTB-XL metadata CSV, parse SCP codes, keep useful columns."""
    csv_path = os.path.join(dataset_dir, "ptbxl_database.csv")
    df = pd.read_csv(csv_path, index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)
    return df


def load_scp_statements(dataset_dir: str = PTBXL_DIR) -> pd.DataFrame:
    """Load the SCP statement lookup table (diagnostic rows only)."""
    path = os.path.join(dataset_dir, "scp_statements.csv")
    scp = pd.read_csv(path, index_col=0)
    scp = scp[scp.diagnostic == 1]
    return scp


def _aggregate_diagnostic(scp_codes: dict, scp_df: pd.DataFrame) -> str:
    """Return 'NORM' if the only diagnostic superclass is NORM, else 'ABNORM'."""
    superclasses = set()
    for code, likelihood in scp_codes.items():
        if likelihood >= 50 and code in scp_df.index:
            superclasses.add(scp_df.loc[code].diagnostic_class)
    if superclasses == {"NORM"}:
        return "NORM"
    return "ABNORM"


def build_manifest(dataset_dir: str = PTBXL_DIR) -> pd.DataFrame:
    """Build a manifest DataFrame with columns: patient_id, filename_lr, label.

    Uses ``filename_lr`` (100 Hz records).
    label: 0 = NORM, 1 = ABNORM.
    Records with ambiguous labels (no diagnostic superclass ≥ 50) are dropped.

    Returns
    -------
    pd.DataFrame indexed by ecg_id with columns [patient_id, filename_lr, label].
    """
    df = load_metadata(dataset_dir)
    scp_df = load_scp_statements(dataset_dir)

    df["diag"] = df.scp_codes.apply(lambda x: _aggregate_diagnostic(x, scp_df))
    df["label"] = (df["diag"] != "NORM").astype(np.int64)

    # Keep only records that have at least one diagnostic superclass
    def _has_diag(scp_codes: dict) -> bool:
        for code, lh in scp_codes.items():
            if lh >= 50 and code in scp_df.index:
                return True
        return False

    mask = df.scp_codes.apply(_has_diag)
    df = df[mask].copy()

    manifest = df[["patient_id", "filename_lr", "label"]].copy()
    logger.info(
        "Manifest built: %d records, NORM=%d, ABNORM=%d",
        len(manifest),
        (manifest.label == 0).sum(),
        (manifest.label == 1).sum(),
    )
    return manifest


# ──────────────────────────────────────────────────────────────
# 2. Signal loading (lazy, 12-lead)
# ──────────────────────────────────────────────────────────────
def load_record_12lead(
    filename_lr: str,
    dataset_dir: str = PTBXL_DIR,
    target_length: int = DATA_CFG.signal_length,
) -> np.ndarray:
    """Load a single 12-lead ECG record from wfdb.

    Parameters
    ----------
    filename_lr : str
        Relative path from dataset_dir (e.g. ``records100/00000/00001_lr``).
    dataset_dir : str
        Root directory of PTB-XL dataset.
    target_length : int
        Expected number of samples (1000 for 100 Hz × 10 s).

    Returns
    -------
    np.ndarray of shape (target_length, 12), dtype float32.
    """
    rec_path = os.path.join(dataset_dir, filename_lr)
    record = wfdb.rdrecord(rec_path)
    signal = record.p_signal  # (samples, 12)

    # Truncate or pad if needed (should be 1000 for 100 Hz records)
    if signal.shape[0] > target_length:
        signal = signal[:target_length, :]
    elif signal.shape[0] < target_length:
        pad = np.zeros((target_length - signal.shape[0], signal.shape[1]), dtype=signal.dtype)
        signal = np.concatenate([signal, pad], axis=0)

    return signal.astype(np.float32)


def preprocess_signal(signal: np.ndarray) -> np.ndarray:
    """Apply per-lead z-score normalisation.

    Parameters
    ----------
    signal : np.ndarray of shape (1000, 12)

    Returns
    -------
    np.ndarray of shape (1000, 12), dtype float32.
    """
    mu = signal.mean(axis=0, keepdims=True)
    sigma = signal.std(axis=0, keepdims=True)
    sigma = np.where(sigma > 0, sigma, 1.0)
    normalised = (signal - mu) / sigma
    return normalised.astype(np.float32)


def load_and_preprocess_record(
    filename_lr: str,
    dataset_dir: str = PTBXL_DIR,
) -> np.ndarray:
    """Load a single record and return preprocessed 12-lead signal.

    Returns
    -------
    np.ndarray of shape (1000, 12), dtype float32.
    """
    signal = load_record_12lead(filename_lr, dataset_dir)
    return preprocess_signal(signal)
