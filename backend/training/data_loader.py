"""
Data loading module for the ECG Arrhythmia Detection pipeline.

Provides:
    - Patient-wise stratified train / validation / test splitting
    - Lazy-loading PyTorch Dataset (loads records on demand via wfdb)
    - DataLoader construction with pin_memory and num_workers
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from training.config import DATA_CFG, PTBXL_DIR, TRAIN_CFG
from training.preprocessing import load_and_preprocess_record

logger = logging.getLogger("ecg_pipeline.data_loader")


# ──────────────────────────────────────────────────────────────
# Patient-wise stratified split
# ──────────────────────────────────────────────────────────────
def patient_wise_split(
    manifest: pd.DataFrame,
    train_ratio: float = DATA_CFG.train_ratio,
    val_ratio: float = DATA_CFG.val_ratio,
    test_ratio: float = DATA_CFG.test_ratio,
    seed: int = DATA_CFG.seed,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split manifest by patient_id ensuring no patient leakage.

    Stratification is done at the patient level using the majority
    label for each patient.

    Parameters
    ----------
    manifest : pd.DataFrame
        Must have columns [patient_id, filename_lr, label].

    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # Determine majority label per patient for stratification
    patient_labels = (
        manifest.groupby("patient_id")["label"]
        .apply(lambda x: int(x.mode().iloc[0]))
        .reset_index()
    )
    patient_labels.columns = ["patient_id", "majority_label"]

    patient_ids = patient_labels["patient_id"].values
    majority_labels = patient_labels["majority_label"].values

    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_pids, tmp_pids, _, tmp_labels = train_test_split(
        patient_ids,
        majority_labels,
        test_size=val_test_ratio,
        stratify=majority_labels,
        random_state=seed,
    )

    # Second split: val vs test
    relative_test = test_ratio / val_test_ratio
    # Get majority labels for tmp patients
    tmp_patient_labels = patient_labels[
        patient_labels["patient_id"].isin(tmp_pids)
    ]["majority_label"].values
    val_pids, test_pids, _, _ = train_test_split(
        tmp_pids,
        tmp_patient_labels,
        test_size=relative_test,
        stratify=tmp_patient_labels,
        random_state=seed,
    )

    train_set = set(train_pids)
    val_set = set(val_pids)
    test_set = set(test_pids)

    # Sanity: no overlap
    assert len(train_set & val_set) == 0
    assert len(train_set & test_set) == 0
    assert len(val_set & test_set) == 0

    train_df = manifest[manifest["patient_id"].isin(train_set)].copy()
    val_df = manifest[manifest["patient_id"].isin(val_set)].copy()
    test_df = manifest[manifest["patient_id"].isin(test_set)].copy()

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        pos = df.label.sum()
        neg = len(df) - pos
        n_patients = df.patient_id.nunique()
        logger.info(
            "  %s → %d records, %d patients, NORM=%d, ABNORM=%d (pos_rate=%.2f%%)",
            name, len(df), n_patients, neg, pos, 100.0 * pos / len(df),
        )

    return train_df, val_df, test_df


# ──────────────────────────────────────────────────────────────
# Lazy-loading PyTorch Dataset
# ──────────────────────────────────────────────────────────────
class ECGDataset(Dataset):
    """Lazy-loading dataset that reads records on demand via wfdb.

    Each __getitem__ call loads the record from disk, applies
    preprocessing (per-lead z-score normalisation), and returns
    a channel-first tensor of shape (12, 1000).
    """

    def __init__(
        self,
        filenames: List[str],
        labels: np.ndarray,
        dataset_dir: str = PTBXL_DIR,
    ) -> None:
        self.filenames = filenames
        self.labels = labels.astype(np.float32)
        self.dataset_dir = dataset_dir

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        filename = self.filenames[idx]
        label = self.labels[idx]

        # Load and preprocess: returns (1000, 12)
        signal = load_and_preprocess_record(filename, self.dataset_dir)

        # Convert to channel-first: (12, 1000)
        tensor = torch.from_numpy(signal.T)  # (12, 1000)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return tensor, label_tensor


# ──────────────────────────────────────────────────────────────
# DataLoader construction
# ──────────────────────────────────────────────────────────────
def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_dir: str = PTBXL_DIR,
    batch_size: int = TRAIN_CFG.batch_size,
    num_workers: int = TRAIN_CFG.num_workers,
    pin_memory: bool = TRAIN_CFG.pin_memory,
) -> Dict[str, DataLoader]:
    """Build PyTorch DataLoaders for train / val / test.

    Returns dict with keys "train", "val", "test".
    """
    loaders: Dict[str, DataLoader] = {}

    for split, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        filenames = df["filename_lr"].tolist()
        labels = df["label"].values

        dataset = ECGDataset(filenames, labels, dataset_dir)
        shuffle = split == "train"

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=num_workers > 0,
        )
        logger.info(
            "DataLoader[%s]: %d samples, %d batches",
            split, len(dataset), len(loaders[split]),
        )

    return loaders
