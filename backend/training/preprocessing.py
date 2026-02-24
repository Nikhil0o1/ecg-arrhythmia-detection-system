"""
Preprocessing module for the ECG Arrhythmia Detection pipeline.

Responsibilities:
    1. Download PTB-XL dataset from PhysioNet (streaming ZIP download).
    2. Load raw 12-lead ECG records; extract Lead I.
    3. Resample to 100 Hz (if needed).
    4. Apply per-signal z-score normalisation.
    5. Convert multi-label diagnoses to binary (NORM vs non-NORM).
    6. Persist processed arrays to disk.
"""

import ast
import logging
import os
import zipfile
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests
import wfdb
from scipy.signal import resample
from tqdm import tqdm

from training.config import DATA_CFG, PROCESSED_DATA_DIR, RAW_DATA_DIR

logger = logging.getLogger("ecg_pipeline.preprocessing")

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
_PTBXL_FOLDER_NAME = (
    "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)
_PTBXL_ZIP_URL = (
    "https://physionet.org/static/published-projects/ptb-xl/"
    "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
)
_DOWNLOAD_CHUNK_SIZE = 1 * 1024 * 1024  # 1 MB


# ──────────────────────────────────────────────────────────────
# 1. Download
# ──────────────────────────────────────────────────────────────
def download_ptbxl(data_dir: str = RAW_DATA_DIR) -> str:
    """Download the PTB-XL dataset from PhysioNet if not already present.

    The dataset is downloaded as a ZIP archive using a streaming request,
    extracted safely (with path-traversal protection), and the ZIP is
    deleted afterwards.

    Parameters
    ----------
    data_dir : str
        Parent directory where the dataset folder will reside.
        Defaults to ``RAW_DATA_DIR`` from config.

    Returns
    -------
    str
        Absolute path to the extracted dataset folder.

    Raises
    ------
    RuntimeError
        If the download fails (non-200 status) or extraction fails.
    """
    dataset_dir = os.path.join(data_dir, _PTBXL_FOLDER_NAME)

    # ── Already downloaded ────────────────────────────────────
    if os.path.isdir(dataset_dir):
        logger.info(
            "PTB-XL already present at %s – skipping download.", dataset_dir
        )
        return os.path.abspath(dataset_dir)

    os.makedirs(data_dir, exist_ok=True)

    zip_filename = _PTBXL_FOLDER_NAME + ".zip"
    zip_path = os.path.join(data_dir, zip_filename)

    # ── Streaming download ────────────────────────────────────
    logger.info("Download started – fetching PTB-XL from %s", _PTBXL_ZIP_URL)
    try:
        response = requests.get(_PTBXL_ZIP_URL, stream=True, timeout=600)
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Failed to connect to PhysioNet: {exc}"
        ) from exc

    if response.status_code != 200:
        raise RuntimeError(
            f"Download failed: HTTP {response.status_code} from {_PTBXL_ZIP_URL}"
        )

    total_size = int(response.headers.get("content-length", 0))
    try:
        with open(zip_path, "wb") as fh, tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc="Downloading PTB-XL",
        ) as pbar:
            for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_SIZE):
                if chunk:
                    fh.write(chunk)
                    pbar.update(len(chunk))
    except Exception as exc:
        # Clean up partial download
        if os.path.isfile(zip_path):
            os.remove(zip_path)
        raise RuntimeError(
            f"Download interrupted: {exc}"
        ) from exc

    logger.info("Download completed – saved to %s", zip_path)

    # ── Safe extraction ───────────────────────────────────────
    logger.info("Extraction started – unpacking %s", zip_path)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Path-traversal protection
            for member in zf.infolist():
                member_path = os.path.normpath(member.filename)
                abs_target = os.path.normpath(
                    os.path.join(data_dir, member_path)
                )
                if not abs_target.startswith(os.path.normpath(data_dir)):
                    raise RuntimeError(
                        f"Path traversal detected in ZIP member: {member.filename}"
                    )
            zf.extractall(path=data_dir)
    except zipfile.BadZipFile as exc:
        raise RuntimeError(
            f"Extraction failed – corrupt ZIP file: {exc}"
        ) from exc
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Extraction failed: {exc}"
        ) from exc
    finally:
        # Delete ZIP regardless of extraction outcome
        if os.path.isfile(zip_path):
            os.remove(zip_path)
            logger.info("Deleted ZIP archive %s", zip_path)

    logger.info("Extraction completed – dataset at %s", dataset_dir)

    if not os.path.isdir(dataset_dir):
        raise RuntimeError(
            f"Expected dataset folder not found after extraction: {dataset_dir}"
        )

    logger.info("Dataset ready at %s", dataset_dir)
    return os.path.abspath(dataset_dir)


# ──────────────────────────────────────────────────────────────
# 1b. Record discovery (records100 only)
# ──────────────────────────────────────────────────────────────
def get_record_list(dataset_dir: str) -> List[str]:
    """Build a list of record IDs from the ``records100`` sub-folder.

    Only ``records100`` is used (100 Hz resolution).  Each record ID is
    returned *without* file extensions and is expressed as a path
    relative to ``dataset_dir`` (e.g. ``records100/00000/00001_lr``).

    Parameters
    ----------
    dataset_dir : str
        Root directory of the extracted PTB-XL dataset.

    Returns
    -------
    List[str]
        Sorted list of record IDs (without extensions).

    Raises
    ------
    FileNotFoundError
        If the ``records100`` folder is missing.
    """
    records100_dir = os.path.join(dataset_dir, "records100")
    if not os.path.isdir(records100_dir):
        raise FileNotFoundError(
            f"records100 folder not found at {records100_dir}"
        )

    record_ids: List[str] = []
    for sub_dir in sorted(os.listdir(records100_dir)):
        sub_path = os.path.join(records100_dir, sub_dir)
        if not os.path.isdir(sub_path):
            continue
        seen_basenames: set = set()
        for fname in sorted(os.listdir(sub_path)):
            basename, _ = os.path.splitext(fname)
            if basename in seen_basenames:
                continue
            seen_basenames.add(basename)
            # Relative record ID: records100/<sub>/<basename>
            record_id = os.path.join("records100", sub_dir, basename)
            record_ids.append(record_id)

    logger.info("Found %d records in records100.", len(record_ids))
    return record_ids


# ──────────────────────────────────────────────────────────────
# 2. Load metadata & labels
# ──────────────────────────────────────────────────────────────
def _load_metadata(dataset_dir: str) -> pd.DataFrame:
    """Load the PTB-XL metadata CSV and parse the SCP codes column."""
    csv_path = os.path.join(dataset_dir, "ptbxl_database.csv")
    df = pd.read_csv(csv_path, index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)
    return df


def _load_scp_statements(dataset_dir: str) -> pd.DataFrame:
    """Load the SCP statement lookup table."""
    path = os.path.join(dataset_dir, "scp_statements.csv")
    scp = pd.read_csv(path, index_col=0)
    scp = scp[scp.diagnostic == 1]
    return scp


def _aggregate_diagnostic(scp_codes: dict, scp_df: pd.DataFrame) -> str:
    """
    Map a record's SCP codes to its diagnostic superclass.

    Returns 'NORM' if the only superclass is NORM, else 'ABNORM'.
    """
    superclasses = set()
    for code, likelihood in scp_codes.items():
        if likelihood >= 50 and code in scp_df.index:
            superclasses.add(scp_df.loc[code].diagnostic_class)
    if superclasses == {"NORM"}:
        return "NORM"
    return "ABNORM"


def _binarise_labels(df: pd.DataFrame, scp_df: pd.DataFrame) -> np.ndarray:
    """
    Create binary labels: 0 = NORM, 1 = any non-NORM.
    """
    df = df.copy()
    df["diag"] = df.scp_codes.apply(lambda x: _aggregate_diagnostic(x, scp_df))
    labels = (df["diag"] != "NORM").astype(np.int64).values
    return labels


# ──────────────────────────────────────────────────────────────
# 3. Signal loading & processing
# ──────────────────────────────────────────────────────────────
def _load_signal(record_path: str, target_length: int, lead_idx: int) -> np.ndarray:
    """
    Load a single ECG record, extract one lead, resample if necessary,
    and apply z-score normalisation.

    Parameters
    ----------
    record_path : str
        Path to the wfdb record (without extension).
    target_length : int
        Desired number of samples (e.g. 1000 for 100 Hz × 10 s).
    lead_idx : int
        Index of the lead to extract.

    Returns
    -------
    np.ndarray of shape (target_length, 1)
    """
    record = wfdb.rdrecord(record_path)
    signal = record.p_signal[:, lead_idx]  # single lead

    # Resample if length differs from target
    if len(signal) != target_length:
        signal = resample(signal, target_length)

    # Z-score normalisation
    mu = np.mean(signal)
    sigma = np.std(signal)
    if sigma > 0:
        signal = (signal - mu) / sigma
    else:
        signal = signal - mu

    return signal.reshape(-1, 1).astype(np.float32)


def load_and_preprocess(
    dataset_dir: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all records, preprocess signals, and create binary labels.

    Returns
    -------
    X : np.ndarray, shape (N, 1000, 1)
    y : np.ndarray, shape (N,)
    """
    logger.info("Loading metadata …")
    df = _load_metadata(dataset_dir)
    scp_df = _load_scp_statements(dataset_dir)

    logger.info("Binarising labels …")
    y = _binarise_labels(df, scp_df)

    # Choose the 100 Hz file column
    filename_col = "filename_lr"  # 100 Hz records

    logger.info("Loading & preprocessing %d ECG records (Lead I, 100 Hz) …", len(df))
    signals = []
    valid_indices = []

    for idx, (ecg_id, row) in enumerate(df.iterrows()):
        rec_path = os.path.join(dataset_dir, row[filename_col])
        try:
            sig = _load_signal(
                rec_path,
                target_length=DATA_CFG.signal_length,
                lead_idx=DATA_CFG.lead_index,
            )
            signals.append(sig)
            valid_indices.append(idx)
        except Exception as exc:
            logger.warning("Skipping ecg_id=%s: %s", ecg_id, exc)

    X = np.stack(signals, axis=0)
    y = y[valid_indices]

    logger.info("Preprocessing complete: X=%s  y=%s  pos_rate=%.2f%%",
                X.shape, y.shape, 100 * y.mean())
    return X, y


# ──────────────────────────────────────────────────────────────
# 4. Persist / load processed data
# ──────────────────────────────────────────────────────────────
def save_processed(X: np.ndarray, y: np.ndarray, out_dir: str = PROCESSED_DATA_DIR) -> None:
    """Save processed arrays to disk."""
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X.npy"), X)
    np.save(os.path.join(out_dir, "y.npy"), y)
    logger.info("Saved processed data to %s", out_dir)


def load_processed(data_dir: str = PROCESSED_DATA_DIR) -> Tuple[np.ndarray, np.ndarray]:
    """Load previously saved processed arrays."""
    X = np.load(os.path.join(data_dir, "X.npy"))
    y = np.load(os.path.join(data_dir, "y.npy"))
    logger.info("Loaded processed data: X=%s  y=%s", X.shape, y.shape)
    return X, y


def processed_data_exists(data_dir: str = PROCESSED_DATA_DIR) -> bool:
    """Check whether processed data already exists on disk."""
    return (
        os.path.isfile(os.path.join(data_dir, "X.npy"))
        and os.path.isfile(os.path.join(data_dir, "y.npy"))
    )
