"""
ECG Arrhythmia Predictor — production inference module.

Loads a trained model checkpoint, applies the same preprocessing used
during training (Lead I extraction, z-score normalisation), and returns
calibrated predictions.

Thread-safe: the model is loaded once and kept in eval() mode.
All inference runs under ``torch.no_grad()``.
"""

import logging
import os
import threading
from typing import Any, Dict

import numpy as np
import torch

from training.config import MODEL_CFG, SAVED_MODELS_DIR
from training.models import build_model

logger = logging.getLogger("ecg_pipeline.inference")

# Expected signal length (10 s @ 100 Hz)
_SIGNAL_LENGTH = MODEL_CFG.input_length  # 1000


class ECGPredictor:
    """Lazy-loaded, thread-safe ECG arrhythmia predictor.

    Parameters
    ----------
    model_name : str
        Registered model name (``CNN1D``, ``LSTMClassifier``,
        ``TransformerClassifier``).
    device : str | None
        ``"cuda"``, ``"cpu"``, or ``None`` for auto-detection.
    checkpoint_dir : str
        Directory containing ``<model_name>_best.pt`` checkpoint files.
    """

    def __init__(
        self,
        model_name: str = "CNN1D",
        device: str | None = None,
        checkpoint_dir: str = SAVED_MODELS_DIR,
    ) -> None:
        self._model_name = model_name
        self._checkpoint_dir = checkpoint_dir
        self._lock = threading.Lock()

        # ── Device ────────────────────────────────────────────
        if device is None:
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self._device = torch.device(device)

        # ── Model (loaded once) ───────────────────────────────
        self._model = self._load_model()
        logger.info(
            "Model loaded: %s on %s", self._model_name, self._device
        )

    # ──────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────
    def _load_model(self) -> torch.nn.Module:
        """Instantiate the architecture and load saved weights."""
        checkpoint_path = os.path.join(
            self._checkpoint_dir, f"{self._model_name}_best.pt"
        )
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        model = build_model(self._model_name)
        state_dict = torch.load(
            checkpoint_path,
            map_location=self._device,
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        model.to(self._device)
        model.eval()
        return model

    @staticmethod
    def _preprocess(signal: np.ndarray) -> np.ndarray:
        """Apply the same z-score normalisation used during training.

        Parameters
        ----------
        signal : np.ndarray, shape ``(1000,)`` or ``(1000, 1)``

        Returns
        -------
        np.ndarray of shape ``(1000, 1)``, dtype float32
        """
        signal = np.asarray(signal, dtype=np.float64).squeeze()
        if signal.shape != (_SIGNAL_LENGTH,):
            raise ValueError(
                f"Expected signal of length {_SIGNAL_LENGTH}, "
                f"got shape {signal.shape}"
            )

        mu = signal.mean()
        sigma = signal.std()
        if sigma > 0:
            signal = (signal - mu) / sigma
        else:
            signal = signal - mu

        return signal.reshape(-1, 1).astype(np.float32)

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict(self, signal: np.ndarray) -> Dict[str, Any]:
        """Run inference on a single ECG signal.

        Parameters
        ----------
        signal : np.ndarray, shape ``(1000,)``
            Raw Lead-I samples (10 s @ 100 Hz).

        Returns
        -------
        dict
            ``probability`` – sigmoid probability of non-NORM class.
            ``prediction``  – 0 (NORM) or 1 (non-NORM).
            ``confidence``  – how confident the model is in the predicted
            class (always ≥ 0.5).
        """
        processed = self._preprocess(signal)

        tensor = torch.from_numpy(processed).unsqueeze(0).to(self._device)

        with self._lock:
            logit = self._model(tensor).squeeze()

        probability = torch.sigmoid(logit).item()
        prediction = int(probability >= 0.5)
        confidence = probability if prediction == 1 else 1.0 - probability

        return {
            "probability": round(probability, 6),
            "prediction": prediction,
            "confidence": round(confidence, 6),
        }

    @property
    def device_name(self) -> str:
        """Return device string for health-check endpoints."""
        return str(self._device)
