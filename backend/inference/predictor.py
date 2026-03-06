"""
ECG Arrhythmia Predictor — production inference module.

Loads the IndustryCNN checkpoint, applies the same preprocessing used
during training (12-lead, per-lead z-score normalisation), and returns
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
from training.preprocessing import preprocess_signal

logger = logging.getLogger("ecg_pipeline.inference")

# Expected signal parameters
_SIGNAL_LENGTH = MODEL_CFG.input_length  # 1000
_N_LEADS = MODEL_CFG.input_channels  # 12


class ECGPredictor:
    """Lazy-loaded, thread-safe ECG arrhythmia predictor.

    Parameters
    ----------
    model_name : str
        Registered model name (default: ``IndustryCNN``).
    device : str | None
        ``"cuda"``, ``"cpu"``, or ``None`` for auto-detection.
    checkpoint_dir : str
        Directory containing ``<model_name>_best.pt`` checkpoint files.
    """

    def __init__(
        self,
        model_name: str = "IndustryCNN",
        device: str | None = None,
        checkpoint_dir: str = SAVED_MODELS_DIR,
    ) -> None:
        self._model_name = model_name
        self._checkpoint_dir = checkpoint_dir
        self._lock = threading.Lock()

        if device is None:
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self._device = torch.device(device)

        self._model = self._load_model()
        logger.info(
            "Model loaded: %s on %s", self._model_name, self._device
        )

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
        """Apply the same preprocessing used during training.

        Accepts:
            - (1000, 12) — time-first, 12 leads
            - (12, 1000) — channel-first, 12 leads

        Returns
        -------
        np.ndarray of shape ``(12, 1000)``, dtype float32 (channel-first).
        """
        signal = np.asarray(signal, dtype=np.float32)

        # Handle shape
        if signal.shape == (_N_LEADS, _SIGNAL_LENGTH):
            # Already channel-first -> convert to time-first for preprocessing
            signal = signal.T  # (1000, 12)
        elif signal.shape == (_SIGNAL_LENGTH, _N_LEADS):
            pass  # Already (1000, 12)
        else:
            raise ValueError(
                f"Expected signal shape ({_SIGNAL_LENGTH}, {_N_LEADS}) or "
                f"({_N_LEADS}, {_SIGNAL_LENGTH}), got {signal.shape}"
            )

        # Per-lead z-score normalisation
        normalised = preprocess_signal(signal)  # (1000, 12)

        # Return channel-first: (12, 1000)
        return normalised.T.astype(np.float32)

    @torch.no_grad()
    def predict(self, signal: np.ndarray) -> Dict[str, Any]:
        """Run inference on a single 12-lead ECG signal.

        Parameters
        ----------
        signal : np.ndarray
            Shape ``(1000, 12)`` or ``(12, 1000)`` — 12-lead ECG (10 s @ 100 Hz).

        Returns
        -------
        dict
            ``probability`` — sigmoid probability of ABNORM class.
            ``prediction``  — 0 (NORM) or 1 (ABNORM).
            ``confidence``  — confidence in the predicted class (always >= 0.5).
        """
        processed = self._preprocess(signal)  # (12, 1000)

        tensor = torch.from_numpy(processed).unsqueeze(0).to(self._device)  # (1, 12, 1000)

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
