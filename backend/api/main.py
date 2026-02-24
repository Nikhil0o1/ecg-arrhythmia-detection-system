"""
FastAPI application for ECG Arrhythmia Detection inference.

Endpoints
---------
GET  /health       – Service health check.
POST /predict      – Predict from a JSON array of 1000 floats.
POST /predict-file – Predict from an uploaded ``.npy`` file.
POST /simulate     – Real-time streaming simulation over a 10-second signal.
"""

import io
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, field_validator

from inference.predictor import ECGPredictor

logger = logging.getLogger("ecg_pipeline.api")

# ──────────────────────────────────────────────────────────────
# Singleton predictor (loaded once at startup)
# ──────────────────────────────────────────────────────────────
_predictor: ECGPredictor | None = None

_SIGNAL_LENGTH = 1000  # 10 s @ 100 Hz
_CHUNK_SIZE = 100      # sliding-window chunk for simulation


def _get_predictor() -> ECGPredictor:
    """Return the global predictor; raise 503 if not yet loaded."""
    if _predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet. Please try again shortly.",
        )
    return _predictor


# ──────────────────────────────────────────────────────────────
# Lifespan – lazy model loading
# ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _predictor
    logger.info("Loading ECG model at startup …")

    try:
        _predictor = ECGPredictor()
        logger.info("Model ready on device=%s", _predictor.device_name)
    except Exception:
        logger.exception("Model failed to load. API will start without model.")
        _predictor = None

    yield

    logger.info("Shutting down ECG inference service.")

# ──────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="ECG Arrhythmia Detection API",
    version="1.0.0",
    lifespan=_lifespan,
)


# ──────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    signal: List[float]

    @field_validator("signal")
    @classmethod
    def validate_signal_length(cls, v: List[float]) -> List[float]:
        if len(v) != _SIGNAL_LENGTH:
            raise ValueError(
                f"Signal must contain exactly {_SIGNAL_LENGTH} samples, "
                f"got {len(v)}."
            )
        return v


class PredictResponse(BaseModel):
    probability: float
    prediction: int
    confidence: float


class HealthResponse(BaseModel):
    status: str
    device: str


class TimelineEntry(BaseModel):
    chunk_index: int
    start_sample: int
    end_sample: int
    probability: float
    prediction: int
    confidence: float


class SimulateResponse(BaseModel):
    timeline_predictions: List[TimelineEntry]
    final_prediction: PredictResponse


class SimulateRequest(BaseModel):
    signal: List[float]

    @field_validator("signal")
    @classmethod
    def validate_signal_length(cls, v: List[float]) -> List[float]:
        if len(v) != _SIGNAL_LENGTH:
            raise ValueError(
                f"Signal must contain exactly {_SIGNAL_LENGTH} samples, "
                f"got {len(v)}."
            )
        return v


# ──────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health() -> Dict[str, str]:
    """Service health check."""
    predictor = _get_predictor()
    return {"status": "ok", "device": predictor.device_name}


@app.post("/predict", response_model=PredictResponse)
async def predict(body: PredictRequest) -> Dict[str, Any]:
    """Run inference on a JSON-encoded 1000-sample ECG signal."""
    predictor = _get_predictor()
    try:
        signal = np.array(body.signal, dtype=np.float64)
        result = predictor.predict(signal)
        logger.info(
            "Prediction: prob=%.4f pred=%d conf=%.4f",
            result["probability"],
            result["prediction"],
            result["confidence"],
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict-file", response_model=PredictResponse)
async def predict_file(
    file: UploadFile = File(..., description="A .npy file with shape (1000,)"),
) -> Dict[str, Any]:
    """Run inference on an uploaded ``.npy`` file."""
    predictor = _get_predictor()

    if not file.filename or not file.filename.endswith(".npy"):
        raise HTTPException(
            status_code=400,
            detail="Only .npy files are accepted.",
        )

    try:
        contents = await file.read()
        signal = np.load(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read .npy file: {exc}",
        ) from exc

    signal = np.squeeze(signal)
    if signal.shape != (_SIGNAL_LENGTH,):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Expected shape ({_SIGNAL_LENGTH},), "
                f"got {signal.shape}."
            ),
        )

    try:
        result = predictor.predict(signal)
        logger.info(
            "Prediction (file): prob=%.4f pred=%d conf=%.4f",
            result["probability"],
            result["prediction"],
            result["confidence"],
        )
        return result
    except Exception as exc:
        logger.exception("Prediction failed (file upload)")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/simulate", response_model=SimulateResponse)
async def simulate(body: SimulateRequest) -> Dict[str, Any]:
    """Simulate real-time streaming inference over a 10-second signal.

    The signal is split into 100-sample chunks.  For each chunk a
    sliding window is formed from samples ``[0 … end_of_chunk]``,
    zero-padded on the right to 1000 samples, and passed through the
    model so that the caller can observe how the prediction evolves
    as more of the signal arrives.

    The final entry uses the full 1000-sample signal without padding.
    """
    predictor = _get_predictor()
    full_signal = np.array(body.signal, dtype=np.float64)

    n_chunks = _SIGNAL_LENGTH // _CHUNK_SIZE
    timeline: List[Dict[str, Any]] = []

    try:
        for i in range(n_chunks):
            start = 0
            end = (i + 1) * _CHUNK_SIZE
            partial = full_signal[start:end]

            # Zero-pad to full length for model compatibility
            padded = np.zeros(_SIGNAL_LENGTH, dtype=np.float64)
            padded[: len(partial)] = partial

            result = predictor.predict(padded)
            timeline.append(
                {
                    "chunk_index": i,
                    "start_sample": start,
                    "end_sample": end,
                    "probability": result["probability"],
                    "prediction": result["prediction"],
                    "confidence": result["confidence"],
                }
            )

        # Final full-signal prediction
        final_result = predictor.predict(full_signal)

        logger.info(
            "Simulation complete: %d chunks, final pred=%d (prob=%.4f)",
            n_chunks,
            final_result["prediction"],
            final_result["probability"],
        )

        return {
            "timeline_predictions": timeline,
            "final_prediction": final_result,
        }
    except Exception as exc:
        logger.exception("Simulation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
