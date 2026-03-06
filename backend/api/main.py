"""
FastAPI application for ECG Arrhythmia Detection inference (12-lead version).
"""

import io
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from inference.predictor import ECGPredictor

logger = logging.getLogger("ecg_pipeline.api")

# --------------------------------------------------
# Global predictor (loaded once)
# --------------------------------------------------
_predictor: ECGPredictor | None = None

_SIGNAL_LENGTH = 1000
_NUM_LEADS = 12
_CHUNK_SIZE = 100


def _get_predictor() -> ECGPredictor:
    if _predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet."
        )
    return _predictor


# --------------------------------------------------
# Lifespan
# --------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor
    logger.info("Loading IndustryCNN model...")
    _predictor = ECGPredictor(model_name="IndustryCNN")
    logger.info("Model loaded on device=%s", _predictor.device_name)
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="ECG Arrhythmia Detection API",
    version="2.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        # Local development
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        # Possible Render frontend domains (include variants created by Render)
        "https://ecg-frontend.onrender.com",
        "https://ecg-arrhythmia-detection-system.onrender.com",
        "https://ecg-arrhythmia-detection-system-1.onrender.com",
        "https://ecg-arrhythmia-detection-system-sj19.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Schemas
# --------------------------------------------------
class PredictRequest(BaseModel):
    signal: List[List[float]]

    @field_validator("signal")
    @classmethod
    def validate_shape(cls, v):
        if len(v) != _SIGNAL_LENGTH:
            raise ValueError(f"Signal must have {_SIGNAL_LENGTH} time steps.")

        for row in v:
            if len(row) != _NUM_LEADS:
                raise ValueError(
                    f"Each time step must contain {_NUM_LEADS} leads."
                )
        return v


class PredictResponse(BaseModel):
    probability: float
    prediction: int
    confidence: float


class HealthResponse(BaseModel):
    status: str
    device: str


class SimulateRequest(BaseModel):
    signal: List[List[float]]

    @field_validator("signal")
    @classmethod
    def validate_shape(cls, v):
        if len(v) != _SIGNAL_LENGTH:
            raise ValueError(f"Signal must have {_SIGNAL_LENGTH} time steps.")

        for row in v:
            if len(row) != _NUM_LEADS:
                raise ValueError(
                    f"Each time step must contain {_NUM_LEADS} leads."
                )
        return v


class ModelMetricsResponse(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    sensitivity: Optional[float] = None
    specificity: Optional[float] = None


class AllMetricsResponse(BaseModel):
    primary: ModelMetricsResponse
    primary_model_name: str
    comparison: Dict[str, ModelMetricsResponse]


class ROCCurveResponse(BaseModel):
    fpr: List[float]
    tpr: List[float]
    auc: float
    model_name: str
    n_samples: int


class ConfusionMatrixResponse(BaseModel):
    matrix: List[List[int]]
    labels: List[str]
    tn: int
    fp: int
    fn: int
    tp: int
    total: int
    model_name: str


class PRCurveResponse(BaseModel):
    precision: List[float]
    recall: List[float]
    auc: float
    model_name: str


_ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "evaluation" / "artifacts"


# --------------------------------------------------
# Endpoints
# --------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health():
    predictor = _get_predictor()
    return {"status": "ok", "device": predictor.device_name}


@app.get("/metrics", response_model=AllMetricsResponse)
async def metrics():
    """Return real evaluation metrics from training artifacts."""
    try:
        primary_path = _ARTIFACTS_DIR / "test_metrics.json"
        comparison_path = _ARTIFACTS_DIR / "comparison_metrics.json"

        if not primary_path.exists():
            raise HTTPException(status_code=404, detail="Primary metrics not found.")

        with open(primary_path) as f:
            primary_data = json.load(f)

        comparison_data: Dict[str, Any] = {}
        if comparison_path.exists():
            with open(comparison_path) as f:
                comparison_data = json.load(f)

        return {
            "primary": primary_data,
            "primary_model_name": "IndustryCNN",
            "comparison": comparison_data,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to load metrics")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/roc-curve", response_model=ROCCurveResponse)
async def roc_curve_data():
    """Return real ROC curve data points from training evaluation."""
    path = _ARTIFACTS_DIR / "roc_curve_data.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="ROC curve data not found.")
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.exception("Failed to load ROC curve data")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/confusion-matrix", response_model=ConfusionMatrixResponse)
async def confusion_matrix_data():
    """Return real confusion matrix from training evaluation."""
    path = _ARTIFACTS_DIR / "confusion_matrix_data.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Confusion matrix data not found.")
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.exception("Failed to load confusion matrix data")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pr-curve", response_model=PRCurveResponse)
async def pr_curve_data():
    """Return real Precision-Recall curve data from training evaluation."""
    path = _ARTIFACTS_DIR / "pr_curve_data.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="PR curve data not found.")
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.exception("Failed to load PR curve data")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictResponse)
async def predict(body: PredictRequest):
    predictor = _get_predictor()

    try:
        signal = np.array(body.signal, dtype=np.float64)

        if signal.shape != (_SIGNAL_LENGTH, _NUM_LEADS):
            raise HTTPException(
                status_code=400,
                detail=f"Expected shape (1000, 12), got {signal.shape}"
            )

        result = predictor.predict(signal)

        return result

    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-file", response_model=PredictResponse)
async def predict_file(file: UploadFile = File(...)):
    predictor = _get_predictor()

    if not file.filename.endswith(".npy"):
        raise HTTPException(
            status_code=400,
            detail="Only .npy files allowed."
        )

    try:
        contents = await file.read()
        signal = np.load(io.BytesIO(contents))

        # Accept both (1000,12) and (12,1000)
        if signal.shape == (_NUM_LEADS, _SIGNAL_LENGTH):
            signal = signal.T

        if signal.shape != (_SIGNAL_LENGTH, _NUM_LEADS):
            raise HTTPException(
                status_code=400,
                detail=f"Expected shape (1000, 12), got {signal.shape}"
            )

        return predictor.predict(signal)

    except Exception as e:
        logger.exception("File prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simulate")
async def simulate(body: SimulateRequest):
    predictor = _get_predictor()
    signal = np.array(body.signal, dtype=np.float64)

    timeline = []

    try:
        for i in range(_SIGNAL_LENGTH // _CHUNK_SIZE):
            end = (i + 1) * _CHUNK_SIZE
            partial = signal[:end]

            padded = np.zeros((_SIGNAL_LENGTH, _NUM_LEADS))
            padded[:end] = partial

            result = predictor.predict(padded)

            timeline.append({
                "chunk_index": i,
                "end_sample": end,
                "probability": result["probability"],
                "prediction": result["prediction"],
                "confidence": result["confidence"],
            })

        final_result = predictor.predict(signal)

        return {
            "timeline_predictions": timeline,
            "final_prediction": final_result,
        }

    except Exception as e:
        logger.exception("Simulation failed")
        raise HTTPException(status_code=500, detail=str(e))
