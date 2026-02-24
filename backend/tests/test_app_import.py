"""
Smoke test: verify the FastAPI app object can be imported without
triggering heavy model loading or crashing at import time.
"""

import sys
import types


def test_app_import():
    """Importing api.main must expose a FastAPI `app` instance."""
    # Stub out inference.predictor so the real model is never loaded
    fake_inference = types.ModuleType("inference")
    fake_predictor = types.ModuleType("inference.predictor")

    class _FakePredictor:
        device_name = "cpu"

    fake_predictor.ECGPredictor = _FakePredictor
    sys.modules.setdefault("inference", fake_inference)
    sys.modules.setdefault("inference.predictor", fake_predictor)

    from api.main import app  # noqa: E402

    assert app is not None
    assert hasattr(app, "openapi")
    assert callable(app.openapi)


def test_app_routes_registered():
    """Core routes (/health, /predict) must be present."""
    sys.modules.setdefault("inference", __import__("types"))
    fake_predictor = types.ModuleType("inference.predictor")

    class _FakePredictor:
        device_name = "cpu"

    fake_predictor.ECGPredictor = _FakePredictor
    sys.modules.setdefault("inference.predictor", fake_predictor)

    from api.main import app  # noqa: E402

    route_paths = [r.path for r in app.routes]
    assert "/health" in route_paths, f"/health not in {route_paths}"
    assert "/predict" in route_paths, f"/predict not in {route_paths}"
