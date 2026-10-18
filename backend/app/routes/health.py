"""
routes/health.py — Health check, liveness, and readiness endpoints.
"""

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from app.services.model_loader import get_model_state, get_models, is_model_ready

router = APIRouter()


@router.get("/", summary="Health check & API status", status_code=status.HTTP_200_OK)
async def root():
    """
    Returns the general health status of the API, loaded emotion classes, and
    the device being used for inference. Preserves full backward compatibility.
    """
    ready = is_model_ready()
    models = get_models() if ready else {}
    classes = models.get("label_encoder", None)
    emotion_classes = classes.classes_.tolist() if classes else []

    return {
        "status": "online" if ready else "initializing",
        "api": "Emotion Detection API",
        "version": "1.0.0",
        "device": models.get("device", "unknown"),
        "emotion_classes": emotion_classes,
        "frontend": "/ui/index.html",
        "evaluation": "/ui/evaluation.html",
        "endpoints": {
            "health": "GET  /",
            "liveness": "GET  /health/live",
            "readiness": "GET  /health/ready",
            "predict": "POST /predict  { text: string }",
            "predict_batch": "POST /predict/batch  [multipart/form-data CSV]",
        },
    }


@router.get(
    "/health/live",
    summary="Liveness probe",
    status_code=status.HTTP_200_OK,
)
async def liveness():
    """Liveness probe: reports whether the application server process is running."""
    return {"status": "alive"}


@router.get(
    "/health/ready",
    summary="Readiness probe",
    status_code=status.HTTP_200_OK,
)
async def readiness():
    """
    Readiness probe: reports whether all ML models are loaded and ready to serve traffic.
    Returns HTTP 200 when ready, HTTP 503 when initializing or failed.
    """
    if is_model_ready():
        models = get_models()
        return {
            "status": "ready",
            "state": get_model_state().value,
            "device": models.get("device", "unknown"),
        }

    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": "not_ready",
            "state": get_model_state().value,
            "detail": "Model artifacts are still initializing or failed to load.",
        },
    )
