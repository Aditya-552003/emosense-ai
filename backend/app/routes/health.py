"""
routes/health.py — Health check and status endpoint.
"""

from fastapi import APIRouter

from app.services.model_loader import get_models

router = APIRouter()


@router.get("/", summary="Health check & API status")
async def root():
    """
    Returns the health status of the API, loaded emotion classes, and
    the device being used for inference.
    """
    models = get_models()
    classes = models.get("label_encoder", None)
    emotion_classes = classes.classes_.tolist() if classes else []
    return {
        "status": "online",
        "api": "Emotion Detection API",
        "version": "1.0.0",
        "device": models.get("device", "unknown"),
        "emotion_classes": emotion_classes,
        "frontend": "http://127.0.0.1:8080/ui/index.html",
        "evaluation": "http://127.0.0.1:8080/ui/evaluation.html",
        "endpoints": {
            "health": "GET  /",
            "predict": "POST /predict  { text: string }",
        },
    }
