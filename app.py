"""
app.py — Hugging Face Spaces entry point for EmoSense AI.

Mounts all FastAPI REST API routes directly onto Gradio's underlying FastAPI
application (demo.app) and runs demo.launch(), which satisfies Hugging Face's
ZeroGPU startup scanner while serving the Gradio UI (/), static frontend (/ui),
and REST API (/predict, /health/ready, /analytics) on the exact same port.
"""

try:
    import spaces
except ImportError:
    class _MockSpaces:
        @staticmethod
        def GPU(fn=None, **kwargs):
            if fn is not None and callable(fn):
                return fn
            def decorator(func):
                return func
            return decorator
    spaces = _MockSpaces()

import logging
import os
import sys

# Ensure backend/ is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.join(_PROJECT_ROOT, "backend")
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Dynamically load backend/app package to prevent name collision with root app.py
import importlib.util

_backend_init = os.path.join(_BACKEND_DIR, "app", "__init__.py")
_backend_spec = importlib.util.spec_from_file_location(
    "app",
    _backend_init,
    submodule_search_locations=[os.path.join(_BACKEND_DIR, "app")],
)
_backend_mod = importlib.util.module_from_spec(_backend_spec)
sys.modules["app"] = _backend_mod
_backend_spec.loader.exec_module(_backend_mod)

from app import config
from app.routes.analytics import router as analytics_router
from app.routes.health import router as health_router
from app.routes.predict import router as predict_router
from app.services import model_loader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import gradio as gr

log = logging.getLogger("emosense.gradio")

# ── Load ML models at startup ─────────────────────────────────────────────
log.info("=" * 60)
log.info("  EmoSense AI — Initializing Models")
log.info("=" * 60)
try:
    model_loader.load_models()
except Exception as _exc:
    log.critical("Model initialization failed: %s", _exc, exc_info=True)

# ── Emotion config ────────────────────────────────────────────────────────
EMOTION_EMOJI = {
    "anger": "😡",
    "fear": "😨",
    "joy": "😄",
    "neutral": "😐",
    "sadness": "😢",
    "surprise": "😲",
}

EMOTION_COLORS = {
    "anger": "#EF4444",
    "fear": "#A855F7",
    "joy": "#F59E0B",
    "neutral": "#6B7280",
    "sadness": "#3B82F6",
    "surprise": "#10B981",
}


@spaces.GPU
def predict_emotion(text: str) -> tuple:
    """Run ensemble inference with length validation and safe exception handling."""
    if not text or not text.strip():
        return "⚠️ Please enter some text.", None

    clean_input = text.strip()
    if len(clean_input) > 2000:
        return (
            f"⚠️ Input text is too long ({len(clean_input)} chars). "
            f"Maximum allowed length is 2000 characters.",
            None,
        )

    try:
        from app.services.inference import run_inference

        result = run_inference(clean_input, alpha=0.7)

        emotion = result["predicted_emotion"]
        emoji = EMOTION_EMOJI.get(emotion, "🤔")
        confidence = result["confidence"]
        time_ms = result["processing_time_ms"]

        # Build headline
        headline = (
            f"## {emoji} **{emotion.upper()}**\n\n"
            f"**Confidence:** {confidence:.1%}  \n"
            f"**Processing time:** {time_ms:.1f}ms"
        )

        # Build probability bars
        prob_dict = {
            f"{EMOTION_EMOJI.get(k, '')} {k}": v
            for k, v in result["probabilities"].items()
        }

        return headline, prob_dict
    except Exception as exc:
        log.exception("Gradio inference failed: %s", exc)
        return "⚠️ Emotion analysis could not be completed. Please try again later.", None


# ── Build Gradio UI ──────────────────────────────────────────────────────
with gr.Blocks(
    title="EmoSense AI — Emotion Detection",
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="blue",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
    css="""
        .main-title { text-align: center; margin-bottom: 0.5em; }
        .subtitle { text-align: center; color: #64748b; margin-top: 0; font-size: 1.05em; }
        footer { display: none !important; }
    """,
) as demo:
    gr.Markdown(
        "# 🧠 EmoSense AI",
        elem_classes=["main-title"],
    )
    gr.Markdown(
        "Hybrid ensemble emotion detection — RoBERTa + XGBoost + NRC Lexicon",
        elem_classes=["subtitle"],
    )

    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Enter your text (max 2000 characters)",
                placeholder="Type something like: I am absolutely thrilled about this project!",
                lines=4,
                max_lines=10,
            )
            with gr.Row():
                submit_btn = gr.Button("🔍 Analyze Emotion", variant="primary", scale=2)
                clear_btn = gr.ClearButton([text_input], value="🗑️ Clear", scale=1)

            gr.Examples(
                examples=[
                    ["I am absolutely thrilled about this project!"],
                    ["This makes me so angry, I can't believe it happened."],
                    ["I'm really scared about what might happen next."],
                    ["The weather is fine today, nothing special."],
                    ["I miss my old friends so much, it hurts."],
                    ["Wow, I never expected that to happen!"],
                ],
                inputs=text_input,
                label="💡 Try these examples",
            )

        with gr.Column(scale=1):
            result_md = gr.Markdown(
                value="*Results will appear here...*",
                label="Prediction",
            )
            prob_label = gr.Label(
                label="Emotion Probabilities",
                num_top_classes=6,
            )

    submit_btn.click(
        fn=predict_emotion,
        inputs=[text_input],
        outputs=[result_md, prob_label],
    )
    text_input.submit(
        fn=predict_emotion,
        inputs=[text_input],
        outputs=[result_md, prob_label],
    )

    gr.Markdown(
        "---\n"
        "**API Endpoints** also available: "
        "`POST /predict` · `POST /predict/batch` · `GET /health/ready` · `GET /ui`  \n"
        "Built with ❤️ using FastAPI + Gradio"
    )


create_app = _backend_mod.create_app
fastapi_app = create_app()

# ── Attach REST API & Middleware directly to Gradio's underlying FastAPI app ──
demo.app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=config.CORS_ALLOW_CREDENTIALS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept", "X-Requested-With"],
)

demo.app.include_router(health_router)
demo.app.include_router(predict_router)
demo.app.include_router(analytics_router)

# Mount FastAPI app at /api and /gradio_api/api for dedicated REST routing
demo.app.mount("/api", fastapi_app)
demo.app.mount("/gradio_api/api", fastapi_app)

# Mount frontend static UI at /ui
frontend_dir = str(config.FRONTEND_DIR)
if os.path.isdir(frontend_dir):
    demo.app.mount(
        "/ui",
        StaticFiles(directory=frontend_dir, html=True),
        name="frontend",
    )
    log.info("Serving frontend at /ui -> %s", frontend_dir)

# Expose app for ASGI servers if needed
app = demo.app

# ── Enable Gradio Queue for ZeroGPU compatibility ─────────────────────────
demo.queue()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True,
    )
