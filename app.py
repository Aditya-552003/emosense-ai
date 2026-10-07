"""
app.py — Hugging Face Spaces entry point.

Mounts the full FastAPI backend (with all existing REST endpoints)
inside a Gradio app using gr.mount_gradio_app, so BOTH the Gradio UI
and the JSON API coexist on the same port.

HF Spaces (Gradio SDK) looks for this file at the project root.
"""

import sys
import os

# Add backend/ to sys.path so `app.*` imports resolve correctly
_BACKEND_DIR = os.path.join(os.path.dirname(__file__), "backend")
sys.path.insert(0, _BACKEND_DIR)

import gradio as gr
from app import create_app  # noqa: E402 — FastAPI factory

# ── Create the FastAPI app (loads models via lifespan) ────────────────────
fastapi_app = create_app()

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


def predict_emotion(text: str) -> tuple:
    """Run the ensemble inference and return formatted results."""
    if not text or not text.strip():
        return "⚠️ Please enter some text.", None

    from app.services.inference import run_inference

    result = run_inference(text.strip())

    emotion = result["predicted_emotion"]
    emoji = EMOTION_EMOJI.get(emotion, "🤔")
    confidence = result["confidence"]
    time_ms = result["processing_time_ms"]

    # Build the headline
    headline = (
        f"## {emoji} **{emotion.upper()}**\n\n"
        f"**Confidence:** {confidence:.1%}  \n"
        f"**Processing time:** {time_ms:.0f}ms"
    )

    # Build probability bars (dict for gr.Label)
    prob_dict = {
        f"{EMOTION_EMOJI.get(k, '')} {k}": v
        for k, v in result["probabilities"].items()
    }

    return headline, prob_dict


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
                label="Enter your text",
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
        "`POST /predict` · `POST /predict/batch` · `GET /` (health check)  \n"
        "Built with ❤️ using FastAPI + Gradio"
    )


# ── Mount Gradio interface at /gradio so REST API remains on / ────────────
app = gr.mount_gradio_app(fastapi_app, demo, path="/gradio")

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)



