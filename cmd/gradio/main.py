"""
cmd/gradio/main.py

Gradio app entry point for SafeSpace AI.
Provides a tabbed UI for the diagnosis assistant, dashboard, logs, and feedback.
"""

import logging
import gradio as gr
from transformers import pipeline
from internal.audio.transcriber import Transcriber
from internal.diagnosis.engine import DiagnosisEngine
from internal.feedback.logger import log_diagnosis, log_feedback
from internal.dashboard.analytics import diagnosis_distribution_chart, get_recent_logs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load models once at startup
transcriber = Transcriber()
engine = DiagnosisEngine()
emotion_classifier = pipeline(
    "text-classification",
    model="nateraw/bert-base-uncased-emotion",
    top_k=1,
)

# Store session state per request
_session: dict = {}


def detect_emotion(text: str) -> str:
    """Detect the dominant emotion from the user's input text."""
    try:
        result = emotion_classifier(text[:512])
        return result[0][0]["label"]
    except Exception as e:
        logger.warning(f"Emotion detection failed: {e}")
        return "neutral"


def handle_text(user_query: str):
    """Handle a text query through the diagnosis pipeline."""
    if not user_query.strip():
        return "Please enter a message.", None

    emotion = detect_emotion(user_query)
    result = engine.run(user_query, emotion=emotion)

    session_id = log_diagnosis(
        query=user_query,
        response=result["response"],
        diagnosis=result["diagnosis"],
        confidence=result["confidence"],
        emotion=emotion,
        input_type="text",
    )
    _session["last_id"] = session_id

    output = (
        f"{result['response']}\n\n"
        f"**Detected Emotion:** {emotion}\n"
        f"**Confidence:** {result['confidence']}\n"
        f"**Session ID:** {session_id}"
    )
    return output, None


def handle_audio(audio_path: str):
    """Transcribe audio and pass the text through the diagnosis pipeline."""
    if not audio_path:
        return "No audio received.", None

    query = transcriber.transcribe(audio_path)
    if not query:
        return "Could not transcribe audio. Please try again.", None

    return handle_text(query)


def unified_handler(audio, text):
    """Route to audio or text handler depending on what the user provided."""
    if audio:
        return handle_audio(audio)
    return handle_text(text)


def submit_feedback(feedback: str):
    """Save user feedback tied to the last session."""
    session_id = _session.get("last_id", "unknown")
    log_feedback(session_id, feedback)
    return "Thank you for your feedback!"


def build_ui():
    """Build and return the Gradio tabbed interface."""
    with gr.Blocks(title="SafeSpace AI") as app:
        gr.Markdown("# 🧠 SafeSpace AI — Mental Health Diagnosis Assistant")

        with gr.Tab("Assistant"):
            with gr.Row():
                audio_input = gr.Audio(type="filepath", label="Voice Input")
                text_input = gr.Textbox(lines=3, label="Text Input")
            submit_btn = gr.Button("Submit")
            response_out = gr.Markdown(label="Response")
            download_out = gr.File(label="Download Report", visible=False)
            submit_btn.click(
                unified_handler,
                inputs=[audio_input, text_input],
                outputs=[response_out, download_out],
            )

        with gr.Tab("Dashboard"):
            refresh_btn = gr.Button("Refresh Chart")
            chart_out = gr.Plot()
            refresh_btn.click(diagnosis_distribution_chart, outputs=chart_out)

        with gr.Tab("Logs"):
            logs_btn = gr.Button("Load Logs")
            logs_out = gr.Dataframe()
            logs_btn.click(get_recent_logs, outputs=logs_out)

        with gr.Tab("Feedback"):
            feedback_input = gr.Textbox(lines=2, label="Your Feedback")
            feedback_btn = gr.Button("Submit Feedback")
            feedback_out = gr.Textbox(label="Status")
            feedback_btn.click(submit_feedback, inputs=feedback_input, outputs=feedback_out)

    return app


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(share=False)