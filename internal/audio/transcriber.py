"""
internal/audio/transcriber.py

Handles audio transcription using OpenAI Whisper.
Converts voice input from the Gradio UI into text
that gets passed to the diagnosis engine.
"""

import logging
import whisper
from config.config import settings

logger = logging.getLogger(__name__)


class Transcriber:
    """
    Loads the Whisper model and transcribes audio files to text.
    Instantiate once and reuse across requests.
    """

    def __init__(self):
        logger.info(f"Loading Whisper model: {settings.whisper_model_size}")
        self.model = whisper.load_model(settings.whisper_model_size)
        logger.info("Whisper model ready.")

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe an audio file and return the text.

        Returns an empty string if transcription fails or
        the audio path is not provided.
        """
        if not audio_path:
            logger.warning("No audio path provided.")
            return ""

        try:
            result = self.model.transcribe(audio_path)
            text = result.get("text", "").strip()
            logger.info(f"Transcribed {len(text)} characters from audio.")
            return text
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return ""