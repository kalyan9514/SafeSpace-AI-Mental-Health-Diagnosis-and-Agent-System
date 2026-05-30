"""
internal/feedback/logger.py

Handles thread-safe CSV logging for diagnosis sessions
and user feedback submissions.
"""

import csv
import logging
import threading
import uuid
from datetime import datetime
from pathlib import Path
from config.config import settings

logger = logging.getLogger(__name__)

# Lock ensures no two threads write to the same file at the same time
_write_lock = threading.Lock()


def _append_row(file_path: str, row: dict) -> None:
    """Append a single row to a CSV file, creating it with headers if it doesn't exist."""
    path = Path(file_path)
    file_exists = path.exists()

    with _write_lock:
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)


def log_diagnosis(
    query: str,
    response: str,
    diagnosis: str,
    confidence: float,
    emotion: str,
    input_type: str,
) -> str:
    """
    Log a diagnosis session to the diagnosis CSV file.
    Returns the session ID for reference.
    """
    session_id = str(uuid.uuid4())[:8]
    row = {
        "session_id": session_id,
        "timestamp": datetime.utcnow().isoformat(),
        "input_type": input_type,
        "emotion": emotion,
        "query": query,
        "diagnosis": diagnosis,
        "confidence": confidence,
        "response": response,
    }
    _append_row(settings.diagnosis_log_path, row)
    logger.info(f"Logged diagnosis session: {session_id}")
    return session_id


def log_feedback(session_id: str, feedback: str) -> None:
    """Log a user feedback submission to the feedback CSV file."""
    row = {
        "session_id": session_id,
        "timestamp": datetime.utcnow().isoformat(),
        "feedback": feedback,
    }
    _append_row(settings.feedback_log_path, row)
    logger.info(f"Logged feedback for session: {session_id}")
    