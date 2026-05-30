"""
internal/diagnosis/parser.py

Parses the LLM response to extract the diagnosed mental disorder
and any other structured fields from the generated text.
"""

import re
import logging

logger = logging.getLogger(__name__)


def extract_diagnosis(response_text: str) -> str:
    """
    Extract the diagnosed mental disorder from the LLM response.

    Looks for a line starting with 'Diagnosed Mental Disorder:' and
    returns the value after it. Falls back to 'Unknown' if not found.
    """
    pattern = r"Diagnosed Mental Disorder[:\s]+(.+)"
    match = re.search(pattern, response_text, re.IGNORECASE)

    if match:
        diagnosis = match.group(1).strip()
        logger.info(f"Extracted diagnosis: {diagnosis}")
        return diagnosis

    logger.warning("Could not extract diagnosis from response.")
    return "Unknown"


def clean_response(response_text: str) -> str:
    """
    Remove any system prompt artifacts or repeated tokens
    from the raw LLM output before displaying to the user.
    """
    # Strip leading/trailing whitespace and repeated newlines
    cleaned = re.sub(r"\n{3,}", "\n\n", response_text.strip())
    return cleaned