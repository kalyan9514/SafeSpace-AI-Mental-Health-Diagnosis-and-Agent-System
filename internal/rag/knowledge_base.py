"""
internal/rag/knowledge_base.py

Loads and preprocesses the NIMH mental health text chunks
that serve as the RAG knowledge base.
"""

import logging
from pathlib import Path
from config.config import settings

logger = logging.getLogger(__name__)


def load_chunks(path: str | None = None) -> list[str]:
    """
    Load text chunks from the knowledge base file.

    Each chunk is separated by the configured separator.
    Returns a list of non-empty text chunks.
    """
    file_path = Path(path or settings.knowledge_base_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Knowledge base not found at: {file_path}")

    raw = file_path.read_text(encoding="utf-8")
    chunks = [c.strip() for c in raw.split(settings.rag_chunk_separator) if c.strip()]

    logger.info(f"Loaded {len(chunks)} chunks from {file_path.name}")
    return chunks