"""
internal/diagnosis/confidence.py

Calculates a RAG confidence score based on how well
the retrieved chunks match the original query.
"""

import logging
import numpy as np
from config.config import settings

logger = logging.getLogger(__name__)


def calculate_confidence(
    query_embedding: np.ndarray,
    chunk_embeddings: np.ndarray,
) -> float:
    """
    Calculate a confidence score between 0 and 1.

    Uses cosine similarity between the query and retrieved chunks,
    weighted by the retriever and generation weights from config.
    """
    if chunk_embeddings.size == 0:
        logger.warning("No chunk embeddings provided, returning zero confidence.")
        return 0.0

    # Average cosine similarity across all retrieved chunks
    similarities = np.dot(chunk_embeddings, query_embedding.T).flatten()
    avg_similarity = float(np.mean(similarities))

    # Weighted combination of retriever and generation scores
    score = settings.rag_alpha * avg_similarity + settings.rag_beta * avg_similarity
    return round(min(max(score, 0.0), 1.0), 4)