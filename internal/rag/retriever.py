"""
internal/rag/retriever.py

Loads the FAISS index and handles semantic search
against the knowledge base chunks.
"""

import logging
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config.config import settings
from internal.rag.knowledge_base import load_chunks

logger = logging.getLogger(__name__)


class Retriever:
    """Handles embedding queries and retrieving top-k relevant chunks."""

    def __init__(self):
        logger.info("Loading embedding model...")
        self.model = SentenceTransformer(settings.embedding_model_id)

        logger.info("Loading FAISS index...")
        self.index = faiss.read_index(settings.faiss_index_path)

        self.chunks = load_chunks()
        logger.info(f"Retriever ready with {len(self.chunks)} chunks.")

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string and return as a float32 array."""
        return self.model.encode([text], normalize_embeddings=True).astype("float32")

    def search(self, query: str, top_k: int | None = None) -> tuple[list[str], np.ndarray]:
        """
        Search the FAISS index for the most relevant chunks.

        Returns the top-k chunks and their embedding vectors.
        """
        k = top_k or settings.rag_top_k
        query_embedding = self.embed(query)

        distances, indices = self.index.search(query_embedding, k)
        top_chunks = [self.chunks[i] for i in indices[0] if i < len(self.chunks)]

        # Embed each retrieved chunk for confidence scoring later
        chunk_embeddings = self.model.encode(top_chunks, normalize_embeddings=True).astype("float32")

        return top_chunks, chunk_embeddings
    