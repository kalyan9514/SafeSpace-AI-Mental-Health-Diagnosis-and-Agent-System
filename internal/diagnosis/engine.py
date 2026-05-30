"""
internal/diagnosis/engine.py

Core diagnosis engine that ties together the retriever,
LLM generation, confidence scoring, and response parsing.
"""

import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config.config import settings
from internal.rag.retriever import Retriever
from internal.diagnosis.confidence import calculate_confidence
from internal.diagnosis.parser import extract_diagnosis, clean_response

logger = logging.getLogger(__name__)


class DiagnosisEngine:
    """
    Loads the LLM and runs the full RAG-based diagnosis pipeline.
    Instantiate once and reuse across requests.
    """

    def __init__(self):
        logger.info("Loading tokenizer and LLM...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.llm_model_id,
            token=settings.huggingface_token,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.llm_model_id,
            token=settings.huggingface_token,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.retriever = Retriever()
        logger.info("Diagnosis engine ready.")

    def run(self, query: str, emotion: str = "neutral") -> dict:
        """
        Run the full diagnosis pipeline for a given user query.

        Returns a dict with the response text, diagnosis label,
        confidence score, and retrieved context chunks.
        """
        # Retrieve relevant context chunks
        top_chunks, chunk_embeddings = self.retriever.search(query)
        context = "\n\n".join(top_chunks)

        # Build the prompt
        prompt = self._build_prompt(query, context, emotion)

        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=settings.max_new_tokens,
                temperature=settings.temperature,
                do_sample=True,
            )
        raw_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = clean_response(raw_response[len(prompt):])

        # Score and parse
        query_embedding = self.retriever.embed(query)
        confidence = calculate_confidence(query_embedding, chunk_embeddings)
        diagnosis = extract_diagnosis(response)

        return {
            "response": response,
            "diagnosis": diagnosis,
            "confidence": confidence,
            "context_chunks": top_chunks,
        }

    def _build_prompt(self, query: str, context: str, emotion: str) -> str:
        """Build the chat prompt from query, retrieved context, and detected emotion."""
        return (
            f"You are a compassionate mental health assistant.\n"
            f"The user appears to be feeling: {emotion}.\n\n"
            f"Relevant information:\n{context}\n\n"
            f"User: {query}\n\n"
            f"Provide a supportive response and end with:\n"
            f"Diagnosed Mental Disorder: <disorder name or None>\n\n"
            f"Assistant:"
        )
    