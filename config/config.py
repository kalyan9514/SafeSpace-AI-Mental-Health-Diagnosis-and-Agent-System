"""
config/config.py

Central configuration for SafeSpace AI.
All settings are loaded from environment variables or a .env file.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
LOGS_DIR = ROOT_DIR / "logs"


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Hugging Face
    huggingface_token: str = ""

    # OpenAI (used by LangChain agent in Streamlit app)
    openai_api_key: str = ""

    # Google Maps (used by clinic finder)
    google_maps_api_key: str = ""

    # Model identifiers
    llm_model_id: str = "google/gemma-2-9b-it"
    embedding_model_id: str = "BAAI/bge-base-en-v1.5"
    emotion_model_id: str = "nateraw/bert-base-uncased-emotion"
    whisper_model_size: str = "base"

    # RAG settings
    faiss_index_path: str = str(DATA_DIR / "qa_faiss_embedding.index")
    knowledge_base_path: str = str(DATA_DIR / "chunked_text_RAG_text.txt")
    rag_top_k: int = 5
    rag_chunk_separator: str = "\n\n---\n\n"

    # Confidence scoring weights
    rag_alpha: float = 0.6
    rag_beta: float = 0.4

    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 0.7

    # Log file paths
    diagnosis_log_path: str = str(LOGS_DIR / "diagnosis_logs.csv")
    feedback_log_path: str = str(LOGS_DIR / "feedback_logs.csv")


# Single instance used across the application
settings = Settings()

# Make sure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)