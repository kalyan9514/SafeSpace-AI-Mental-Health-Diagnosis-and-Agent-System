# SafeSpace AI

A production-grade mental health diagnosis system that combines voice input, retrieval-augmented generation, and a conversational agent to provide AI-powered emotional support and early disorder identification.

---

## Architecture

```
User Input (text or voice)
   ↓
Whisper (speech-to-text)
   ↓
Emotion Classifier (BERT)
   ↓
RAG Retriever (FAISS + BGE embeddings)
   ↓
Gemma-2 9B LLM (diagnosis + response generation)
   ↓
Confidence Scorer → CSV Logger
   ↓
Gradio UI (diagnosis assistant)

User Input (text)
   ↓
LangChain Agent
   ↓
Tool Router
   ├── SummarizedDocSearch (OpenAI + FAISS)
   ├── EmergencyHelpline
   └── NearbyMentalHealthClinics (Google Maps API)
   ↓
Streamlit UI (conversational agent + clinic map)
```

---

## Features

- Voice and text input via Whisper speech-to-text
- RAG pipeline using FAISS index and BGE embeddings for context retrieval
- Gemma-2 9B LLM for diagnosis, symptom matching, and treatment suggestions
- Emotion detection on user input using BERT classifier
- Confidence scoring on retrieved context
- LangChain agent with three tools: document search, helpline suggestions, clinic finder
- Interactive clinic map rendered with Folium in the Streamlit app
- Thread-safe CSV logging for diagnosis sessions and user feedback
- Plotly dashboard for diagnosis distribution analytics
- Two independent deployable apps: Gradio and Streamlit
- GitHub Actions CI on every push

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Gemma-2 9B (google/gemma-2-9b-it) |
| Embeddings | BAAI/bge-base-en-v1.5 |
| Voice Input | OpenAI Whisper |
| Retrieval | FAISS |
| Emotion Detection | nateraw/bert-base-uncased-emotion |
| Agent Framework | LangChain |
| Gradio UI | Gradio 4.x |
| Streamlit UI | Streamlit 1.30+ |
| Clinic Search | Google Maps API |
| Containerisation | Docker Compose |
| CI/CD | GitHub Actions |

---

## Prerequisites

- Python 3.11+
- Docker Desktop
- Hugging Face account with access to Gemma-2
- OpenAI API key
- Google Maps API key

---

## Running Locally

**1. Clone the repo**
```bash
git clone https://github.com/kalyan9514/SafeSpace-AI-Mental-Health-Diagnosis-and-Agent-System.git
cd SafeSpace-AI-Mental-Health-Diagnosis-and-Agent-System
```

**2. Create your .env file**
```bash
cp .env.example .env
```

**3. Add your credentials to .env**
```bash
HUGGINGFACE_TOKEN=your_token_here
OPENAI_API_KEY=your_key_here
GOOGLE_MAPS_API_KEY=your_key_here
```

**4. Start with Docker**
```bash
docker compose up -d
```

Or run individually:

**5. Run the Gradio app**
```bash
python cmd/gradio/main.py
```

**6. Run the Streamlit app**
```bash
streamlit run cmd/streamlit/main.py
```

---

## Services

| Service | URL |
|---------|-----|
| Gradio diagnosis assistant | http://localhost:7860 |
| Streamlit agent chat | http://localhost:8501 |

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| HUGGINGFACE_TOKEN | Token for loading Gemma-2 and other gated models |
| OPENAI_API_KEY | Used by the LangChain agent in the Streamlit app |
| GOOGLE_MAPS_API_KEY | Used by the clinic finder tool |
| LLM_MODEL_ID | LLM model identifier (default: google/gemma-2-9b-it) |
| EMBEDDING_MODEL_ID | Embedding model (default: BAAI/bge-base-en-v1.5) |
| WHISPER_MODEL_SIZE | Whisper model size (default: base) |
| RAG_TOP_K | Number of chunks to retrieve (default: 5) |
| MAX_NEW_TOKENS | Max tokens for LLM generation (default: 512) |

---

## Project Structure

```
├── cmd/
│   ├── gradio/             # Gradio diagnosis assistant entry point
│   └── streamlit/          # Streamlit agent chat entry point
├── internal/
│   ├── rag/                # FAISS retriever and knowledge base loader
│   ├── diagnosis/          # Engine, confidence scorer, response parser
│   ├── audio/              # Whisper transcriber
│   ├── feedback/           # Thread-safe CSV logger
│   ├── dashboard/          # Plotly analytics
│   └── agent/              # LangChain agent with tools
├── config/
│   └── config.py           # Pydantic settings and path constants
├── data/                   # FAISS index and knowledge base (not committed)
├── logs/                   # Diagnosis and feedback CSV logs (not committed)
├── tests/                  # Unit tests
├── .github/workflows/
│   └── ci.yml              # GitHub Actions CI
├── Dockerfile.gradio       # Docker image for Gradio app
├── Dockerfile.streamlit    # Docker image for Streamlit app
├── docker-compose.yml      # All local services
├── DECISIONS.md            # Architecture decision records
├── .env.example            # Safe credentials template
└── requirements.txt        # Python dependencies
```

---

## Contact

[LinkedIn — Kalyan Kumar Chenchu Malakondaiah](https://www.linkedin.com/in/kalyan-kumar-8170a111b/)
