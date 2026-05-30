# Architecture Decision Records

This document explains the key design decisions made during the refactor
of SafeSpace AI from a Colab notebook to a production-grade project.

---

## 1. Modular `internal/` package structure

**Decision:** Split all core logic into separate modules under `internal/`.

**Why:** The original code had everything in two flat files (`app 1.py` and `utils.py`).
This made it hard to test, reuse, or change individual pieces. Each module now has
a single responsibility, mirroring how production ML systems are structured.

---

## 2. Centralized config with Pydantic Settings

**Decision:** All environment variables and constants live in `config/config.py`.

**Why:** Hardcoded API keys and paths were scattered across the original files.
Pydantic Settings validates types at startup and loads from `.env` automatically,
making the app safe and easy to configure across environments.

---

## 3. Two separate entry points (`cmd/gradio`, `cmd/streamlit`)

**Decision:** Each app has its own `main.py` under `cmd/`.

**Why:** The Gradio app handles voice and text diagnosis. The Streamlit app handles
conversational agent interactions with clinic search. They serve different use cases
and should be independently runnable and deployable.

---

## 4. Thread-safe CSV logging

**Decision:** Used a `threading.Lock` in `internal/feedback/logger.py`.

**Why:** The original app wrote to CSV files without any concurrency protection.
Under multiple simultaneous requests this would corrupt the log files.

---

## 5. Separate Dockerfiles per app

**Decision:** `Dockerfile.gradio` and `Dockerfile.streamlit` are kept separate.

**Why:** Each app has different startup commands and could have different
resource requirements in production. Keeping them separate allows independent
scaling and deployment.

---

## 6. `refactor/production-structure` branch strategy

**Decision:** All refactor work is done on a feature branch, not directly on `main`.

**Why:** Keeps `main` stable while the refactor is in progress. Once reviewed,
the branch gets merged via a pull request.