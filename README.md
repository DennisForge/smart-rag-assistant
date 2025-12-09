# Smart RAG Assistant (FREE STACK)

Smart RAG Assistant is a local-first, free-stack Retrieval-Augmented Generation (RAG) backend.

## Tech stack

- Python 3.x
- FastAPI
- ChromaDB (local vector store)
- sentence-transformers (open-source embeddings)
- Local LLMs (Ollama / LM Studio, planned)

## Project structure

- `app/` – FastAPI app, RAG logic, configuration
- `data/` – local documents (raw and processed, **not committed to Git**)
- `storage/chroma/` – ChromaDB persistence (**not committed to Git**)
- `tests/` – unit and integration tests (to be added)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload
