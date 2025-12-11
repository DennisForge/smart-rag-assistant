# Smart RAG Assistant (FREE STACK)

Smart RAG Assistant is a local-first, free-stack Retrieval-Augmented Generation (RAG) backend.

## Tech stack

- Python 3.x
- FastAPI
- ChromaDB (local vector store)
- sentence-transformers (open-source embeddings)
- Local LLMs (Ollama / LM Studio, planned)

## Project structure

```
smart-rag-assistant/
├── app/                        # Main application package
│   ├── __init__.py
│   ├── main.py                # FastAPI application entry point
│   ├── api/                   # API layer
│   │   ├── __init__.py
│   │   └── v1/               # API version 1
│   │       ├── __init__.py
│   │       └── routes_health.py  # Health check endpoints
│   ├── core/                 # Core application components
│   │   ├── __init__.py
│   │   └── config.py         # Application configuration and settings
│   ├── models/               # Data models (Pydantic schemas)
│   │   └── __init__.py
│   └── services/             # Business logic and services
│       └── __init__.py
├── tests/                     # Test suite
│   ├── __init__.py
│   └── test_health.py        # Health endpoint tests
├── requirements.txt           # Python dependencies
└── README.md                 # Project documentation
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload
