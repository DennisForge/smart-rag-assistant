# Smart RAG Assistant

Local RAG backend that runs on your machine. No cloud, no API keys.

## What's inside

**Stack:**
- FastAPI + Python 3.14
- ChromaDB (vector store)
- sentence-transformers (embeddings)
- No external APIs

**Next up:**
- Local LLM integration (Ollama/LM Studio)

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
│   ├── services/             # Business logic and services
│   │   └── __init__.py
│   └── rag/                  # RAG (Retrieval-Augmented Generation) module
│       ├── __init__.py
│       ├── models/           # RAG data models
│       │   ├── __init__.py
│       │   ├── documents.py  # Document, chunk, and storage models
│       │   └── embeddings.py # Embedding vector and metadata models
│       ├── interfaces/       # Abstract base classes
│       │   ├── __init__.py
│       │   ├── embeddings.py # Embedding interface contract
│       │   └── vector_store.py # Vector store interface contract
│       ├── services/         # RAG orchestration services
│       │   ├── __init__.py
│       │   └── indexing.py   # Document indexing and retrieval service
│       └── vectorstores/     # Vector database implementations
│           ├── __init__.py
│           └── chroma.py     # ChromaDB implementation
├── tests/                     # Test suite
│   ├── __init__.py
│   ├── test_health.py        # Health endpoint tests
│   └── rag/                  # RAG module tests
│       ├── __init__.py
│       ├── test_embeddings_interface.py
│       ├── test_vector_store_interface.py
│       └── test_indexing_service.py
├── requirements.txt           # Python dependencies
└── README.md                 # Project documentation
```

## Quick Start

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Create .env file
cat > .env << EOF
APP_NAME=Smart RAG Assistant
DEBUG=true
ENVIRONMENT=local
API_V1_PREFIX=/api/v1
HOST=0.0.0.0
PORT=8000
EOF

# Run server
uvicorn app.main:app --reload

# Test it
curl http://127.0.0.1:8000/api/v1/health
```

Browse to `http://127.0.0.1:8000/docs` for API documentation.

## Development

**Run tests:**
```bash
pytest -v
```

**RAG tests only:**
```bash
pytest tests/rag/ -v
```

## Project Progress

- ✅ **F1**: Backend skeleton, health endpoints
- ✅ **F2**: RAG architecture (interfaces, models)
- ✅ **F3**: Embeddings + ChromaDB indexing
- ✅ **F4**: Retrieval + context building
- ⏳ **F5**: Local LLM integration

Current status: **21 tests passing**, ready for LLM integration.
