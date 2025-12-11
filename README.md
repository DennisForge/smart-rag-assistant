# Smart RAG Assistant (FREE STACK)

Smart RAG Assistant is a local-first, free-stack Retrieval-Augmented Generation (RAG) backend.

## Tech stack

### Core Framework
- **Python 3.14** - Programming language
- **FastAPI** - Modern async web framework for building APIs
- **Uvicorn** - Lightning-fast ASGI server
- **Pydantic** - Data validation and settings management

### RAG Components
- **ChromaDB** - Local-first vector store for embeddings
- **Sentence Transformers** - Open-source embedding models
- **PyTorch** - Deep learning framework for ML models
- **Transformers (HuggingFace)** - Pre-trained language models

### Development Tools
- **pytest** - Testing framework
- **Black** - Code formatter
- **Ruff** - Fast Python linter
- **python-dotenv** - Environment variable management

### Planned Features
- **Local LLMs** - Ollama / LM Studio integration for text generation

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
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Phase F1 – Backend Skeleton

This phase establishes the foundational backend architecture with project layout, configuration system, minimal API, and automated tests.

### Configuration

Configuration is managed through **Pydantic Settings** in `app/core/config.py`.

Create a `.env` file in the project root:

```bash
APP_NAME=Smart RAG Assistant
DEBUG=true
ENVIRONMENT=local
API_V1_PREFIX=/api/v1
HOST=0.0.0.0
PORT=8000
```

### Running the API

Start the development server:

```bash
uvicorn app.main:app --reload
```

Or without activating venv:

```bash
.venv/bin/uvicorn app.main:app --reload
```

### Available Endpoints

| Endpoint | Description | Response |
|----------|-------------|----------|
| `GET /api/v1/health` | Health check | `{"status": "ok", "app_name": "...", "environment": "..."}` |
| `GET /docs` | Swagger UI (Interactive API docs) | Web interface |
| `GET /redoc` | ReDoc (Alternative API docs) | Web interface |

**Example request:**

```bash
curl http://127.0.0.1:8000/api/v1/health
```

**Expected response:**

```json
{
  "status": "ok",
  "app_name": "Smart RAG Assistant",
  "environment": "local"
}
```

### Tests

Run all tests:

```bash
pytest
```

Run with verbose output:

```bash
pytest -v
```

**Expected output:** `2 passed`

### Core Dependencies (F1)

- `fastapi` - Web framework
- `uvicorn[standard]` - ASGI server
- `pydantic-settings` - Configuration management
- `pytest` - Testing framework
- `httpx` - HTTP client for testing

---

**F1 Phase Complete** ✅ 

The backend skeleton is ready for the next phase: RAG integration with ChromaDB and embedding models
