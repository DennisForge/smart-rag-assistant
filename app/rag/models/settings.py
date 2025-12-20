"""
RAG module settings model.

Centralized configuration for RAG operations with stable defaults.
"""

from pydantic import BaseModel


class RAGSettings(BaseModel):
    """RAG configuration with sensible defaults."""

    collection_name: str = "default"
    default_top_k: int = 5
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_persist_dir: str | None = "chroma_data"
    chroma_collection_name: str = "documents"
