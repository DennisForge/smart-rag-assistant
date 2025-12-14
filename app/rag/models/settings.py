"""
RAG module settings model.

Centralized configuration for RAG operations with stable defaults.
"""

from pydantic import BaseModel


class RAGSettings(BaseModel):
    """RAG configuration with sensible defaults."""

    collection_name: str = "default"
    default_top_k: int = 5
