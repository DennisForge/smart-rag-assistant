"""
RAG (Retrieval-Augmented Generation) Module

This module contains all RAG-related components:
- models: Data structures for documents and embeddings
- interfaces: Abstract contracts for embeddings and vector stores
- services: Business logic for indexing and retrieval
- vectorstores: Vector database implementations (ChromaDB)
"""
from app.rag.models.settings import RAGSettings

__all__ = ["RAGSettings"]