"""Test idempotent upsert behavior for indexing service."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.rag.embeddings.sentence_transformer_provider import (
    SentenceTransformerEmbeddingProvider,
)
from app.rag.models.documents import DocumentBase
from app.rag.models.settings import RAGSettings
from app.rag.services.indexing import IndexingService
from app.rag.vectorstores.chroma import ChromaVectorStore


def test_idempotent_upsert(tmp_path: Path):
    """Test that indexing the same document twice doesn't duplicate records."""
    settings = RAGSettings()

    # Setup components
    provider = SentenceTransformerEmbeddingProvider(
        model_name=settings.embedding_model_name
    )
    persist_dir = str(tmp_path / "chroma_idempotent")
    store = ChromaVectorStore(
        collection_name=settings.chroma_collection_name,
        persist_directory=persist_dir,
    )
    indexing_service = IndexingService(
        embedder=provider,
        vector_store=store,
    )

    # Create document with deterministic ID
    document = DocumentBase(
        id="stable_doc_id",
        content="This document will be indexed twice.",
        metadata={"test": "idempotent"},
    )

    # Index same document twice
    indexing_service.index_documents([document])
    indexing_service.index_documents([document])

    # Assert: count remains 1 (upsert, not insert)
    assert store._collection.count() == 1
