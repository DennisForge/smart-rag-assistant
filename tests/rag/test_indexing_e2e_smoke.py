"""End-to-end smoke test for indexing pipeline."""

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


def test_indexing_e2e_smoke(tmp_path: Path):
    """End-to-end smoke test: index document and verify it's stored."""
    settings = RAGSettings()

    # Setup components
    provider = SentenceTransformerEmbeddingProvider(
        model_name=settings.embedding_model_name
    )
    persist_dir = str(tmp_path / "chroma_e2e")
    store = ChromaVectorStore(
        collection_name=settings.collection_name,
        persist_directory=persist_dir,
    )
    indexing_service = IndexingService(
        embedder=provider,
        vector_store=store,
    )

    # Create and index one document
    document = DocumentBase(
        id="test_doc_1",
        content="This is a test document for indexing.",
        metadata={"category": "test", "priority": 1},
    )
    indexing_service.index_documents([document])

    # Assert: collection has exactly 1 record (1 doc = 1 chunk in F3)
    assert store._collection.count() == 1


def test_indexing_multiple_documents(tmp_path: Path):
    """Test indexing multiple documents creates multiple chunks."""
    settings = RAGSettings()

    provider = SentenceTransformerEmbeddingProvider(
        model_name=settings.embedding_model_name
    )
    persist_dir = str(tmp_path / "chroma_multi")
    store = ChromaVectorStore(
        collection_name="multi_test",
        persist_directory=persist_dir,
    )
    indexing_service = IndexingService(embedder=provider, vector_store=store)

    # Create multiple documents
    documents = [
        DocumentBase(id="doc1", content="First document", metadata={}),
        DocumentBase(id="doc2", content="Second document", metadata={}),
        DocumentBase(id="doc3", content="Third document", metadata={}),
    ]
    indexing_service.index_documents(documents)

    # Assert: collection has 3 records (3 docs = 3 chunks)
    assert store._collection.count() == 3
