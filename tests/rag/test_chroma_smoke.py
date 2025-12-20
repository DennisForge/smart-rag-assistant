"""Smoke test for ChromaVectorStore."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.rag.models.documents import DocumentChunk
from app.rag.models.embeddings import EmbeddingVector
from app.rag.vectorstores.chroma import ChromaVectorStore


def test_chroma_vector_store_smoke(tmp_path: Path):
    """Smoke test: ChromaVectorStore can store and count chunks."""
    # Create store with temporary directory
    persist_dir = str(tmp_path / "chroma_test")
    store = ChromaVectorStore(
        collection_name="test_collection",
        persist_directory=persist_dir,
    )

    # Create test chunk and embedding
    chunk = DocumentChunk(
        id="doc1::0",
        document_id="doc1",
        content="Test document content",
        index=0,
        metadata={"source": "test"},
    )
    embedding = EmbeddingVector(vector=[0.1, 0.2, 0.3, 0.4, 0.5])

    # Upsert chunk
    store.add_chunks([chunk], [embedding])

    # Assert: collection has at least 1 item
    assert store._collection.count() > 0


def test_chroma_vector_store_guard_check():
    """Test that add_chunks fails fast if collection is None."""
    store = ChromaVectorStore(
        collection_name="test",
        persist_directory=None,
    )
    # Manually break the collection to test guard
    store._collection = None

    chunk = DocumentChunk(
        id="doc1::0",
        document_id="doc1",
        content="Test",
        index=0,
        metadata={},
    )
    embedding = EmbeddingVector(vector=[0.1, 0.2, 0.3])

    with pytest.raises(RuntimeError, match="collection not initialized"):
        store.add_chunks([chunk], [embedding])
