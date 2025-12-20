"""Smoke tests for RetrievalService."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.rag.embeddings.sentence_transformer_provider import (
    SentenceTransformerEmbeddingProvider,
)
from app.rag.models.documents import DocumentBase, ScoredDocumentChunk
from app.rag.services.indexing import IndexingService
from app.rag.services.retrieval_service import RetrievalService
from app.rag.vectorstores.chroma import ChromaVectorStore

# Constant model name used across tests
TEST_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture(scope="module")
def provider():
    """Shared embedding provider for all tests (loads model once)."""
    return SentenceTransformerEmbeddingProvider(model_name=TEST_MODEL_NAME)


def test_retrieval_smoke(tmp_path: Path, provider):
    """Smoke test: retrieval returns valid results with correct properties."""
    # Setup components
    persist_dir = str(tmp_path / "chroma_retrieval")
    store = ChromaVectorStore(
        collection_name="retrieval_test",
        persist_directory=persist_dir,
    )

    # Index some documents
    indexing_service = IndexingService(embedder=provider, vector_store=store)
    documents = [
        DocumentBase(
            id="doc1",
            content="Python is a programming language",
            metadata={},
        ),
        DocumentBase(
            id="doc2",
            content="JavaScript is used for web development",
            metadata={},
        ),
        DocumentBase(
            id="doc3",
            content="Machine learning models use Python",
            metadata={},
        ),
    ]
    indexing_service.index_documents(documents)

    # Create retrieval service
    retrieval_service = RetrievalService(embedder=provider, vector_store=store)

    # Retrieve results (use top_k=3 since we only have 3 documents)
    results = retrieval_service.retrieve("Python programming", top_k=3)

    # Assertions
    assert isinstance(results, list)
    assert len(results) <= 3
    assert len(results) > 0  # Should have at least some results

    # Every item is ScoredDocumentChunk
    for result in results:
        assert isinstance(result, ScoredDocumentChunk)

    # Results are sorted by score ascending (distance: lower=better)
    scores = [r.score for r in results]
    assert scores == sorted(scores)

    # Each result has non-empty content
    for result in results:
        assert result.chunk.content.strip() != ""


def test_retrieval_empty_query_raises_error(provider):
    """Test that empty query raises ValueError."""
    # No need for store or tmp_path - just testing validation
    store = ChromaVectorStore(
        collection_name="empty_query_test",
        persist_directory=None,  # In-memory for validation test
    )

    retrieval_service = RetrievalService(embedder=provider, vector_store=store)

    with pytest.raises(ValueError, match="Query text cannot be empty"):
        retrieval_service.retrieve("")

    with pytest.raises(ValueError, match="Query text cannot be empty"):
        retrieval_service.retrieve("   ")

    with pytest.raises(ValueError, match="top_k must be greater than 0"):
        retrieval_service.retrieve("test", top_k=0)

    with pytest.raises(ValueError, match="top_k must be greater than 0"):
        retrieval_service.retrieve("test", top_k=-1)


def test_retrieval_similarity_wins(tmp_path: Path, provider):
    """Test that more similar document ranks higher (lower distance score)."""
    # Setup components
    persist_dir = str(tmp_path / "chroma_similarity")
    store = ChromaVectorStore(
        collection_name="similarity_test",
        persist_directory=persist_dir,
    )

    # Index two documents with very distinct terms
    indexing_service = IndexingService(embedder=provider, vector_store=store)
    documents = [
        DocumentBase(
            id="doc_fruit",
            content="apple banana apple",
            metadata={"category": "fruit"},
        ),
        DocumentBase(
            id="doc_industrial",
            content="welding cnc trumatic",
            metadata={"category": "industrial"},
        ),
    ]
    indexing_service.index_documents(documents)

    # Create retrieval service
    retrieval_service = RetrievalService(embedder=provider, vector_store=store)

    # Query for "apple"
    results = retrieval_service.retrieve("apple", top_k=2)

    # Assertions
    assert len(results) == 2

    # Top result should be the fruit document (contains "apple")
    top_result = results[0]
    assert "apple" in top_result.chunk.content

    # Top result should have lower (better) score than second result
    assert results[0].score < results[1].score

    # Second result should be the industrial document
    second_result = results[1]
    assert "welding" in second_result.chunk.content or "cnc" in second_result.chunk.content


def test_retrieve_with_context_builds_context(tmp_path: Path, provider):
    """Test retrieve_with_context() returns results and context string."""
    # Setup components
    persist_dir = str(tmp_path / "chroma_context")
    store = ChromaVectorStore(
        collection_name="context_test",
        persist_directory=persist_dir,
    )

    # Index documents
    indexing_service = IndexingService(embedder=provider, vector_store=store)
    documents = [
        DocumentBase(
            id="doc1",
            content="Python is a programming language used for data science",
            metadata={},
        ),
        DocumentBase(
            id="doc2",
            content="JavaScript is used for web development and frontend applications",
            metadata={},
        ),
        DocumentBase(
            id="doc3",
            content="Python has extensive libraries for machine learning",
            metadata={},
        ),
    ]
    indexing_service.index_documents(documents)

    # Create retrieval service
    retrieval_service = RetrievalService(embedder=provider, vector_store=store)

    # Call retrieve_with_context
    results, context = retrieval_service.retrieve_with_context(
        "Python programming",
        top_k=3,
        max_chars=8000,
    )

    # Assertions
    assert isinstance(results, list)
    assert len(results) > 0  # Should have results

    assert isinstance(context, str)
    assert len(context) <= 8000

    # Context should contain content from top chunk
    top_chunk_content = results[0].chunk.content
    assert top_chunk_content in context

    # If multiple results, separator should be present
    if len(results) > 1:
        assert "\n\n---\n\n" in context
