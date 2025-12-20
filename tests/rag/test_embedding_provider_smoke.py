"""Smoke test for SentenceTransformerEmbeddingProvider."""

from __future__ import annotations

import pytest

from app.rag.embeddings.sentence_transformer_provider import (
    SentenceTransformerEmbeddingProvider,
)
from app.rag.models.settings import RAGSettings


def test_embedding_provider_smoke():
    """Smoke test: embedding provider can embed text and returns valid vectors."""
    settings = RAGSettings()
    provider = SentenceTransformerEmbeddingProvider(
        model_name=settings.embedding_model_name
    )

    # Embed single text
    result = provider.embed_text("Hello, world!")

    # Assert: vector is not empty
    assert len(result.vector) > 0
    assert all(isinstance(v, float) for v in result.vector)


def test_embedding_provider_batch():
    """Smoke test: batch embedding returns correct number of vectors."""
    settings = RAGSettings()
    provider = SentenceTransformerEmbeddingProvider(
        model_name=settings.embedding_model_name
    )

    texts = ["First text", "Second text", "Third text"]
    results = provider.embed_texts(texts)

    # Assert: same number of vectors as input texts
    assert len(results) == len(texts)
    assert all(len(vec.vector) > 0 for vec in results)
