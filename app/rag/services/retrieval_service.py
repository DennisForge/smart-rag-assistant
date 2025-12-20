"""Retrieval service for RAG query execution."""
# Phase: F4 (retrieval + context prep, no LLM)

from __future__ import annotations

from app.rag.interfaces.embeddings import EmbeddingInterface
from app.rag.interfaces.vector_store import VectorStoreInterface
from app.rag.models.documents import ScoredDocumentChunk
from app.rag.services.context_builder import ContextBuilder


class RetrievalService:
    """Service for retrieving relevant document chunks based on queries."""

    def __init__(
        self,
        embedder: EmbeddingInterface,
        vector_store: VectorStoreInterface,
    ) -> None:
        """Initialize with embedding provider and vector store."""
        self._embedder = embedder
        self._vector_store = vector_store

    def retrieve(
        self,
        query_text: str,
        *,
        top_k: int = 5,
    ) -> list[ScoredDocumentChunk]:
        """
        Retrieve relevant document chunks for a query.

        Args:
            query_text: Query string to search for
            top_k: Maximum number of results to return

        Returns:
            List of scored chunks, sorted by score ascending (lower=better)

        Raises:
            ValueError: If query_text is empty or top_k <= 0
        """
        # Validate inputs
        if not query_text.strip():
            raise ValueError("Query text cannot be empty")
        
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        # Embed query
        query_embedding = self._embedder.embed_text(query_text)

        # Query vector store
        results = self._vector_store.query(query_embedding, top_k=top_k)

        # Sort by score ascending (distance: lower is better)
        # Defensive sorting even if store returns sorted results
        sorted_results = sorted(results, key=lambda x: x.score)

        return sorted_results

    def retrieve_with_context(
        self,
        query_text: str,
        *,
        top_k: int = 5,
        max_chars: int = 8000,
    ) -> tuple[list[ScoredDocumentChunk], str]:
        """
        Retrieve relevant chunks and build context string.

        Args:
            query_text: Query string to search for
            top_k: Maximum number of results to return
            max_chars: Maximum characters in context string

        Returns:
            Tuple of (scored chunks, context string)

        Raises:
            ValueError: If query_text is empty or top_k <= 0
        """
        # Retrieve chunks
        results = self.retrieve(query_text, top_k=top_k)

        # Build context string
        context_builder = ContextBuilder()
        context = context_builder.build(results, max_chars=max_chars)

        return (results, context)
