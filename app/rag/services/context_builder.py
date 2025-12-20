"""Context builder service for RAG retrieval results."""
# Phase: F4 (retrieval + context prep, no LLM)

from __future__ import annotations

from typing import Sequence

from app.rag.models.documents import ScoredDocumentChunk


class ContextBuilder:
    """Builds context string from retrieved document chunks."""

    SEPARATOR = "\n\n---\n\n"

    def build(
        self,
        chunks: Sequence[ScoredDocumentChunk],
        *,
        max_chars: int = 8000,
    ) -> str:
        """
        Build context string from scored chunks.

        Args:
            chunks: Retrieved chunks in retrieval order
            max_chars: Maximum total characters allowed (hard cap)

        Returns:
            Concatenated chunk contents, or empty string if no chunks
        """
        if not chunks:
            return ""

        # Build incrementally, stop before exceeding limit
        parts = []
        total_length = 0

        for i, scored_chunk in enumerate(chunks):
            content = scored_chunk.chunk.content

            # Calculate what the length would be with this chunk
            # Include separator length for all chunks except the first
            separator_length = len(self.SEPARATOR) if i > 0 else 0
            chunk_contribution = separator_length + len(content)

            # Stop if adding this chunk would exceed the limit
            if total_length + chunk_contribution > max_chars:
                break

            parts.append(content)
            total_length += chunk_contribution

        return self.SEPARATOR.join(parts)
