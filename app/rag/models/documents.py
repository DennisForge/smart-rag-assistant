from __future__ import annotations

from typing import Any, Mapping
from pydantic import BaseModel, Field


class DocumentBase(BaseModel):
    """Base document schema with metadata."""

    id: str
    content: str
    metadata: Mapping[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    """Document chunk with position and metadata."""

    id: str
    document_id: str
    content: str
    index: int
    metadata: Mapping[str, Any] = Field(default_factory=dict)


class ScoredDocumentChunk(BaseModel):
    """Chunk with score from vector search (lower is better for distance metrics)."""

    chunk: DocumentChunk
    score: float
