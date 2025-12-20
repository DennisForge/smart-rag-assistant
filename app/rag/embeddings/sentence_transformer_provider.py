"""Sentence Transformers embedding provider implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from app.rag.interfaces.embeddings import EmbeddingInterface
from app.rag.models.embeddings import EmbeddingVector

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbeddingProvider(EmbeddingInterface):
    """Embedding provider using sentence-transformers library."""

    def __init__(self, model_name: str) -> None:
        """Initialize with model name (lazy loading)."""
        self._model_name = model_name
        self._model: SentenceTransformer | None = None

    def _load_model(self) -> SentenceTransformer:
        """Lazy load the model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed_text(self, text: str) -> EmbeddingVector:
        """Embed single text into vector."""
        model = self._load_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return EmbeddingVector(vector=embedding.tolist())

    def embed_texts(self, texts: List[str]) -> List[EmbeddingVector]:
        """Embed multiple texts into vectors (batch operation)."""
        model = self._load_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return [EmbeddingVector(vector=emb.tolist()) for emb in embeddings]
