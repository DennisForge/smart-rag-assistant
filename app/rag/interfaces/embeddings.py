from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from app.rag.models.embeddings import EmbeddingVector


class EmbeddingInterface(ABC):

    @abstractmethod
    def embed_text(self, text: str) -> EmbeddingVector:
        raise NotImplementedError

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[EmbeddingVector]:
        raise NotImplementedError