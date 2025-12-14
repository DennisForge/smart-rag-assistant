from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from app.rag.models.documents import DocumentChunk, ScoredDocumentChunk
from app.rag.models.embeddings import EmbeddingVector


class VectorStoreInterface(ABC):

    @abstractmethod
    def add_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[EmbeddingVector],
    ) -> None:
        
        raise NotImplementedError

    @abstractmethod
    def query(
        self,
        embedding: EmbeddingVector,
        top_k: int = 5,
    ) -> List[ScoredDocumentChunk]:

        raise NotImplementedError

    @abstractmethod
    def delete_by_document_ids(self, document_ids: List[str]) -> None:

        raise NotImplementedError