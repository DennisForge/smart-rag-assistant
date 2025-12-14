from __future__ import annotations

from typing import List

from app.rag.interfaces.vector_store import VectorStoreInterface
from app.rag.models.documents import DocumentChunk, ScoredDocumentChunk
from app.rag.models.embeddings import EmbeddingVector


class ChromaVectorStore(VectorStoreInterface):
    

    def __init__(self, collection_name: str) -> None:
        self._collection_name = collection_name
        self._client = None  # placeholder za budući Chroma klijent

    def add_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[EmbeddingVector],
    ) -> None:
        raise NotImplementedError

    def query(
        self,
        embedding: EmbeddingVector,
        top_k: int = 5,
    ) -> List[ScoredDocumentChunk]:
        raise NotImplementedError

    def delete_by_document_ids(self, document_ids: List[str]) -> None:
        raise NotImplementedError