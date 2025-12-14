from __future__ import annotations

from typing import List

from app.rag.interfaces.embeddings import EmbeddingInterface
from app.rag.interfaces.vector_store import VectorStoreInterface
from app.rag.models.documents import DocumentBase, DocumentChunk


class IndexingService:

    def __init__(
        self,
        embedder: EmbeddingInterface,
        vector_store: VectorStoreInterface,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store

    def index_documents(self, documents: List[DocumentBase]) -> None:
        
        raise NotImplementedError

    def _chunk_document(self, document: DocumentBase) -> List[DocumentChunk]:
        
        raise NotImplementedError