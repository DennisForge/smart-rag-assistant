"""Document indexing service implementation."""

from __future__ import annotations

from typing import List

from app.rag.interfaces.embeddings import EmbeddingInterface
from app.rag.interfaces.vector_store import VectorStoreInterface
from app.rag.models.documents import DocumentBase, DocumentChunk


class IndexingService:
    """Service for indexing documents into vector store."""

    def __init__(
        self,
        embedder: EmbeddingInterface,
        vector_store: VectorStoreInterface,
    ) -> None:
        """Initialize with embedding provider and vector store."""
        self._embedder = embedder
        self._vector_store = vector_store

    def index_documents(self, documents: List[DocumentBase]) -> None:
        """Index multiple documents by chunking, embedding, and storing."""
        if not documents:
            return

        # Chunk all documents
        all_chunks: List[DocumentChunk] = []
        for document in documents:
            chunks = self._chunk_document(document)
            all_chunks.extend(chunks)

        if not all_chunks:
            return

        # Batch embed all chunks
        chunk_texts = [chunk.content for chunk in all_chunks]
        embeddings = self._embedder.embed_texts(chunk_texts)

        # Store in vector store
        self._vector_store.add_chunks(all_chunks, embeddings)

    def _chunk_document(self, document: DocumentBase) -> List[DocumentChunk]:
        """Chunk a single document (minimal: 1 document = 1 chunk for F3)."""
        # Minimal chunking for F3: entire document as single chunk
        # Deterministic ID format: {document_id}::chunk:{index}
        return [
            DocumentChunk(
                id=f"{document.id}::chunk:0",
                document_id=document.id,
                content=document.content,
                index=0,
                metadata=dict(document.metadata),
            )
        ]