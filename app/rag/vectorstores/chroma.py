"""ChromaDB vector store implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from app.rag.interfaces.vector_store import VectorStoreInterface
from app.rag.models.documents import DocumentChunk, ScoredDocumentChunk
from app.rag.models.embeddings import EmbeddingVector

if TYPE_CHECKING:
    import chromadb
    from chromadb import Collection


class ChromaVectorStore(VectorStoreInterface):
    """Vector store implementation using ChromaDB."""

    def __init__(
        self,
        collection_name: str,
        persist_directory: str | None = None,
    ) -> None:
        """Initialize ChromaDB client and collection."""
        self._collection_name = collection_name
        self._persist_directory = persist_directory
        self._client: chromadb.Client | None = None
        self._collection: Collection | None = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize ChromaDB client and get/create collection."""
        import chromadb
        from chromadb.config import Settings

        if self._persist_directory:
            self._client = chromadb.Client(
                Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=self._persist_directory,
                )
            )
        else:
            self._client = chromadb.Client()

        self._collection = self._client.get_or_create_collection(
            name=self._collection_name
        )

    def add_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[EmbeddingVector],
    ) -> None:
        """Add document chunks with embeddings to vector store."""
        if not chunks:
            return

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks count ({len(chunks)}) must match embeddings count ({len(embeddings)})"
            )

        # Guard: fail-fast if collection not initialized
        if self._collection is None:
            raise RuntimeError(
                "ChromaDB collection not initialized. Call _initialize_client() first."
            )

        ids = [chunk.id for chunk in chunks]
        embedding_vectors = [emb.vector for emb in embeddings]
        documents = [chunk.content for chunk in chunks]
        # TODO: Metadata must be JSON-serializable (str, int, float, bool, None)
        # Non-primitive types will cause ChromaDB serialization errors
        metadatas = [
            {
                "document_id": chunk.document_id,
                "index": chunk.index,
                **dict(chunk.metadata),
            }
            for chunk in chunks
        ]

        self._collection.upsert(
            ids=ids,
            embeddings=embedding_vectors,
            documents=documents,
            metadatas=metadatas,
        )

    def query(
        self,
        embedding: EmbeddingVector,
        top_k: int = 5,
    ) -> List[ScoredDocumentChunk]:
        """Query vector store for similar chunks."""
        # TODO (F3): Implement query retrieval
        raise NotImplementedError

    def delete_by_document_ids(self, document_ids: List[str]) -> None:
        """Delete all chunks belonging to specified documents."""
        # TODO (F3): Implement deletion (optional for initial F3)
        raise NotImplementedError