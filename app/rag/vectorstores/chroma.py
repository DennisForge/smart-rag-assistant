"""ChromaDB vector store implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
        chunks: list[DocumentChunk],
        embeddings: list[EmbeddingVector],
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
    ) -> list[ScoredDocumentChunk]:
        """Query vector store for similar chunks."""
        # Guard: fail-fast if collection not initialized
        if self._collection is None:
            raise RuntimeError(
                "ChromaDB collection not initialized. Call _initialize_client() first."
            )

        # Query ChromaDB
        results = self._collection.query(
            query_embeddings=[embedding.vector],
            n_results=top_k,
        )

        # Guard: ensure expected fields are present
        required_fields = ["ids", "distances", "documents", "metadatas"]
        for field in required_fields:
            if field not in results:
                raise RuntimeError(
                    f"Chroma query did not return expected fields. Missing: {field}"
                )

        # Convert ChromaDB results to ScoredDocumentChunk
        scored_chunks = []

        # ChromaDB returns results in this structure:
        # results = {
        #     'ids': [['id1', 'id2', ...]],
        #     'distances': [[0.1, 0.2, ...]],
        #     'documents': [['doc1', 'doc2', ...]],
        #     'metadatas': [[{...}, {...}, ...]],
        # }

        if not results["ids"] or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        distances = results["distances"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        for i in range(len(ids)):
            # Extract metadata and pop reserved keys
            meta = dict(metadatas[i])
            document_id = meta.pop("document_id", "")
            index = meta.pop("index", 0)

            chunk = DocumentChunk(
                id=ids[i],
                document_id=document_id,
                content=documents[i],
                index=index,
                metadata=meta,  # Only extra metadata
            )
            scored_chunk = ScoredDocumentChunk(
                chunk=chunk,
                score=distances[i],  # Distance metric: lower is better
            )
            scored_chunks.append(scored_chunk)

        return scored_chunks

    def delete_by_document_ids(self, document_ids: list[str]) -> None:
        """Delete all chunks belonging to specified documents."""
        # TODO (F3): Implement deletion (optional for initial F3)
        raise NotImplementedError