from app.rag.interfaces.embeddings import EmbeddingInterface
from app.rag.interfaces.vector_store import VectorStoreInterface
from app.rag.services.indexing import IndexingService


class _DummyEmbedder(EmbeddingInterface):
    def embed_text(self, text: str):
        raise NotImplementedError

    def embed_texts(self, texts):
        raise NotImplementedError


class _DummyVectorStore(VectorStoreInterface):
    def add_chunks(self, chunks, embeddings):
        raise NotImplementedError

    def query(self, embedding, top_k: int = 5):
        raise NotImplementedError

    def delete_by_document_ids(self, document_ids):
        raise NotImplementedError


def test_indexing_service_can_be_constructed() -> None:
    service = IndexingService(
        embedder=_DummyEmbedder(),
        vector_store=_DummyVectorStore(),
    )
    assert service is not None