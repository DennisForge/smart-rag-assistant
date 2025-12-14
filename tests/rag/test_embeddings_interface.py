from app.rag.interfaces.embeddings import EmbeddingInterface


def test_embedding_interface_methods_exist() -> None:
    assert hasattr(EmbeddingInterface, "embed_text")
    assert hasattr(EmbeddingInterface, "embed_texts")