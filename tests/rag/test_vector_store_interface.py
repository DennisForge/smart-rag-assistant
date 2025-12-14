from app.rag.interfaces.vector_store import VectorStoreInterface


def test_vector_store_interface_methods_exist() -> None:
    method_names = [m for m in dir(VectorStoreInterface) if not m.startswith("_")]

    assert "add_chunks" in method_names
    assert "query" in method_names
    assert "delete_by_document_ids" in method_names