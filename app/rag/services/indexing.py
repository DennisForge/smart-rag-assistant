# TODO (F3): Define IndexingService class that uses EmbeddingInterface and VectorStoreInterface
# Methods to implement:
# - __init__(embedding_model, vector_store, chunk_size, chunk_overlap)
# - index_document(document: DocumentIn) -> DocumentStored
# - index_documents_batch(documents: List[DocumentIn]) -> List[DocumentStored]
# - search(query: str, top_k: int, filters) -> List[SimilarityResult]
# - delete_document(document_id: str) -> bool
