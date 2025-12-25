"""F5 smoke test: RAG LLM service integration."""

from app.rag.services.rag_llm_service import RAGLLMService
from app.rag.llm.interfaces.llm_interface import LLMInterface


class FakeLLM(LLMInterface):
    """Fake LLM for testing without network calls."""
    
    def generate(self, prompt: str) -> str:
        """Return fixed response."""
        return "This is a test answer."


class FakeRetrievalService:
    """Fake retrieval service for testing."""
    
    def retrieve_with_context(self, query: str) -> tuple[list, str]:
        """Return dummy context."""
        return ([], "dummy context")


def test_rag_llm_service_returns_string():
    """Test that RAG LLM service returns a non-empty string."""
    fake_retrieval = FakeRetrievalService()
    fake_llm = FakeLLM()
    
    service = RAGLLMService(retrieval_service=fake_retrieval, llm=fake_llm)
    
    result = service.answer("any question")
    
    assert isinstance(result, str)
    assert result != ""
