"""RAG orchestration service combining retrieval and LLM generation."""

from app.rag.services.retrieval_service import RetrievalService
from app.rag.prompts.prompt_builder import PromptBuilder
from app.rag.llm.interfaces.llm_interface import LLMInterface


class RAGLLMService:
    """Orchestrates RAG pipeline: retrieval → prompt building → LLM generation."""
    
    def __init__(
        self,
        retrieval_service: RetrievalService,
        llm: LLMInterface
    ) -> None:
        """Initialize RAG LLM service.
        
        Args:
            retrieval_service: Service for retrieving relevant context (F4).
            llm: LLM provider interface for generating responses.
        """
        self.retrieval_service = retrieval_service
        self.llm = llm
    
    def answer(self, query: str) -> str:
        """Generate answer for a query using RAG pipeline.
        
        Retrieves relevant context, builds a prompt, and generates an answer
        using the configured LLM provider.
        
        Args:
            query: User's question or query.
            
        Returns:
            Generated answer from the LLM.
        """
        results, context = self.retrieval_service.retrieve_with_context(query)
        prompt = PromptBuilder.build(context=context, query=query)
        answer = self.llm.generate(prompt)
        return answer
