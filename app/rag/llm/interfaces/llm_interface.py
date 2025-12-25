"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod


class LLMInterface(ABC):
    """Interface contract for LLM providers.
    
    Defines the contract that all LLM provider implementations must follow.
    Providers must implement text generation from prompts without knowledge
    of RAG-specific concerns.
    """
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate text completion from a prompt.
        
        Args:
            prompt: Input text prompt to send to the LLM.
            
        Returns:
            Generated text response from the LLM.
        """
        pass
