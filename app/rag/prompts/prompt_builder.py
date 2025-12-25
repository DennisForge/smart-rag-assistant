"""Prompt construction for RAG queries."""


class PromptBuilder:
    """Builds prompts for LLM generation with retrieved context."""
    
    @staticmethod
    def build(context: str, query: str) -> str:
        """Build a prompt combining retrieved context and user query.
        
        Creates a simple, deterministic prompt that instructs the LLM to answer
        based solely on the provided context without using external knowledge.
        
        Args:
            context: Retrieved document chunks joined as context.
            query: User's question or query.
            
        Returns:
            Formatted prompt string ready for LLM generation.
        """
        return f"""Answer the question using only the information from the context below. If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {query}

Answer:"""
