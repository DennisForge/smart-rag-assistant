"""Ollama LLM provider implementation."""

import json
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from ..interfaces.llm_interface import LLMInterface


class OllamaProvider(LLMInterface):
    """Ollama local LLM provider."""
    
    def __init__(self, base_url: str, model: str) -> None:
        """Initialize Ollama provider.
        
        Args:
            base_url: Base URL of the Ollama server (e.g., "http://localhost:11434").
            model: Name of the model to use (e.g., "llama2", "mistral").
        """
        self.base_url = base_url
        self.model = model
    
    def generate(self, prompt: str) -> str:
        """Generate text completion from a prompt.
        
        Args:
            prompt: Input text prompt to send to the LLM.
            
        Returns:
            Generated text response from the LLM.
            
        Raises:
            RuntimeError: If Ollama request fails or returns unexpected response.
        """
        url = f"{self.base_url.rstrip('/')}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            req = Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            with urlopen(req, timeout=30) as response:
                if response.status != 200:
                    raise RuntimeError(f"Ollama returned status {response.status}")
                
                data = json.loads(response.read().decode('utf-8'))
                
                if "response" not in data:
                    raise RuntimeError("Unexpected Ollama response: missing 'response' field")
                
                return data["response"]
                
        except HTTPError as e:
            raise RuntimeError(f"Ollama HTTP error: {e.code} {e.reason}")
        except URLError as e:
            raise RuntimeError(f"Ollama connection error: {e.reason}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON from Ollama: {e}")
