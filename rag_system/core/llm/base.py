"""
Base LLM service abstraction.

This module defines the abstract interface that all LLM services must implement.
This allows the system to support multiple LLM backends (Ollama, LiteLLM, etc.)
without changing the pipeline code.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any


class BaseLLMService(ABC):
    """
    Abstract base class for LLM services.
    
    All LLM implementations (Ollama, LiteLLM, etc.) must inherit from this class
    and implement the required methods.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        stream: bool = False,
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            stream: Whether to stream the response

        Returns:
            Generated text (or generator if streaming)
        """
        pass

    @abstractmethod
    def generate_with_context(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Generate a response with retrieved context and optional chat history.

        Args:
            query: User query
            context: Retrieved context from vector store
            system_prompt: Optional system prompt
            chat_history: Optional list of previous messages [{"role": "user/assistant", "content": "..."}]

        Returns:
            Generated text
        """
        pass

    @abstractmethod
    def _verify_model(self) -> None:
        """
        Verify that the LLM model is available and accessible.
        
        Raises:
            ValueError: If model is not available
            ConnectionError: If cannot connect to LLM service
        """
        pass

