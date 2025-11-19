"""
LLM service factory.

This module provides a factory function to create the appropriate LLM service
based on configuration settings.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...config.settings import Settings

from .base import BaseLLMService
from .ollama_llm import OllamaLLMService


def get_llm_service(settings: "Settings") -> BaseLLMService:
    """
    Factory function to create the appropriate LLM service based on settings.

    Args:
        settings: Application settings containing llm_backend configuration

    Returns:
        An instance of BaseLLMService (OllamaLLMService or LiteLLMService)

    Raises:
        ValueError: If llm_backend is not supported
        ImportError: If required package is not installed
    """
    backend = settings.llm_backend.lower()

    if backend == "ollama":
        return OllamaLLMService(
            base_url=settings.ollama_base_url,
            model=settings.ollama_llm_model,
            context_window=settings.llm_context_window,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
    elif backend == "litellm":
        # Import here to avoid requiring litellm if not used
        try:
            from .litellm_llm import LiteLLMService
        except ImportError:
            raise ImportError(
                "LiteLLM backend requires 'litellm' package. "
                "Install it with: pip install litellm"
            )

        return LiteLLMService(
            model=settings.litellm_model,
            api_key=settings.litellm_api_key,
            api_base=settings.litellm_api_base,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
    else:
        raise ValueError(
            f"Unsupported LLM backend: '{backend}'. "
            f"Supported backends: 'ollama', 'litellm'"
        )

