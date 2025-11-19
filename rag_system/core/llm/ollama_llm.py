"""
Ollama LLM service with proper context window configuration.

CRITICAL: Sets num_ctx=8192 to handle large retrieved contexts.
Default 2048 would make RAG pipeline useless for large files.

Updated to use latest ollama>=0.4.4 API with chat() method.
"""

from typing import Optional, Dict, Any, Generator, List

import ollama

from .base import BaseLLMService


class OllamaLLMService(BaseLLMService):
    """
    LLM service using Ollama with gpt-oss:20b or alternative models.

    Configured with large context window (8192) to handle retrieved
    code chunks effectively.

    Uses the latest ollama.chat() API for better compatibility.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "gpt-oss:20b",
        context_window: int = 8192,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        """
        Initialize the Ollama LLM service.

        Args:
            base_url: Ollama API base URL
            model: LLM model name
            context_window: Context window size (num_ctx)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.base_url = base_url
        self.model = model
        self.context_window = context_window
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize Ollama client
        self.client = ollama.Client(host=base_url)

        # Verify model is available
        self._verify_model()

    def _verify_model(self) -> None:
        """Verify that the LLM model is available."""
        try:
            models = self.client.list()

            # Extract models list from the response
            # New API returns ListResponse object with 'models' attribute
            if hasattr(models, 'models'):
                models_list = models.models
            elif isinstance(models, dict):
                models_list = models.get("models", [])
            else:
                models_list = []

            # Extract model names from Model objects
            # New API: Model objects have 'model' attribute (not 'name')
            model_names = []
            for m in models_list:
                if hasattr(m, 'model'):
                    model_names.append(m.model)
                elif isinstance(m, dict):
                    model_names.append(m.get("model", m.get("name", "")))
                else:
                    model_names.append(str(m))

            if self.model not in model_names:
                raise ValueError(
                    f"Model '{self.model}' not found in Ollama. "
                    f"Available models: {model_names}. "
                    f"Please run: ollama pull {self.model}"
                )
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to Ollama at {self.base_url}: {e}"
            )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        stream: bool = False,
    ) -> str:
        """
        Generate a response from the LLM using the new ollama.chat() API.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            stream: Whether to stream the response

        Returns:
            Generated text (or generator if streaming)
        """
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })

        messages.append({
            "role": "user",
            "content": prompt,
        })

        # Generate response using the new chat() API
        response = self.client.chat(
            model=self.model,
            messages=messages,
            stream=stream,
            options={
                "num_ctx": self.context_window,  # CRITICAL: Large context window
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        )

        if stream:
            # Return generator for streaming
            return response
        else:
            # Handle different response formats
            if isinstance(response, dict):
                return response.get("message", {}).get("content", "")
            else:
                # Handle ChatResponse object
                return getattr(getattr(response, "message", None), "content", "")

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
        messages = []

        # Add system prompt
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })

        # Add chat history if provided
        if chat_history:
            messages.extend(chat_history)

        # Build current prompt with context
        prompt = f"""Context from codebase:

{context}

Question: {query}

Please provide a detailed answer based on the context above. Include specific file paths and line numbers when referencing code."""

        messages.append({
            "role": "user",
            "content": prompt,
        })

        # Generate response
        response = self.client.chat(
            model=self.model,
            messages=messages,
            stream=False,
            options={
                "num_ctx": self.context_window,
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        )

        # Extract response text
        if isinstance(response, dict):
            return response.get("message", {}).get("content", "")
        else:
            return getattr(getattr(response, "message", None), "content", "")

