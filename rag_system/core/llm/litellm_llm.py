"""
LiteLLM service for unified access to 100+ LLM providers.

Supports OpenAI, Anthropic, Google, Azure, AWS Bedrock, Cohere, and many more
through a single unified interface.

Uses litellm.completion() which provides OpenAI-compatible API for all providers.
"""

from typing import Optional, List, Dict, Any

try:
    import litellm
except ImportError:
    raise ImportError(
        "litellm is not installed. Install it with: pip install litellm"
    )

from .base import BaseLLMService


class LiteLLMService(BaseLLMService):
    """
    LLM service using LiteLLM for unified access to multiple providers.
    
    Supports 100+ LLM providers including:
    - OpenAI (gpt-4, gpt-3.5-turbo, etc.)
    - Anthropic (claude-3-opus, claude-3-sonnet, etc.)
    - Google (gemini-pro, palm-2, etc.)
    - Azure OpenAI
    - AWS Bedrock
    - Cohere
    - Hugging Face
    - Ollama (as alternative to direct ollama client)
    - And many more...
    
    Model format: "provider/model-name"
    Examples:
        - "openai/gpt-4"
        - "anthropic/claude-3-opus-20240229"
        - "gemini/gemini-pro"
        - "ollama/llama2"
    """

    def __init__(
        self,
        model: str = "openai/gpt-4",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        """
        Initialize the LiteLLM service.

        Args:
            model: Model identifier in format "provider/model-name"
            api_key: API key for the provider (can also be set via environment variables)
            api_base: Optional custom API base URL
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Set API key if provided
        if api_key:
            # LiteLLM uses environment variables, but we can pass api_key directly
            pass

        # Verify model is accessible
        self._verify_model()

    def _verify_model(self) -> None:
        """
        Verify that the LLM model is accessible.
        
        Note: LiteLLM doesn't have a list models API, so we do a minimal test call.
        """
        try:
            # Make a minimal test call to verify the model works
            test_response = litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
                api_key=self.api_key,
                api_base=self.api_base,
            )
            # If we get here, the model is accessible
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to model '{self.model}': {e}\n"
                f"Make sure your API key is set correctly and the model name is valid."
            )

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

        # Generate response using litellm.completion()
        response = litellm.completion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=stream,
            api_key=self.api_key,
            api_base=self.api_base,
        )

        if stream:
            # Return generator for streaming
            return response
        else:
            # Extract content from response
            return response.choices[0].message.content

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
        response = litellm.completion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=False,
            api_key=self.api_key,
            api_base=self.api_base,
        )

        # Extract response text
        return response.choices[0].message.content


