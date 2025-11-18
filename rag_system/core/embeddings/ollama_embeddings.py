"""
Ollama embedding service with Matryoshka dimension consistency.

CRITICAL: Ensures consistent 768-dimensional embeddings for both indexing
and querying to avoid dimension mismatch errors in Qdrant.

Updated to use latest ollama>=0.4.4 API with batch embedding support.
"""

from typing import List

import ollama
from tqdm import tqdm


class OllamaEmbeddingService:
    """
    Embedding service using Ollama with nomic-embed-text.

    Handles batching and ensures consistent dimensionality (768) for
    Matryoshka embeddings to prevent dimension mismatch errors.

    Uses the latest ollama.embed() API with batch support for better performance.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text:latest",
        batch_size: int = 32,
        expected_dimension: int = 768,
    ):
        """
        Initialize the Ollama embedding service.

        Args:
            base_url: Ollama API base URL
            model: Embedding model name
            batch_size: Number of texts to embed in one batch
            expected_dimension: Expected embedding dimension (for validation)
        """
        self.base_url = base_url
        self.model = model
        self.batch_size = batch_size
        self.expected_dimension = expected_dimension

        # Initialize Ollama client
        self.client = ollama.Client(host=base_url)

        # Verify model is available
        self._verify_model()

    def _verify_model(self) -> None:
        """Verify that the embedding model is available."""
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

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts using the new ollama.embed() API.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        # Use the new embed() API with batch support
        response = self.client.embed(
            model=self.model,
            input=texts,  # Can be a single string or list of strings
        )

        # Extract embeddings from response
        if isinstance(response, dict):
            embeddings = response.get("embeddings", [])
        else:
            embeddings = getattr(response, "embeddings", [])

        # Validate dimensions for all embeddings
        for i, embedding in enumerate(embeddings):
            if len(embedding) != self.expected_dimension:
                raise ValueError(
                    f"Embedding dimension mismatch at index {i}: "
                    f"expected {self.expected_dimension}, got {len(embedding)}. "
                    f"Ensure consistent Matryoshka dimensions."
                )

        return embeddings

    def _embed_single(self, text: str) -> List[float]:
        """
        Embed a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        embeddings = self._embed_batch([text])
        return embeddings[0]

    def embed_texts(
        self, texts: List[str], show_progress: bool = True
    ) -> List[List[float]]:
        """
        Embed multiple texts with batching using the new ollama.embed() API.

        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar

        Returns:
            List of embedding vectors
        """
        embeddings = []

        # Process in batches using the new batch API
        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(
                iterator,
                desc="Generating embeddings",
                total=(len(texts) + self.batch_size - 1) // self.batch_size,
            )

        for i in iterator:
            batch = texts[i : i + self.batch_size]

            # Use batch embedding API for better performance
            batch_embeddings = self._embed_batch(batch)
            embeddings.extend(batch_embeddings)

        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query.
        
        Args:
            query: Query text to embed
            
        Returns:
            Embedding vector
        """
        return self._embed_single(query)

    def get_dimension(self) -> int:
        """
        Get the embedding dimension.
        
        Returns:
            Embedding dimension
        """
        return self.expected_dimension

