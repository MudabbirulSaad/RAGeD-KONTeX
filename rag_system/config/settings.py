"""
Configuration settings for the RAG system.

Loads settings from environment variables with sensible defaults.
"""

from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Ollama Configuration
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for Ollama API",
    )
    ollama_embedding_model: str = Field(
        default="nomic-embed-text:latest",
        description="Ollama embedding model name",
    )
    ollama_llm_model: str = Field(
        default="gpt-oss:20b",
        description="Ollama LLM model name",
    )

    # Qdrant Configuration
    qdrant_host: str = Field(
        default="localhost",
        description="Qdrant server host",
    )
    qdrant_port: int = Field(
        default=6333,
        description="Qdrant server port",
    )

    # Embedding Configuration
    embedding_dimension: int = Field(
        default=768,
        description="Embedding vector dimension (Matryoshka consistency)",
    )
    embedding_batch_size: int = Field(
        default=32,
        description="Batch size for embedding generation",
    )

    # Chunking Configuration
    chunk_size: int = Field(
        default=1024,
        description="Target chunk size in tokens",
    )
    chunk_overlap_percent: int = Field(
        default=10,
        description="Overlap percentage between chunks",
    )
    min_chunk_size: int = Field(
        default=512,
        description="Minimum chunk size in tokens",
    )

    # Retrieval Configuration
    top_k_results: int = Field(
        default=10,
        description="Number of top results to retrieve",
    )
    search_score_threshold: float = Field(
        default=0.3,  # Lowered from 0.7 - cosine similarity scores are typically lower
        description="Minimum similarity score for retrieval",
    )

    # File Context Reconstruction Configuration
    enable_file_reconstruction: bool = Field(
        default=True,
        description="Enable two-step retrieval with file I/O reads for complete context",
    )
    context_lines_before_after: int = Field(
        default=10,
        description="Number of lines to expand before/after retrieved chunks",
    )
    max_lines_per_file: int = Field(
        default=100,
        description="Maximum lines to include per file in reconstructed context",
    )

    # LLM Configuration
    llm_context_window: int = Field(
        default=8192,
        description="LLM context window size (num_ctx)",
    )
    llm_temperature: float = Field(
        default=0.1,
        description="LLM temperature for generation",
    )
    llm_max_tokens: int = Field(
        default=2048,
        description="Maximum tokens for LLM response",
    )

    # File Filtering
    supported_extensions: List[str] = Field(
        default=[
            # Code files
            ".py", ".js", ".ts", ".jsx", ".tsx",
            ".java", ".cpp", ".c", ".h", ".hpp",
            ".go", ".rs", ".rb", ".php",
            # Config files
            ".json", ".yaml", ".yml", ".toml", ".env",
            # Web files
            ".html", ".css", ".scss",
            # Documentation
            ".md", ".txt",
            # Scripts
            ".sh", ".sql",
        ],
        description="Supported file extensions for indexing",
    )
    ignore_patterns: List[str] = Field(
        default=[
            "node_modules",
            ".git",
            "__pycache__",
            ".venv",
            "venv",
            "dist",
            "build",
            ".next",
            "target",
            "*.pyc",
            "*.pyo",
            "*.so",
            "*.dylib",
            "*.dll",
        ],
        description="Patterns to ignore during file loading",
    )

    @property
    def chunk_overlap_size(self) -> int:
        """Calculate chunk overlap size based on percentage."""
        return int(self.chunk_size * self.chunk_overlap_percent / 100)

    @property
    def qdrant_url(self) -> str:
        """Get full Qdrant URL."""
        return f"http://{self.qdrant_host}:{self.qdrant_port}"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Returns:
        Settings: Application settings
    """
    return Settings()

