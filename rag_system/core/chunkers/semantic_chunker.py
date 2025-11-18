"""
AST-based semantic chunker for code.

Uses language-specific splitters to ensure chunks don't break in the middle
of functions or classes. Critical for maintaining semantic integrity.

Updated to use langchain-text-splitters package (v0.3.8+).
"""

from typing import List, Optional

from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)
from pydantic import BaseModel, Field

from ..loaders.code_loader import CodeDocument


class CodeChunk(BaseModel):
    """Represents a semantically meaningful chunk of code."""

    content: str = Field(description="Chunk content")
    file_path: str = Field(description="Source file path")
    relative_path: str = Field(description="Relative path from codebase root")
    language: str = Field(description="Programming language")
    start_line: int = Field(description="Starting line number (1-indexed)")
    end_line: int = Field(description="Ending line number (1-indexed)")
    chunk_index: int = Field(description="Index of this chunk in the document")

    class Config:
        frozen = True


class SemanticChunker:
    """
    AST-based semantic chunker that respects code structure.
    
    Uses LangChain's RecursiveCharacterTextSplitter with language-specific
    separators to avoid breaking functions/classes mid-definition.
    """

    # Map our language names to LangChain's Language enum
    LANGUAGE_MAP = {
        "python": Language.PYTHON,
        "javascript": Language.JS,
        "typescript": Language.TS,
        "java": Language.JAVA,
        "cpp": Language.CPP,
        "c": Language.CPP,  # Use CPP splitter for C
        "go": Language.GO,
        "rust": Language.RUST,
        "markdown": Language.MARKDOWN,
    }

    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 102,  # 10% of 1024
        min_chunk_size: int = 512,
    ):
        """
        Initialize the semantic chunker.
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap size between chunks
            min_chunk_size: Minimum chunk size
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self._splitters = {}

    def _get_splitter(self, language: str) -> RecursiveCharacterTextSplitter:
        """
        Get or create a language-specific text splitter.
        
        Args:
            language: Programming language name
            
        Returns:
            RecursiveCharacterTextSplitter configured for the language
        """
        if language in self._splitters:
            return self._splitters[language]

        # Get LangChain language enum
        lang_enum = self.LANGUAGE_MAP.get(language)

        if lang_enum:
            # Use language-specific splitter (AST-aware)
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=lang_enum,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        else:
            # Fallback to generic splitter for unknown languages
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", " ", ""],
            )

        self._splitters[language] = splitter
        return splitter

    def _calculate_line_numbers(
        self, content: str, chunk_text: str, previous_end_line: int
    ) -> tuple[int, int]:
        """
        Calculate start and end line numbers for a chunk.
        
        Args:
            content: Full document content
            chunk_text: Chunk text
            previous_end_line: End line of previous chunk
            
        Returns:
            Tuple of (start_line, end_line)
        """
        # Find chunk position in content
        chunk_start_pos = content.find(chunk_text, 0)
        
        if chunk_start_pos == -1:
            # Fallback if exact match not found
            return previous_end_line + 1, previous_end_line + chunk_text.count("\n") + 1
        
        # Count newlines before chunk
        lines_before = content[:chunk_start_pos].count("\n")
        start_line = lines_before + 1
        
        # Count newlines in chunk
        lines_in_chunk = chunk_text.count("\n")
        end_line = start_line + lines_in_chunk
        
        return start_line, end_line

    def chunk_document(self, document: CodeDocument) -> List[CodeChunk]:
        """
        Chunk a code document into semantic chunks.
        
        Args:
            document: CodeDocument to chunk
            
        Returns:
            List of CodeChunk objects
        """
        # Get language-specific splitter
        splitter = self._get_splitter(document.language)
        
        # Split the content
        texts = splitter.split_text(document.content)
        
        # Create CodeChunk objects with line numbers
        chunks = []
        previous_end_line = 0
        
        for idx, text in enumerate(texts):
            # Skip chunks that are too small
            if len(text.strip()) < self.min_chunk_size:
                continue
            
            start_line, end_line = self._calculate_line_numbers(
                document.content, text, previous_end_line
            )
            
            chunk = CodeChunk(
                content=text,
                file_path=document.file_path,
                relative_path=document.relative_path,
                language=document.language,
                start_line=start_line,
                end_line=end_line,
                chunk_index=idx,
            )
            
            chunks.append(chunk)
            previous_end_line = end_line
        
        return chunks

