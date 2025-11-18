"""
File context reconstructor for complete code retrieval.

Implements two-step retrieval:
1. Vector search finds relevant chunks (fast semantic filtering)
2. File I/O reads complete context around chunks (syntactic completeness)

This solves the "Code Chunking Trap" by ensuring the LLM sees complete
functions, classes, imports, and surrounding context.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ExpandedContext:
    """
    Represents expanded context from a file.
    
    Contains complete code context reconstructed from the file system,
    not just the fragmented chunk from vector search.
    """
    file_path: str
    relative_path: str
    language: str
    start_line: int
    end_line: int
    content: str
    is_complete_file: bool
    original_chunk_lines: Tuple[int, int]  # Original chunk boundaries from vector search
    score: float  # Similarity score from vector search


class FileContextReconstructor:
    """
    Reconstructs complete code context from file system.
    
    Uses metadata from vector search (file_path, start_line, end_line) to:
    1. Read the raw file from disk
    2. Expand context around the retrieved chunk
    3. Ensure syntactic completeness (full functions, imports, etc.)
    
    This is the "Step 2" of the two-step retrieval system.
    """
    
    def __init__(self, codebase_root: str, context_lines: int = 10):
        """
        Initialize reconstructor.
        
        Args:
            codebase_root: Root directory of the codebase
            context_lines: Number of lines to expand before/after chunk
        """
        self.codebase_root = Path(codebase_root)
        self.context_lines = context_lines
        self._file_cache: Dict[str, List[str]] = {}
    
    def reconstruct_context(
        self,
        search_results: List[Dict[str, Any]],
        max_lines_per_file: int = 100,
    ) -> List[ExpandedContext]:
        """
        Reconstruct complete context from search results.
        
        Args:
            search_results: Results from vector search (with metadata)
            max_lines_per_file: Maximum lines to include per file
            
        Returns:
            List of expanded contexts with complete code
        """
        expanded_contexts = []
        
        for result in search_results:
            payload = result["payload"]
            score = result.get("score", 0.0)
            
            # Extract metadata from vector search
            file_path = payload["file_path"]
            start_line = payload["start_line"]
            end_line = payload["end_line"]
            
            # Read file and expand context
            expanded = self._expand_chunk_context(
                file_path=file_path,
                relative_path=payload["relative_path"],
                language=payload["language"],
                chunk_start=start_line,
                chunk_end=end_line,
                max_lines=max_lines_per_file,
                score=score,
            )
            
            if expanded:
                expanded_contexts.append(expanded)
        
        return expanded_contexts
    
    def _expand_chunk_context(
        self,
        file_path: str,
        relative_path: str,
        language: str,
        chunk_start: int,
        chunk_end: int,
        max_lines: int,
        score: float,
    ) -> Optional[ExpandedContext]:
        """
        Expand context around a chunk by reading the raw file.
        
        Args:
            file_path: Absolute path to the file
            relative_path: Relative path from codebase root
            language: Programming language
            chunk_start: Start line of the chunk (1-indexed)
            chunk_end: End line of the chunk (1-indexed)
            max_lines: Maximum lines to include
            score: Similarity score from vector search
            
        Returns:
            ExpandedContext with complete code, or None if file not found
        """
        # Read file (with caching)
        lines = self._read_file(file_path)
        if not lines:
            return None
        
        total_lines = len(lines)
        
        # Calculate expansion boundaries
        expand_start = max(1, chunk_start - self.context_lines)
        expand_end = min(total_lines, chunk_end + self.context_lines)
        
        # If file is small, include entire file
        if total_lines <= max_lines:
            expand_start = 1
            expand_end = total_lines
            is_complete = True
        else:
            # Ensure we don't exceed max_lines
            if (expand_end - expand_start + 1) > max_lines:
                # Center around the chunk
                mid = (chunk_start + chunk_end) // 2
                expand_start = max(1, mid - max_lines // 2)
                expand_end = min(total_lines, expand_start + max_lines - 1)
            is_complete = False

        # Extract lines (1-indexed to 0-indexed)
        content_lines = lines[expand_start - 1:expand_end]
        content = "".join(content_lines)  # Lines already have \n

        return ExpandedContext(
            file_path=file_path,
            relative_path=relative_path,
            language=language,
            start_line=expand_start,
            end_line=expand_end,
            content=content,
            is_complete_file=is_complete,
            original_chunk_lines=(chunk_start, chunk_end),
            score=score,
        )

    def _read_file(self, file_path: str) -> Optional[List[str]]:
        """
        Read file with caching.

        Args:
            file_path: Absolute path to the file

        Returns:
            List of lines (with newlines), or None if file not found
        """
        if file_path in self._file_cache:
            return self._file_cache[file_path]

        try:
            path = Path(file_path)
            if not path.exists():
                return None

            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            self._file_cache[file_path] = lines
            return lines
        except Exception:
            return None

    def clear_cache(self):
        """Clear the file cache."""
        self._file_cache.clear()

    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        return {
            "cached_files": len(self._file_cache),
            "total_lines": sum(len(lines) for lines in self._file_cache.values()),
        }

