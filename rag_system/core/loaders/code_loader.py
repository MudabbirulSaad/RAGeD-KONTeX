"""
Code file loader with filtering and ignore patterns.

Loads code files from a local directory while respecting ignore patterns
and file extension filters.
"""

import fnmatch
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field


class CodeDocument(BaseModel):
    """Represents a loaded code document."""

    file_path: str = Field(description="Absolute path to the file")
    relative_path: str = Field(description="Path relative to the codebase root")
    content: str = Field(description="File content")
    language: str = Field(description="Programming language (inferred from extension)")
    size_bytes: int = Field(description="File size in bytes")

    class Config:
        frozen = True


class CodeLoader:
    """
    Loads code files from a directory with filtering.
    
    Filters files by extension and ignores patterns like node_modules, .git, etc.
    """

    # Language mapping from file extensions
    LANGUAGE_MAP = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".java": "java",
        ".cpp": "cpp",
        ".c": "c",
        ".h": "c",
        ".hpp": "cpp",
        ".md": "markdown",
        ".txt": "text",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".xml": "xml",
        ".html": "html",
        ".css": "css",
        ".scss": "scss",
        ".sql": "sql",
        ".sh": "bash",
        ".env": "text",
    }

    def __init__(
        self,
        supported_extensions: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None,
    ):
        """
        Initialize the code loader.
        
        Args:
            supported_extensions: List of file extensions to load (e.g., ['.py', '.js'])
            ignore_patterns: List of patterns to ignore (e.g., ['node_modules', '*.pyc'])
        """
        self.supported_extensions = supported_extensions or [
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
        ]
        self.ignore_patterns = ignore_patterns or [
            "node_modules",
            ".git",
            "__pycache__",
            ".venv",
            "venv",
            "dist",
            "build",
            ".next",
            "target",
        ]

    def _should_ignore(self, path: Path, root: Path) -> bool:
        """
        Check if a path should be ignored based on ignore patterns.
        
        Args:
            path: Path to check
            root: Root directory for relative path calculation
            
        Returns:
            True if path should be ignored, False otherwise
        """
        relative = path.relative_to(root)
        path_str = str(relative)
        
        # Check each part of the path against ignore patterns
        for part in relative.parts:
            for pattern in self.ignore_patterns:
                if fnmatch.fnmatch(part, pattern):
                    return True
        
        # Check full path against patterns
        for pattern in self.ignore_patterns:
            if fnmatch.fnmatch(path_str, pattern):
                return True
        
        return False

    def _get_language(self, file_path: Path) -> str:
        """
        Infer programming language from file extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Language name
        """
        suffix = file_path.suffix.lower()
        return self.LANGUAGE_MAP.get(suffix, "unknown")

    def load_documents(self, directory: str) -> List[CodeDocument]:
        """
        Load all code documents from a directory.
        
        Args:
            directory: Path to the directory to load
            
        Returns:
            List of CodeDocument objects
        """
        root = Path(directory).resolve()
        
        if not root.exists():
            raise ValueError(f"Directory does not exist: {directory}")
        
        if not root.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")
        
        documents = []
        
        # Walk through directory
        for file_path in root.rglob("*"):
            # Skip directories
            if not file_path.is_file():
                continue
            
            # Check if should be ignored
            if self._should_ignore(file_path, root):
                continue
            
            # Check extension
            if file_path.suffix.lower() not in self.supported_extensions:
                continue
            
            # Try to read file
            try:
                content = file_path.read_text(encoding="utf-8")
                
                doc = CodeDocument(
                    file_path=str(file_path),
                    relative_path=str(file_path.relative_to(root)),
                    content=content,
                    language=self._get_language(file_path),
                    size_bytes=file_path.stat().st_size,
                )
                
                documents.append(doc)
                
            except (UnicodeDecodeError, PermissionError) as e:
                # Skip files that can't be read
                print(f"Warning: Could not read {file_path}: {e}")
                continue
        
        return documents

