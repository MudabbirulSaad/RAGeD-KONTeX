"""
Codebase metadata manager using SQLite.

Manages multiple indexed codebases with isolated Qdrant collections.
"""

import sqlite3
import hashlib
import re
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass
from datetime import datetime


@dataclass
class CodebaseInfo:
    """Information about an indexed codebase."""
    id: int
    name: str
    path: str
    collection_name: str
    indexed_at: str
    file_count: int
    chunk_count: int
    graph_path: Optional[str]
    is_active: bool


class CodebaseManager:
    """
    Manages multiple indexed codebases.
    
    Features:
    - Unique collection names per codebase
    - Active codebase selection
    - Metadata tracking
    - CRUD operations
    """
    
    def __init__(self, db_path: str = ".rag_codebases.db"):
        """
        Initialize codebase manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self) -> None:
        """Create database schema if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS codebases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                path TEXT NOT NULL UNIQUE,
                collection_name TEXT NOT NULL UNIQUE,
                indexed_at TIMESTAMP NOT NULL,
                file_count INTEGER NOT NULL DEFAULT 0,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                graph_path TEXT,
                is_active BOOLEAN NOT NULL DEFAULT 0
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_active ON codebases(is_active)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_name ON codebases(name)")
        
        conn.commit()
        conn.close()
    
    def generate_collection_name(self, codebase_path: str) -> str:
        """
        Generate unique collection name from codebase path.
        
        Args:
            codebase_path: Absolute path to codebase
            
        Returns:
            Collection name (e.g., "codebase_portfolio_z1_a3f2b1c4")
        """
        # Normalize path
        path = Path(codebase_path).resolve()
        
        # Get directory name
        dir_name = path.name
        
        # Sanitize name (alphanumeric + underscore only)
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', dir_name).lower()
        sanitized = re.sub(r'_+', '_', sanitized).strip('_')
        
        # Generate hash of full path for uniqueness
        path_hash = hashlib.sha256(str(path).encode()).hexdigest()[:8]
        
        return f"codebase_{sanitized}_{path_hash}"
    
    def register_codebase(
        self,
        name: str,
        path: str,
        collection_name: str,
        file_count: int,
        chunk_count: int,
        graph_path: Optional[str] = None,
        set_active: bool = True,
    ) -> int:
        """
        Register a new codebase or update existing one.
        
        Args:
            name: User-friendly name
            path: Absolute path to codebase
            collection_name: Qdrant collection name
            file_count: Number of indexed files
            chunk_count: Number of chunks created
            graph_path: Path to dependency graph JSON
            set_active: Set as active codebase
            
        Returns:
            Codebase ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Normalize path
        path = str(Path(path).resolve())

        # Check if codebase already exists by name or path
        cursor.execute("SELECT id FROM codebases WHERE name = ? OR path = ?", (name, path))
        existing = cursor.fetchone()

        if existing:
            # Update existing
            codebase_id = existing[0]
            cursor.execute("""
                UPDATE codebases
                SET name = ?, path = ?, collection_name = ?, indexed_at = ?,
                    file_count = ?, chunk_count = ?, graph_path = ?
                WHERE id = ?
            """, (name, path, collection_name, datetime.now().isoformat(),
                  file_count, chunk_count, graph_path, codebase_id))
        else:
            # Insert new
            cursor.execute("""
                INSERT INTO codebases (name, path, collection_name, indexed_at,
                                     file_count, chunk_count, graph_path, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0)
            """, (name, path, collection_name, datetime.now().isoformat(),
                  file_count, chunk_count, graph_path))
            codebase_id = cursor.lastrowid
        
        if set_active:
            # Deactivate all others
            cursor.execute("UPDATE codebases SET is_active = 0")
            # Activate this one
            cursor.execute("UPDATE codebases SET is_active = 1 WHERE id = ?", (codebase_id,))
        
        conn.commit()
        conn.close()

        return codebase_id

    def get_active_codebase(self) -> Optional[CodebaseInfo]:
        """
        Get the currently active codebase.

        Returns:
            CodebaseInfo or None if no active codebase
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, name, path, collection_name, indexed_at,
                   file_count, chunk_count, graph_path, is_active
            FROM codebases
            WHERE is_active = 1
            LIMIT 1
        """)

        row = cursor.fetchone()
        conn.close()

        if row:
            return CodebaseInfo(*row)
        return None

    def get_codebase_by_name(self, name: str) -> Optional[CodebaseInfo]:
        """Get codebase by name."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, name, path, collection_name, indexed_at,
                   file_count, chunk_count, graph_path, is_active
            FROM codebases
            WHERE name = ?
        """, (name,))

        row = cursor.fetchone()
        conn.close()

        if row:
            return CodebaseInfo(*row)
        return None

    def list_codebases(self) -> List[CodebaseInfo]:
        """List all registered codebases."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, name, path, collection_name, indexed_at,
                   file_count, chunk_count, graph_path, is_active
            FROM codebases
            ORDER BY indexed_at DESC
        """)

        rows = cursor.fetchall()
        conn.close()

        return [CodebaseInfo(*row) for row in rows]

    def set_active_codebase(self, name: str) -> bool:
        """
        Set a codebase as active.

        Args:
            name: Codebase name

        Returns:
            True if successful, False if codebase not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if exists
        cursor.execute("SELECT id FROM codebases WHERE name = ?", (name,))
        if not cursor.fetchone():
            conn.close()
            return False

        # Deactivate all
        cursor.execute("UPDATE codebases SET is_active = 0")

        # Activate target
        cursor.execute("UPDATE codebases SET is_active = 1 WHERE name = ?", (name,))

        conn.commit()
        conn.close()

        return True

    def delete_codebase(self, name: str) -> bool:
        """
        Delete a codebase from registry.

        Note: This does NOT delete the Qdrant collection.
        Use QdrantVectorStore.delete_collection() separately.

        Args:
            name: Codebase name

        Returns:
            True if deleted, False if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM codebases WHERE name = ?", (name,))
        deleted = cursor.rowcount > 0

        conn.commit()
        conn.close()

        return deleted

