"""
Qdrant vector store with rich payload metadata.

CRITICAL: Stores file_path, start_line, end_line, and content in payload
so the LLM can point users to exact locations in the codebase.
"""

from typing import List, Optional, Dict, Any
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from tqdm import tqdm

from ..chunkers.semantic_chunker import CodeChunk


class QdrantVectorStore:
    """
    Qdrant vector store for code chunks.
    
    Stores vectors with rich metadata including file paths and line numbers
    for precise code location references.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "codebase_v1",
        vector_dimension: int = 768,
    ):
        """
        Initialize Qdrant vector store.
        
        Args:
            host: Qdrant server host
            port: Qdrant server port
            collection_name: Name of the collection
            vector_dimension: Dimension of embedding vectors
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_dimension = vector_dimension
        
        # Initialize client
        self.client = QdrantClient(host=host, port=port)
        
        # Ensure collection exists
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_dimension,
                    distance=Distance.COSINE,  # Cosine similarity
                ),
            )
            print(f"Created collection: {self.collection_name}")
        else:
            print(f"Collection already exists: {self.collection_name}")

    def upsert_chunks(
        self,
        chunks: List[CodeChunk],
        embeddings: List[List[float]],
        show_progress: bool = True,
    ) -> None:
        """
        Upsert code chunks with embeddings into Qdrant.
        
        Args:
            chunks: List of CodeChunk objects
            embeddings: List of embedding vectors
            show_progress: Whether to show progress bar
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks and embeddings length mismatch: {len(chunks)} vs {len(embeddings)}"
            )
        
        points = []
        
        iterator = zip(chunks, embeddings)
        if show_progress:
            iterator = tqdm(
                iterator,
                total=len(chunks),
                desc="Preparing points for Qdrant",
            )
        
        for chunk, embedding in iterator:
            # Create payload with all metadata
            payload = {
                "file_path": chunk.file_path,
                "relative_path": chunk.relative_path,
                "language": chunk.language,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
            }
            
            point = PointStruct(
                id=str(uuid4()),
                vector=embedding,
                payload=payload,
            )
            
            points.append(point)
        
        # Upsert in batches
        batch_size = 100
        for i in tqdm(
            range(0, len(points), batch_size),
            desc="Upserting to Qdrant",
            disable=not show_progress,
        ):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        score_threshold: Optional[float] = None,
        file_path_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar code chunks.

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            file_path_filter: Optional file path to filter results

        Returns:
            List of search results with payload and score
        """
        # Build filter if file path specified
        query_filter = None
        if file_path_filter:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="relative_path",
                        match=MatchValue(value=file_path_filter),
                    )
                ]
            )

        # Perform search using new API (qdrant-client v1.15+)
        # Changed from client.search() to client.query_points()
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            score_threshold=score_threshold,
        )

        # Format results
        formatted_results = []
        for point in response.points:
            formatted_results.append({
                "id": point.id,
                "score": point.score,
                "payload": point.payload,
            })

        return formatted_results

    def delete_collection(self) -> None:
        """Delete the collection."""
        self.client.delete_collection(collection_name=self.collection_name)
        print(f"Deleted collection: {self.collection_name}")

    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information."""
        info = self.client.get_collection(collection_name=self.collection_name)
        return {
            "name": self.collection_name,
            "vectors_count": info.indexed_vectors_count,  # Updated for qdrant-client v1.15+
            "points_count": info.points_count,
            "status": info.status,
            "segments_count": info.segments_count,
        }

