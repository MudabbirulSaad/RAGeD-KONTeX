"""
Context expander for multi-file intelligent retrieval.

Expands initial search results by following dependency relationships.
"""

from typing import List, Dict, Any, Optional, Set
from enum import Enum
from collections import defaultdict

from ..graph import DependencyGraph


class ExpansionStrategy(Enum):
    """Strategy for context expansion."""
    
    NONE = "none"  # No expansion, use only initial results
    IMPORTS = "imports"  # Expand to imported files
    IMPORTED_BY = "imported_by"  # Expand to files that import these
    BIDIRECTIONAL = "bidirectional"  # Both imports and imported_by
    MULTI_HOP = "multi_hop"  # Follow multiple hops in the graph


class ContextExpander:
    """
    Expands search results using dependency graph.
    
    Takes initial semantic search results and intelligently expands
    to related files for more complete context.
    """
    
    def __init__(self, dependency_graph: Optional[DependencyGraph] = None):
        """
        Initialize context expander.
        
        Args:
            dependency_graph: Dependency graph for expansion (optional)
        """
        self.dependency_graph = dependency_graph
    
    def expand_results(
        self,
        initial_results: List[Dict[str, Any]],
        vector_store,
        query_vector: List[float],
        strategy: ExpansionStrategy = ExpansionStrategy.BIDIRECTIONAL,
        max_additional_files: int = 5,
        max_chunks_per_file: int = 2,
        score_threshold: float = 0.25,
    ) -> List[Dict[str, Any]]:
        """
        Expand search results to include related files.
        
        Args:
            initial_results: Initial search results from vector store
            vector_store: Vector store for additional queries
            query_vector: Original query embedding
            strategy: Expansion strategy
            max_additional_files: Maximum number of additional files to include
            max_chunks_per_file: Maximum chunks per additional file
            score_threshold: Minimum score for additional chunks
            
        Returns:
            Expanded list of search results
        """
        if not self.dependency_graph or strategy == ExpansionStrategy.NONE:
            return initial_results
        
        if not initial_results:
            return []
        
        # Extract files from initial results
        initial_files = self._extract_files(initial_results)
        
        # Find related files based on strategy
        related_files = self._find_related_files(initial_files, strategy, max_additional_files)
        
        if not related_files:
            return initial_results
        
        # Retrieve additional chunks from related files
        additional_results = self._retrieve_from_files(
            related_files,
            vector_store,
            query_vector,
            max_chunks_per_file,
            score_threshold,
        )
        
        # Combine and deduplicate results
        combined_results = self._combine_results(initial_results, additional_results)
        
        return combined_results
    
    def _extract_files(self, results: List[Dict[str, Any]]) -> Set[str]:
        """Extract unique file paths from search results."""
        files = set()
        for result in results:
            file_path = result["payload"].get("relative_path")
            if file_path:
                files.add(file_path)
        return files
    
    def _find_related_files(
        self,
        seed_files: Set[str],
        strategy: ExpansionStrategy,
        max_files: int,
    ) -> List[str]:
        """
        Find related files based on expansion strategy.
        
        Args:
            seed_files: Initial set of files
            strategy: Expansion strategy
            max_files: Maximum number of related files to return
            
        Returns:
            List of related file paths
        """
        if not self.dependency_graph:
            return []
        
        related = set()
        
        for file_path in seed_files:
            if strategy == ExpansionStrategy.IMPORTS:
                # Get files that this file imports
                neighbors = self.dependency_graph.get_neighbors(file_path, direction="outgoing")
                related.update(neighbors)
            
            elif strategy == ExpansionStrategy.IMPORTED_BY:
                # Get files that import this file
                neighbors = self.dependency_graph.get_neighbors(file_path, direction="incoming")
                related.update(neighbors)
            
            elif strategy == ExpansionStrategy.BIDIRECTIONAL:
                # Get both imports and imported_by
                neighbors = self.dependency_graph.get_neighbors(file_path, direction="both")
                related.update(neighbors)
            
            elif strategy == ExpansionStrategy.MULTI_HOP:
                # Multi-hop expansion
                expanded = self.dependency_graph.multi_hop_expansion(
                    seed_files=[file_path],
                    max_hops=2,
                    max_files=max_files,
                    direction="both",
                )
                related.update(expanded)
        
        # Remove files that were in the initial results
        related = related - seed_files
        
        # Rank by importance and return top-k
        ranked = self._rank_files_by_importance(list(related))
        return ranked[:max_files]

    def _rank_files_by_importance(self, files: List[str]) -> List[str]:
        """
        Rank files by importance score.

        Args:
            files: List of file paths

        Returns:
            Sorted list of file paths (most important first)
        """
        if not self.dependency_graph:
            return files

        # Calculate importance score for each file
        scored_files = [
            (file_path, self.dependency_graph.get_file_importance(file_path))
            for file_path in files
        ]

        # Sort by score (descending)
        scored_files.sort(key=lambda x: x[1], reverse=True)

        return [file_path for file_path, _ in scored_files]

    def _retrieve_from_files(
        self,
        file_paths: List[str],
        vector_store,
        query_vector: List[float],
        max_chunks_per_file: int,
        score_threshold: float,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks from specific files.

        Args:
            file_paths: List of file paths to retrieve from
            vector_store: Vector store instance
            query_vector: Query embedding
            max_chunks_per_file: Maximum chunks per file
            score_threshold: Minimum similarity score

        Returns:
            List of search results
        """
        all_results = []

        for file_path in file_paths:
            # Search with file path filter
            results = vector_store.search(
                query_vector=query_vector,
                top_k=max_chunks_per_file,
                score_threshold=score_threshold,
                file_path_filter=file_path,
            )
            all_results.extend(results)

        return all_results

    def _combine_results(
        self,
        initial_results: List[Dict[str, Any]],
        additional_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Combine and deduplicate results.

        Args:
            initial_results: Initial search results
            additional_results: Additional expanded results

        Returns:
            Combined and deduplicated results
        """
        # Use set to track seen IDs
        seen_ids = set()
        combined = []

        # Add initial results first (higher priority)
        for result in initial_results:
            result_id = result.get("id")
            if result_id not in seen_ids:
                seen_ids.add(result_id)
                combined.append(result)

        # Add additional results
        for result in additional_results:
            result_id = result.get("id")
            if result_id not in seen_ids:
                seen_ids.add(result_id)
                # Mark as expanded result
                result["is_expanded"] = True
                combined.append(result)

        return combined

    def get_expansion_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get summary of expansion results.

        Args:
            results: Search results (possibly expanded)

        Returns:
            Summary dictionary
        """
        files_by_type = defaultdict(list)

        for result in results:
            file_path = result["payload"].get("relative_path")
            is_expanded = result.get("is_expanded", False)

            if is_expanded:
                files_by_type["expanded"].append(file_path)
            else:
                files_by_type["initial"].append(file_path)

        return {
            "total_results": len(results),
            "initial_results": len(files_by_type["initial"]),
            "expanded_results": len(files_by_type["expanded"]),
            "unique_files": len(set(r["payload"].get("relative_path") for r in results)),
            "initial_files": list(set(files_by_type["initial"])),
            "expanded_files": list(set(files_by_type["expanded"])),
        }
