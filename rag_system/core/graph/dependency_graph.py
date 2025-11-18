"""
Dependency graph for tracking code relationships.

Builds and maintains a graph of file dependencies for intelligent context expansion.
"""

from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json


@dataclass
class DependencyEdge:
    """Represents a dependency relationship between files."""
    
    source: str  # Source file path
    target: str  # Target file path
    edge_type: str  # "import", "call", "inherit", "config"
    weight: float = 1.0  # Edge weight for ranking


@dataclass
class FileNode:
    """Represents a file in the dependency graph."""
    
    file_path: str
    language: str
    imports: List[str] = field(default_factory=list)  # Files this file imports
    imported_by: List[str] = field(default_factory=list)  # Files that import this
    defines: List[str] = field(default_factory=list)  # Functions/classes defined
    chunk_count: int = 0  # Number of chunks from this file


class DependencyGraph:
    """
    Tracks dependencies between code files.
    
    Enables multi-hop context expansion by following import relationships.
    """
    
    def __init__(self):
        """Initialize empty dependency graph."""
        self.nodes: Dict[str, FileNode] = {}
        self.edges: List[DependencyEdge] = []
        self._adjacency: Dict[str, Set[str]] = defaultdict(set)  # file -> {imported files}
        self._reverse_adjacency: Dict[str, Set[str]] = defaultdict(set)  # file -> {files that import it}
    
    def add_node(self, file_path: str, language: str) -> FileNode:
        """
        Add a file node to the graph.
        
        Args:
            file_path: Path to the file
            language: Programming language
            
        Returns:
            The created or existing FileNode
        """
        if file_path not in self.nodes:
            self.nodes[file_path] = FileNode(
                file_path=file_path,
                language=language,
            )
        return self.nodes[file_path]
    
    def add_edge(self, source: str, target: str, edge_type: str = "import", weight: float = 1.0):
        """
        Add a dependency edge between two files.
        
        Args:
            source: Source file path
            target: Target file path
            edge_type: Type of dependency
            weight: Edge weight
        """
        edge = DependencyEdge(
            source=source,
            target=target,
            edge_type=edge_type,
            weight=weight,
        )
        self.edges.append(edge)
        
        # Update adjacency lists
        self._adjacency[source].add(target)
        self._reverse_adjacency[target].add(source)
        
        # Update node metadata
        if source in self.nodes:
            if target not in self.nodes[source].imports:
                self.nodes[source].imports.append(target)
        
        if target in self.nodes:
            if source not in self.nodes[target].imported_by:
                self.nodes[target].imported_by.append(source)
    
    def get_neighbors(self, file_path: str, direction: str = "outgoing") -> List[str]:
        """
        Get neighboring files in the dependency graph.
        
        Args:
            file_path: File to get neighbors for
            direction: "outgoing" (imports), "incoming" (imported by), or "both"
            
        Returns:
            List of neighboring file paths
        """
        neighbors = set()
        
        if direction in ("outgoing", "both"):
            neighbors.update(self._adjacency.get(file_path, set()))
        
        if direction in ("incoming", "both"):
            neighbors.update(self._reverse_adjacency.get(file_path, set()))
        
        return list(neighbors)
    
    def multi_hop_expansion(
        self,
        seed_files: List[str],
        max_hops: int = 2,
        max_files: int = 20,
        direction: str = "both",
    ) -> List[str]:
        """
        Expand context by following dependency edges multiple hops.
        
        Args:
            seed_files: Initial files to expand from
            max_hops: Maximum number of hops to traverse
            max_files: Maximum number of files to return
            direction: "outgoing", "incoming", or "both"
            
        Returns:
            List of related file paths (including seed files)
        """
        visited = set(seed_files)
        queue = deque([(f, 0) for f in seed_files])  # (file, hop_count)
        result = list(seed_files)
        
        while queue and len(result) < max_files:
            current_file, hop_count = queue.popleft()
            
            if hop_count >= max_hops:
                continue
            
            # Get neighbors
            neighbors = self.get_neighbors(current_file, direction=direction)
            
            for neighbor in neighbors:
                if neighbor not in visited and len(result) < max_files:
                    visited.add(neighbor)
                    result.append(neighbor)
                    queue.append((neighbor, hop_count + 1))
        
        return result[:max_files]

    def get_file_importance(self, file_path: str) -> float:
        """
        Calculate importance score for a file based on graph centrality.

        Files that are imported by many others are more important.

        Args:
            file_path: File to calculate importance for

        Returns:
            Importance score (higher = more important)
        """
        if file_path not in self.nodes:
            return 0.0

        # Simple centrality: number of incoming edges (files that import this)
        incoming_count = len(self._reverse_adjacency.get(file_path, set()))

        # Bonus for files that also import many others (hub files)
        outgoing_count = len(self._adjacency.get(file_path, set()))

        return incoming_count + (outgoing_count * 0.5)

    def to_dict(self) -> dict:
        """Export graph to dictionary for serialization."""
        return {
            "nodes": {
                path: {
                    "file_path": node.file_path,
                    "language": node.language,
                    "imports": node.imports,
                    "imported_by": node.imported_by,
                    "defines": node.defines,
                    "chunk_count": node.chunk_count,
                }
                for path, node in self.nodes.items()
            },
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "edge_type": edge.edge_type,
                    "weight": edge.weight,
                }
                for edge in self.edges
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DependencyGraph":
        """Load graph from dictionary."""
        graph = cls()

        # Load nodes
        for path, node_data in data.get("nodes", {}).items():
            node = graph.add_node(node_data["file_path"], node_data["language"])
            node.imports = node_data.get("imports", [])
            node.imported_by = node_data.get("imported_by", [])
            node.defines = node_data.get("defines", [])
            node.chunk_count = node_data.get("chunk_count", 0)

        # Load edges
        for edge_data in data.get("edges", []):
            graph.add_edge(
                source=edge_data["source"],
                target=edge_data["target"],
                edge_type=edge_data.get("edge_type", "import"),
                weight=edge_data.get("weight", 1.0),
            )

        return graph

    def save_to_file(self, file_path: str):
        """Save graph to JSON file."""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, file_path: str) -> "DependencyGraph":
        """Load graph from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def get_stats(self) -> dict:
        """Get graph statistics."""
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "avg_imports_per_file": sum(len(node.imports) for node in self.nodes.values()) / max(len(self.nodes), 1),
            "most_imported_files": sorted(
                [(path, len(self._reverse_adjacency.get(path, set())))
                 for path in self.nodes.keys()],
                key=lambda x: x[1],
                reverse=True
            )[:10],
        }
