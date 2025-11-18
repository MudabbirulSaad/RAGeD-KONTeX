"""
Ingestion pipeline for indexing codebase.

Orchestrates: Load → Chunk → Embed → Store
"""

from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..config import Settings
from ..core.loaders import CodeLoader
from ..core.chunkers import SemanticChunker
from ..core.embeddings import OllamaEmbeddingService
from ..core.vectorstore import QdrantVectorStore
from ..core.graph import DependencyGraph, ImportParser


class IngestionPipeline:
    """
    Offline batch ingestion pipeline.
    
    Loads code files, chunks them semantically, generates embeddings,
    and stores them in Qdrant with rich metadata.
    """

    def __init__(self, settings: Optional[Settings] = None, collection_name: Optional[str] = None):
        """
        Initialize the ingestion pipeline.

        Args:
            settings: Application settings (uses defaults if not provided)
            collection_name: Override collection name (for multi-codebase support)
        """
        from ..config import get_settings

        self.settings = settings or get_settings()
        self.console = Console()

        # Use provided collection name or default from settings
        self.collection_name = collection_name or self.settings.qdrant_collection_name

        # Initialize components
        self.loader = CodeLoader(
            supported_extensions=self.settings.supported_extensions,
            ignore_patterns=self.settings.ignore_patterns,
        )

        self.chunker = SemanticChunker(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap_size,
            min_chunk_size=self.settings.min_chunk_size,
        )

        self.embedding_service = OllamaEmbeddingService(
            base_url=self.settings.ollama_base_url,
            model=self.settings.ollama_embedding_model,
            batch_size=self.settings.embedding_batch_size,
            expected_dimension=self.settings.embedding_dimension,
        )

        self.vector_store = QdrantVectorStore(
            host=self.settings.qdrant_host,
            port=self.settings.qdrant_port,
            collection_name=self.collection_name,
            vector_dimension=self.settings.embedding_dimension,
        )

        # Initialize dependency graph
        self.dependency_graph = DependencyGraph()
        self.import_parser = ImportParser()

    def run(self, codebase_path: str) -> None:
        """
        Run the ingestion pipeline.
        
        Args:
            codebase_path: Path to the codebase directory
        """
        self.console.print(f"\n[bold blue]Starting ingestion pipeline for: {codebase_path}[/bold blue]\n")
        
        # Step 1: Load documents
        self.console.print("[bold]Step 1/4:[/bold] Loading code files...")
        documents = self.loader.load_documents(codebase_path)
        self.console.print(f"✓ Loaded {len(documents)} documents\n")
        
        if not documents:
            self.console.print("[yellow]No documents found. Check your file filters.[/yellow]")
            return
        
        # Step 2: Build dependency graph and chunk documents
        self.console.print("[bold]Step 2/5:[/bold] Building dependency graph...")
        self._build_dependency_graph(documents)
        self.console.print(f"✓ Built graph with {len(self.dependency_graph.nodes)} nodes\n")

        # Step 3: Chunk documents
        self.console.print("[bold]Step 3/5:[/bold] Chunking documents (AST-based)...")
        all_chunks = []
        for doc in documents:
            chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(chunks)

            # Update chunk count in graph
            if doc.relative_path in self.dependency_graph.nodes:
                self.dependency_graph.nodes[doc.relative_path].chunk_count = len(chunks)

        self.console.print(f"✓ Created {len(all_chunks)} semantic chunks\n")
        
        if not all_chunks:
            self.console.print("[yellow]No chunks created. Documents may be too small.[/yellow]")
            return
        
        # Step 4: Generate embeddings
        self.console.print("[bold]Step 4/5:[/bold] Generating embeddings...")
        chunk_texts = [chunk.content for chunk in all_chunks]
        embeddings = self.embedding_service.embed_texts(chunk_texts, show_progress=True)
        self.console.print(f"✓ Generated {len(embeddings)} embeddings\n")

        # Step 5: Store in Qdrant with graph metadata
        self.console.print("[bold]Step 5/5:[/bold] Storing in Qdrant...")
        self.vector_store.upsert_chunks(all_chunks, embeddings, show_progress=True)

        # Save dependency graph
        from pathlib import Path

        # Save in the codebase directory
        graph_path_in_codebase = f"{codebase_path}/.rag_dependency_graph.json"
        try:
            self.dependency_graph.save_to_file(graph_path_in_codebase)
            self.console.print(f"✓ Saved dependency graph to {graph_path_in_codebase}")
        except Exception as e:
            self.console.print(f"[yellow]Warning: Could not save dependency graph to codebase: {e}[/yellow]")

        # Also save in current directory for convenience
        graph_path_current = ".rag_dependency_graph.json"
        try:
            self.dependency_graph.save_to_file(graph_path_current)
            self.console.print(f"✓ Saved dependency graph to {Path.cwd() / graph_path_current}\n")
        except Exception as e:
            self.console.print(f"[yellow]Warning: Could not save dependency graph to current directory: {e}[/yellow]\n")
        
        # Show collection info
        info = self.vector_store.get_collection_info()
        self.console.print(f"\n[bold green]✓ Ingestion complete![/bold green]")
        self.console.print(f"Collection: {info['name']}")
        self.console.print(f"Total vectors: {info['vectors_count']}")
        self.console.print(f"Total points: {info['points_count']}\n")

    def _build_dependency_graph(self, documents) -> None:
        """
        Build dependency graph from loaded documents.

        Args:
            documents: List of CodeDocument objects
        """
        # First pass: Add all nodes
        for doc in documents:
            self.dependency_graph.add_node(doc.relative_path, doc.language)

        # Second pass: Parse imports and add edges
        total_imports = 0
        resolved_imports = 0

        for doc in documents:
            imports = self.import_parser.parse_imports(doc.content, doc.language)
            total_imports += len(imports)

            for import_info in imports:
                # Try to resolve import to a file in the codebase
                target_file = self._resolve_import_to_file(
                    doc.relative_path,
                    import_info.module,
                    import_info.is_relative,
                    import_info.level,
                    documents,
                )

                if target_file:
                    self.dependency_graph.add_edge(
                        source=doc.relative_path,
                        target=target_file,
                        edge_type="import",
                    )
                    resolved_imports += 1

        # Show import resolution statistics
        if total_imports > 0:
            resolution_rate = (resolved_imports / total_imports) * 100
            self.console.print(f"  Parsed {total_imports} imports, resolved {resolved_imports} ({resolution_rate:.1f}%) to local files")

    def _resolve_import_to_file(
        self,
        current_file: str,
        import_module: str,
        is_relative: bool,
        level: int,
        documents,
    ) -> Optional[str]:
        """
        Resolve an import statement to an actual file in the codebase.

        Args:
            current_file: File containing the import
            import_module: Module being imported
            is_relative: Whether it's a relative import
            level: Relative import level
            documents: List of all documents

        Returns:
            Relative path to the imported file, or None if not found
        """
        # Build set of available files for quick lookup
        # Normalize all paths to use forward slashes
        available_files = {doc.relative_path.replace('\\', '/') for doc in documents}

        if is_relative:
            # Resolve relative import using string manipulation (not Path objects)
            # to avoid Windows path issues
            current_file_normalized = current_file.replace('\\', '/')
            parts = current_file_normalized.split('/')

            # Remove filename, keep directory parts
            dir_parts = parts[:-1]

            # Go up 'level' directories
            for _ in range(level):
                if dir_parts:
                    dir_parts.pop()

            # Combine with import module
            if import_module:
                # Replace dots with slashes for module path
                module_parts = import_module.split('/')
                target_parts = dir_parts + module_parts
            else:
                target_parts = dir_parts

            target_path = '/'.join(target_parts)
        else:
            # Absolute import - try to find matching file
            # For Next.js style imports like '@/utilities/...'
            if import_module.startswith('@/'):
                # Remove '@/' prefix and treat as relative to src or root
                target_path = import_module[2:]
            else:
                # Regular absolute import
                target_path = import_module.replace('.', '/')

        # Try different file extensions
        for ext in ['.tsx', '.ts', '.jsx', '.js', '.py', '/__init__.py', '/index.tsx', '/index.ts', '/index.jsx', '/index.js']:
            candidate = target_path + ext

            if candidate in available_files:
                return candidate.replace('/', '\\') if '\\' in current_file else candidate

        # Try as directory with index file
        for init_file in ['__init__.py', 'index.tsx', 'index.ts', 'index.jsx', 'index.js']:
            candidate = f"{target_path}/{init_file}"
            if candidate in available_files:
                return candidate.replace('/', '\\') if '\\' in current_file else candidate

        return None

    def clear_collection(self) -> None:
        """Clear the vector store collection."""
        self.console.print("[yellow]Clearing collection...[/yellow]")
        self.vector_store.delete_collection()
        # Recreate collection
        self.vector_store._ensure_collection()
        self.console.print("[green]✓ Collection cleared[/green]")

