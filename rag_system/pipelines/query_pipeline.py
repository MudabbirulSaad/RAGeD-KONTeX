"""
Query pipeline for RAG retrieval and generation.

Orchestrates: Query → Embed → Search → Assemble → Generate → Response
"""

from typing import Optional, List, Dict, Any

from rich.console import Console

from ..config import Settings
from ..core.embeddings import OllamaEmbeddingService
from ..core.vectorstore import QdrantVectorStore
from ..core.llm.factory import get_llm_service
from ..core.graph import DependencyGraph
from ..core.expansion import ContextExpander, ExpansionStrategy
from ..core.reconstruction import FileContextReconstructor, ExpandedContext


class ContextAssembler:
    """
    Assembles retrieved chunks into formatted context for LLM.

    Formats chunks with file metadata (path, line numbers) for precise
    code location references.
    """

    @staticmethod
    def assemble_context(search_results: List[Dict[str, Any]]) -> str:
        """
        Assemble search results into formatted context (legacy method).

        Args:
            search_results: List of search results from Qdrant

        Returns:
            Formatted context string
        """
        if not search_results:
            return "No relevant code found."

        context_parts = []

        for idx, result in enumerate(search_results, 1):
            payload = result["payload"]
            score = result["score"]

            # Format each chunk with metadata
            chunk_context = f"""
--- Result {idx} (Similarity: {score:.3f}) ---
File: {payload['relative_path']}
Lines: {payload['start_line']}-{payload['end_line']}
Language: {payload['language']}

```{payload['language']}
{payload['content']}
```
"""
            context_parts.append(chunk_context)

        return "\n".join(context_parts)

    @staticmethod
    def assemble_expanded_context(expanded_contexts: List[ExpandedContext]) -> str:
        """
        Assemble expanded contexts from file reconstruction.

        Args:
            expanded_contexts: List of ExpandedContext objects

        Returns:
            Formatted context string with complete code
        """
        if not expanded_contexts:
            return "No relevant code found."

        context_parts = []

        for idx, ctx in enumerate(expanded_contexts, 1):
            # Show expansion indicator
            expansion_note = ""
            if ctx.is_complete_file:
                expansion_note = " [COMPLETE FILE]"
            else:
                original_start, original_end = ctx.original_chunk_lines
                expansion_note = f" [Expanded from lines {original_start}-{original_end}]"

            # Format with complete context
            chunk_context = f"""
--- Result {idx} (Similarity: {ctx.score:.3f}){expansion_note} ---
File: {ctx.relative_path}
Lines: {ctx.start_line}-{ctx.end_line}
Language: {ctx.language}

```{ctx.language}
{ctx.content}
```
"""
            context_parts.append(chunk_context)

        return "\n".join(context_parts)


class QueryPipeline:
    """
    Online real-time query pipeline.
    
    Embeds query, searches vector store, assembles context,
    and generates response using LLM.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        collection_name: str = "codebase_default",
        dependency_graph: Optional[DependencyGraph] = None,
        codebase_root: Optional[str] = None,
    ):
        """
        Initialize the query pipeline.

        Args:
            settings: Application settings (uses defaults if not provided)
            collection_name: Collection name to query (required for multi-codebase support)
            dependency_graph: Dependency graph for context expansion (optional)
            codebase_root: Root directory of the codebase for file reconstruction (optional)
        """
        from ..config import get_settings

        self.settings = settings or get_settings()
        self.console = Console()
        self.codebase_root = codebase_root

        # Initialize components
        self.embedding_service = OllamaEmbeddingService(
            base_url=self.settings.ollama_base_url,
            model=self.settings.ollama_embedding_model,
            batch_size=self.settings.embedding_batch_size,
            expected_dimension=self.settings.embedding_dimension,
        )

        self.vector_store = QdrantVectorStore(
            host=self.settings.qdrant_host,
            port=self.settings.qdrant_port,
            collection_name=collection_name,
            vector_dimension=self.settings.embedding_dimension,
        )

        # Use factory to create LLM service based on settings
        self.llm_service = get_llm_service(self.settings)

        self.context_assembler = ContextAssembler()

        # Initialize context expander
        self.dependency_graph = dependency_graph
        self.context_expander = ContextExpander(dependency_graph=dependency_graph)

        # Initialize file context reconstructor
        self.file_reconstructor = None
        if codebase_root and self.settings.enable_file_reconstruction:
            self.file_reconstructor = FileContextReconstructor(
                codebase_root=codebase_root,
                context_lines=self.settings.context_lines_before_after,
            )

        # Chat history for interactive mode
        self.chat_history: List[Dict[str, str]] = []

    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        file_path_filter: Optional[str] = None,
        enable_expansion: bool = True,
        expansion_strategy: ExpansionStrategy = ExpansionStrategy.BIDIRECTIONAL,
        verbose: bool = True,
        use_chat_history: bool = False,
    ) -> str:
        """
        Execute a query against the indexed codebase.

        Args:
            query_text: User query
            top_k: Number of results to retrieve (uses config default if None)
            file_path_filter: Optional file path to filter results
            enable_expansion: Whether to expand context using dependency graph
            expansion_strategy: Strategy for context expansion
            verbose: Whether to print progress
            use_chat_history: Whether to include chat history in LLM context (for interactive mode)

        Returns:
            Generated response
        """
        if verbose:
            self.console.print(f"\n[bold blue]Query:[/bold blue] {query_text}\n")
        
        # Use config default if not specified
        top_k = top_k or self.settings.top_k_results
        
        # Determine total steps
        total_steps = 6 if self.file_reconstructor else 5

        # Step 1: Embed query
        if verbose:
            self.console.print(f"[bold]Step 1/{total_steps}:[/bold] Embedding query...")
        query_vector = self.embedding_service.embed_query(query_text)
        if verbose:
            self.console.print("✓ Query embedded\n")

        # Step 2: Search vector store
        if verbose:
            self.console.print(f"[bold]Step 2/{total_steps}:[/bold] Searching vector store...")
        search_results = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            score_threshold=self.settings.search_score_threshold,
            file_path_filter=file_path_filter,
        )
        if verbose:
            self.console.print(f"✓ Found {len(search_results)} relevant chunks\n")

        if not search_results:
            return "No relevant code found for your query. Try rephrasing or check if the codebase is indexed."

        # Step 3: Expand context using dependency graph
        if enable_expansion and self.dependency_graph:
            if verbose:
                self.console.print(f"[bold]Step 3/{total_steps}:[/bold] Expanding context via dependency graph...")

            search_results = self.context_expander.expand_results(
                initial_results=search_results,
                vector_store=self.vector_store,
                query_vector=query_vector,
                strategy=expansion_strategy,
                max_additional_files=5,
                max_chunks_per_file=2,
                score_threshold=self.settings.search_score_threshold * 0.8,  # Slightly lower threshold for expanded results
            )

            if verbose:
                expansion_summary = self.context_expander.get_expansion_summary(search_results)
                self.console.print(
                    f"✓ Expanded to {expansion_summary['unique_files']} files "
                    f"({expansion_summary['initial_results']} initial + {expansion_summary['expanded_results']} expanded)\n"
                )
        else:
            if verbose:
                if enable_expansion:
                    self.console.print(f"[yellow]Step 3/{total_steps}: Context expansion skipped (no dependency graph loaded)[/yellow]\n")
                else:
                    self.console.print(f"[dim]Step 3/{total_steps}: Context expansion disabled[/dim]\n")

        # Step 4: Reconstruct complete context from files (NEW!)
        if self.file_reconstructor:
            if verbose:
                self.console.print(f"[bold]Step 4/{total_steps}:[/bold] Reconstructing complete context from files...")

            expanded_contexts = self.file_reconstructor.reconstruct_context(
                search_results=search_results,
                max_lines_per_file=self.settings.max_lines_per_file,
            )

            if verbose:
                complete_files = sum(1 for ctx in expanded_contexts if ctx.is_complete_file)
                self.console.print(
                    f"✓ Reconstructed {len(expanded_contexts)} file contexts "
                    f"({complete_files} complete files, {len(expanded_contexts) - complete_files} expanded)\n"
                )

            # Use expanded contexts for assembly
            if verbose:
                self.console.print(f"[bold]Step 5/{total_steps}:[/bold] Assembling context...")
            context = self.context_assembler.assemble_expanded_context(expanded_contexts)
            if verbose:
                self.console.print("✓ Context assembled\n")
        else:
            # Fallback to legacy chunk-based assembly
            if verbose:
                self.console.print(f"[bold]Step 4/{total_steps}:[/bold] Assembling context...")
            context = self.context_assembler.assemble_context(search_results)
            if verbose:
                self.console.print("✓ Context assembled\n")

        # Step 5/6: Generate response
        if verbose:
            self.console.print(f"[bold]Step {total_steps}/{total_steps}:[/bold] Generating response...\n")
        
        system_prompt = """You are an expert code assistant. Your task is to answer questions about a codebase based on the provided context.

Always reference specific files and line numbers when discussing code.
Be precise and technical in your explanations.
If the context doesn't contain enough information, say so clearly."""

        # Pass chat history if enabled
        chat_history_to_use = self.chat_history if use_chat_history else None

        response = self.llm_service.generate_with_context(
            query=query_text,
            context=context,
            system_prompt=system_prompt,
            chat_history=chat_history_to_use,
        )

        # Update chat history if enabled
        if use_chat_history:
            self.chat_history.append({"role": "user", "content": query_text})
            self.chat_history.append({"role": "assistant", "content": response})

        return response

    def clear_chat_history(self):
        """Clear the chat history."""
        self.chat_history.clear()

    def get_chat_history_length(self) -> int:
        """Get the number of messages in chat history."""
        return len(self.chat_history)

