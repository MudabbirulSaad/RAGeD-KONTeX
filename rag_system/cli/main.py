"""
Command-line interface for RAG system.

Provides commands for indexing codebases and querying the system.
"""

import click
from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from ..config import get_settings
from ..pipelines import IngestionPipeline, QueryPipeline
from ..core.graph import DependencyGraph
from ..core.storage import CodebaseManager


console = Console()


@click.group()
@click.version_option(version="2.0.0")
def cli():
    """
    Agentic RAG System for Local Codebase Indexing.

    Industry-standard RAG architecture using fully self-hosted stack.
    """
    pass


@cli.command()
@click.argument("codebase_path", type=click.Path(exists=True))
@click.option(
    "--name",
    type=str,
    default=None,
    help="Name for this codebase (default: directory name)",
)
@click.option(
    "--clear",
    is_flag=True,
    help="Clear existing collection before indexing",
)
def index(codebase_path: str, name: str, clear: bool):
    """
    Index a local codebase.

    CODEBASE_PATH: Path to the directory containing the code to index.

    Example:
        rag-system index /path/to/my/project --name "MyProject"
    """
    try:
        settings = get_settings()
        codebase_manager = CodebaseManager()

        # Resolve absolute path
        abs_path = str(Path(codebase_path).resolve())

        # Generate collection name
        collection_name = codebase_manager.generate_collection_name(abs_path)

        # Use directory name if no name provided
        if not name:
            name = Path(abs_path).name

        # Create pipeline with dynamic collection name
        pipeline = IngestionPipeline(settings, collection_name=collection_name)

        if clear:
            pipeline.clear_collection()

        # Run indexing
        pipeline.run(codebase_path)

        # Get stats from pipeline
        info = pipeline.vector_store.get_collection_info()
        file_count = len(pipeline.dependency_graph.nodes)
        chunk_count = info['points_count']

        # Save dependency graph path
        graph_path = str(Path.cwd() / ".rag_dependency_graph.json")

        # Register codebase
        codebase_manager.register_codebase(
            name=name,
            path=abs_path,
            collection_name=collection_name,
            file_count=file_count,
            chunk_count=chunk_count,
            graph_path=graph_path,
            set_active=True,
        )

        console.print(f"\n[bold green]✓ Registered codebase:[/bold green] {name}")
        console.print(f"Collection: {collection_name}")
        console.print(f"Active: Yes\n")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise click.Abort()


@cli.command()
@click.argument("query_text", required=False)
@click.option(
    "--top-k",
    type=int,
    default=None,
    help="Number of results to retrieve (default: from config)",
)
@click.option(
    "--file-filter",
    type=str,
    default=None,
    help="Filter results to specific file path",
)
@click.option(
    "--interactive",
    "-i",
    is_flag=True,
    help="Start interactive query mode",
)
@click.option(
    "--no-expansion",
    is_flag=True,
    help="Disable context expansion (use only semantic search)",
)
@click.option(
    "--graph-path",
    type=click.Path(exists=True),
    default=None,
    help="Path to dependency graph JSON file",
)
def query(query_text: str, top_k: int, file_filter: str, interactive: bool, no_expansion: bool, graph_path: str):
    """
    Query the indexed codebase.
    
    QUERY_TEXT: The question to ask about the codebase.
    
    Examples:
        rag-system query "Explain the authentication logic"
        rag-system query "How does the database connection work?" --top-k 5
        rag-system query --interactive
    """
    try:
        settings = get_settings()
        codebase_manager = CodebaseManager()

        # Get active codebase
        active_codebase = codebase_manager.get_active_codebase()

        if not active_codebase:
            console.print("\n[bold red]Error:[/bold red] No active codebase found.")
            console.print("Please index a codebase first:")
            console.print("  rag-system index /path/to/code --name 'MyProject'\n")
            raise click.Abort()

        console.print(f"[dim]Using codebase: {active_codebase.name}[/dim]")
        console.print(f"[dim]Collection: {active_codebase.collection_name}[/dim]\n")

        # Override settings with active codebase's collection
        settings.qdrant_collection_name = active_codebase.collection_name

        # Load dependency graph if available
        dependency_graph = None
        if not no_expansion:
            if graph_path:
                # Use specified graph path
                try:
                    dependency_graph = DependencyGraph.load_from_file(graph_path)
                    console.print(f"[green]✓ Loaded dependency graph from {graph_path}[/green]\n")
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not load graph from {graph_path}: {e}[/yellow]\n")
            elif active_codebase.graph_path and Path(active_codebase.graph_path).exists():
                # Use graph path from active codebase
                try:
                    dependency_graph = DependencyGraph.load_from_file(active_codebase.graph_path)
                    console.print(f"[green]✓ Loaded dependency graph from {active_codebase.graph_path}[/green]\n")
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not load graph: {e}[/yellow]\n")
            else:
                # Try to find graph in common locations
                # Check current directory and parent directories
                current = Path.cwd()
                possible_paths = []

                # Add current directory and up to 3 parent directories
                for i in range(4):
                    possible_paths.append(current / ".rag_dependency_graph.json")
                    if current.parent == current:  # Reached root
                        break
                    current = current.parent

                for path in possible_paths:
                    if path.exists():
                        try:
                            dependency_graph = DependencyGraph.load_from_file(str(path))
                            console.print(f"[green]✓ Loaded dependency graph from {path}[/green]\n")
                            break
                        except Exception as e:
                            console.print(f"[yellow]Warning: Found graph at {path} but could not load: {e}[/yellow]\n")
                            continue

                # If still not found, show helpful message
                if dependency_graph is None:
                    console.print("[yellow]Note: No dependency graph found. Context expansion disabled.[/yellow]")
                    console.print("[yellow]To enable expansion, index a codebase first or use --graph-path[/yellow]\n")

        pipeline = QueryPipeline(settings, dependency_graph=dependency_graph)
        
        if interactive:
            # Interactive mode
            console.print("\n[bold blue]Interactive Query Mode[/bold blue]")
            console.print("Type 'exit' or 'quit' to exit\n")

            # If query_text was provided with --interactive, ignore it
            # (user wants interactive mode, not a single query)

            while True:
                try:
                    user_query = click.prompt("Query", type=str)

                    if user_query.lower() in ["exit", "quit"]:
                        console.print("\n[yellow]Goodbye![/yellow]\n")
                        break

                    response = pipeline.query(
                        query_text=user_query,
                        top_k=top_k,
                        file_path_filter=file_filter,
                        enable_expansion=not no_expansion,
                        verbose=True,
                    )

                    console.print("\n[bold green]Response:[/bold green]")
                    console.print(Markdown(response))
                    console.print("\n" + "=" * 80 + "\n")

                except KeyboardInterrupt:
                    console.print("\n\n[yellow]Goodbye![/yellow]\n")
                    break
        else:
            # Single query mode
            if not query_text:
                console.print("[bold red]Error:[/bold red] Query text is required in non-interactive mode")
                raise click.Abort()
            
            response = pipeline.query(
                query_text=query_text,
                top_k=top_k,
                file_path_filter=file_filter,
                enable_expansion=not no_expansion,
                verbose=True,
            )
            
            console.print("\n[bold green]Response:[/bold green]")
            console.print(Markdown(response))
            console.print()
        
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise click.Abort()


@cli.command()
def info():
    """Show collection information."""
    try:
        settings = get_settings()
        from ..core.vectorstore import QdrantVectorStore
        
        vector_store = QdrantVectorStore(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection_name=settings.qdrant_collection_name,
            vector_dimension=settings.embedding_dimension,
        )
        
        info = vector_store.get_collection_info()
        
        console.print("\n[bold blue]Collection Information[/bold blue]")
        console.print(f"Name: {info['name']}")
        console.print(f"Vectors: {info['vectors_count']}")
        console.print(f"Points: {info['points_count']}\n")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise click.Abort()


@cli.command(name="list-codebases")
def list_codebases():
    """List all indexed codebases."""
    try:
        codebase_manager = CodebaseManager()
        codebases = codebase_manager.list_codebases()

        if not codebases:
            console.print("\n[yellow]No codebases indexed yet.[/yellow]")
            console.print("Use 'rag-system index <path> --name <name>' to index a codebase.\n")
            return

        # Create table
        table = Table(title="Indexed Codebases", show_header=True, header_style="bold magenta")
        table.add_column("Name", style="cyan")
        table.add_column("Path", style="white")
        table.add_column("Collection", style="green")
        table.add_column("Files", justify="right", style="yellow")
        table.add_column("Chunks", justify="right", style="yellow")
        table.add_column("Indexed", style="blue")
        table.add_column("Active", justify="center", style="bold green")

        for cb in codebases:
            table.add_row(
                cb.name,
                cb.path,
                cb.collection_name,
                str(cb.file_count),
                str(cb.chunk_count),
                cb.indexed_at.split('T')[0] if 'T' in cb.indexed_at else cb.indexed_at,
                "✓" if cb.is_active else "",
            )

        console.print()
        console.print(table)
        console.print()

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise click.Abort()


@cli.command(name="switch-codebase")
@click.argument("name", type=str)
def switch_codebase(name: str):
    """
    Switch to a different codebase.

    NAME: Name of the codebase to activate.

    Example:
        rag-system switch-codebase "MyProject"
    """
    try:
        codebase_manager = CodebaseManager()

        if codebase_manager.set_active_codebase(name):
            console.print(f"\n[bold green]✓ Switched to codebase:[/bold green] {name}\n")
        else:
            console.print(f"\n[bold red]Error:[/bold red] Codebase '{name}' not found.")
            console.print("Use 'rag-system list-codebases' to see available codebases.\n")
            raise click.Abort()

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise click.Abort()


@cli.command(name="delete-codebase")
@click.argument("name", type=str)
@click.option(
    "--yes",
    is_flag=True,
    help="Skip confirmation prompt",
)
def delete_codebase(name: str, yes: bool):
    """
    Delete a codebase from the registry.

    NAME: Name of the codebase to delete.

    Note: This also deletes the Qdrant collection.

    Example:
        rag-system delete-codebase "MyProject"
    """
    try:
        codebase_manager = CodebaseManager()

        # Get codebase info
        codebase = codebase_manager.get_codebase_by_name(name)
        if not codebase:
            console.print(f"\n[bold red]Error:[/bold red] Codebase '{name}' not found.\n")
            raise click.Abort()

        # Confirm deletion
        if not yes:
            console.print(f"\n[bold yellow]Warning:[/bold yellow] This will delete:")
            console.print(f"  - Codebase: {codebase.name}")
            console.print(f"  - Collection: {codebase.collection_name}")
            console.print(f"  - {codebase.chunk_count} chunks")

            if not click.confirm("\nAre you sure?", default=False):
                console.print("\n[yellow]Cancelled.[/yellow]\n")
                return

        # Delete Qdrant collection
        settings = get_settings()
        from ..core.vectorstore import QdrantVectorStore

        vector_store = QdrantVectorStore(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection_name=codebase.collection_name,
            vector_dimension=settings.embedding_dimension,
        )

        vector_store.delete_collection()

        # Delete from registry
        codebase_manager.delete_codebase(name)

        console.print(f"\n[bold green]✓ Deleted codebase:[/bold green] {name}\n")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise click.Abort()


if __name__ == "__main__":
    cli()

