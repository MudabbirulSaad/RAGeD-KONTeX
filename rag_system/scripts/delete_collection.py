#!/usr/bin/env python3
"""Script to delete the Qdrant collection."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from qdrant_client import QdrantClient
from rich.console import Console

console = Console()

def delete_collection():
    """Delete the collection."""
    console.print("\n[bold]Deleting Collection[/bold]\n")
    
    try:
        client = QdrantClient(url="http://localhost:6333")
        
        # Check if collection exists
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if "codebase_v1" in collection_names:
            console.print("Deleting collection: codebase_v1...")
            client.delete_collection(collection_name="codebase_v1")
            console.print("✓ Collection deleted successfully!\n")
        else:
            console.print("Collection 'codebase_v1' does not exist.\n")
        
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    delete_collection()

