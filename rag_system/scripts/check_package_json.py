#!/usr/bin/env python3
"""Check if package.json was indexed."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from qdrant_client import QdrantClient
from rich.console import Console

console = Console()

def check_package_json():
    """Check if package.json is in the collection."""
    console.print("\n[bold]Checking for package.json in collection[/bold]\n")
    
    try:
        client = QdrantClient(url="http://localhost:6333")
        
        # Scroll through all points and look for package.json
        offset = None
        package_json_points = []
        
        while True:
            points, next_offset = client.scroll(
                collection_name="codebase_v1",
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            
            if not points:
                break
            
            for point in points:
                if 'package.json' in point.payload.get('file_path', ''):
                    package_json_points.append(point)
            
            if next_offset is None:
                break
            offset = next_offset
        
        console.print(f"Found {len(package_json_points)} chunks from package.json\n")
        
        if package_json_points:
            for i, point in enumerate(package_json_points, 1):
                console.print(f"[bold]Chunk {i}:[/bold]")
                console.print(f"  File: {point.payload.get('relative_path', 'N/A')}")
                console.print(f"  Lines: {point.payload.get('start_line', 'N/A')}-{point.payload.get('end_line', 'N/A')}")
                console.print(f"  Content preview: {point.payload.get('content', '')[:200]}...")
                console.print()
        else:
            console.print("[yellow]No package.json chunks found in the collection![/yellow]")
            
            # Check if any JSON files are indexed
            json_points = []
            offset = None
            
            while True:
                points, next_offset = client.scroll(
                    collection_name="codebase_v1",
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                
                if not points:
                    break
                
                for point in points:
                    if point.payload.get('language') == 'json':
                        json_points.append(point)
                
                if next_offset is None:
                    break
                offset = next_offset
            
            console.print(f"\nFound {len(json_points)} JSON chunks total:")
            for point in json_points[:5]:  # Show first 5
                console.print(f"  - {point.payload.get('relative_path', 'N/A')}")
        
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    check_package_json()

