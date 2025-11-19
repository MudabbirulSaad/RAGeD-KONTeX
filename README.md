# RAG System for Local Codebase Indexing

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

RAG (Retrieval-Augmented Generation) system for indexing and querying local codebases using a self-hosted stack (Ollama + Qdrant). No external API dependencies.

## Features

- **Two-Step Retrieval**: Vector search for semantic filtering + file I/O for complete code context
- **Multi-Codebase Support**: Index multiple projects with isolated Qdrant collections
- **Graph-Based Expansion**: Includes related files via import dependency analysis
- **AST-Based Chunking**: Uses LangChain's language-specific splitters to respect code structure
- **File Context Reconstruction**: Expands ±10 lines around retrieved chunks for syntactic completeness
- **Interactive Mode**: Chat interface with conversation history
- **Local-Only**: All processing happens on your machine (Ollama + Qdrant)
- **Multi-Language**: Python, JavaScript, TypeScript, Java, C++, Go, Rust, etc.

## Technology Stack

| Component      | Technology              | Purpose                  |
| -------------- | ----------------------- | ------------------------ |
| LLM            | Ollama (gpt-oss:20b)    | Code understanding       |
| Embeddings     | nomic-embed-text:latest | 768-dim vectors          |
| Vector Store   | Qdrant                  | Vector database          |
| Chunking       | LangChain               | AST-based code splitting |
| Metadata Store | SQLite                  | Codebase registry        |

## Architecture

### Ingestion Pipeline

```
Codebase → Load → Build Graph → Chunk (AST) → Embed → Store (Qdrant)
                        ↓
                Dependency Graph (.json)
```

1. Load files (filter by extension, ignore patterns)
2. Parse imports and build dependency graph
3. Chunk using AST-based splitters (1024 chars, 10% overlap)
4. Generate embeddings (768-dim via Ollama)
5. Store in Qdrant with metadata (file_path, start_line, end_line)

### Query Pipeline (Two-Step Retrieval)

```
Query → Embed → Search → Expand → Reconstruct → Assemble → LLM → Response
                   ↓        ↓          ↓
              Top-K    Related    Complete
              Chunks   Files      Context
```

1. Embed query (768-dim)
2. Search Qdrant (cosine similarity, top-K)
3. Expand via dependency graph (1-2 hops)
4. Reconstruct: Read raw files, expand ±10 lines around chunks
5. Assemble context with file/line metadata
6. Generate response with LLM

**Two-Step Retrieval**: Vector search finds relevant locations (fast), then file I/O reads complete code (syntactically correct). Avoids fragmented function signatures and imports.

## Installation

### Prerequisites

- Conda (Anaconda or Miniconda)
- Ollama ([ollama.ai](https://ollama.ai))
- Docker (for Qdrant)

### Setup

```bash
# 1. Create conda environment
conda create -n rag python=3.11
conda activate rag

# 2. Install dependencies
cd rag_system
pip install -r requirements.txt

# 3. Pull Ollama models
ollama pull nomic-embed-text:latest
ollama pull gpt-oss:20b

# 4. Start Qdrant
docker-compose up -d

# 5. Verify
python scripts/test_system.py
```

## Usage

### Index a Codebase

```bash
python -m rag_system index /path/to/project --name "MyProject"
```

Creates Qdrant collection `codebase_myproject_abc123`, builds dependency graph, indexes files.

### Query

```bash
# Single query
python -m rag_system query "How does authentication work?"

# Interactive mode
python -m rag_system query --interactive
```

### Multi-Codebase Management

```bash
# Index multiple projects
python -m rag_system index ~/frontend --name "Frontend"
python -m rag_system index ~/backend --name "Backend"

# List codebases
python -m rag_system list-codebases

# Switch active codebase
python -m rag_system switch-codebase "Backend"

# Delete codebase
python -m rag_system delete-codebase "Frontend" --yes
```

## Interactive Mode

```bash
python -m rag_system query --interactive
```

Features:

- **Chat history**: LLM remembers previous questions/answers
- **Commands**: `clear` (reset history), `exit`/`quit`
- **Follow-up questions**: "What about that function?" works

## CLI Options

```bash
# Query options
--top-k N              # Number of results (default: 10)
--no-expansion         # Disable graph-based expansion
--file-filter PATTERN  # Filter by file path
--graph-path PATH      # Custom dependency graph path

# Re-index after changes
python -m rag_system index /path/to/project --name "MyProject" --clear
```

## Configuration

Optional `.env` file in `rag_system/`:

```bash
# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text:latest
OLLAMA_LLM_MODEL=gpt-oss:20b

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Chunking
CHUNK_SIZE=1024
CHUNK_OVERLAP_PERCENT=10
MIN_CHUNK_SIZE=512

# Retrieval
TOP_K_RESULTS=10
SEARCH_SCORE_THRESHOLD=0.3
ENABLE_FILE_RECONSTRUCTION=true
CONTEXT_LINES_BEFORE_AFTER=10
MAX_LINES_PER_FILE=100

# LLM
LLM_CONTEXT_WINDOW=8192
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=2048
```

## Implementation Details

### Collection Naming

Format: `codebase_{sanitized_name}_{hash}`

Example: `/path/to/my-app` → `codebase_my_app_a1b2c3d4`

### Dependency Graph

Stored as `.rag_dependency_graph.json`:

```json
{
  "nodes": {
    "src/auth/login.py": {
      "language": "python",
      "imports": ["src/auth/user.py"],
      "imported_by": ["src/api/routes.py"]
    }
  }
}
```

### Metadata Storage

SQLite database `.rag_codebases.db`:

```sql
CREATE TABLE codebases (
    name TEXT UNIQUE,
    path TEXT UNIQUE,
    collection_name TEXT UNIQUE,
    indexed_at TIMESTAMP,
    file_count INTEGER,
    chunk_count INTEGER,
    is_active BOOLEAN
);
```

## Performance

- Indexing: ~100-500 chunks/min
- Query: 2-5 seconds
- Memory: ~4-8GB (20B model)
- Storage: ~1KB per chunk

## Troubleshooting

### No active codebase

```bash
python -m rag_system index /path/to/code --name "MyProject"
```

### Model not found

```bash
ollama pull nomic-embed-text:latest
ollama pull gpt-oss:20b
```

### Qdrant connection error

```bash
docker ps | grep qdrant
docker-compose restart
```

### Poor responses

- Lower threshold: `SEARCH_SCORE_THRESHOLD=0.3` in `.env`
- Increase results: `--top-k 15`
- Try different model: `codellama:13b`

## License

MIT
