# Agentic RAG System for Local Codebase Indexing

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Industry-standard RAG (Retrieval-Augmented Generation) architecture for indexing and querying local codebases using a **fully self-hosted stack** with no external API dependencies.

## 🚀 Features

- **AST-Based Semantic Chunking**: Respects code structure (functions, classes) for meaningful embeddings
- **Matryoshka Embeddings**: Consistent 768-dimensional vectors to prevent dimension mismatch errors
- **Large Context Window**: 8192 tokens for handling multiple code chunks
- **Rich Metadata**: Stores file paths and line numbers for precise code references
- **Fully Local**: No data leaves your machine, complete privacy
- **Multi-Language Support**: Python, JavaScript, TypeScript, Java, C++, Go, Rust, and more

## 📋 Technology Stack

| Component         | Technology                    | Purpose                           |
| ----------------- | ----------------------------- | --------------------------------- |
| **Orchestration** | Ollama                        | Local LLM server                  |
| **LLM**           | gpt-oss:20b (or alternatives) | Code understanding and generation |
| **Embeddings**    | nomic-embed-text:latest       | 768-dim semantic vectors          |
| **Vector Store**  | Qdrant                        | Self-hosted vector database       |
| **Framework**     | LangChain                     | AST-based code splitting          |

## 🏗️ Architecture

### Ingestion Pipeline (Offline/Batch)

```
Local Codebase → Load → Chunk (AST) → Embed → Store (Qdrant)
```

1. **Load**: Scan directory, filter by extension, ignore patterns
2. **Chunk**: AST-based splitting (respects function/class boundaries)
3. **Embed**: Generate 768-dim vectors via Ollama
4. **Store**: Upsert to Qdrant with rich metadata (file, lines, content)

### Query Pipeline (Online/Real-time)

```
User Query → Embed → Search (Qdrant) → Assemble Context → Generate (LLM) → Response
```

1. **Embed**: Convert query to 768-dim vector
2. **Search**: Find top-K similar chunks (cosine similarity)
3. **Assemble**: Format chunks with file/line metadata
4. **Generate**: LLM generates answer with context

## 📁 Project Structure

```
rag_system/
├── config/
│   └── settings.py              # Pydantic settings with env vars
├── core/
│   ├── loaders/
│   │   └── code_loader.py       # File system loader with filters
│   ├── chunkers/
│   │   └── semantic_chunker.py  # AST-based code chunker
│   ├── embeddings/
│   │   └── ollama_embeddings.py # Ollama embedding service
│   ├── vectorstore/
│   │   └── qdrant_store.py      # Qdrant vector store
│   └── llm/
│       └── ollama_llm.py        # Ollama LLM service
├── pipelines/
│   ├── ingestion_pipeline.py    # Batch indexing pipeline
│   └── query_pipeline.py        # Real-time query pipeline
├── cli/
│   └── main.py                  # Click-based CLI
├── scripts/
│   ├── setup.sh                 # Automated setup script
│   └── test_system.py           # System verification
└── tests/                       # Test suite
```

## 🔧 Installation

### Prerequisites

1. **Conda** (Anaconda or Miniconda)
2. **Ollama** - Download from [ollama.ai](https://ollama.ai)
3. **Docker** - For running Qdrant

### Quick Setup

```bash
# 1. Activate conda environment
conda activate rag

# 2. Run automated setup (Linux/Mac)
cd rag_system
chmod +x scripts/setup.sh
./scripts/setup.sh

# Or manual setup:

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Pull Ollama models
ollama pull nomic-embed-text:latest
ollama pull codellama:13b  # or gpt-oss:20b, mistral:latest

# 5. Start Qdrant
docker-compose up -d

# 6. Verify system
python scripts/test_system.py
```

## 🎯 Usage

### Command-Line Interface

#### Index a Codebase

```bash
# Index your project
python -m rag_system.cli index /path/to/your/codebase

# Clear existing index and re-index
python -m rag_system.cli index /path/to/your/codebase --clear

# Example: Index this RAG system itself
python -m rag_system.cli index .
```

#### Query the System

```bash
# Single query
python -m rag_system.cli query "Explain how the embedding service works"

# Interactive mode (recommended)
python -m rag_system.cli query --interactive

# Query with custom top-k
python -m rag_system.cli query "How does chunking work?" --top-k 5

# Filter by file path
python -m rag_system.cli query "Show authentication code" --file-filter "auth.py"
```

#### Check Collection Info

```bash
python -m rag_system.cli info
```

### Programmatic Usage

```python
from rag_system.pipelines import IngestionPipeline, QueryPipeline
from rag_system.config import get_settings

# Get settings
settings = get_settings()

# Index a codebase
ingestion = IngestionPipeline(settings)
ingestion.run("/path/to/codebase")

# Query the system
query_pipeline = QueryPipeline(settings)
response = query_pipeline.query(
    "Explain the authentication logic",
    top_k=10,
    verbose=True
)
print(response)
```

## ⚙️ Configuration

Edit `.env` file to customize settings:

```bash
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text:latest
OLLAMA_LLM_MODEL=codellama:13b

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=codebase_v1

# Embedding Configuration
EMBEDDING_DIMENSION=768
EMBEDDING_BATCH_SIZE=32

# Chunking Configuration
CHUNK_SIZE=1024
CHUNK_OVERLAP_PERCENT=10
MIN_CHUNK_SIZE=512

# Retrieval Configuration
TOP_K_RESULTS=10
SEARCH_SCORE_THRESHOLD=0.7

# LLM Configuration
LLM_CONTEXT_WINDOW=8192
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=2048
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_loader.py -v

# Run with coverage
pytest tests/ --cov=rag_system --cov-report=html
```

## 🔍 Critical Design Decisions

### 1. AST-Based Chunking

**Problem**: Character-based splitting breaks functions/classes mid-definition.
**Solution**: Use LangChain's language-specific splitters to respect code structure.

### 2. Matryoshka Dimension Consistency

**Problem**: Variable dimensions cause Qdrant errors.
**Solution**: Enforce 768 dimensions for both indexing and querying.

### 3. Large Context Window

**Problem**: Default 2048 tokens is too small for RAG.
**Solution**: Set `num_ctx=8192` explicitly in LLM options.

### 4. Rich Payload Metadata

**Problem**: Without line numbers, responses are vague.
**Solution**: Store `file_path`, `start_line`, `end_line` in Qdrant payload.

## 📊 Performance

- **Indexing Speed**: ~100-500 chunks/minute (GPU-dependent)
- **Query Latency**: 2-5 seconds (embedding + search + generation)
- **Memory Usage**: ~4-8GB (20B model + embeddings)
- **Storage**: ~1KB per chunk (vector + metadata)

## 🐛 Troubleshooting

### Model Not Found

```bash
ollama list  # Check available models
ollama pull nomic-embed-text:latest
```

### Qdrant Connection Error

```bash
docker ps | grep qdrant  # Check if running
docker-compose restart qdrant
```

### Dimension Mismatch

Clear and re-index:

```bash
python -m rag_system.cli index /path/to/codebase --clear
```

## 📚 Documentation

- [USAGE.md](USAGE.md) - Detailed usage guide
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture and design
- [.env.example](.env.example) - Configuration template

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai) - Local LLM inference
- [Qdrant](https://qdrant.tech) - Vector database
- [LangChain](https://langchain.com) - LLM framework
- [nomic-embed-text](https://huggingface.co/nomic-ai/nomic-embed-text-v1) - Embedding model

---

**Built with ❤️ for developers who value privacy and control over their data.**
