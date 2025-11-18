#!/bin/bash

# Setup script for RAG system

set -e

echo "========================================="
echo "RAG System Setup"
echo "========================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Anaconda or Miniconda."
    exit 1
fi

# Check if ollama is available
if ! command -v ollama &> /dev/null; then
    echo "Error: ollama not found. Please install Ollama from https://ollama.ai"
    exit 1
fi

# Check if docker is available
if ! command -v docker &> /dev/null; then
    echo "Warning: docker not found. You'll need to install Docker to run Qdrant."
fi

echo "Step 1: Activating conda environment 'rag'..."
eval "$(conda shell.bash hook)"
conda activate rag || {
    echo "Error: Could not activate conda environment 'rag'"
    echo "Please create it first: conda create -n rag python=3.10"
    exit 1
}

echo "Step 2: Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "Step 3: Pulling Ollama models..."
echo "This may take a while depending on your internet connection..."
echo ""

echo "Pulling nomic-embed-text:latest..."
ollama pull nomic-embed-text:latest

echo ""
echo "Checking for gpt-oss:20b..."
if ollama list | grep -q "gpt-oss:20b"; then
    echo "✓ gpt-oss:20b already available"
else
    echo "Warning: gpt-oss:20b not found."
    echo "You can use an alternative model. Popular options:"
    echo "  - codellama:13b (good for code)"
    echo "  - mistral:latest (fast and accurate)"
    echo "  - llama2:13b (general purpose)"
    echo ""
    read -p "Would you like to pull codellama:13b instead? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ollama pull codellama:13b
        echo "Remember to update .env: OLLAMA_LLM_MODEL=codellama:13b"
    fi
fi

echo ""
echo "Step 4: Starting Qdrant with Docker..."
if command -v docker &> /dev/null; then
    docker-compose up -d
    echo "✓ Qdrant started on http://localhost:6333"
else
    echo "Skipping Qdrant setup (Docker not available)"
fi

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Verify Ollama models: ollama list"
echo "2. Check Qdrant: curl http://localhost:6333"
echo "3. Index a codebase: python -m rag_system.cli index /path/to/code"
echo "4. Query the system: python -m rag_system.cli query --interactive"
echo ""
echo "For more information, see USAGE.md"
echo ""

