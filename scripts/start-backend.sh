#!/bin/bash
# Start the backend API server
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR/backend"

# Check for .env
if [ ! -f "$PROJECT_DIR/.env" ]; then
    echo "⚠️  No .env found. Copying from .env.example..."
    cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
fi

# Create storage dirs
mkdir -p "$PROJECT_DIR/storage/images" "$PROJECT_DIR/storage/models" "$PROJECT_DIR/storage/exports"

echo "🚀 Starting ModelGenerator API on port 8000..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
