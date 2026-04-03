#!/bin/bash
# Start the background worker
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR/backend"

echo "⚙️  Starting ModelGenerator Worker..."
echo "   Press Ctrl+C to stop gracefully."
python -m app.workers.runner
