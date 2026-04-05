#!/bin/bash
#
# ModelGenerator — Start the application
#
# Usage:
#   ./start.sh          # Docker mode (recommended for Bazzite/immutable OS)
#   ./start.sh --local  # Local venv mode (needs Python headers + CUDA toolkit)
#
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

# ── .env ─────────────────────────────────────────────────────

if [ ! -f "$ROOT_DIR/.env" ]; then
    cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
fi

mkdir -p "$ROOT_DIR/storage/images" "$ROOT_DIR/storage/models" "$ROOT_DIR/storage/exports"

# ── Mode selection ───────────────────────────────────────────

if [ "$1" = "--local" ]; then
    # ── Local mode (venv) ────────────────────────────────────
    if [ ! -d "$ROOT_DIR/backend/venv" ]; then
        echo "Run ./setup.sh first for local mode."
        exit 1
    fi

    # Kill previous instance
    if command -v lsof &>/dev/null && lsof -ti:8000 &>/dev/null; then
        echo "▸ Stopping previous instance on port 8000..."
        kill $(lsof -ti:8000) 2>/dev/null
        sleep 2
    fi

    # Rebuild frontend if needed
    if [ ! -d "$ROOT_DIR/frontend/out" ]; then
        echo "▸ Building frontend..."
        cd "$ROOT_DIR/frontend" && npx next build
        cd "$ROOT_DIR"
    fi

    source "$ROOT_DIR/backend/venv/bin/activate"

    # Setup CUDA_HOME from pip nvidia packages (needed for nvdiffrast/InstantMesh)
    NVCC_PATH=$(find "$ROOT_DIR/backend/venv" -name "nvcc" -type f -executable 2>/dev/null | head -1)
    if [ -n "$NVCC_PATH" ]; then
        export CUDA_HOME=$(dirname $(dirname "$NVCC_PATH"))
        export PATH="$CUDA_HOME/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_HOME/lib:$LD_LIBRARY_PATH"
        # Ensure libcudart.so symlink exists
        if [ -f "$CUDA_HOME/lib/libcudart.so.13" ] && [ ! -f "$CUDA_HOME/lib/libcudart.so" ]; then
            ln -sf "$CUDA_HOME/lib/libcudart.so.13" "$CUDA_HOME/lib/libcudart.so"
        fi
    fi

    echo ""
    echo "╔══════════════════════════════════════════╗"
    echo "║       ModelGenerator (local)             ║"
    echo "║       http://localhost:8000              ║"
    echo "║       Press Ctrl+C to stop               ║"
    echo "╚══════════════════════════════════════════╝"
    echo ""

    if command -v xdg-open &>/dev/null; then
        (sleep 3 && xdg-open "http://localhost:8000") &
    fi

    cd "$ROOT_DIR/backend"
    exec uvicorn app.main:app --host 0.0.0.0 --port 8000

else
    # ── Docker mode (default) ────────────────────────────────
    if ! command -v docker &>/dev/null; then
        echo "Docker not found. Install Docker or use: ./start.sh --local"
        exit 1
    fi

    if ! docker info &>/dev/null; then
        echo "Docker daemon not running. Start it with: sudo systemctl start docker"
        exit 1
    fi

    echo ""
    echo "╔══════════════════════════════════════════╗"
    echo "║       ModelGenerator (Docker)            ║"
    echo "║                                          ║"
    echo "║  Building container...                   ║"
    echo "║  First build takes ~10 min               ║"
    echo "║  (downloads CUDA toolkit + Python deps)  ║"
    echo "║                                          ║"
    echo "║  After build: http://localhost:8000      ║"
    echo "╚══════════════════════════════════════════╝"
    echo ""

    # Build and start
    docker compose up --build

fi
