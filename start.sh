#!/bin/bash
#
# ModelGenerator — Start the application
#
# Single command to run everything:
#   ./start.sh
#
# The app runs on http://localhost:8000
# API + Worker + Frontend — all in one process.
#
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

# ── Check setup ──────────────────────────────────────────────

if [ ! -d "$ROOT_DIR/backend/venv" ]; then
    echo "First time? Run ./setup.sh first."
    exit 1
fi

if [ ! -f "$ROOT_DIR/.env" ]; then
    cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
fi

# ── Rebuild frontend if needed ───────────────────────────────

FRONTEND_OUT="$ROOT_DIR/frontend/out"

if [ ! -d "$FRONTEND_OUT" ]; then
    echo "▸ Frontend not built yet, building..."
    cd "$ROOT_DIR/frontend"
    npm run build 2>&1 | tail -3
    cd "$ROOT_DIR"
fi

# ── Kill any previous instance ───────────────────────────────

if lsof -ti:8000 &>/dev/null; then
    echo "▸ Port 8000 in use, stopping previous instance..."
    kill $(lsof -ti:8000) 2>/dev/null
    sleep 2
fi

# ── Create storage dirs ─────────────────────────────────────

mkdir -p "$ROOT_DIR/storage/images" "$ROOT_DIR/storage/models" "$ROOT_DIR/storage/exports"

# ── Activate venv ────────────────────────────────────────────

source "$ROOT_DIR/backend/venv/bin/activate"

# ── Start ────────────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║       ModelGenerator                     ║"
echo "║       http://localhost:8000              ║"
echo "║                                          ║"
echo "║       API docs: /docs                    ║"
echo "║       Press Ctrl+C to stop               ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# Open browser (best effort, don't fail if can't)
if command -v xdg-open &>/dev/null; then
    (sleep 3 && xdg-open "http://localhost:8000") &
elif command -v open &>/dev/null; then
    (sleep 3 && open "http://localhost:8000") &
fi

cd "$ROOT_DIR/backend"
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
