#!/bin/bash
#
# ModelGenerator — First-time setup
# Run once: ./setup.sh
#
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║       ModelGenerator — Setup             ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# ── 1. Check prerequisites ──────────────────────────────────

echo "▸ Checking prerequisites..."

check_cmd() {
    if ! command -v "$1" &>/dev/null; then
        echo "  ✗ $1 not found. $2"
        return 1
    fi
    echo "  ✓ $1 found: $(command -v "$1")"
    return 0
}

MISSING=0
check_cmd python3 "Install Python 3.11+" || MISSING=1
check_cmd node "Install Node.js 20+" || MISSING=1
check_cmd npm "Comes with Node.js" || MISSING=1

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "ERROR: Missing prerequisites. Install them and re-run."
    exit 1
fi

# Check Python version
PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  Python version: $PY_VERSION"

# Check CUDA (optional)
echo ""
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    echo "  ✓ CUDA available: $GPU_NAME"
elif command -v nvidia-smi &>/dev/null; then
    echo "  ⚠ nvidia-smi found but PyTorch CUDA not yet available (will install next)"
else
    echo "  ⚠ No CUDA detected. Will use mock ML services (works for testing)."
fi

# ── 2. Environment config ───────────────────────────────────

echo ""
echo "▸ Setting up environment..."

if [ ! -f "$ROOT_DIR/.env" ]; then
    cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
    echo "  Created .env from .env.example"
else
    echo "  .env already exists, keeping it"
fi

# ── 3. Storage directories ──────────────────────────────────

mkdir -p "$ROOT_DIR/storage/images" "$ROOT_DIR/storage/models" "$ROOT_DIR/storage/exports"
echo "  Created storage directories"

# ── 4. Python venv + deps ───────────────────────────────────

echo ""
echo "▸ Setting up Python backend..."

if [ ! -d "$ROOT_DIR/backend/venv" ]; then
    echo "  Creating virtual environment..."
    python3 -m venv "$ROOT_DIR/backend/venv"
fi

source "$ROOT_DIR/backend/venv/bin/activate"

echo "  Installing PyTorch with CUDA..."
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -1

echo "  Installing backend dependencies..."
pip install -q -r "$ROOT_DIR/backend/requirements.txt" 2>&1 | tail -1

echo "  ✓ Backend ready"

# ── 5. Frontend ──────────────────────────────────────────────

echo ""
echo "▸ Setting up frontend..."

cd "$ROOT_DIR/frontend"

if [ ! -d "node_modules" ]; then
    echo "  Installing npm packages..."
    npm install --silent 2>&1 | tail -3
fi

echo "  Building static frontend..."
npm run build 2>&1 | tail -3

echo "  ✓ Frontend built"

# ── 6. Done ──────────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║            Setup complete!               ║"
echo "║                                          ║"
echo "║  Start the app:  ./start.sh              ║"
echo "║  Then open:       http://localhost:8000   ║"
echo "╚══════════════════════════════════════════╝"
echo ""
