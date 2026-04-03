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

PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
NODE_VERSION=$(node --version)
echo "  Python $PY_VERSION, Node $NODE_VERSION"

# ── 2. Environment config ───────────────────────────────────

echo ""
echo "▸ Setting up environment..."

if [ ! -f "$ROOT_DIR/.env" ]; then
    cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
    echo "  Created .env from .env.example"
else
    echo "  .env already exists, keeping it"
fi

mkdir -p "$ROOT_DIR/storage/images" "$ROOT_DIR/storage/models" "$ROOT_DIR/storage/exports"
echo "  Created storage directories"

# ── 3. Python venv + deps ───────────────────────────────────

echo ""
echo "▸ Setting up Python backend..."

if [ ! -d "$ROOT_DIR/backend/venv" ]; then
    echo "  Creating virtual environment..."
    python3 -m venv "$ROOT_DIR/backend/venv"
fi

source "$ROOT_DIR/backend/venv/bin/activate"
pip install --upgrade pip -q 2>&1 | tail -1

# Try CUDA-specific PyTorch first, fallback to default PyPI
echo "  Installing PyTorch..."
if pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 2>/dev/null; then
    echo "  ✓ PyTorch installed (CUDA 12.4 index)"
else
    echo "  ⚠ CUDA 12.4 index failed, installing from default PyPI..."
    pip install torch torchvision
    echo "  ✓ PyTorch installed (default)"
fi

echo "  Installing backend dependencies..."
pip install -r "$ROOT_DIR/backend/requirements.txt"
echo "  ✓ Backend ready"

# Check GPU now that torch is installed
echo ""
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
    echo "  ✓ GPU: $GPU_NAME"
else
    echo "  ⚠ No CUDA GPU detected. Mock services will be used (app still works)."
fi

# ── 4. Frontend ──────────────────────────────────────────────

echo ""
echo "▸ Setting up frontend..."

cd "$ROOT_DIR/frontend"

echo "  Installing npm packages..."
npm install

echo "  Building static frontend..."
npx next build

if [ -d "$ROOT_DIR/frontend/out" ]; then
    echo "  ✓ Frontend built → frontend/out/"
else
    echo "  ✗ Frontend build failed (frontend/out/ not found)"
    echo "    Try manually: cd frontend && npx next build"
    exit 1
fi

# ── 5. Done ──────────────────────────────────────────────────

cd "$ROOT_DIR"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║            Setup complete!               ║"
echo "║                                          ║"
echo "║  Start the app:  ./start.sh              ║"
echo "║  Then open:       http://localhost:8000   ║"
echo "╚══════════════════════════════════════════╝"
echo ""
