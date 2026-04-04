# ModelGenerator — Single container with full CUDA dev toolkit
# Supports: InstantMesh (nvdiffrast), xatlas UV unwrap, TripoSR, SDXL, ControlNet
#
# Build:  docker compose build
# Run:    docker compose up
# Open:   http://localhost:8000

# ── Stage 1: Build frontend ─────────────────────────────────
FROM node:20-alpine AS frontend-build

WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --silent 2>/dev/null || npm install
COPY frontend/ .
RUN npm run build

# ── Stage 2: Python + CUDA devel ─────────────────────────────
# CUDA 12.8 devel — supports RTX 5090 (Blackwell, sm_120)
# Using Ubuntu 24.04 for Python 3.12+
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYOPENGL_PLATFORM=egl

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip python3-venv \
    libgl1 libglib2.0-0 libegl1 libgles2 libglx-mesa0 \
    git cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python deps (cached layer) ───────────────────────────────
COPY backend/requirements.txt .

# PyTorch with CUDA — use default PyPI which has sm_120 support (Blackwell/RTX 5090)
# Do NOT use --index-url cu124 — those wheels only support up to sm_90
RUN pip3 install --no-cache-dir --break-system-packages torch torchvision

# Core deps
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# Build-requiring deps (xatlas, nvdiffrast, pyrender)
# These need Python.h + nvcc which are available in the devel image
RUN pip3 install --no-cache-dir --break-system-packages \
    xatlas \
    pyrender \
    PyOpenGL==3.1.0 \
    bvh \
    opencv-python-headless

# nvdiffrast (NVIDIA differentiable rasterizer — needed by InstantMesh for UV textures)
RUN pip3 install --no-cache-dir --break-system-packages git+https://github.com/NVlabs/nvdiffrast.git || \
    echo "WARNING: nvdiffrast install failed — InstantMesh will use vertex-colors mode"

# ── InstantMesh ──────────────────────────────────────────────
RUN git clone --depth 1 https://github.com/TencentARC/InstantMesh.git /app/instantmesh && \
    pip3 install --no-cache-dir --break-system-packages -r /app/instantmesh/requirements.txt || \
    echo "WARNING: InstantMesh install failed — will fall back to TripoSR"

# ── Backend code ─────────────────────────────────────────────
COPY backend/ ./backend/

# ── Frontend static build ────────────────────────────────────
COPY --from=frontend-build /frontend/out ./frontend/out

# ── Storage ──────────────────────────────────────────────────
RUN mkdir -p /app/storage/images /app/storage/models /app/storage/exports /app/storage/uploads

# ── Runtime ──────────────────────────────────────────────────
EXPOSE 8000

WORKDIR /app/backend
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
