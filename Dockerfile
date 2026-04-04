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
# Using devel image (not runtime) — includes nvcc, cuda headers, Python dev headers
# This enables building: nvdiffrast, xatlas, torchmcubes, etc.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYOPENGL_PLATFORM=egl

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    libgl1-mesa-glx libglib2.0-0 libegl1 libgles2 \
    git cmake ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

# ── Python deps (cached layer) ───────────────────────────────
COPY backend/requirements.txt .

# PyTorch with CUDA
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Core deps
RUN pip3 install --no-cache-dir -r requirements.txt

# Build-requiring deps (xatlas, nvdiffrast, pyrender)
# These need Python.h + nvcc which are available in the devel image
RUN pip3 install --no-cache-dir \
    xatlas \
    pyrender \
    PyOpenGL==3.1.0 \
    bvh \
    opencv-python-headless

# nvdiffrast (NVIDIA differentiable rasterizer — needed by InstantMesh for UV textures)
RUN pip3 install --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git || \
    echo "WARNING: nvdiffrast install failed — InstantMesh will use vertex-colors mode"

# ── Backend code ─────────────────────────────────────────────
COPY backend/ ./backend/

# ── Frontend static build ────────────────────────────────────
COPY --from=frontend-build /frontend/out ./frontend/out

# ── Mocap data ───────────────────────────────────────────────
# BVH animation files are embedded in backend/app/services/mocap_data/
# (copied as part of backend/)

# ── Storage ──────────────────────────────────────────────────
RUN mkdir -p /app/storage/images /app/storage/models /app/storage/exports /app/storage/uploads

# ── Runtime ──────────────────────────────────────────────────
EXPOSE 8000

WORKDIR /app/backend
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
