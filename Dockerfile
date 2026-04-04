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
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYOPENGL_PLATFORM=egl

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip python3-venv \
    libgl1 libglib2.0-0 libegl1 libgles2 libglx-mesa0 \
    git cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python deps ──────────────────────────────────────────────
COPY backend/requirements.txt .

# PyTorch — default PyPI has sm_120 support (RTX 5090 Blackwell)
RUN pip3 install --no-cache-dir --break-system-packages torch torchvision

# Core deps
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# rembg with onnxruntime (needed by both TripoSR and InstantMesh for background removal)
RUN pip3 install --no-cache-dir --break-system-packages "rembg[cpu]" onnxruntime

# Build-requiring deps
RUN pip3 install --no-cache-dir --break-system-packages \
    xatlas \
    pyrender \
    PyOpenGL==3.1.0 \
    PyOpenGL-accelerate \
    bvh \
    opencv-python-headless

# nvdiffrast (needed by InstantMesh for UV textures)
RUN pip3 install --no-cache-dir --break-system-packages git+https://github.com/NVlabs/nvdiffrast.git || \
    echo "WARNING: nvdiffrast install failed — InstantMesh will use vertex-colors mode"

# InstantMesh explicit deps (its requirements.txt can conflict with ours)
RUN pip3 install --no-cache-dir --break-system-packages \
    pytorch-lightning \
    einops \
    omegaconf \
    huggingface-hub

# ── InstantMesh ──────────────────────────────────────────────
RUN git clone --depth 1 https://github.com/TencentARC/InstantMesh.git /app/instantmesh

# ── Validate key imports ─────────────────────────────────────
RUN python3 -c "\
import torch; print(f'PyTorch {torch.__version__}, CUDA archs: {torch.cuda.get_arch_list()[-3:]}'); \
import rembg; print('rembg OK'); \
import trimesh; print('trimesh OK'); \
import xatlas; print('xatlas OK'); \
import pyrender; print('pyrender OK'); \
import bvh; print('bvh OK'); \
import cv2; print('opencv OK'); \
import pytorch_lightning; print('pytorch_lightning OK'); \
import einops; print('einops OK'); \
import omegaconf; print('omegaconf OK'); \
print('All imports validated.'); \
"

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
