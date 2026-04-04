# ModelGenerator — Single container with full CUDA dev toolkit
# Supports: InstantMesh, TripoSR, SDXL, ControlNet, xatlas, nvdiffrast
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
# CUDA 13.0 to match PyTorch cu130 (required for nvdiffrast compilation)
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYOPENGL_PLATFORM=egl

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip python3-venv \
    libgl1 libglib2.0-0 libegl1 libgles2 libglx-mesa0 \
    git cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Our requirements ─────────────────────────────────────────
COPY backend/requirements.txt .

# PyTorch — default PyPI has sm_120 (RTX 5090 Blackwell)
RUN pip3 install --no-cache-dir --break-system-packages torch torchvision torchaudio

# Our core deps
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# ── ALL InstantMesh dependencies (from its requirements.txt) ─
# Pinning versions where InstantMesh requires them, relaxing where we can
RUN pip3 install --no-cache-dir --break-system-packages \
    pytorch-lightning==2.1.2 \
    einops \
    omegaconf \
    torchmetrics \
    webdataset \
    accelerate \
    tensorboard \
    PyMCubes \
    trimesh \
    "rembg[cpu]" \
    onnxruntime \
    transformers \
    diffusers \
    bitsandbytes \
    "imageio[ffmpeg]" \
    xatlas \
    plyfile \
    tqdm \
    huggingface-hub

# ── Build-from-source deps ───────────────────────────────────
# nvdiffrast needs --no-build-isolation + explicit CUDA arch flags
# (no GPU visible during docker build, so PyTorch can't auto-detect)
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0;10.0;12.0"
RUN pip3 install --no-cache-dir --break-system-packages --no-build-isolation \
    git+https://github.com/NVlabs/nvdiffrast.git

# ── Additional tools ─────────────────────────────────────────
RUN pip3 install --no-cache-dir --break-system-packages \
    pyrender \
    PyOpenGL==3.1.0 \
    PyOpenGL-accelerate \
    bvh \
    opencv-python-headless \
    scipy

# ── Clone InstantMesh repo ──────────────────────────────────
RUN git clone --depth 1 https://github.com/TencentARC/InstantMesh.git /app/instantmesh

# ── Validate ALL imports ─────────────────────────────────────
# PYOPENGL_PLATFORM=egl needed for pyrender in headless container
RUN PYOPENGL_PLATFORM=egl python3 -c "\
import sys; print(f'Python {sys.version}'); \
import torch; print(f'PyTorch {torch.__version__}, CUDA archs: {torch.cuda.get_arch_list()[-3:]}'); \
import rembg; print('rembg OK'); \
import onnxruntime; print(f'onnxruntime {onnxruntime.__version__}'); \
import trimesh; print('trimesh OK'); \
import xatlas; print('xatlas OK'); \
import nvdiffrast; print('nvdiffrast OK'); \
from pyrender.offscreen import OffscreenRenderer; print('pyrender OffscreenRenderer OK'); \
import bvh; print('bvh OK'); \
import cv2; print(f'opencv {cv2.__version__}'); \
import pytorch_lightning; print(f'pytorch_lightning {pytorch_lightning.__version__}'); \
import einops; print('einops OK'); \
import omegaconf; print('omegaconf OK'); \
import diffusers; print(f'diffusers {diffusers.__version__}'); \
import transformers; print(f'transformers {transformers.__version__}'); \
import accelerate; print('accelerate OK'); \
import huggingface_hub; print('huggingface_hub OK'); \
import imageio; print('imageio OK'); \
import plyfile; print('plyfile OK'); \
import mcubes; print('PyMCubes OK'); \
import scipy; print(f'scipy {scipy.__version__}'); \
print(); \
print('=== ALL IMPORTS OK ==='); \
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
