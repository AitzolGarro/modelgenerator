FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3-pip \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124

# App code
COPY . .

# Storage volume
RUN mkdir -p /app/storage/images /app/storage/models /app/storage/exports

EXPOSE 8000

# Default: run API server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
