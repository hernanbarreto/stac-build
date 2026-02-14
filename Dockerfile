# ===========================================================================
# STAC-Builder — Unified Docker Image
# Single container for DA3 + SAM3 + CloudComPy + InternVL3 scene analyzer
#
# Build:
#   docker build -t stac-builder .
#
# Run (GPU required):
#   docker run --gpus all -p 8000:8000 -v $(pwd)/weights:/app/weights stac-builder
#
# Dev mode (mount source for live editing):
#   docker run --gpus all -p 8000:8000 \
#     -v $(pwd)/server:/app/server \
#     -v $(pwd)/weights:/app/weights \
#     stac-builder
# ===========================================================================

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LC_NUMERIC=C

# ── System dependencies ─────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    git \
    wget \
    curl \
    ffmpeg \
    openssl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set python3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# ── Working directory ────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ─────────────────────────────────────────────
# Install PyTorch first (specific CUDA version)
RUN pip3 install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Copy and install requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# ── Copy project ────────────────────────────────────────────────────
# Vendor dependencies (source code, ~280MB)
COPY vendor/ ./vendor/

# Server code
COPY server/ ./server/

# Config and scripts
COPY setup_weights.sh .
COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh setup_weights.sh

# ── Install vendored packages in editable/path mode ─────────────────
# SAM3 needs to be importable as a package
RUN cd vendor/sam3 && pip3 install --no-cache-dir -e . 2>/dev/null || true

# ── CloudComPy runtime setup ────────────────────────────────────────
ENV PYTHONPATH="/app/vendor/cloudcompy/lib/cloudcompare:${PYTHONPATH}"
ENV LD_LIBRARY_PATH="/app/vendor/cloudcompy/lib/cloudcompare:/app/vendor/cloudcompy/lib/cloudcompare/plugins:${LD_LIBRARY_PATH}"

# ── Weights volume (mounted at runtime) ─────────────────────────────
# Model weights are NOT baked into the image — mount or download at start
VOLUME /app/weights

# ── Health check ─────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# ── Entrypoint ───────────────────────────────────────────────────────
EXPOSE 8765

# Default: start via entrypoint (handles SSL + uvicorn)
ENTRYPOINT ["./docker-entrypoint.sh"]
