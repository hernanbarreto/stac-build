# ===========================================================================
# STAC-Builder — Unified Docker Image
# Single container for DA3 + SAM3 + CloudComPy + InternVL3 + PotreeConverter
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

# ── Stage 1: Build UI ───────────────────────────────────────────────
FROM node:20-slim AS ui-builder
WORKDIR /build
COPY ui/package.json ui/package-lock.json* ./
RUN npm ci --silent
COPY ui/ .
# Only build the web assets (skip electron-builder, not needed in Docker)
RUN npx tsc && npx vite build

# ── Stage 2: Main image ─────────────────────────────────────────────
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
    # CloudComPy runtime deps (pre-built .so need these)
    libqt5core5a \
    libqt5widgets5 \
    libqt5gui5 \
    libqt5opengl5 \
    libqt5svg5 \
    libqt5concurrent5 \
    libglew2.2 \
    libboost-filesystem1.74.0 \
    libboost-program-options1.74.0 \
    libtbb2 \
    # PotreeConverter build dependencies
    cmake \
    make \
    g++ \
    libtbb-dev \
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
RUN pip3 install --upgrade pip && pip3 install --no-cache-dir -r requirements.txt

# FlashAttention2 — accelerates VLM inference (~2x faster attention)
RUN pip3 install --no-cache-dir flash-attn --no-build-isolation

# Potree converter needs laspy for PLY→LAS conversion
RUN pip3 install --no-cache-dir laspy

# ── Copy project ────────────────────────────────────────────────────
# Vendor dependencies (source code, ~280MB)
COPY vendor/ ./vendor/

# Build PotreeConverter from source (C++20, needs cmake + tbb)
RUN cd vendor/PotreeConverter \
    && mkdir -p build && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release .. \
    && make -j$(nproc) \
    && echo '✅ PotreeConverter built successfully'

# Server code
COPY server/ ./server/

# Static files (viewer, camera HTML/JS)
COPY static/ ./static/

# UI build output (from Stage 1)
COPY --from=ui-builder /build/dist ./static/ui/

# Docs and scripts
COPY docs/ ./docs/ 
COPY scripts/ ./scripts/

# Config and scripts
COPY setup_weights.sh .
COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh setup_weights.sh

# ── Install vendored packages in editable/path mode ─────────────────
# SAM3 needs to be importable as a package
# Use --no-deps to avoid numpy==1.26 pin conflicting with our numpy<2
RUN cd vendor/sam3 && pip3 install --no-cache-dir --no-deps -e .

# ── Verify critical dependencies ────────────────────────────────────
RUN python3 -c "import numba; print('✅ numba:', numba.__version__)" && \
    python3 -c "import pypose; print('✅ pypose:', pypose.__version__)" && \
    python3 -c "import e3nn; print('✅ e3nn:', e3nn.__version__)" && \
    python3 -c "from sam3.model_builder import build_sam3_video_predictor; print('✅ SAM3 importable')" && \
    python3 -c "import ifcopenshell; print('✅ ifcopenshell:', ifcopenshell.version)" && \
    echo '✅ All critical dependencies verified'

# ── CloudComPy runtime setup ────────────────────────────────────────
ENV PYTHONPATH="/app/vendor/cloudcompy/lib/cloudcompare:${PYTHONPATH}"
ENV LD_LIBRARY_PATH="/app/vendor/cloudcompy/lib/cloudcompare:/app/vendor/cloudcompy/lib/cloudcompare/plugins:${LD_LIBRARY_PATH}"

# ── Volumes ──────────────────────────────────────────────────────────
# Model weights (mounted at runtime)
VOLUME /app/weights
# Auth database (persists across container restarts)
VOLUME /app/server/data

# ── Health check ─────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# ── Entrypoint ───────────────────────────────────────────────────────
EXPOSE 8765

# Default: start via entrypoint (handles SSL + uvicorn)
ENTRYPOINT ["./docker-entrypoint.sh"]
