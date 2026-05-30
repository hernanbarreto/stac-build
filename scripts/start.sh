#!/bin/bash
# STAC-Builder Startup — POD (RunPod)
set -e

STAC_ROOT="/workspace/stac-build"
SERVER_DIR="$STAC_ROOT/server"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CERT_FILE="$SCRIPT_DIR/cert.pem"
KEY_FILE="$SCRIPT_DIR/key.pem"
PORT=8765

# Conda
source /workspace/miniforge3/etc/profile.d/conda.sh
export CONDA_ROOT=/workspace/miniforge3
conda activate da3

# HF cache persistente
export HF_HOME=/workspace/hf_cache
# torch.hub cache persistente (DINOv2 del selector de keyframes). Por defecto usa
# ~/.cache/torch, que vive en el fs efímero del pod y se borra al reiniciar →
# re-clonaría dinov2 de GitHub en cada arranque. /workspace persiste.
export TORCH_HOME=/workspace/torch_cache

# Cert autofirmado si no existe
if [ ! -f "$CERT_FILE" ] || [ ! -f "$KEY_FILE" ]; then
    openssl req -x509 -newkey rsa:4096 -keyout "$KEY_FILE" -out "$CERT_FILE" \
        -days 365 -nodes -subj "/CN=localhost"
fi

cd "$SERVER_DIR"
echo "🚀 STAC server en puerto $PORT"

exec python -m uvicorn main:app --host 0.0.0.0 --port $PORT \
    --ssl-keyfile "$KEY_FILE" \
    --ssl-certfile "$CERT_FILE" \
    --ws-ping-interval 30 \
    --ws-ping-timeout 600