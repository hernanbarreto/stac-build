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