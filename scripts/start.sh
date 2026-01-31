#!/bin/bash
set -e

# Configuration
USER_HOME="/home/hernan"
# DA3 paths
DA3_PATH="$USER_HOME/Depth-Anything-3/src"
DA3_STREAMING_PATH="$USER_HOME/Depth-Anything-3/da3_streaming"
SAM3_PATH="$USER_HOME/sam3"

# --- NUEVO: Capturamos la ruta donde está este script y los certificados ---
# Esto asegura que uvicorn encuentre los .pem aunque el script cambie de carpeta después
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CERT_FILE="$SCRIPT_DIR/cert.pem"
KEY_FILE="$SCRIPT_DIR/key.pem"

echo "🚀 Starting STAC-BUILD Room Scanner (DA3-Streaming) [HTTPS MODE]..."

# Check Conda
source $(conda info --base)/etc/profile.d/conda.sh

# Use 'da3' environment (Main environment)
conda activate da3
# previously: conda activate da3

# Set PYTHONPATH explicitly
export PYTHONPATH="$PYTHONPATH:$DA3_PATH:$DA3_STREAMING_PATH:$SAM3_PATH"

echo "📍 PYTHONPATH: $PYTHONPATH"
echo "📍 Python: $(which python)"
echo "📍 Certificado SSL: $CERT_FILE"

# Navigate to server directory
cd "$(dirname "$0")/../server"

# Run Uvicorn Server with SSL
# IMPORTANT: --reload disabled for GPU models
echo "🌐 Launching Server at https://0.0.0.0:8765"
echo "📷 Camera: https://192.168.101.12:8765/static/camera.html"
exec python -m uvicorn main:app --host 0.0.0.0 --port 8765 \
    --ssl-keyfile "$KEY_FILE" \
    --ssl-certfile "$CERT_FILE"