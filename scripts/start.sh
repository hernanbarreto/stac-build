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
# NOTA: torch.hub (DINOv2 del selector) se deja en su default ~/.cache/torch
# (disco local rápido). Apuntarlo a /workspace lo hacía persistente pero el
# volumen de red es lento → el model load de DINOv2 saltaba a ~40s por corrida.
# El costo es re-clonar dinov2 de GitHub tras un reinicio del pod (una sola vez).

# Cert autofirmado si no existe
if [ ! -f "$CERT_FILE" ] || [ ! -f "$KEY_FILE" ]; then
    openssl req -x509 -newkey rsa:4096 -keyout "$KEY_FILE" -out "$CERT_FILE" \
        -days 365 -nodes -subj "/CN=localhost"
fi

# Logs persistentes: toda la salida del server (incluye los logs de los workers
# DA3/SAM3/CloudComPy reenviados por el pipeline) se guarda a un archivo y, a la
# vez, se sigue mostrando en consola. logs/latest.log siempre apunta al último.
LOG_DIR="$STAC_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/server_$(date +%Y%m%d_%H%M%S).log"
ln -sf "$LOG_FILE" "$LOG_DIR/latest.log"

cd "$SERVER_DIR"
echo "🚀 STAC server en puerto $PORT"
echo "📝 logs → $LOG_FILE (latest: $LOG_DIR/latest.log)"

# Redirigir stdout/stderr del proceso a tee (consola + archivo) antes del exec,
# así uvicorn hereda esos fds y todo queda registrado.
exec > >(tee -a "$LOG_FILE") 2>&1

# Unbuffered stdout/stderr so plain print() (e.g. [Shape] logs) flush immediately
# through the tee above, instead of getting stuck in the block buffer while only
# uvicorn's logging output (which flushes) appears. Without this we were debugging
# half-blind: ShapeR could start/crash and its prints stayed invisible.
export PYTHONUNBUFFERED=1

exec python -m uvicorn main:app --host 0.0.0.0 --port $PORT \
    --ssl-keyfile "$KEY_FILE" \
    --ssl-certfile "$CERT_FILE" \
    --ws-ping-interval 30 \
    --ws-ping-timeout 600