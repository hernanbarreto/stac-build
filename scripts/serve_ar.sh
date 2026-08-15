#!/bin/bash
# STAC-Builder — standalone AR/XR server launcher (phone surface).
#
# Own process on 127.0.0.1:8766 (plain HTTP — only tailscaled talks to it;
# tailscale serve --https=8443 publishes it with a valid TLS cert). Fully
# decoupled from the main backend: restart freely, the pipeline never notices.
#
#   bash scripts/serve_ar.sh
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC
set -e

STAC_ROOT="/workspace/stac-build"
source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate da3

LOG_DIR="$STAC_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/arserver_$(date +%Y%m%d_%H%M%S).log"
ln -sf "$LOG_FILE" "$LOG_DIR/arserver_latest.log"
echo "📱 STAC AR server (phone XR) → http://127.0.0.1:8766"
echo "📝 logs → $LOG_FILE"

cd "$STAC_ROOT/server"
exec > >(tee -a "$LOG_FILE") 2>&1
export PYTHONUNBUFFERED=1
exec python -m uvicorn ar_server:app --host 127.0.0.1 --port 8766
