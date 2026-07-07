#!/bin/bash
# STAC-Builder — Semantic Service launcher (vLLM / Qwen3-VL)
#
# Runs the persistent semantic endpoint in the `semantic` conda env. All model
# flags come from server/config.yaml (`semantic:` block) via semantic.serve.
#
#   bash scripts/serve_semantic.sh                 # default backend (qwen_local)
#   bash scripts/serve_semantic.sh qwen_local_large
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC
set -e

STAC_ROOT="/workspace/stac-build"
SERVER_DIR="$STAC_ROOT/server"
BACKEND="${1:-}"

source /workspace/miniforge3/etc/profile.d/conda.sh
export CONDA_ROOT=/workspace/miniforge3
conda activate semantic

# HF cache on the network volume (offline once weights are local).
export HF_HOME=/workspace/hf_cache
export VLLM_LOGGING_LEVEL=INFO

# Persistent logs, mirroring scripts/start.sh.
LOG_DIR="$STAC_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/semantic_$(date +%Y%m%d_%H%M%S).log"
ln -sf "$LOG_FILE" "$LOG_DIR/semantic_latest.log"
echo "🧠 STAC semantic service (vLLM/Qwen3-VL)"
echo "📝 logs → $LOG_FILE (latest: $LOG_DIR/semantic_latest.log)"

cd "$SERVER_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1
export PYTHONUNBUFFERED=1

if [ -n "$BACKEND" ]; then
    exec python -m semantic.serve --backend "$BACKEND"
else
    exec python -m semantic.serve
fi
