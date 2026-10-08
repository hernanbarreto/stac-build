#!/bin/bash
# STAC-Builder — Semantic Service launcher (vLLM / Qwen3-VL)
#
# Runs the persistent semantic endpoint in the `semantic` conda env. All model
# flags come from the `semantic:` block via semantic.serve — of server/config.yaml,
# or of the job's FROZEN output/run_config.yaml when the VLM stage launches its
# own engine (docs/plan_determinismo.md point 155):
#
#   bash scripts/serve_semantic.sh                              # default backend (qwen_local)
#   bash scripts/serve_semantic.sh qwen_local_large             # a named backend
#   bash scripts/serve_semantic.sh --backend qwen_local --config <session>/output/run_config.yaml
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC
set -e

STAC_ROOT="/workspace/stac-build"
SERVER_DIR="$STAC_ROOT/server"

# a bare first argument is the backend name (the old form); `--...` arguments go to
# semantic.serve as they are
ARGS=()
if [ -n "$1" ] && [ "${1#--}" = "$1" ]; then
    ARGS+=("--backend" "$1")
    shift
fi
ARGS+=("$@")

source /workspace/miniforge3/etc/profile.d/conda.sh
export CONDA_ROOT=/workspace/miniforge3
conda activate semantic

# Guard: never start a second vLLM instance (two at gpu_mem_util 0.5 fill the
# whole card and starve the reconstruction with CUDA OOM). A second launch is an
# ERROR for the caller (exit 1), not a silent success — semantic.service checks
# the process table before launching and never relies on this.
SEM_PORT="$(python - <<'EOF'
import yaml
cfg = yaml.safe_load(open("/workspace/stac-build/server/config.yaml"))
# The real key is semantic.service.port (semantic/semantic_config.py) — the old
# read of semantic.port only worked because its fallback matched the default.
print(cfg.get("semantic", {}).get("service", {}).get("port", 8799))
EOF
)"
if curl -s -o /dev/null -m 3 "http://127.0.0.1:${SEM_PORT}/health"; then
    echo "⚠️  Semantic service already running on port ${SEM_PORT} — not starting a second instance."
    echo "    Kill it first if you really want to restart: pkill -f 'vllm serve'"
    exit 1
fi

# HF cache on the network volume; OFFLINE: the weights are local and verified by
# semantic.serve against their pinned sha256 (point 94) — nothing is ever fetched.
export HF_HOME=/workspace/hf_cache
export HF_HUB_OFFLINE=1
export VLLM_LOGGING_LEVEL=INFO
# The deterministic environment of every STAC process (server/repro.py DETERMINISTIC_ENV,
# point 152): Python's string hashing fixed, the cuBLAS workspace pinned. semantic.serve
# re-asserts both (repro.deterministic_env refuses another value) and adds
# VLLM_BATCH_INVARIANT=1 (point 80).
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export VLLM_BATCH_INVARIANT=1

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

exec python -m semantic.serve "${ARGS[@]}"
