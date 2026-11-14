#!/bin/bash
# STAC-Builder — STAC's voice service (speech → text, text → speech), in the `voice` env.
#
#   bash scripts/serve_voice.sh          (the backend launches it: server/voice/service.py)
#
# Whisper + Kokoro on the device of config.yaml voice.device, HTTP on voice.service.host:port.
# Build the env once with scripts/build_voice_env.sh.
set -e
STAC_ROOT="/workspace/stac-build"
source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate voice
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export HF_HUB_ENABLE_HF_TRANSFER=0      # the env has no hf_transfer; the weights are fetched once
# the pod grants ~26 CPUs and shows 252: a torch job left to its own thread count crawls
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONHASHSEED=0
cd "$STAC_ROOT/server"
exec python -m voice.serve "$@"
