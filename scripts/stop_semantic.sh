#!/bin/bash
# STAC-Builder — stop the semantic service and free the GPU.
# vLLM spawns an EngineCore child that must be killed too, or ~23 GiB stays
# pinned. Kills the port holder + any vLLM engine procs. Idempotent.
#
#   bash scripts/stop_semantic.sh
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

PORT="${1:-8799}"

# 1) whatever listens on the service port
for pid in $(ss -ltnp 2>/dev/null | grep ":${PORT}" | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u); do
    kill -9 "$pid" 2>/dev/null && echo "killed port ${PORT} holder pid=$pid"
done

# 2) the vLLM API server + EngineCore worker + its resource tracker
pkill -9 -f "vllm serve"        2>/dev/null && echo "killed vllm serve"
pkill -9 -f "VLLM::EngineCore"  2>/dev/null && echo "killed EngineCore"
pkill -9 -f "semantic\.serve"   2>/dev/null

sleep 2
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -1)
echo "GPU memory used now: ${USED}"
