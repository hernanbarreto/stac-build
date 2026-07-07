#!/bin/bash
# STAC-Builder — build the Phase 0 `semantic` conda env (vLLM / Qwen3-VL).
#
# Standalone quick spin-up on a fresh cloud box (setup_pod_envs.sh also creates
# it via docs/migration/environment_semantic.yml). Idempotent: skips if present.
#
#   bash scripts/build_semantic_env.sh
#
# Then download weights and launch:
#   bash setup_weights.sh semantic            # or: semantic-large
#   bash scripts/serve_semantic.sh            # 127.0.0.1:8799
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC
set -e
source "${CONDA_ROOT:-/workspace/miniforge3}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "semantic"; then
    echo "[semantic] env already exists — skipping (conda env update to refresh)"
    exit 0
fi

echo "[$(date +%T)] creating env semantic (python 3.11)"
conda create -y -n semantic python=3.11
conda activate semantic

echo "[$(date +%T)] pip install vllm==0.19.0 (brings torch/transformers)"
pip install --no-input "vllm==0.19.0"

echo "[$(date +%T)] pip install semantic_client deps"
pip install --no-input openai pydantic pillow requests pyyaml numpy

echo "[$(date +%T)] DONE. versions:"
python -c "import vllm, torch, transformers; print('vllm', vllm.__version__, 'torch', torch.__version__, 'tf', transformers.__version__)"
