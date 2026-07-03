#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# MeshFlow batch launcher (replaces run_shaper.sh)
# Activates the `meshflow` conda env (torch 2.8 cu126, A100 sm_80),
# puts vendor/meshflow on PYTHONPATH and runs run_meshflow_batch.py.
# Uses `exec` so the python process inherits PDEATHSIG from the
# backend and never outlives it (no orphaned GPU jobs).
# ─────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MESHFLOW_DIR="${PROJECT_ROOT}/vendor/meshflow"

CONDA_ENV="meshflow"
CONDA_ROOT="${CONDA_ROOT:-/workspace/miniforge3}"

if [ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
else
    echo "[MeshFlow] ⚠️ Conda not found at ${CONDA_ROOT}, trying without activation"
fi

export PYTHONPATH="${MESHFLOW_DIR}:${PYTHONPATH}"
export PYTHONUNBUFFERED=1

exec python -u "${SCRIPT_DIR}/run_meshflow_batch.py" \
    --model_path "${MESHFLOW_DIR}/ckpt/meshflow" "$@"
