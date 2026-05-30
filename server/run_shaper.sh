#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# ShapeR Batch Inference Launcher
# Activates the `shaper` conda env, sets PYTHONPATH to vendor/ShapeR,
# and runs run_shaper_batch.py with all forwarded arguments.
# ─────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SHAPER_DIR="${PROJECT_ROOT}/vendor/ShapeR"

CONDA_ENV="${SHAPER_CONDA_ENV:-shaper}"
CONDA_ROOT="${CONDA_ROOT:-/home/hernan/miniforge3}"

# Prefer conda when it's available (the pod is itself a container, so /.dockerenv
# exists there too — only the real single-env Docker image lacks miniforge).
if [ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
elif [ -f "/.dockerenv" ]; then
    echo "[Shaper] Running in Docker mode (no conda)"
else
    echo "[Shaper] ⚠️ Conda not found at ${CONDA_ROOT}, trying without activation"
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${SHAPER_DIR}:${PYTHONPATH}"

# Run from vendor/ShapeR so relative checkpoint paths in infer_shape.py work
cd "${SHAPER_DIR}"
python -u "${SCRIPT_DIR}/run_shaper_batch.py" --ckpt_root "${SHAPER_DIR}" "$@"
