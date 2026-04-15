#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# DA3 Streaming Launcher
# Activates the da3 conda environment and runs da3_streaming.py
# Same pattern as run_mapanything.sh
# ─────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DA3_DIR="${PROJECT_ROOT}/vendor/depth-anything-3/da3_streaming"

# ── Conda environment activation ──
CONDA_ENV="${DA3_CONDA_ENV:-da3}"
CONDA_ROOT="${CONDA_ROOT:-/home/hernan/miniforge3}"

if [ -f "/.dockerenv" ]; then
    echo "[DA3] Running in Docker mode (no conda)"
else
    if [ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
        source "${CONDA_ROOT}/etc/profile.d/conda.sh"
        conda activate "${CONDA_ENV}"
    else
        echo "[DA3] ⚠️ Conda not found at ${CONDA_ROOT}, trying without activation"
    fi
fi

export PYTHONUNBUFFERED=1

# Add depth_anything_3 package to PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}/vendor/depth-anything-3/src:${PYTHONPATH}"

# Force CPU if CUDA_VISIBLE_DEVICES is empty
if [ -z "${CUDA_VISIBLE_DEVICES+x}" ]; then
    :  # Not set, let DA3 auto-detect
fi

# Run DA3 Streaming with all passed arguments (unbuffered output)
cd "${DA3_DIR}"
python -u "${DA3_DIR}/da3_streaming.py" "$@"
