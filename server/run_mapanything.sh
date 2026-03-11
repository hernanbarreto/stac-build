#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# MapAnything (VGGT-Long) Launcher
# Activates the mapanything conda environment and runs vggt_long.py
# Follows the same pattern as run_cloudcompy.sh
# ─────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VGGT_DIR="${PROJECT_ROOT}/vendor/VGGT-Long"

# ── Conda environment activation ──
CONDA_ENV="${MAPANYTHING_CONDA_ENV:-mapanything}"
CONDA_ROOT="${CONDA_ROOT:-/home/hernan/miniforge3}"

if [ -f "/.dockerenv" ]; then
    echo "[MapAnything] Running in Docker mode (no conda)"
else
    if [ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
        source "${CONDA_ROOT}/etc/profile.d/conda.sh"
        conda activate "${CONDA_ENV}"
    else
        echo "[MapAnything] ⚠️ Conda not found at ${CONDA_ROOT}, trying without activation"
    fi
fi

export PYTHONUNBUFFERED=1

# Run VGGT-Long with all passed arguments (unbuffered output)
cd "${VGGT_DIR}"
python -u "${VGGT_DIR}/vggt_long.py" "$@"
