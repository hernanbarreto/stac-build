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

# Prefer conda when it's available (the pod is itself a container, so /.dockerenv
# exists there too — only the real single-env Docker image lacks miniforge).
if [ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
elif [ -f "/.dockerenv" ]; then
    echo "[DA3] Running in Docker mode (no conda)"
else
    echo "[DA3] ⚠️ Conda not found at ${CONDA_ROOT}, trying without activation"
fi

export PYTHONUNBUFFERED=1

# Add depth_anything_3 package + da3_streaming + server dir to PYTHONPATH.
# (run_da3_main.py imports da3_streaming/loop_utils from DA3_DIR and the STAC
#  subclass stray_da3_streaming from SCRIPT_DIR.)
export PYTHONPATH="${PROJECT_ROOT}/vendor/depth-anything-3/src:${DA3_DIR}:${SCRIPT_DIR}:${PYTHONPATH}"

# Force CPU if CUDA_VISIBLE_DEVICES is empty
if [ -z "${CUDA_VISIBLE_DEVICES+x}" ]; then
    :  # Not set, let DA3 auto-detect
fi

# Run DA3 Streaming via the STAC entry point (honors --selected_frames without
# patching the vendored da3_streaming.py). cwd = DA3_DIR for loop_utils imports.
cd "${DA3_DIR}"
python -u "${SCRIPT_DIR}/run_da3_main.py" "$@"
