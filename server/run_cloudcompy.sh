#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# CloudCompPy Post-Processing Launcher
# Sets up the CloudComPy310 environment and runs the Python script
# Works in both Docker (no conda) and native (conda) environments
# ─────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CLOUDCOMPY_ROOT="${PROJECT_ROOT}/vendor/cloudcompy"

# Detect if we're inside Docker (/.dockerenv exists or PYTHONPATH already set)
if [ -f /.dockerenv ] || echo "$PYTHONPATH" | grep -q "cloudcompare"; then
    # Docker mode: env vars already set by Dockerfile, no conda needed
    echo "[CloudComPy] Running in Docker mode (no conda)"
else
    # Native mode: activate conda environment
    CONDA_ENV="CloudComPy310"
    CONDA_ROOT="${CONDA_ROOT:-/home/hernan/miniforge3}"

    if [ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
        source "${CONDA_ROOT}/etc/profile.d/conda.sh"
        conda activate "${CONDA_ENV}"
    else
        echo "[CloudComPy] ⚠️ Conda not found at ${CONDA_ROOT}, trying without activation"
    fi
fi

# Set CloudCompPy paths (safe to set even if already set)
export PYTHONPATH="${CLOUDCOMPY_ROOT}/lib/cloudcompare:${CLOUDCOMPY_ROOT}/doc/PythonAPI_test:${PYTHONPATH}"
export LD_LIBRARY_PATH="${CLOUDCOMPY_ROOT}/lib/cloudcompare:${CLOUDCOMPY_ROOT}/lib/cloudcompare/plugins:${LD_LIBRARY_PATH}"
export LC_NUMERIC=C
export PYTHONUNBUFFERED=1

# Run the post-processing script with all passed arguments (unbuffered output)
python -u "${SCRIPT_DIR}/cloudcompy_postprocess.py" "$@"

