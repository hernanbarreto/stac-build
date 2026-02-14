#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# CloudCompPy Post-Processing Launcher
# Sets up the CloudComPy310 environment and runs the Python script
# ─────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CLOUDCOMPY_ROOT="${PROJECT_ROOT}/vendor/cloudcompy"
CONDA_ENV="CloudComPy310"
CONDA_ROOT="/home/hernan/miniforge3"

# Activate conda
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Set CloudCompPy paths
export PYTHONPATH="${CLOUDCOMPY_ROOT}/lib/cloudcompare:${CLOUDCOMPY_ROOT}/doc/PythonAPI_test:${PYTHONPATH}"
export LD_LIBRARY_PATH="${CLOUDCOMPY_ROOT}/lib/cloudcompare:${CLOUDCOMPY_ROOT}/lib/cloudcompare/plugins:${LD_LIBRARY_PATH}"
export LC_NUMERIC=C
export PYTHONUNBUFFERED=1

# Run the post-processing script with all passed arguments (unbuffered output)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python -u "${SCRIPT_DIR}/cloudcompy_postprocess.py" "$@"
