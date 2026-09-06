#!/usr/bin/env bash
# Run ANY python script inside the CloudComPy310 environment (same setup as
# run_cloudcompy.sh, which is hard-wired to cloudcompy_postprocess.py).
#   run_cloudcompy_script.sh <script.py> [args...]
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CLOUDCOMPY_ROOT="${PROJECT_ROOT}/vendor/CloudComPy310"
CONDA_ENV="CloudComPy310"
CONDA_ROOT="${CONDA_ROOT:-/workspace/miniforge3}"
if [ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi
export PYTHONPATH="${CLOUDCOMPY_ROOT}/lib/cloudcompare:${CLOUDCOMPY_ROOT}/doc/PythonAPI_test:${PYTHONPATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CLOUDCOMPY_ROOT}/lib/cloudcompare:${CLOUDCOMPY_ROOT}/lib/cloudcompare/plugins:${LD_LIBRARY_PATH}"
export QT_QPA_PLATFORM=offscreen
export LC_NUMERIC=C
export PYTHONUNBUFFERED=1
SCRIPT="$1"; shift
python -u "${SCRIPT}" "$@"
