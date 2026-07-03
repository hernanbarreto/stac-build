#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# CGAL WLOP launcher (surface_fit stage-1 consolidation)
# Activates the CloudComPy310 env — the only one with CGAL Python
# bindings — and runs cgal_wlop.py. Mirrors run_cloudcompy.sh.
# ─────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONDA_ENV="CloudComPy310"
CONDA_ROOT="${CONDA_ROOT:-/workspace/miniforge3}"

if [ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
else
    echo "[WLOP] ⚠️ Conda not found at ${CONDA_ROOT}, trying without activation"
fi

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
export LC_NUMERIC=C
export PYTHONUNBUFFERED=1

exec python -u "${SCRIPT_DIR}/cgal_wlop.py" "$@"
