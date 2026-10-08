#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# MapAnything (VGGT-Long) Launcher
# Runs vggt_long.py on the mapanything interpreter — FIXED BY ABSOLUTE PATH
# (docs/plan_determinismo.md point 153, 2026-10-08): the interpreter is
# STAC_PYTHON_MAPANYTHING (exported by the pipeline manager from the job's frozen
# configuration, reconstruction.precision.runner.python_mapanything), else the
# conda env's own python under CONDA_ROOT. It must exist: there is NO fallback to
# whatever `python` the PATH holds (that was the launcher's silent branch when
# CONDA_ROOT pointed to a machine this pod is not). The interpreter used is echoed.
# Follows the same pattern as run_cloudcompy.sh
# ─────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VGGT_DIR="${PROJECT_ROOT}/vendor/VGGT-Long"

# ── The interpreter (absolute path, verified) ──
CONDA_ENV="${MAPANYTHING_CONDA_ENV:-mapanything}"
CONDA_ROOT="${CONDA_ROOT:-/workspace/miniforge3}"
PY="${STAC_PYTHON_MAPANYTHING:-${CONDA_ROOT}/envs/${CONDA_ENV}/bin/python}"

if [ -z "${STAC_PYTHON_MAPANYTHING:-}" ]; then
    echo "[MapAnything] DECLARED: STAC_PYTHON_MAPANYTHING not set — using the conda env's interpreter ${PY}"
fi
if [ ! -x "${PY}" ]; then
    echo "[MapAnything] ERROR: interpreter ${PY} does not exist or is not executable" \
         "(STAC_PYTHON_MAPANYTHING / reconstruction.precision.runner.python_mapanything)" \
         "— no fallback to the PATH's python (plan point 153)" >&2
    exit 2
fi

# ── Conda environment activation — only the env's own activation hooks (library
# paths); the interpreter that runs is the absolute one above either way ──
if [ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ] && [ "${PY}" = "${CONDA_ROOT}/envs/${CONDA_ENV}/bin/python" ]; then
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
else
    echo "[MapAnything] no conda activation (interpreter outside ${CONDA_ROOT}/envs/${CONDA_ENV}, or no conda.sh) — running ${PY} directly"
fi

export PYTHONUNBUFFERED=1
echo "[MapAnything] interpreter: ${PY}"

# Run VGGT-Long with all passed arguments (unbuffered output)
cd "${VGGT_DIR}"
exec "${PY}" -u "${VGGT_DIR}/vggt_long.py" "$@"
