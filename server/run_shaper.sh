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
# Auto-detect the miniforge/conda root instead of hard-coding one host's path.
# Without this, a backend started without CONDA_ROOT exported falls through to
# "trying without activation" and runs ShapeR under whatever python is on PATH
# (not the `shaper` env), which is how GPU/torch-version mismatches sneak in.
if [ -z "${CONDA_ROOT:-}" ]; then
    for _root in /workspace/miniforge3 "${HOME}/miniforge3" /home/hernan/miniforge3 /opt/conda; do
        if [ -f "${_root}/etc/profile.d/conda.sh" ]; then
            CONDA_ROOT="${_root}"
            break
        fi
    done
fi
CONDA_ROOT="${CONDA_ROOT:-/workspace/miniforge3}"

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
# `exec` so python REPLACES this shell (same PID). The backend sets
# PR_SET_PDEATHSIG on the process it spawns (this shell); without `exec`, python
# is a grandchild that does NOT inherit PDEATHSIG and survives as an orphan when
# the backend dies (burning CPU/GPU). With `exec`, python keeps this shell's PID
# and its PDEATHSIG, so killing the backend kills ShapeR too.
exec python -u "${SCRIPT_DIR}/run_shaper_batch.py" --ckpt_root "${SHAPER_DIR}" "$@"
