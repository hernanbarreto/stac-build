#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# NKSR mesher launcher (research evaluation, 2026-08-18)
#
# Runs reconstruction/nksr_run.py in the `nksr` conda env (torch 2.1 cu121 +
# NVIDIA NKSR wheels). Same pattern as run_pgsr.sh: the server-env orchestrator
# (reconstruction/nksr_scene.py) calls this with the cloud + pose paths; the
# script prints progress lines and writes the raw mesh PLY.
#
# LICENSE NOTE: NKSR ships under the NVIDIA Source Code License-NC —
# research/evaluation use only. Fine for this evaluation phase; revisit
# before any commercial deployment.
# ─────────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NKSR_ENV="${NKSR_CONDA_ENV:-nksr}"
CONDA_BASE="${CONDA_BASE:-/workspace/miniforge3}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# large clouds fragment the allocator badly (12+ GB reserved-unallocated seen
# on an 11M-point scene) — expandable segments reclaims it
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${SCRIPT_DIR}"
exec "${CONDA_BASE}/envs/${NKSR_ENV}/bin/python" -m reconstruction.nksr_run "$@"
