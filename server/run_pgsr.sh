#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# PGSR precision-mode trainer launcher (precision task, Phase D)
#
# Runs reconstruction/pgsr_train.py in the `pgsr` conda env (torch cu128 +
# compiled diff-plane-rasterization / simple-knn). Same pattern as
# run_da3.sh / run_mapanything.sh: the map_worker calls this with the
# scene/model/render dirs; everything else comes from config.yaml via CLI.
#
# Usage:
#   bash run_pgsr.sh --scene <output>/pgsr_scene --model_dir <output>/pgsr_model \
#       --render_dir <output>/pgsr_render [--iterations N] [--pose_refine] [--quick]
# ─────────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PGSR_ENV="${PGSR_CONDA_ENV:-pgsr}"
CONDA_BASE="${CONDA_BASE:-/workspace/miniforge3}"

export STAC_PGSR_VENDOR="${PROJECT_ROOT}/vendor/pgsr"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6}"
export PYTHONUNBUFFERED=1

cd "${SCRIPT_DIR}"
exec "${CONDA_BASE}/envs/${PGSR_ENV}/bin/python" -m reconstruction.pgsr_train "$@"
