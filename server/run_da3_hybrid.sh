#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# DA3 Hybrid / LiDAR Streaming Launcher
# Activates the da3 conda environment and runs da3_streaming.py
# using StrayDA3Streaming (hybrid) or StrayLiDAROnly (lidar) subclass.
#
# Usage:
#   bash run_da3_hybrid.sh --mode hybrid --image_dir ... --data_dir ... --config ... --output_dir ...
#   bash run_da3_hybrid.sh --mode lidar  --image_dir ... --data_dir ... --config ... --output_dir ...
#
# --mode:     "hybrid" (DA3 + LiDAR injection) or "lidar" (LiDAR-only, no inference)
# --data_dir: Path to Stray Scanner raw data (contains depth/, odometry.csv, etc.)
# --stride:   Frame stride for Stray Scanner extraction (default: 4)
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
    echo "[DA3-Hybrid] Running in Docker mode (no conda)"
else
    echo "[DA3-Hybrid] ⚠️ Conda not found at ${CONDA_ROOT}, trying without activation"
fi

export PYTHONUNBUFFERED=1

# Add depth_anything_3 package + server dir to PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}/vendor/depth-anything-3/src:${SCRIPT_DIR}:${PYTHONPATH}"

# Force CPU if CUDA_VISIBLE_DEVICES is empty
if [ -z "${CUDA_VISIBLE_DEVICES+x}" ]; then
    :  # Not set, let DA3 auto-detect
fi

# Run the hybrid/lidar runner script
cd "${DA3_DIR}"
python -u "${SCRIPT_DIR}/run_da3_hybrid_main.py" "$@"
