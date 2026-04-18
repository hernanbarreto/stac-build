#!/bin/bash
# Unified Cloud: Merge hybrid DA3 + LiDAR (backprojected with DA3-streaming poses)
# Then clean with CloudCompPy (dedup + SOR + voxel)
#
# Prerequisites: run_da3_hybrid.sh must have completed first
#
# Usage: cd server && bash tests/run_unified_cloud.sh [--skip_cloudcompy]
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

source /home/hernan/miniforge3/etc/profile.d/conda.sh
conda activate da3

export PYTHONPATH="$SERVER_DIR:$SCRIPT_DIR:$PYTHONPATH"

echo "================================================"
echo "Unified Cloud: DA3 Hybrid + LiDAR"
echo "================================================"

python "$SCRIPT_DIR/test_unified_cloud.py" \
    --data_dir "$SERVER_DIR/test2" \
    --hybrid_dir "$SERVER_DIR/test2_da3_hybrid" \
    --output_dir "$SERVER_DIR/test2_da3_hybrid/unified" \
    --stride 4 \
    --max_frames 0 \
    --voxel_size 0.002 \
    --sor_knn 8 \
    --sor_sigma 3.0 \
    "$@"
