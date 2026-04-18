#!/bin/bash
# Unified Hybrid Pipeline: VGGT-Long loop closure + LiDAR + DA3 depth
# Usage: cd server && bash tests/run_hybrid.sh [--max_frames N]
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VGGT_DIR="$SERVER_DIR/../vendor/VGGT-Long"

source /home/hernan/miniforge3/etc/profile.d/conda.sh
conda activate mapanything

export PYTHONPATH="$VGGT_DIR:$VGGT_DIR/base_models:$SERVER_DIR/../vendor/MapAnything2:$SERVER_DIR:$PYTHONPATH"

# CPU (VGGT-Long uses StrayLiDAR adapter — no GPU needed)
export CUDA_VISIBLE_DEVICES=""

# VGGT-Long needs to resolve relative paths from its dir
cd "$VGGT_DIR"

echo "================================================"
echo "Hybrid Pipeline (VGGT-Long poses + LiDAR + DA3)"
echo "================================================"

python "$SCRIPT_DIR/test_hybrid_backproject.py" \
    --data_dir "$SERVER_DIR/test2" \
    --output_dir "$SERVER_DIR/test2_hybrid_output" \
    --stride 4 \
    --max_frames 0 \
    "$@"
