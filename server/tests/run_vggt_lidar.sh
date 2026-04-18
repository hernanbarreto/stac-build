#!/bin/bash
# Run VGGT-Long poses + LiDAR depth pipeline
# Usage: bash run_vggt_lidar.sh [--device cpu|cuda] [--max_frames N]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VGGT_DIR="$SERVER_DIR/../vendor/VGGT-Long"

# Activate conda environment
source /home/hernan/miniforge3/etc/profile.d/conda.sh
conda activate mapanything

export PYTHONPATH="$VGGT_DIR:$VGGT_DIR/base_models:$SERVER_DIR/../vendor/MapAnything2:$PYTHONPATH"

# Force CPU — same pattern as map_worker.py (VGGT-Long ignores --device flag)
export CUDA_VISIBLE_DEVICES=""

# DINOv2 loop detector uses relative paths from VGGT-Long dir
cd "$VGGT_DIR"

echo "================================================"
echo "VGGT-Long (Loop Closure) + LiDAR Backprojection"
echo "================================================"

python "$SCRIPT_DIR/test_vggt_lidar.py" \
    --data_dir "$SERVER_DIR/test2" \
    --output_dir "$SERVER_DIR/test2_output" \
    --stride 4 \
    --max_frames 0 \
    "$@"
