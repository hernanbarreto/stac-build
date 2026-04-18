#!/bin/bash
# Hybrid Backprojection: ARKit poses + LiDAR depth + DA3 calibrated
# Usage: cd server && bash tests/run_hybrid_backproject.sh [--max_frames N]
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

source /home/hernan/miniforge3/etc/profile.d/conda.sh
conda activate mapanything

export PYTHONPATH="$SERVER_DIR:$PYTHONPATH"

# CPU only (DA3 extraction runs in its own env via subprocess)
export CUDA_VISIBLE_DEVICES=""

echo "================================================"
echo "Hybrid Backprojection (ARKit + LiDAR + DA3)"
echo "================================================"

python "$SCRIPT_DIR/test_hybrid_backproject.py" \
    --data_dir "$SERVER_DIR/test2" \
    --output_dir "$SERVER_DIR/test2_hybrid_output" \
    --stride 4 \
    --max_frames 0 \
    "$@"
