#!/bin/bash
# DA3-streaming + Stray Scanner hybrid
# Auto-detects: if Stray data exists → hybrid, else → pure DA3
# Usage: cd server && bash tests/run_da3_hybrid.sh [--max_frames N]
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DA3_STREAM_DIR="$SERVER_DIR/../vendor/depth-anything-3/da3_streaming"
DA3_SRC_DIR="$SERVER_DIR/../vendor/depth-anything-3/src"

source /home/hernan/miniforge3/etc/profile.d/conda.sh
conda activate da3

export PYTHONPATH="$DA3_STREAM_DIR:$DA3_SRC_DIR:$SERVER_DIR:$SCRIPT_DIR:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=""
export XFORMERS_DISABLED=1  # Force DINOv2 to use standard attention on CPU

echo "================================================"
echo "DA3-streaming Hybrid Pipeline"
echo "================================================"

cd "$DA3_STREAM_DIR"

python "$SCRIPT_DIR/test_da3_hybrid.py" \
    --data_dir "$SERVER_DIR/test2" \
    --output_dir "$SERVER_DIR/test2_da3_hybrid" \
    --stride 4 \
    --max_frames 0 \
    "$@"
