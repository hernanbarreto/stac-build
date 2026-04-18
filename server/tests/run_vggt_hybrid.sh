#!/bin/bash
# Run VGGT-Long Hybrid pipeline (LiDAR + DA3 Giant + ARKit + SALAD Loop Closure)
# Usage: bash run_vggt_hybrid.sh [--max_frames N]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VGGT_DIR="$SERVER_DIR/../vendor/VGGT-Long"

# Activate conda environment
source /home/hernan/miniforge3/etc/profile.d/conda.sh
conda activate mapanything

export PYTHONPATH="$VGGT_DIR:$VGGT_DIR/base_models:$SERVER_DIR/../vendor/MapAnything2:$PYTHONPATH"

# Force CPU (to avoid OOM on DA3/MapAnything unless we are confident, 
# but DA3 Giant requires a lot of VRAM. Let's start with CPU just in case, 
# or comment this out to use GPU if 24GB is enough for DA3 Giant + VGGT)
export CUDA_VISIBLE_DEVICES=""

cd "$VGGT_DIR"

echo "================================================"
echo "VGGT-Long HYBRID Pipeline (DA3 + LiDAR)"
echo "================================================"

python "$SCRIPT_DIR/test_vggt_hybrid.py" \
    --data_dir "$SERVER_DIR/test2" \
    --output_dir "$SERVER_DIR/test2_hybrid_output" \
    --stride 4 \
    --max_frames 0 \
    "$@"
