#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Activate DA3 environment natively
source /workspace/miniforge3/etc/profile.d/conda.sh
conda activate da3

export PYTHONPATH="$SCRIPT_DIR/../vendor/depth-anything-3/src:$PYTHONPATH"

# Run the python script
python "$SCRIPT_DIR/extract_da3_depth.py" "$@"
