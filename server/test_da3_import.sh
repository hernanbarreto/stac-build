#!/bin/bash
# Quick test: verify DA3 API imports correctly in da3 env
source ~/miniforge3/etc/profile.d/conda.sh
conda activate da3
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
python -c "
import sys
sys.path.insert(0, '${PROJECT_ROOT}/vendor/depth_anything_3/src')
from depth_anything_3.api import DepthAnything3
print('DA3 API import: OK')
print('DepthAnything3 class found')
"
