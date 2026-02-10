#!/bin/bash
# Quick test: verify DA3 API imports correctly in da3 env
source ~/miniforge3/etc/profile.d/conda.sh
conda activate da3
python -c "
import sys
sys.path.insert(0, '/home/hernan/Depth-Anything-3/src')
from depth_anything_3.api import DepthAnything3
print('DA3 API import: OK')
print('DepthAnything3 class found')
"
