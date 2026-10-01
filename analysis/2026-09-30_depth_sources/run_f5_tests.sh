#!/bin/bash
# F5 R1 poses: epoch 2 = F5 camera (fx 392), epoch 3 = DA3's own camera — no confidence gate, fused.
cd /workspace/stac-build/server
export HF_HOME=/workspace/hf_cache OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 STREAM_CONF_DROP_PCT=0
PY=/workspace/miniforge3/envs/da3/bin/python
S=/workspace/stac-build/analysis/2026-09-30_depth_sources/da3_stream_f5.py
STREAM_EPOCH=2 STREAM_POSES=f5    taskset -c 0-7 nice -n 5 env -u HF_HUB_ENABLE_HF_TRANSFER $PY -u $S
STREAM_EPOCH=3 STREAM_POSES=f5da3 taskset -c 0-7 nice -n 5 env -u HF_HUB_ENABLE_HF_TRANSFER $PY -u $S
