#!/bin/bash
# THE NEW ORDER, step 2: Omega per cube + the DA3 lock per cube with the pipeline's own code, stopped before the tie.
# Launches it detached on the card (the chat and the voice stop while it runs and come back at the end).
# Log: /workspace/tmp/new_sequence/logs/step2.log
set -e
E=/workspace/stac-build/experiments/new_sequence
D=/workspace/tmp/new_sequence
S=/workspace/stac-build/server/projects/bufferStop_cubos/scans/2026-08-11/src_default
mkdir -p "$D/run" "$D/logs"
: > "$D/logs/step2.log"
cd /workspace/stac-build/server
setsid nohup /workspace/miniforge3/envs/da3/bin/python -u "$E/run_omega_step.py" "$D/logs/step2.log" \
    "step 2 - Omega per cube + DA3 lock" "$E/step2_omega.py" -- \
    --image_dir "$S/frames" --config "$S/output/vggt_omega_config.yaml" \
    --save_dir "$D/run/maplong_run" --selected_frames "$S/frames/selected_frames.json" \
    > "$D/logs/step2.driver.log" 2>&1 < /dev/null &
echo "step 2 launched; log: $D/logs/step2.log"
