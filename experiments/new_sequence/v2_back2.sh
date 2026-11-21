#!/bin/bash
# VERSION 2, step 11's solve again (the seams' identity written exactly) and step 12 again on it.
E=/workspace/stac-build/experiments/new_sequence
D=/workspace/tmp/new_sequence/v2
S=/workspace/stac-build/server/projects/bufferStop_v2/scans/2026-08-11/src_default
PY=/workspace/miniforge3/envs/da3/bin/python
export NEW_SEQUENCE_SESSION=bufferStop_v2
cd /workspace/stac-build/server || exit 1
say() { echo "$(date -u +%H:%M:%S) $*"; }
say "step 11 again: the second tie (bridge fits reused; the graph solved from the exact identity)"
V2_FIRST_GRAPH=$S/output/maplong_run/chunk_graph.json $PY -u $E/run_omega_step.py $D/logs/step11b.log "v2 step 11 - second tie (solve again)" $E/v2_tie.py -- \
    --image_dir $S/frames --config $S/output/vggt_omega_config.yaml --save_dir $D/tie2 --selected_frames $S/frames/selected_frames.json || { say "step 11 FAILED"; exit 11; }
say "step 12 again: the vote, the cleaner, the publication — with the second tie applied"
$PY -u $E/run_on_card.py $D/logs/step12b.log "v2 step 12 - vote, cleaner, publication (with the second tie)" da3 f6_bend -- $E/v2_f6.py publish $S $D/tie2/chunk_graph.json $D/step12.json || { say "step 12 FAILED"; exit 12; }
/usr/bin/curl -sk --max-time 30 -X POST https://127.0.0.1:8765/api/sessions/bufferStop_v2/cloud_changed; echo
say "done"
