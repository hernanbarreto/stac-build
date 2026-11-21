#!/bin/bash
# VERSION 2, steps 10-12 in order, each on the card; stops at the first failure.
# Log of the sequence: /workspace/tmp/new_sequence/v2/logs/back.log (its last line: "exit <code>").
E=/workspace/stac-build/experiments/new_sequence
D=/workspace/tmp/new_sequence/v2
S=/workspace/stac-build/server/projects/bufferStop_v2/scans/2026-08-11/src_default
PY=/workspace/miniforge3/envs/da3/bin/python
export NEW_SEQUENCE_SESSION=bufferStop_v2
cd /workspace/stac-build/server || exit 1
say() { echo "$(date -u +%H:%M:%S) $*"; }
say "step 10: F6 depth (the factor of each keyframe; PointDiT off)"
$PY -u $E/run_on_card.py $D/logs/step10.log "v2 step 10 - F6 depth" da3 f6_bend -- $E/v2_f6.py depth $S $D/state.npz || { say "step 10 FAILED"; exit 10; }
say "step 11: the cubes in the session's state"
$PY -u $E/v2_tie_cubes.py $S $D/state.npz $D/tie2 > $D/logs/step11_cubes.log 2>&1 || { say "step 11 (cubes) FAILED"; exit 11; }
say "step 11: the second tie (the fork's bridges, seams, closures, graph)"
V2_FIRST_GRAPH=$S/output/maplong_run/chunk_graph.json $PY -u $E/run_omega_step.py $D/logs/step11.log "v2 step 11 - second tie" $E/v2_tie.py -- \
    --image_dir $S/frames --config $S/output/vggt_omega_config.yaml --save_dir $D/tie2 --selected_frames $S/frames/selected_frames.json || { say "step 11 (tie) FAILED"; exit 11; }
say "step 12: the vote, the cleaner, the publication"
$PY -u $E/run_on_card.py $D/logs/step12.log "v2 step 12 - vote, cleaner, publication" da3 f6_bend -- $E/v2_f6.py publish $S $D/tie2/chunk_graph.json $D/step12.json || { say "step 12 FAILED"; exit 12; }
/usr/bin/curl -sk --max-time 30 -X POST https://127.0.0.1:8765/api/sessions/bufferStop_v2/cloud_changed; echo
say "steps 10-12 done"
