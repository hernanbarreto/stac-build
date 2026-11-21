#!/bin/bash
# VERSION 3, after cloud A (PointDiT off) is published:
#   a copy of the session at cloud A  ->  F6 again with PointDiT on (cloud B)  ->  a copy at cloud B  ->
#   each copy painted by its distance to the other cloud ("como la sabana").
# Log of the sequence: /workspace/tmp/new_sequence/v3/logs/back.log (its last line: "exit <code>").
E=/workspace/stac-build/experiments/new_sequence
D=/workspace/tmp/new_sequence/v3
P=/workspace/stac-build/server/projects
S=$P/bufferStop_v3/scans/2026-08-11/src_default
PY=/workspace/miniforge3/envs/da3/bin/python
export NEW_SEQUENCE_SESSION=bufferStop_v3
cd /workspace/stac-build/server || exit 1
say() { echo "$(date -u +%H:%M:%S) $*"; }
copy_session() {
    local N=$1
    if [ -e "$P/$N" ]; then say "$P/$N exists already - not overwritten"; return 1; fi
    mkdir -p "$P/$N/bim_comparison" "$P/$N/coverage_history" "$P/$N/ifcs" "$P/$N/merged" "$P/$N/segmentation" "$P/$N/scans/2026-08-11" || return 1
    cp -rp "$S" "$P/$N/scans/2026-08-11/src_default" || return 1
    $PY - "$P" "$N" <<'PYEOF' || return 1
import json, sys, datetime
P, N = sys.argv[1], sys.argv[2]
d = json.load(open(f"{P}/bufferStop_v3/project.json"))
d["name"] = d["slug"] = N
now = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
d["created_at"] = now
for s in d["scans"]:
    s["added_at"] = now
json.dump(d, open(f"{P}/{N}/project.json", "w"), indent=2)
PYEOF
}
say "a copy of the session at cloud A (PointDiT off)"
copy_session bufferStop_v3_pintada_A || { say "the copy at A FAILED"; exit 21; }
say "cloud B: F6 again with PointDiT on"
$PY -u $E/v3_pointdit.py $S $D/cloud_A.ply > $D/logs/pointdit.log 2>&1 || { say "cloud B FAILED"; exit 22; }
/usr/bin/curl -sk --max-time 30 -X POST https://127.0.0.1:8765/api/sessions/bufferStop_v3/cloud_changed; echo
say "a copy of the session at cloud B (PointDiT on)"
copy_session bufferStop_v3_pintada_B || { say "the copy at B FAILED"; exit 23; }
say "the two copies painted by their distance to the other cloud"
$PY -u $E/run_on_card.py $D/logs/compare.log "v3 - the comparison painted" da3 f6_bend -- $E/v3_compare.py paint $D/cloud_A.ply $S/output/cleaned_cloud.ply $P/bufferStop_v3_pintada_A/scans/2026-08-11/src_default A || { say "painting A FAILED"; exit 24; }
$PY -u $E/run_on_card.py $D/logs/compare.log "v3 - the comparison painted" da3 f6_bend -- $E/v3_compare.py paint $D/cloud_A.ply $S/output/cleaned_cloud.ply $P/bufferStop_v3_pintada_B/scans/2026-08-11/src_default B || { say "painting B FAILED"; exit 25; }
say "done"
