#!/bin/bash
# STAC-Builder — Phase 7 reproducible end-to-end demo (Pitch 2).
#
# video-in (already framed) -> auto-segmentation -> reconstruction with semantic
# anchoring (Phase R store) -> classification (Ph2) -> findings (Ph3) ->
# coverage (Ph4) -> 6-8 spatial questions answered WITH traces (Ph5) ->
# bilingual report draft (Ph6). Reconstruction (DA3/VGGT) is assumed already run
# for the session (its depth/poses live under output/); this script drives the
# semantic pipeline that consumes it.
#
# Usage:
#   scripts/demo_pitch2.sh <session_dir> [scene.db] [out_dir]
# The semantic service must be up (scripts/serve_semantic.sh).
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC
set -euo pipefail

SESSION="${1:?usage: demo_pitch2.sh <session_dir> [scene.db] [out_dir]}"
SCENE="${2:-/tmp/stac_demo/scene.db}"
OUT="${3:-/tmp/stac_demo}"
PY="${PY:-/workspace/miniforge3/envs/stac-build/bin/python}"
SERVER="$(cd "$(dirname "$0")/../server" && pwd)"
mkdir -p "$OUT" "$(dirname "$SCENE")"
cd "$SERVER"

echo "== 0. semantic service healthcheck =="
curl -sf -m 5 http://127.0.0.1:8799/v1/models >/dev/null && echo "  service UP" \
  || { echo "  service DOWN — start scripts/serve_semantic.sh"; exit 1; }

echo "== 1. auto-segmentation (understanding-driven, open-vocabulary) =="
"$PY" -m segmentation.autoprompt.session_builder --session "$SESSION" \
  --output "$SESSION/output" || echo "  (segmentation.json expected under output/)"

echo "== 2. Phase R — build the canonical instance store (poses+depth+masklets) =="
"$PY" - "$SESSION" "$SCENE" <<'PYEOF'
import sys
from phase_r.pipeline import PhaseRPipeline
session, scene = sys.argv[1], sys.argv[2]
PhaseRPipeline(session, f"{session}/output", scene, config={"phase_r": {}}).run()
print("  store:", scene)
PYEOF

echo "== 3. Phase 2 — classify / enrich instances =="
"$PY" -m phase2_classify.cli --scene "$SCENE" --session "$SESSION" || true

echo "== 4. Phase 3 — 3D-anchored visual findings =="
"$PY" -m phase3_findings.cli --scene "$SCENE" --session "$SESSION" \
  --output "$SESSION/output" --max-frames 24 || true

echo "== 5. Phase 4 — coverage report + recapture checklist =="
"$PY" -m phase4_qc.cli coverage --scene "$SCENE" --session "$SESSION" \
  --output "$SESSION/output" --out "$OUT/coverage" || true

echo "== 6. Phase 5 — 8 showcase questions answered WITH traces =="
"$PY" - "$SCENE" <<'PYEOF'
import sys
from phase5_qa.orchestrator import SpatialQA
qa = SpatialQA(sys.argv[1], backend="qwen_local")
qs = [
    "How many objects are in the scene?",
    "What is the bounding-box size of the floor?",
    "What is the minimum clearance between the wall and the door?",
    "Is the wall plumb, and by how much?",
    "What defects have been reported in the scene?",
    "What is the alignment health of the floor?",
    "What is the distance between the floor and object 9999?",   # trap
    "Which object is the largest by volume?",
]
for q in qs:
    r = qa.ask(q)
    tools = [t["tool"] for t in r["tool_trace"]]
    print(f"\nQ: {q}\nA: {r['answer'][:200]}\n   trace: {tools}")
qa.close()
PYEOF

echo "== 7. Phase 6 — bilingual report draft (ES/FR) =="
"$PY" -m phase6_report.cli --scene "$SCENE" --session "$SESSION" \
  --out "$OUT/report" --lang both \
  --coverage "$OUT/coverage/coverage_report.json" || \
  "$PY" -m phase6_report.cli --scene "$SCENE" --session "$SESSION" \
  --out "$OUT/report" --lang both

echo "== DONE =="
echo "  store   : $SCENE"
echo "  reports : $OUT/report/report_es.md , report_fr.md"
echo "  coverage: $OUT/coverage/recapture_checklist.txt"
