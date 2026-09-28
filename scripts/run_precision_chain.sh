#!/usr/bin/env bash
# The precision chain F0 → F7 (claude_stac.txt §4) on a session the pipeline has
# reconstructed (reconstruction → CloudCompy → VLM → SAM3, epoch 0). Until F9 wires
# these stages into "Reconstruir", this script is the one way to run them, in order,
# each in its own env; the first failure stops the chain with its reason.
#
#   bash scripts/run_precision_chain.sh <session_dir>
#
# e.g. bash scripts/run_precision_chain.sh server/projects/pccr/scans/2026-08-24/src_default
#
# GPU steps (F4 tracks, F3 probe, F6 sweep, F6 COLMAP) run one at a time. Every
# published stage (gauge, refine, fuse) is a selectable geometry epoch: epoch 0
# stays intact. Log: <session>/output/precision/chain.log
set -euo pipefail

SESSION="$(realpath "${1:?usage: run_precision_chain.sh <session_dir>}")"
ROOT="/workspace/stac-build"
ENVS="/workspace/miniforge3/envs"
DA3="$ENVS/da3/bin/python"
MAP="$ENVS/mapanything/bin/python"
LOG="$SESSION/output/precision/chain.log"
mkdir -p "$(dirname "$LOG")"

export CUBLAS_WORKSPACE_CONFIG=":4096:8"     # deterministic cuBLAS
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_CBWR=COMPATIBLE

step() {                                    # step <label> <python> <module> [args...]
    local label="$1" py="$2" mod="$3"; shift 3
    echo "[$(date '+%F %T')] ── $label: python -m $mod $*" | tee -a "$LOG"
    set +e
    ( cd "$ROOT/server" && "$py" -m "$mod" --session "$SESSION" "$@" ) 2>&1 | tee -a "$LOG"
    local rc=${PIPESTATUS[0]}
    set -e
    if [ "$rc" -ne 0 ]; then
        echo "[$(date '+%F %T')] ✗ $label failed (exit $rc) — the chain stops here" | tee -a "$LOG"
        exit "$rc"
    fi
    echo "[$(date '+%F %T')] ✓ $label" | tee -a "$LOG"
}

echo "[$(date '+%F %T')] precision chain on $SESSION" | tee -a "$LOG"
step "F0 session camera"            "$DA3" precision.camera
step "F3 visit drift (measure)"     "$DA3" correction.visit_drift_run --mode measure
step "F2 continuous gauge (apply)"  "$DA3" precision.gauge
step "F4 native-pixel tracks (GPU)" "$MAP" precision.tracks
step "F3 Omega resolution probe (GPU, report)" "$MAP" precision.omega_probe
step "F5 joint refinement (apply)"  "$MAP" precision.refine
step "F6 native depth sweep (GPU)"  "$DA3" precision.depth_sweep
step "F6 COLMAP reference (GPU, A/B)" "$DA3" precision.depth_colmap
step "F7 witness fusion (publish)"  "$DA3" precision.fuse
echo "[$(date '+%F %T')] precision chain done — the fused cloud is the live epoch; the" \
     "previous ones stay selectable" | tee -a "$LOG"
