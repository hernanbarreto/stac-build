#!/usr/bin/env bash
# ===========================================================================
# STAC-Builder — claude_stac_slam2.txt phase 5, the runs, sequenced ON THE POD
# ---------------------------------------------------------------------------
#   bash scripts/run_slam2_phase5.sh [--scenes pccr,zaragoza] [--variants a,b,c] [--skip-update] [--keep-backend]
#
# One GPU task at a time (prompt rule 9): the backend (and with it vLLM) is stopped first — the
# fork refuses a card that is not free — then stac-build is brought to slam2-omega and the fork to
# stac-slam2 (scripts/update_pod_slam2.sh, with both test batteries), then every variant runs as its
# OWN evaluation run, scene by scene, so one session copy at a time holds the chunk npys (10–40 GB).
# A variant that fails is recorded by the battery and the next one runs. Logs under
# analysis/2026-10-29_slam2/; the runs under runs/eval/slam2_<scene>_<variant>/.
# The backend is NOT restarted here (the viewer would relaunch vLLM onto the card): start it by hand
# (tmux `backend`: bash scripts/start.sh) when the runs are through.
# ===========================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/workspace/miniforge3}"
PY_DA3="${CONDA_ROOT}/envs/da3/bin/python"
LOGDIR="${ROOT}/analysis/2026-10-29_slam2"
SCENES="pccr,zaragoza"
VARIANTS="reference,slam2_focal_common,slam2_attention_gate,slam2_attention_gate_salad,slam2_all"
DO_UPDATE=1; KEEP_BACKEND=0
while [ $# -gt 0 ]; do
  case "$1" in
    --scenes) SCENES="$2"; shift 2 ;;
    --variants) VARIANTS="$2"; shift 2 ;;
    --skip-update) DO_UPDATE=0; shift ;;
    --keep-backend) KEEP_BACKEND=1; shift ;;
    *) echo "usage: $0 [--scenes a,b] [--variants a,b] [--skip-update] [--keep-backend]"; exit 2 ;;
  esac
done
mkdir -p "${LOGDIR}"
MAIN_LOG="${LOGDIR}/phase5_$(date +%Y%m%d_%H%M%S).log"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "${MAIN_LOG}"; }

# ── 0. the card: the backend and vLLM down (USER 2026-10-31: "bájalo vos") ──
if [ "${KEEP_BACKEND}" = 0 ]; then
  if pgrep -f "uvicorn main:app" >/dev/null; then
    log "stopping the backend (uvicorn main:app) — vLLM dies with it"
    pkill -TERM -f "uvicorn main:app" || true
    for _ in $(seq 1 30); do pgrep -f "uvicorn main:app" >/dev/null || break; sleep 1; done
    pgrep -f "uvicorn main:app" >/dev/null && { log "backend still up after 30 s — SIGKILL"; pkill -KILL -f "uvicorn main:app" || true; }
  fi
  if pgrep -f "vllm serve|semantic.serve|VLLM::" >/dev/null; then
    log "stopping vLLM"
    pkill -TERM -f "vllm serve|semantic.serve|VLLM::" || true
    for _ in $(seq 1 60); do pgrep -f "vllm serve|semantic.serve|VLLM::" >/dev/null || break; sleep 1; done
    pgrep -f "vllm serve|semantic.serve|VLLM::" >/dev/null && pkill -KILL -f "vllm serve|semantic.serve|VLLM::" || true
  fi
fi
USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')"
log "card: ${USED} MiB in use"
if [ "${USED:-0}" -gt 100 ]; then
  log "the card is not free (${USED} MiB) — another GPU job is running; refusing to start (one GPU task at a time)"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | tee -a "${MAIN_LOG}"
  exit 3
fi

# ── 1. the branches and the tests ──
if [ "${DO_UPDATE}" = 1 ]; then
  log "update_pod_slam2.sh (branches + tests)"
  if ! bash "${SCRIPT_DIR}/update_pod_slam2.sh" >> "${MAIN_LOG}" 2>&1; then
    log "update_pod_slam2.sh FAILED — see ${MAIN_LOG}; the runs are not started"
    exit 4
  fi
fi
cd "${ROOT}/server" || exit 5
export HF_HOME="${HF_HOME:-/workspace/hf_cache}" PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
log "stac-build $(git -C "${ROOT}" rev-parse --abbrev-ref HEAD) @ $(git -C "${ROOT}" rev-parse --short=12 HEAD); fork $(git -C "${ROOT}/vendor/VGGT-Long" rev-parse --abbrev-ref HEAD) @ $(git -C "${ROOT}/vendor/VGGT-Long" rev-parse --short=12 HEAD)"

# ── 2. the runs: scene by scene, one variant per evaluation run ──
IFS=',' read -r -a SC <<< "${SCENES}"
IFS=',' read -r -a VA <<< "${VARIANTS}"
SUMMARY="${LOGDIR}/phase5_runs.tsv"
[ -f "${SUMMARY}" ] || echo -e "scene\tvariant\trun_id\texit\tseconds\tlog" > "${SUMMARY}"
for scene in "${SC[@]}"; do
  for v in "${VA[@]}"; do
    rid="slam2_${scene}_${v}"
    rlog="${LOGDIR}/${rid}.log"
    log "── ${scene} / ${v} → run ${rid} (log ${rlog})"
    t0=$(date +%s)
    "${PY_DA3}" -m evaluation run --scenes "${scene}" --variants "${v}" --run-id "${rid}" > "${rlog}" 2>&1
    rc=$?
    dt=$(( $(date +%s) - t0 ))
    echo -e "${scene}\t${v}\t${rid}\t${rc}\t${dt}\t${rlog}" >> "${SUMMARY}"
    log "   ${scene} / ${v}: exit ${rc} in ${dt} s"
    df_free=$(python3 -c "import os; s=os.statvfs('/workspace'); print(int(s.f_bavail*s.f_frsize/1e9))" 2>/dev/null || echo "?")
    log "   /workspace used: $(du -sh /workspace 2>/dev/null | cut -f1) (quota 400 GB)"
  done
done
log "phase 5 runs through — ${SUMMARY}"
cat "${SUMMARY}" | tee -a "${MAIN_LOG}"
