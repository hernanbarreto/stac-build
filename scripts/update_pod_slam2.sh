#!/usr/bin/env bash
# ===========================================================================
# STAC-Builder — bring the pod to the VGGT-SLAM 2.0 branches (claude_stac_slam2.txt, phase 5)
# ---------------------------------------------------------------------------
# Idempotent. ON THE POD:
#   1. stac-build on branch slam2-omega (fetch + fast-forward), vendor/VGGT-Long on the fork's
#      branch stac-slam2 at the commit the submodule pointer records;
#   2. the tests of both repos — the fork's (env mapanything) and the server's (env da3) — with
#      their logs under analysis/2026-10-29_slam2/;
#   3. the commands of the phase-5 runs printed, never run here.
# Touches NEITHER the audit branch (audit/precision-geometry-2026-10-23) nor its vendors
# (vendor/mast3r, vendor/RoMa): a `git checkout` of this repo only; the vendors are untracked.
#
#   bash scripts/update_pod_slam2.sh            # everything
#   bash scripts/update_pod_slam2.sh --no-tests # skip the test battery
#   bash scripts/update_pod_slam2.sh --tests    # only the test battery
# One GPU task at a time stays the rule: nothing here uses the card.
# ===========================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BRANCH="slam2-omega"
FORK_BRANCH="stac-slam2"
CONDA_ROOT="${CONDA_ROOT:-/workspace/miniforge3}"
PY_MAP="${CONDA_ROOT}/envs/mapanything/bin/python"
PY_DA3="${CONDA_ROOT}/envs/da3/bin/python"
LOGDIR="${ROOT}/analysis/2026-10-29_slam2"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; RED='\033[0;31m'; NC='\033[0m'
info() { echo -e "${CYAN}[pod]${NC} $1"; }
ok()   { echo -e "${GREEN}[ ok ]${NC} $1"; }
warn() { echo -e "${YELLOW}[warn]${NC} $1"; }
err()  { echo -e "${RED}[fail]${NC} $1"; }

DO_SETUP=1; DO_TESTS=1
for a in "$@"; do
  case "$a" in
    --no-tests) DO_TESTS=0 ;;
    --tests) DO_SETUP=0 ;;
    *) echo "usage: $0 [--no-tests | --tests]"; exit 2 ;;
  esac
done
FAILS=()
step_fail() { err "$1"; FAILS+=("$1"); }
mkdir -p "${LOGDIR}"

if [ "$DO_SETUP" = 1 ]; then
  info "0. stac-build — branch ${BRANCH}"
  cd "${ROOT}"
  if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    step_fail "the working tree has local changes — commit or stash them before switching to ${BRANCH}"
  else
    git fetch -q origin "${BRANCH}" || step_fail "git fetch origin ${BRANCH}"
    if git rev-parse --verify -q "${BRANCH}" >/dev/null; then
      git checkout -q "${BRANCH}" && git pull -q --ff-only origin "${BRANCH}" && ok "on ${BRANCH} @ $(git rev-parse --short=12 HEAD)" \
        || step_fail "git checkout/pull of ${BRANCH}"
    else
      git checkout -q -b "${BRANCH}" "origin/${BRANCH}" && ok "checked out ${BRANCH} @ $(git rev-parse --short=12 HEAD)" \
        || step_fail "git checkout of origin/${BRANCH}"
    fi
  fi

  info "1. the fork — vendor/VGGT-Long on ${FORK_BRANCH} at the submodule pointer"
  WANT="$(git -C "${ROOT}" ls-tree HEAD vendor/VGGT-Long | awk '{print $3}')"
  if [ -z "${WANT}" ]; then
    step_fail "stac-build records no submodule pointer for vendor/VGGT-Long"
  else
    FORK="${ROOT}/vendor/VGGT-Long"
    if [ ! -d "${FORK}/.git" ] && [ ! -f "${FORK}/.git" ]; then
      git -C "${ROOT}" submodule update --init vendor/VGGT-Long || step_fail "submodule init of vendor/VGGT-Long"
    fi
    git -C "${FORK}" fetch -q origin "${FORK_BRANCH}" || step_fail "git fetch of the fork's ${FORK_BRANCH}"
    if git -C "${FORK}" cat-file -e "${WANT}^{commit}" 2>/dev/null; then
      if [ -n "$(git -C "${FORK}" status --porcelain --untracked-files=no)" ]; then
        step_fail "vendor/VGGT-Long has local changes — resolve them before moving it to ${WANT:0:12}"
      else
        git -C "${FORK}" checkout -q -B "${FORK_BRANCH}" "${WANT}" && ok "vendor/VGGT-Long on ${FORK_BRANCH} @ ${WANT:0:12}" \
          || step_fail "checkout of ${WANT:0:12} in vendor/VGGT-Long"
      fi
    else
      step_fail "the fork does not hold ${WANT:0:12} (push ${FORK_BRANCH} to origin first)"
    fi
    [ -f "${FORK}/sim3solve.cpython-310-x86_64-linux-gnu.so" ] || warn "vendor/VGGT-Long/sim3solve*.so missing (the vendor Sim3 optimizer in C++; the STAC path does not use it)"
    [ -f "${FORK}/skyseg.onnx" ] || warn "vendor/VGGT-Long/skyseg.onnx missing — the sky mask is pinned to it (scripts/setup_vendors.sh)"
  fi
  [ -d "${ROOT}/vendor/vggt-omega/vggt_omega" ] && ok "vendor/vggt-omega present (pin 39a0cb8a, never modified)" \
    || step_fail "vendor/vggt-omega missing (scripts/setup_vendors.sh)"
  for p in "${PY_MAP}" "${PY_DA3}"; do [ -x "$p" ] && ok "env $(basename "$(dirname "$(dirname "$p")")")" || step_fail "env missing: $p"; done
fi

if [ "$DO_TESTS" = 1 ]; then
  info "2. tests — the fork (env mapanything) and the server (env da3), logs in ${LOGDIR}"
  export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0
  if [ -x "${PY_MAP}" ]; then
    cd "${ROOT}/vendor/VGGT-Long"
    "${PY_MAP}" -m pytest -q -p no:cacheprovider tests/test_slam2_phase0.py tests/test_seam_radial.py tests/test_common_k.py \
        tests/test_attention_gate.py tests/test_attention_calibrate.py tests/test_stac_loop_stage.py \
        tests/test_loop_detector_gap.py tests/test_loop_bridges.py tests/test_explicit_chunk_ranges.py \
        2>&1 | tee "${LOGDIR}/pytest_fork_mapanything.log" | tail -3
    [ "${PIPESTATUS[0]}" -eq 0 ] && ok "fork tests green (${LOGDIR}/pytest_fork_mapanything.log)" \
      || step_fail "fork tests red — ${LOGDIR}/pytest_fork_mapanything.log"
  fi
  if [ -x "${PY_DA3}" ]; then
    cd "${ROOT}/server"
    "${PY_DA3}" -m pytest -q -p no:cacheprovider tests/test_slam2_config.py tests/test_evaluation_slam2.py \
        tests/test_evaluation_variants.py tests/test_evaluation_cli.py tests/test_evaluation_determinism.py \
        tests/test_vendor_config_wiring.py tests/test_map_worker_job_config.py tests/test_fork_determinism.py \
        2>&1 | tee "${LOGDIR}/pytest_server_da3.log" | tail -3
    [ "${PIPESTATUS[0]}" -eq 0 ] && ok "server tests green (${LOGDIR}/pytest_server_da3.log)" \
      || step_fail "server tests red — ${LOGDIR}/pytest_server_da3.log"
  fi
  cd "${ROOT}"
fi

echo
if [ "${#FAILS[@]}" -eq 0 ]; then ok "pod up to date with ${BRANCH} / ${FORK_BRANCH}"; else err "${#FAILS[@]} step(s) failed:"; for f in "${FAILS[@]}"; do echo "   - $f"; done; fi
cat <<EOF

Next (by hand, ONE GPU task at a time, the backend's vLLM down or left to the runner), pccr first, then observatorio (USER 2026-10-31: zaragoza out):
  1. the offline instruments over an EXISTING run (only if its maplong_run/_tmp_results_unaligned survived):
       cd ${ROOT}/vendor/VGGT-Long
       ${PY_MAP} -m loop_utils.seam_radial --run <session>/output/maplong_run
       ${PY_MAP} -m loop_utils.attention_calibrate --run <session>/output/maplong_run --params <json>   # GPU
     (a run that kept no unaligned chunks is declared by the tools; the reference variant below keeps them)
  2. the variants (every one a full reconstruction on its own copy; the reference runs seam_radial and
     attention_calibrate itself, mirrors their records under the scene's maplong_run/ and the gate variants
     resolve from them). ONE variant per invocation — a session copy with its chunk npys is 10–40 GB and the
     /workspace quota is 400 GB; the keep policy trims a copy only when its run ends:
       cd ${ROOT}/server
       for v in reference slam2_focal_common slam2_attention_gate slam2_attention_gate_salad slam2_all; do
         ${PY_DA3} -m evaluation run --scenes pccr --variants \$v --run-id slam2_pccr_\$v
       done
       for v in reference slam2_focal_common slam2_attention_gate slam2_attention_gate_salad slam2_all; do
         ${PY_DA3} -m evaluation run --scenes observatorio --variants \$v --run-id slam2_observatorio_\$v
       done
  3. the bit-for-bit comparator of the reference against an existing main run of the scene: declare it in
     config.yaml evaluation.slam2.existing_runs (pccr and observatorio are declared) before step 2 — it runs
     with the reference and lands in <run>/<scene>/slam2_comparison.json (bitwise_comparator).
Results: <runs_dir>/<run_id>/<scene>/{slam2_comparison.csv,json,png} and each variant's directory
(variant.json, log.txt, session/output/maplong_run/{seam_radial,attention_gate.json,focal_model.json,attention_calibration}).
Each variant's cloud: project <scene>__eval, scan <variant>/default in the viewer.
EOF
[ "${#FAILS[@]}" -eq 0 ]
