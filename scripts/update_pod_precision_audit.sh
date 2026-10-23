#!/usr/bin/env bash
# ===========================================================================
# STAC-Builder — bring the pod up to the precision audit of 2026-10-23
# ---------------------------------------------------------------------------
# Branch audit/precision-geometry-2026-10-23 (CLAUDE.md top block, claude_stac.txt
# "ACTUALIZACIÓN 2026-10-23") changed the precision core: F5's odometry priors need
# pyceres, F4's loop bridge needs MASt3R (vendor + 2.75 GB checkpoint), F3b's classical
# cross-check needs COLMAP 4 + its vocabulary tree. This script does, idempotently, ON
# THE POD, everything a fresh session needs before the chain can run; then it runs the
# audit's test battery in the two envs and prints the commands that re-run pccr.
#
#   bash scripts/update_pod_precision_audit.sh            # everything
#   bash scripts/update_pod_precision_audit.sh --no-tests # skip the test battery
#   bash scripts/update_pod_precision_audit.sh --tests    # only the test battery
#
# Nothing here touches a session, config.yaml or the card. One GPU job at a time stays
# the rule: run it with the semantic service / a reconstruction NOT running.
# ===========================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BRANCH="audit/precision-geometry-2026-10-23"
CONDA_ROOT="${CONDA_ROOT:-/workspace/miniforge3}"
PY_MAP="${CONDA_ROOT}/envs/mapanything/bin/python"
PY_DA3="${CONDA_ROOT}/envs/da3/bin/python"
COLMAP_BIN="${COLMAP_BIN:-${CONDA_ROOT}/envs/colmap/bin/colmap}"

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

# ── 0. the code: the audit branch, fast-forwarded ────────────────────────────
if [ "$DO_SETUP" = 1 ]; then
  info "0. code — branch ${BRANCH}"
  cd "${ROOT}"
  if git rev-parse --verify -q "${BRANCH}" >/dev/null; then
    git checkout -q "${BRANCH}" && git pull -q --ff-only origin "${BRANCH}" && ok "on ${BRANCH} @ $(git rev-parse --short=12 HEAD)" \
      || step_fail "git checkout/pull of ${BRANCH} (local changes? resolve by hand, then re-run)"
  else
    git fetch -q origin "${BRANCH}" && git checkout -q -b "${BRANCH}" "origin/${BRANCH}" && ok "checked out ${BRANCH} @ $(git rev-parse --short=12 HEAD)" \
      || step_fail "git fetch/checkout of ${BRANCH}"
  fi

  # ── 1. the vendors: MASt3R (+ dust3r submodule) and RoMa at their pinned commits ──
  info "1. vendors — mast3r (naver, CC BY-NC-SA 4.0: USER 2026-10-23 implemented regardless) + RoMa (MIT)"
  bash "${SCRIPT_DIR}/setup_vendors.sh" mast3r || step_fail "vendor mast3r"
  bash "${SCRIPT_DIR}/setup_vendors.sh" RoMa || step_fail "vendor RoMa"
  [ -f "${ROOT}/vendor/mast3r/dust3r/dust3r/__init__.py" ] && ok "vendor/mast3r with its dust3r submodule" \
    || step_fail "vendor/mast3r/dust3r is missing (git -C vendor/mast3r submodule update --init --recursive)"

  # ── 2. python deps, env mapanything (F4 tracks + loop bridge, F5 refine) ──
  info "2. env mapanything — pyceres>=2.6 (F5's odometry priors on pycolmap's Ceres problem) + MASt3R's runtime deps"
  if [ -x "${PY_MAP}" ]; then
    "${PY_MAP}" -m pip install -q "pyceres>=2.6" einops roma huggingface_hub scipy 2>&1 | tail -2
    "${PY_MAP}" - <<'PYCHK' || step_fail "env mapanything: pycolmap 4 + pyceres do not work together (see the message above)"
import sys
import pycolmap, pyceres
from pycolmap import cost_functions as CF
assert hasattr(pycolmap.Camera, "create_from_model_name"), f"pycolmap {pycolmap.__version__} is not the 4.x the refine needs"
assert hasattr(CF, "RelativePosePriorCost"), "pycolmap.cost_functions has no RelativePosePriorCost"
import torch, torchvision
torchvision.ops.nms  # the symbol dust3r's image pipeline pulls; an unmatched torchvision fails here
print(f"  pycolmap {pycolmap.__version__}  pyceres {getattr(pyceres, '__version__', '?')}  torch {torch.__version__}  torchvision {torchvision.__version__}")
PYCHK
    STAC_ROOT="${ROOT}" "${PY_MAP}" - <<'PYCHK2' || step_fail "env mapanything: the MASt3R import path (vendor/mast3r + dust3r, einops, roma) does not import"
import os, sys
ROOT = os.environ["STAC_ROOT"]
for p in (f"{ROOT}/vendor/mast3r", f"{ROOT}/vendor/mast3r/dust3r"):
    sys.path.insert(0, p)
from mast3r.model import AsymmetricMASt3R          # noqa: F401
from mast3r.fast_nn import fast_reciprocal_NNs      # noqa: F401
from dust3r.utils.image import ImgNorm              # noqa: F401
print("  mast3r + dust3r import")
PYCHK2
  else
    step_fail "env mapanything not found at ${PY_MAP}"
  fi

  # ── 3. weights: the MASt3R checkpoint, hash-checked into torch hub's cache ──
  info "3. weights — MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth (2.75 GB) by precision.tracks.verified_checkpoint"
  if [ -x "${PY_MAP}" ]; then
    (cd "${ROOT}/server" && "${PY_MAP}" - <<'PYW') || step_fail "MASt3R checkpoint (download or sha256 — see config.yaml precision.tracks.loop_bridge_weights_*)"
from precision.config import load_precision_config
from precision.tracks import verified_checkpoint
t = load_precision_config().tracks
if t.loop_bridge != "mast3r":
    print(f"  tracks.loop_bridge is '{t.loop_bridge}' — no MASt3R checkpoint to fetch")
else:
    p = verified_checkpoint(t.loop_bridge_weights_url, t.loop_bridge_weights_sha256, log=print)
    print(f"  {p} ({p.stat().st_size / 2**30:.2f} GB) sha256 OK")
PYW
  fi

  # ── 4. COLMAP 4 (F3b classical cross-check + F6 tier 0) and its vocabulary tree ──
  info "4. COLMAP — ${COLMAP_BIN} (config evaluation.colmap.binary)"
  if [ -x "${COLMAP_BIN}" ]; then
    ok "colmap: $("${COLMAP_BIN}" help 2>&1 | head -1)"
    # the vocab tree COLMAP downloads itself on first use (its "<url>;<name>;<sha256>" URI, cache
    # ~/.cache/colmap); fetched here so the first chain run does not wait on GitHub
    VT_URI="$(cd "${ROOT}/server" && "${PY_DA3}" - <<'PYV'
from config import cfg
ev = (cfg.get("evaluation") or {}).get("colmap") or {}
print(((ev.get("vocab_tree") or {}).get("sift")) or "")
PYV
    )"
    VT_URI="$(echo "${VT_URI}" | tail -1)"      # config's own "[Config] Loaded …" line comes first
    if [ -n "${VT_URI}" ]; then
      IFS=';' read -r VT_URL VT_NAME VT_SHA <<< "${VT_URI}"
      VT_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/colmap"; mkdir -p "${VT_DIR}"
      if [ -f "${VT_DIR}/${VT_NAME}" ] && [ "$(sha256sum "${VT_DIR}/${VT_NAME}" | cut -d' ' -f1)" = "${VT_SHA}" ]; then
        ok "vocab tree ${VT_NAME} present (sha256 OK)"
      else
        info "downloading ${VT_NAME} …"
        curl -L --retry 3 -o "${VT_DIR}/${VT_NAME}.part" "${VT_URL}" \
          && [ "$(sha256sum "${VT_DIR}/${VT_NAME}.part" | cut -d' ' -f1)" = "${VT_SHA}" ] \
          && mv "${VT_DIR}/${VT_NAME}.part" "${VT_DIR}/${VT_NAME}" && ok "vocab tree ${VT_NAME} (sha256 OK)" \
          || warn "vocab tree download/sha256 failed — COLMAP will fetch it itself on the first f3_classical run"
      fi
    fi
  else
    step_fail "COLMAP binary not found at ${COLMAP_BIN} (env colmap: conda create -n colmap -c conda-forge colmap=4.*; f3_classical and f6_colmap_tier need it)"
  fi
  cd "${ROOT}"
fi

# ── 5. the audit's test battery (synthetic, no GPU) in both envs ─────────────
if [ "$DO_TESTS" = 1 ]; then
  info "5. tests — the audit's files, env mapanything (pycolmap 4: refine, loops, rolling shutter) and env da3 (the rest)"
  TESTS_MAP=(tests/test_precision_refine.py tests/test_refine_judge.py tests/test_loop_gap.py tests/test_rolling_shutter.py
             tests/test_precision_tracks.py tests/test_loop_bridge.py)
  TESTS_DA3=(tests/test_owner_fusion.py tests/test_depth_on_f5_compute.py tests/test_depth_on_f5.py tests/test_bend_model.py
             tests/test_precision_colmap_tier.py tests/test_classical_check.py tests/test_precision_config.py
             tests/test_precision_chunk_check.py tests/test_precision_runner.py tests/test_precision_camera.py
             tests/test_p3_projection.py tests/test_certify_mask_filter_camera.py tests/test_vote_evidence.py)
  cd "${ROOT}/server"
  export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
  LOGDIR="${ROOT}/analysis/2026-10-23_precision_audit"; mkdir -p "${LOGDIR}"
  if [ -x "${PY_MAP}" ]; then
    "${PY_MAP}" -m pytest -q -p no:cacheprovider "${TESTS_MAP[@]}" 2>&1 | tee "${LOGDIR}/pytest_mapanything.log" | tail -3
    [ "${PIPESTATUS[0]}" -eq 0 ] && ok "env mapanything: tests green (${LOGDIR}/pytest_mapanything.log)" \
      || step_fail "env mapanything: tests red — ${LOGDIR}/pytest_mapanything.log"
  fi
  if [ -x "${PY_DA3}" ]; then
    "${PY_DA3}" -m pytest -q -p no:cacheprovider "${TESTS_DA3[@]}" 2>&1 | tee "${LOGDIR}/pytest_da3.log" | tail -3
    [ "${PIPESTATUS[0]}" -eq 0 ] && ok "env da3: tests green (${LOGDIR}/pytest_da3.log)" \
      || step_fail "env da3: tests red — ${LOGDIR}/pytest_da3.log"
  fi
  cd "${ROOT}"
fi

echo
if [ "${#FAILS[@]}" -eq 0 ]; then
  ok "pod up to date with ${BRANCH}"
else
  err "${#FAILS[@]} step(s) failed:"; for f in "${FAILS[@]}"; do echo "   - $f"; done
fi
cat <<EOF

Next (by hand, one GPU job at a time — stop the semantic service first or let the runner stop it):
  conda activate da3 && cd ${ROOT}/server
  python -m precision.runner --session /workspace/sessions/<pccr> --from f3_classical --refreeze
     --refreeze: the session's frozen output/run_config.yaml is kept beside (run_config.<sha>.before-refreeze.yaml)
                 and config.yaml NOW is frozen — the new keys (refine.prior_model relative, scale_lock, rolling_shutter,
                 tracks.loop_bridge mast3r, fusion.free_space / min_evidence, precision.classical) reach every step.
  Then "Reconstruir" from the viewer for the full chain (VLM/SAM3 projection with member pixels, certification with
  the consumed z-buffers), or the certification by hand (python -m reconstruction.certify.run --help).
What to read afterwards (never a verdict — numbers for Hernán's eye):
  output/precision/refine.json      odometry / prior_sigma_ladder / loops (gap before/after vs control) / scale_lock /
                                    rolling_shutter (tau_frames, taken, search) / classical_cross_check (position_m, worst_keyframes)
  output/precision/tracks.json      classical_loops, loop_bridge (pairs, inliers, inlier ratio, bridged tracks)
  output/precision/depth_on_f5.json fusion: owners / consumed / free_space.rejected / unverified.rejected / representation
  output/precision/classical/       COLMAP's database, sparse models, loop_candidates.json, the cross-check report
EOF
[ "${#FAILS[@]}" -eq 0 ]
