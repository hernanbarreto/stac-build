#!/usr/bin/env bash
# ===========================================================================
# STAC-Builder — Vendor Restore
# ---------------------------------------------------------------------------
# The vendor/ tree is git-ignored (large clones, build trees, gated weights).
# A fresh `git clone` therefore lands WITHOUT it. This script restores every
# git-based vendor at its PINNED commit so a new cloud box is reproducible,
# and prints the provisioning path for the non-git ones (weights / build).
#
# Source of truth for the full inventory: vendor/VENDORS.lock.md
#
#   Usage:
#     bash scripts/setup_vendors.sh            # restore all git vendors + submodules
#     bash scripts/setup_vendors.sh r3d        # restore a single vendor by name
#     bash scripts/setup_vendors.sh --list     # print the manifest and exit
#
# Idempotent: a vendor already present at the right commit is left untouched.
# Hernán Barreto - Ingerop IN3 Session IV - STAC
# ===========================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENDOR="${ROOT}/vendor"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; RED='\033[0;31m'; NC='\033[0m'
info() { echo -e "${CYAN}[vendor]${NC} $1"; }
ok()   { echo -e "${GREEN}[ ok ]${NC}  $1"; }
warn() { echo -e "${YELLOW}[warn]${NC}  $1"; }
err()  { echo -e "${RED}[fail]${NC}  $1"; }

# ── Pinned git vendors: "dir|url|commit" ────────────────────────────────────
# These are git-ignored plain clones (NOT submodules). Pins captured from the
# validated build; bump them here when a vendor is intentionally upgraded.
GIT_VENDORS=(
  "r3d|https://github.com/facebookresearch/r3d.git|9669cacd7993"
  "sam31|https://github.com/facebookresearch/sam3.git|5dd401d1c5c1"
  "nvdiffrast|https://github.com/NVlabs/nvdiffrast.git|253ac4fcea7d"
  "meshflow|https://github.com/facebookresearch/meshflow.git|55f56f60e1bb"
  "mvs-texturing|https://github.com/nmoehrle/mvs-texturing.git|f3374298ac95"
  "oneTBB-src|https://github.com/uxlfoundation/oneTBB.git|e9af1a1b38b8"
  "vggt-omega|https://github.com/facebookresearch/vggt-omega.git|39a0cb8af885"
  "ShapeR|https://github.com/facebookresearch/ShapeR.git|d4402f55dc69"
  "pgsr|https://github.com/zju3dv/PGSR.git|de24f1a38b35"
)

# vendor/pgsr carries ONE local patch (inline quaternion_to_matrix so the whole
# pytorch3d dependency is not needed) — applied after the pinned clone.
apply_pgsr_patch() {
  local patch="${ROOT}/server/patches/pgsr_inline_quaternion_to_matrix.patch"
  [ -d "${VENDOR}/pgsr" ] && [ -f "${patch}" ] || return 0
  if git -C "${VENDOR}/pgsr" apply --check "${patch}" 2>/dev/null; then
    git -C "${VENDOR}/pgsr" apply "${patch}" && info "pgsr: local patch applied"
  else
    info "pgsr: local patch already applied (or not applicable)"
  fi
}

# ── Non-git vendors (documented; not clonable) ──────────────────────────────
# name|how-to-provision
NONGIT_VENDORS=(
  "sam3|DEFAULT segmentation baseline (SAM 3.0). Weights: bash setup_weights.sh sam3 → weights/sam3. Code checkout pinned to the SAM 3.0 release of facebookresearch/sam3 (see vendor/VENDORS.lock.md)."
  "cloudcompy|CloudComPy runtime tree (bin/lib) consumed by server/vendor_paths.py. Provision: conda env create -f docs/migration/environment_CloudComPy310.yml, then place the built install under vendor/cloudcompy."
  "CloudComPy310|CloudComPy build/source tree. See docs/migration/MIGRATION_GUIDE.md (§CloudComPy)."
  "MapAnything2|Optional MapAnything fallback. Provision: conda env create -f docs/migration/environment_mapanything.yml."
  "PotreeConverter|Octree converter. Prebuilt binary shipped historically; rebuild per MIGRATION_GUIDE §6 (cmake + make) if the binary is missing/broken."
  "oneTBB|Compiled install tree built FROM vendor/oneTBB-src (cmake + make install). Platform-specific — do not copy across archs."
  "vggt-omega-weights|Gated HF weights (~4.3 GB) for the VGGT-Ω backbone. Requires HF_TOKEN. See setup_weights.sh / README §weights."
)

print_manifest() {
  echo "── git vendors (pinned, restored by this script) ─────────────────────"
  for e in "${GIT_VENDORS[@]}"; do IFS='|' read -r d u c <<<"$e"; printf "  %-16s %s @ %s\n" "$d" "$u" "$c"; done
  echo "── submodules (restored by: git submodule update --init) ─────────────"
  printf "  %-16s %s\n" "VGGT-Long" "hernanbarreto/VGGT-Long (STAC fork)"
  printf "  %-16s %s\n" "depth-anything-3" "hernanbarreto/Depth-Anything-3 (STAC fork, PRIVATE — needs GitHub access)"
  echo "── non-git (manual: weights / build / env yml) ───────────────────────"
  for e in "${NONGIT_VENDORS[@]}"; do IFS='|' read -r d h <<<"$e"; printf "  %-16s %s\n" "$d" "$h"; done
}

clone_pinned() {
  local dir="$1" url="$2" sha="$3" dest="${VENDOR}/${1}"
  if [ -e "${dest}/.git" ]; then
    local cur; cur="$(git -C "${dest}" rev-parse --short=12 HEAD 2>/dev/null || echo '?')"
    if [ "${cur}" = "${sha}" ]; then ok "${dir} already at ${sha}"; return 0; fi
    warn "${dir} present at ${cur} (want ${sha}); fetching + checking out"
    git -C "${dest}" fetch --quiet origin "${sha}" 2>/dev/null || git -C "${dest}" fetch --quiet origin
    git -C "${dest}" checkout --quiet "${sha}" && ok "${dir} → ${sha}" && return 0
    err "${dir}: could not check out ${sha}"; return 1
  fi
  info "cloning ${dir} …"
  if git clone --quiet "${url}" "${dest}" 2>/dev/null; then
    git -C "${dest}" checkout --quiet "${sha}" 2>/dev/null \
      && ok "${dir} → ${sha}" \
      || { warn "${dir}: cloned but pin ${sha} not found (upstream moved) — left at default HEAD"; }
  else
    err "${dir}: clone failed (network / access). URL: ${url}"; return 1
  fi
}

# ── args ────────────────────────────────────────────────────────────────────
if [ "${1:-}" = "--list" ] || [ "${1:-}" = "-l" ]; then print_manifest; exit 0; fi

ONLY="${1:-}"
mkdir -p "${VENDOR}"
rc=0

# 1) submodules (real, in .gitmodules) — VGGT-Long + depth-anything-3
if [ -z "${ONLY}" ]; then
  if [ -f "${ROOT}/.gitmodules" ]; then
    info "initialising git submodules (VGGT-Long, depth-anything-3)…"
    git -C "${ROOT}" submodule update --init --recursive 2>&1 \
      | sed -E 's/(https:\/\/)[^@]+@/\1***@/g' \
      || warn "submodule init reported issues (depth-anything-3 is a PRIVATE fork — needs GitHub access)"
  fi
fi

# 2) pinned git clones
for e in "${GIT_VENDORS[@]}"; do
  IFS='|' read -r d u c <<<"$e"
  [ -n "${ONLY}" ] && [ "${ONLY}" != "${d}" ] && continue
  clone_pinned "${d}" "${u}" "${c}" || rc=1
done
apply_pgsr_patch

# 3) non-git reminders
if [ -z "${ONLY}" ]; then
  echo
  info "non-git vendors are NOT restored by clone — provision manually:"
  for e in "${NONGIT_VENDORS[@]}"; do IFS='|' read -r d h <<<"$e"; echo -e "   ${YELLOW}${d}${NC}: ${h}"; done
  echo
  info "model weights: bash setup_weights.sh   |   full inventory: vendor/VENDORS.lock.md"
fi

[ "${rc}" -eq 0 ] && ok "vendor restore complete" || err "vendor restore finished with errors (see above)"
exit "${rc}"
