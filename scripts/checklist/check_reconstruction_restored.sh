#!/usr/bin/env bash
# THE RECONSTRUCTION BACK TO ITS STATE OF 2026-11-19 (USER 2026-11-20: "volve a la reconstruccion que teniamos cuando
# hicimos las versiones que estan publicadas … si hace falta clonar y bajar pointdit hacelo tambien" / "ojo que hicimos
# correcciones en el sistema eh!!, solo la parte de reconstruccion tenes que traer").
#
# The written check of that order. The restore point is commit 25636e2 (2026-11-20 08:26 UTC): the reconstruction as it
# was before every change of 2026-11-20 - the validated algorithm with the corrections he ordered up to 2026-11-19.
# It prints what differs and ends with ONE line: "RESTORED - 0 differences" (exit 0) or "NOT RESTORED - n" (exit 1).
# Read only; the same answer at every run:
#
#     bash scripts/checklist/check_reconstruction_restored.sh
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 2
POINT=25636e2
left=0
ok()  { printf '  ok    %s\n' "$*"; }
bad() { left=$((left + 1)); printf '  DIFF  %s\n' "$*"; }

# the reconstruction's files: the pipeline's code and its tests, its configuration, what it shares with the evaluation
# battery and the tools, its documents, the rebuild kit's description of the pod, the vendor trees it runs
PATHS=(server/precision server/tests server/intake server/reconstruction server/workers/map_worker.py
       server/workers/precision_worker.py server/workers/base.py server/config.yaml server/evaluation/cost.py
       server/evaluation/determinism.py server/evaluation/variants.py server/main.py server/repro.py
       server/phase5_qa/tools.py server/tools/amb3r/chain.py server/tools/determinism_smoke.py
       server/tools/genrecon/convert_dinov3.py .gitmodules vendor/VENDORS.lock.md README.md docs/ARCHITECTURE.md
       docs/pipeline_final.md docs/plan_determinismo.md)

echo "1. the reconstruction's files against commit $POINT (the working files and the last commit)"
n=0
while IFS= read -r f; do [ -z "$f" ] && continue; n=$((n + 1)); bad "working file differs: $f"; done < <(/usr/bin/git diff --name-only "$POINT" -- "${PATHS[@]}" 2>/dev/null)
[ "$n" = 0 ] && ok "every working file of the reconstruction is identical to the restore point"
n=0
while IFS= read -r f; do [ -z "$f" ] && continue; n=$((n + 1)); bad "committed file differs: $f"; done < <(/usr/bin/git diff --name-only "$POINT" HEAD -- "${PATHS[@]}" 2>/dev/null)
[ "$n" = 0 ] && ok "every committed file of the reconstruction is identical to the restore point"

echo "2. the vendor trees the reconstruction runs"
want_fork="$(/usr/bin/git ls-tree "$POINT" vendor/VGGT-Long | /usr/bin/awk '{print $3}')"
want_pd="$(/usr/bin/git ls-tree "$POINT" vendor/pointdit | /usr/bin/awk '{print $3}')"
have_fork="$(/usr/bin/git -C vendor/VGGT-Long rev-parse HEAD 2>/dev/null)"
have_pd="$(/usr/bin/git -C vendor/pointdit rev-parse HEAD 2>/dev/null)"
if [ "$have_fork" = "$want_fork" ] && [ -z "$(/usr/bin/git -C vendor/VGGT-Long status --short 2>/dev/null)" ]; then ok "the fork (vendor/VGGT-Long) at ${want_fork:0:8}, no local change"; else bad "the fork is at ${have_fork:0:8}, the restore point has ${want_fork:0:8}"; fi
if [ "$have_pd" = "$want_pd" ]; then ok "the edge-detail model's tree at ${want_pd:0:8}"; else bad "the edge-detail model's tree is at ${have_pd:0:8}, the restore point has ${want_pd:0:8}"; fi
hub="$(/usr/bin/git -C vendor/pointdit/third_party/dinov3 rev-parse HEAD 2>/dev/null)"
if [ "${hub:0:8}" = "6876159a" ]; then ok "its DINOv3 code at 6876159a (the kit's pin)"; else bad "its DINOv3 code is at ${hub:0:8}, the kit pins 6876159a"; fi

echo "3. its weights, each against the fingerprint the kit records"
while IFS=$'\t' read -r _kind dest _src sha size _rest; do
  case "$dest" in vendor/pointdit/*) ;; *) continue ;; esac
  if [ ! -f "$dest" ]; then bad "missing: $dest"; continue; fi
  got="$(/usr/bin/sha256sum "$dest" | /usr/bin/cut -d' ' -f1)"
  if [ "$got" = "$sha" ]; then ok "$(/usr/bin/basename "$dest") ($size bytes)"; else bad "$(/usr/bin/basename "$dest"): another fingerprint"; fi
done < <(/usr/bin/git show "$POINT:scripts/rebuild/models.tsv" | /usr/bin/grep -v '^#')

echo "4. the environment and the chain"
for spec in "torchmetrics==1.9.0" "lightning-utilities==0.15.3"; do
  if /usr/bin/git show "$POINT:scripts/rebuild/locks/da3.pip.txt" | /usr/bin/grep -qx "$spec" && /workspace/miniforge3/envs/da3/bin/python -m pip list --disable-pip-version-check --format=freeze 2>/dev/null | /usr/bin/grep -qix "$spec"; then
    ok "env da3 holds $spec, as its lock of the restore point"; else bad "env da3 does not hold $spec"; fi
done
chain="$(cd server && CUDA_VISIBLE_DEVICES='' /workspace/miniforge3/envs/da3/bin/python -c "
from precision.config import load_precision_config
from precision import runner
p = load_precision_config()
print(','.join(s.key for s in runner.STEPS), '|', p.mono_detail.enabled)" 2>/dev/null | /usr/bin/tail -n 1)"
case "$chain" in
  *f2_gauge*f4_tracks*f5_refine*f6_bend*) ok "the configuration loads and the chain holds F2, F4, F5, F6 (edge detail enabled: ${chain##*| })" ;;
  *) bad "the configuration or the chain does not load as at the restore point (${chain:-no answer})" ;;
esac

echo
if [ "$left" = 0 ]; then echo "THE RECONSTRUCTION AS ON 2026-11-19: RESTORED - 0 differences"; exit 0; fi
echo "THE RECONSTRUCTION AS ON 2026-11-19: NOT RESTORED - $left"; exit 1
