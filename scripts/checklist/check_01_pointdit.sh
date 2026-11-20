#!/usr/bin/env bash
# Checklist line 1 (conclusion 34): PointDiT removed from the chain, the repo, the pod and the rebuild kit
# (USER 2026-11-20: "hay que sacarlo, eliminalo de la cadena, del repo, de todo").
#
# THE WRITTEN CHECK OF THAT LINE (USER 2026-11-20, on a closure that had been declared without one): it looks in the
# repository, on the pod and in the rebuild kit, prints every place where something of PointDiT is left, and ends with
# ONE line: "CLOSED - 0 left" (exit 0) or "NOT CLOSED - n left" (exit 1). Read only; the same answer at every run:
#
#     bash scripts/checklist/check_01_pointdit.sh
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 2
left=0
ok()  { printf '  ok    %s\n' "$*"; }
bad() { left=$((left + 1)); printf '  LEFT  %s\n' "$*"; }

NAMES='pointdit|point_dit|point-dit|mono_detail|mono_ab|mixed_unresolved'
VITH='7c1da9a54b3bdb333f5ebc42e404b7f19b1b5bed504877623c9dc87397f41488'     # DINOv3 ViT-H+: its backbone, used by nothing else
DA3_ONLY_FOR_IT='torchmetrics|lightning_utilities'                           # the two packages env da3 held for it

echo "1. the repository (every file git sees, tracked or not)"
# what may name it: the RECORD of its removal, and the USER's own words quoted in comments about other things
RECORD=(CLAUDE.md docs/pipeline_audit_2026-11-20.md docs/pipeline_checklist.md docs/plan_determinismo.md
        vendor/VENDORS.lock.md scripts/checklist/check_01_pointdit.sh)
QUOTES=(
  "scripts/rebuild/README.md|Revisa pointdit r3d etc etc son muchos"
  "scripts/rebuild/inventory.py|Revisa pointdit r3d etc etc son"
  "server/phase5_qa/tools.py|las vistas de Flayers, PointDit mixed"
  "server/workers/base.py|omega, da3, vlm, sam3, pointdit, todo"
  "ui/src/App.tsx|cubos como en la página de PointDiT"
  "ui/src/components/Viewport.tsx|as PointDiT's page"
  "ui/src/components/Viewport.tsx|cubos como en la página de PointDiT"
  "ui/src/components/viewport/FloatingToolbar.tsx|las vistas de Flayers, PointDit mixed unsolved"
)
excl=()
for f in "${RECORD[@]}"; do excl+=(":!$f"); done
n_repo=0; n_quote=0
while IFS= read -r hit; do
  [ -z "$hit" ] && continue
  file="${hit%%:*}"; quoted=0
  for q in "${QUOTES[@]}"; do
    if [ "$file" = "${q%%|*}" ] && printf '%s' "$hit" | /usr/bin/grep -qF -- "${q#*|}"; then quoted=1; break; fi
  done
  if [ "$quoted" = 1 ]; then n_quote=$((n_quote + 1)); else n_repo=$((n_repo + 1)); bad "${hit:0:170}"; fi
done < <(/usr/bin/git grep --untracked -inE "$NAMES" -- . "${excl[@]}" 2>/dev/null)
[ "$n_repo" = 0 ] && ok "no code, configuration, test or document names it (outside the record of its removal: ${#RECORD[@]} files; ${n_quote} comment lines quote the user's words about other things)"

echo "2. the pod"
if [ -e vendor/pointdit ]; then bad "vendor/pointdit exists"; else ok "vendor/pointdit does not exist"; fi
if [ -e .git/modules/vendor/pointdit ] || [ -e .git/modules/third_party/pointdit ]; then bad "the old submodule's git folder exists"; else ok "no git folder of the old submodule"; fi
if /usr/bin/grep -qi pointdit .gitmodules 2>/dev/null; then bad ".gitmodules names it"; else ok ".gitmodules does not name it"; fi
if [ -n "$(/usr/bin/git config --get-regexp 'pointdit' 2>/dev/null)" ]; then bad "the local git settings name it"; else ok "the local git settings do not name it"; fi
found="$(/usr/bin/find vendor /workspace/hf_cache /workspace/.cache/huggingface /root/.cache/torch -maxdepth 4 \( -iname '*pointdit*' -o -iname '*vith16plus*' \) 2>/dev/null | /usr/bin/head -5)"
if [ -n "$found" ]; then bad "weights or folders: $(printf '%s' "$found" | /usr/bin/tr '\n' ' ')"; else ok "no weight or folder of it in the vendor trees or the model caches"; fi
pk="$(/usr/bin/ls /workspace/miniforge3/envs/da3/lib/python3*/site-packages 2>/dev/null | /usr/bin/grep -iE "^($DA3_ONLY_FOR_IT)" | /usr/bin/tr '\n' ' ')"
if [ -n "$pk" ]; then bad "env da3 still holds: $pk"; else ok "env da3 holds neither of the two packages installed for it"; fi

echo "3. the rebuild kit"
if /usr/bin/grep -rqiE "$VITH|vith16plus" scripts/rebuild --include='*.tsv' --include='*.txt' --include='*.md' --include='*.py' --include='*.sh' 2>/dev/null; then
  bad "the kit lists its backbone weight (DINOv3 ViT-H+)"; else ok "no list of the kit names its backbone weight"; fi
if /usr/bin/ls scripts/rebuild/locks 2>/dev/null | /usr/bin/grep -q '^da3\.' && /usr/bin/grep -qiE "^($DA3_ONLY_FOR_IT|lightning-utilities)" scripts/rebuild/locks/da3.* 2>/dev/null; then
  bad "da3's lock lists a package installed for it"; else ok "da3's lock lists neither of the two packages"; fi
rel="$(PYTHONPATH=scripts/rebuild /usr/bin/timeout 60 /usr/bin/python3 - "$VITH" <<'EOF' 2>/dev/null
import sys
import backup
token = backup.env_value("GIT_TOKEN")
if not token:
    print("unreadable: no GIT_TOKEN"); raise SystemExit
r = backup.release(token, create=False)
if r is None:
    print("none"); raise SystemExit
print("held" if any(a.get("name", "").startswith(sys.argv[1]) for a in r.get("assets", [])) else "clean")
EOF
)"
case "$rel" in
  clean) ok "the repo's draft release does not hold its backbone weight" ;;
  none)  ok "the repo has no draft release of mirrored weights" ;;
  held)  bad "the repo's draft release still holds its backbone weight" ;;
  *)     bad "the repo's draft release could not be read (${rel:-no answer}) - not checked is not closed" ;;
esac

echo
if [ "$left" = 0 ]; then echo "LINE 1 (PointDiT): CLOSED - 0 left"; exit 0; fi
echo "LINE 1 (PointDiT): NOT CLOSED - $left left"; exit 1
