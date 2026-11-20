#!/usr/bin/env bash
# Checklist line 1 (conclusion 34): the edge-detail model removed from the chain, the repo, the pod and the rebuild
# kit (USER 2026-11-20: "hay que sacarlo, eliminalo de la cadena, del repo, de todo").
#
# THE WRITTEN CHECK OF THAT LINE (USER 2026-11-20: a line is closed by what a check shows, never by a word). It looks
# in the repository - the working files AND the last commit, with NO exception: code, comments, quoted words,
# configuration, tests, interface, kit, documents and records -, on the pod and in the rebuild kit; it prints every
# place where something of that model is left and ends with ONE line: "CLOSED - 0 left" (exit 0) or
# "NOT CLOSED - n left" (exit 1). Read only; the same answer at every run:
#
#     bash scripts/checklist/check_01.sh
#
# (The names it looks for are put together from pieces below, so that this file does not hold them either.)
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 2
left=0
ok()  { printf '  ok    %s\n' "$*"; }
bad() { left=$((left + 1)); printf '  LEFT  %s\n' "$*"; }

A="point"; B="dit"; M="mono"; X="mixed"
MODEL="${A}${B}"                                                               # the model's name
NAMES="${MODEL}|${A}_${B}|${A}-${B}|${M}_detail|${M}_ab|${X}_unresolved"       # + its modules and its two layers
VITH='7c1da9a54b3bdb333f5ebc42e404b7f19b1b5bed504877623c9dc87397f41488'       # DINOv3 ViT-H+: its backbone, used by nothing else
DA3_ONLY_FOR_IT='torchmetrics|lightning_utilities'                             # the two packages env da3 held for it

echo "1. the repository: every file git sees (tracked or not) and the last commit - no exception"
n=0
while IFS= read -r hit; do
  [ -z "$hit" ] && continue
  n=$((n + 1)); bad "working files: ${hit:0:160}"
done < <(/usr/bin/git grep --untracked -inE "$NAMES" -- . 2>/dev/null)
[ "$n" = 0 ] && ok "no working file names it"
n=0
while IFS= read -r hit; do
  [ -z "$hit" ] && continue
  n=$((n + 1)); bad "last commit: ${hit:0:160}"
done < <(/usr/bin/git grep -inE "$NAMES" HEAD -- . 2>/dev/null)
[ "$n" = 0 ] && ok "no file of the last commit names it"
if /usr/bin/git ls-files | /usr/bin/grep -qiE "$NAMES"; then bad "a tracked file carries it in its name"; else ok "no tracked file carries it in its name"; fi

echo "2. the pod"
if [ -e "vendor/${MODEL}" ]; then bad "its vendor tree exists"; else ok "its vendor tree does not exist"; fi
if [ -e ".git/modules/vendor/${MODEL}" ] || [ -e ".git/modules/third_party/${MODEL}" ]; then bad "the old submodule's git folder exists"; else ok "no git folder of the old submodule"; fi
if /usr/bin/grep -qi "${MODEL}" .gitmodules 2>/dev/null; then bad ".gitmodules names it"; else ok ".gitmodules does not name it"; fi
if [ -n "$(/usr/bin/git config --get-regexp "${MODEL}" 2>/dev/null)" ]; then bad "the local git settings name it"; else ok "the local git settings do not name it"; fi
found="$(/usr/bin/find vendor /workspace/hf_cache /workspace/.cache/huggingface /root/.cache/torch -maxdepth 4 \( -iname "*${MODEL}*" -o -iname '*vith16plus*' \) 2>/dev/null | /usr/bin/head -5)"
if [ -n "$found" ]; then bad "weights or folders: $(printf '%s' "$found" | /usr/bin/tr '\n' ' ')"; else ok "no weight or folder of it in the vendor trees or the model caches"; fi
pk="$(/usr/bin/ls /workspace/miniforge3/envs/da3/lib/python3*/site-packages 2>/dev/null | /usr/bin/grep -iE "^($DA3_ONLY_FOR_IT)" | /usr/bin/tr '\n' ' ')"
if [ -n "$pk" ]; then bad "env da3 still holds: $pk"; else ok "env da3 holds neither of the two packages installed for it"; fi

echo "3. the rebuild kit"
if /usr/bin/grep -rqiE "$VITH|vith16plus" scripts/rebuild --include='*.tsv' --include='*.txt' --include='*.md' --include='*.py' --include='*.sh' 2>/dev/null; then
  bad "the kit lists its backbone weight (DINOv3 ViT-H+)"; else ok "no list of the kit names its backbone weight"; fi
if /usr/bin/grep -qiE "^($DA3_ONLY_FOR_IT|lightning-utilities)" scripts/rebuild/locks/da3.* 2>/dev/null; then
  bad "da3's lock lists a package installed for it"; else ok "da3's lock lists neither of the two packages"; fi
rel="$(PYTHONPATH=scripts/rebuild /usr/bin/timeout 60 /usr/bin/python3 - "$VITH" <<'PY' 2>/dev/null
import sys
import backup
token = backup.env_value("GIT_TOKEN")
if not token:
    print("unreadable: no GIT_TOKEN"); raise SystemExit
r = backup.release(token, create=False)
if r is None:
    print("none"); raise SystemExit
print("held" if any(a.get("name", "").startswith(sys.argv[1]) for a in r.get("assets", [])) else "clean")
PY
)"
case "$rel" in
  clean) ok "the repo's draft release does not hold its backbone weight" ;;
  none)  ok "the repo has no draft release of mirrored weights" ;;
  held)  bad "the repo's draft release still holds its backbone weight" ;;
  *)     bad "the repo's draft release could not be read (${rel:-no answer}) - not checked is not closed" ;;
esac

echo
if [ "$left" = 0 ]; then echo "LINE 1 (the edge-detail model removed): CLOSED - 0 left"; exit 0; fi
echo "LINE 1 (the edge-detail model removed): NOT CLOSED - $left left"; exit 1
