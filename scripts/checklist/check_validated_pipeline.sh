#!/usr/bin/env bash
# THE VALIDATED PIPELINE STAYS, WITH THREE CHANGES (USER 2026-11-23: "debe quedar el pipeline original solo con la
# correccion de oscuros, y el resto fuera del pipeline, si queres en una rama aparte" / "yo dejaria F2 apagado
# pointdit afuera en el principal" / "el resto campo B, y moge 3 a la rama, ok!" / "acordate de poner F5 configurado
# teniendo en cuenta que no esta F2").
#
# The written check of that order: the reconstruction is the validated one (commit 25636e2) plus ONLY the dark frames,
# F2 off with F5's tolerance from the tie's seams, and PointDiT off; the trial of ViPE's concepts is on its branch and
# nowhere in the pipeline; its numbers are kept. Read only (its tests write under the system's temporary folder); the
# same answer at every run; the last line says CLOSED or NOT CLOSED:
#
#     bash scripts/checklist/check_validated_pipeline.sh
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 2
BASE=25636e2
BRANCH=trial/vipe-concepts
PY=/workspace/miniforge3/envs/da3/bin/python
bad=0
ok() { printf '  ok    %s\n' "$*"; }
no() { bad=$((bad + 1)); printf '  FAIL  %s\n' "$*"; }
code_lines() {   # the changed lines of one file against BASE, comments and blank lines left out
  git diff -U0 "$BASE" -- "$1" | grep -E '^[+-][^+-]|^[+-]$' | sed -E 's/^([+-])[[:space:]]*/\1/; s/[[:space:]]*#.*$//' \
    | grep -vE '^[+-]$'
}

echo "1. server/ against the validated reconstruction ($BASE): only these files differ"
want=$(printf '%s\n' server/config.yaml server/precision/config.py server/precision/runner.py \
  server/precision/tie_tolerance.py server/tests/test_precision_runner.py server/tests/test_tie_tolerance.py | sort)
have=$( (git diff --name-only "$BASE" -- server/; git ls-files --others --exclude-standard -- server/precision server/tests) | sort -u)
if [ "$have" = "$want" ]; then ok "$(echo "$have" | tr '\n' ' ')"; else no "differing files: $(echo "$have" | tr '\n' ' ')"; fi

echo "2. inside them, only the three changes"
cfg=$("$PY" - "$BASE" <<'EOF'
import subprocess, sys, yaml
base = yaml.safe_load(subprocess.run(["git", "show", f"{sys.argv[1]}:server/config.yaml"], capture_output=True,
                                     text=True, check=True).stdout)
now = yaml.safe_load(open("server/config.yaml"))
def flat(d, p=""):
    if not isinstance(d, dict):
        return {p: d}
    out = {}
    for k, v in d.items():
        out.update(flat(v, f"{p}.{k}" if p else str(k)))
    return out
a, b = flat(base), flat(now)
keys = sorted(k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>"))
print("|".join(f"{k}: {a.get(k, '<absent>')} -> {b.get(k, '<absent>')}" for k in keys))
EOF
)
want_cfg='intake.quality.clip_frac_max: 0.25 -> 1.0|intake.quality.luma_lo: 40.0 -> 0.0|reconstruction.precision.gauge.enabled: <absent> -> False|reconstruction.precision.mono_detail.enabled: True -> False'
[ "$cfg" = "$want_cfg" ] && ok "config.yaml (every value compared): ${cfg//|/; }" || no "config.yaml values changed: $cfg"
pc=$(code_lines server/precision/config.py | tr '\n' '|')
want_pc='+enabled: bool = True|+enabled=(_bool(g, "enabled", "gauge") if "enabled" in g else True),|'
[ "$pc" = "$want_pc" ] && ok "precision/config.py: GaugeConfig.enabled (absent = true, the record) and its loader" \
  || no "precision/config.py changes: $pc"
rn_minus=$(code_lines server/precision/runner.py | grep -c '^-')
rn_other=$(code_lines server/precision/runner.py | grep '^+' | grep -vcE 'no_f2|f2_gauge|tie_tolerance|write_tolerance_from_seams|keys = \{s\.key')
[ "$rn_minus" = 1 ] && [ "$rn_other" = 0 ] && ok "precision/runner.py: the chain without f2_gauge and the seams' tolerance for F5" \
  || no "precision/runner.py: $rn_minus line(s) removed, $rn_other added line(s) not about F2"

echo "3. the configuration as the pipeline loads it"
out=$(cd server && "$PY" - <<'EOF' 2>&1
import yaml
from intake.config import load_intake_config
from precision.config import load_precision_config
from precision.runner import chain_steps
raw = yaml.safe_load(open("config.yaml"))
q, p = load_intake_config(raw).quality, load_precision_config(raw)
chain = [s.key for s in chain_steps(p)]
good = (q.luma_lo == 0.0 and q.clip_frac_max == 1.0 and p.gauge.enabled is False and p.mono_detail.enabled is False
        and chain == ["f0_camera", "f4_tracks", "f5_refine", "f6_bend", "f6_check"])
print(("GOOD " if good else "BAD ") + f"luma_lo {q.luma_lo} clip_frac_max {q.clip_frac_max} gauge.enabled "
      f"{p.gauge.enabled} mono_detail.enabled {p.mono_detail.enabled} chain {' '.join(chain)}")
EOF
)
case "$out" in GOOD*) ok "${out#GOOD }";; *) no "$out";; esac

echo "4. F5 without F2: the chain writes its anchors' tolerance from the seams (the tests)"
t=$(cd server && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 "$PY" -m pytest -q -p no:cacheprovider tests/test_tie_tolerance.py \
  "tests/test_precision_runner.py::test_without_f2_the_seams_tolerance_is_written_only_for_a_chain_that_runs_f5" 2>&1 | tail -1)
case "$t" in *passed*) case "$t" in *failed*|*error*) no "$t";; *) ok "$t";; esac;; *) no "$t";; esac

echo "5. the trial is not in the pipeline"
hits=$(grep -rlE --exclude-dir=__pycache__ 'moge3|MoGe|metric_anchor|BendField|choose_field|anchor_dispersions' \
  server/precision server/workers server/tests server/config.yaml 2>/dev/null | tr '\n' ' ')
[ -z "$hits" ] && ok "no MoGe-3, metric anchor or field B in server/precision, server/workers, server/tests, config.yaml" \
  || no "trial names found in: $hits"

echo "6. the trial is on its branch ($BRANCH)"
if git cat-file -e "$BRANCH:server/precision/moge3_runner.py" 2>/dev/null \
   && git grep -q choose_field "$BRANCH" -- server/precision/depth_on_f5.py \
   && git grep -q build_metric_anchor "$BRANCH" -- server/precision/refine.py; then
  ok "$(git log -1 --format='%h %s' "$BRANCH" | cut -c1-110)"
else no "the branch or its code is missing"; fi

echo "7. the numbers kept"
f=experiments/new_sequence/RESULTS_vipe_concepts.md
[ -f "$f" ] && grep -q pccr3108_mejoras "$f" && grep -q bufferStop_mejoras "$f" && ok "$f" || no "$f missing or incomplete"

echo "8. committed on main"
br=$(git rev-parse --abbrev-ref HEAD)
dirty=$(git status --porcelain -- server/ CLAUDE.md "$f" scripts/checklist/check_validated_pipeline.sh | wc -l)
[ "$br" = main ] && [ "$dirty" = 0 ] && ok "main $(git log -1 --format=%h)" || no "branch $br, $dirty uncommitted file(s)"

echo
if [ "$bad" = 0 ]; then echo "CLOSED"; exit 0; else echo "NOT CLOSED - $bad"; exit 1; fi
