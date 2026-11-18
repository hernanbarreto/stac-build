#!/usr/bin/env bash
# STAC Build — CAPTURE: write, from the pod that works TODAY, the locks a rebuild installs from.
#
#   bash scripts/rebuild/capture.sh                 every env: conda's exact packages, pip's pins, the
#                                                   exact wheel files (URL + sha256), the audit
#   bash scripts/rebuild/capture.sh da3 voice       only these envs
#   CHECK=1 bash scripts/rebuild/capture.sh         also DOWNLOAD every wheel and compare its files with
#                                                   the ones installed here (≈ 35 GB on the local disk,
#                                                   ≈ 10 min): the proof that a rebuild gives the same env
#
# Run it again after any `pip install` / `conda install` in an env, and commit scripts/rebuild/locks/.
# It reads the envs and writes only under scripts/rebuild/locks/ (and the cache on the local disk).
#
# USER 2026-11-18: "Prepara todo en .sh para que se creen automáticamente los entornos se descarguen
# todos los modelos y se vuelva a armar en un nuevo volumen en cuestión de poco tiempo … de forma
# completa y sin errores".
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib.sh"

mkdir -p "$LOCKS" "$CACHE/reports" "$CACHE/wheelcache"          # (LOCKS, CACHE: lib.sh)
ENVS=("$@"); [ ${#ENVS[@]} -eq 0 ] && ENVS=("${STAC_ENVS[@]}")

for env in "${ENVS[@]}"; do
    prefix="$CONDA_ROOT/envs/$env"
    [ -d "$prefix" ] || die "env $env does not exist in $CONDA_ROOT/envs"
    log "$env: conda's exact packages"
    "$CONDA_ROOT/bin/conda" list -p "$prefix" --explicit --md5 > "$LOCKS/$env.conda-explicit.txt"
    [ -x "$prefix/bin/python" ] || { log "$env: no python — conda packages only"; continue; }
    "$prefix/bin/python" "$HERE/capture_env.py" "$env" "$LOCKS"

    # pip's own resolution of the pins, one index per group so that a version is never taken from
    # another index than the one it came from: a local label (+cu128) = PyTorch's index of that
    # label; everything else = PyPI alone.
    reports=()
    for label in $(grep -oE '\+cu[0-9]+$' "$LOCKS/$env.pip.txt" | sort -u | tr -d '+') pypi; do
        req="$CACHE/reports/$env.$label.txt"; rep="$CACHE/reports/$env.$label.json"
        if [ "$label" = pypi ]; then
            grep -vE '\+cu[0-9]+$' "$LOCKS/$env.pip.txt" > "$req" || true
            index="https://pypi.org/simple"
        else
            grep -E "\+$label\$" "$LOCKS/$env.pip.txt" > "$req"
            index="https://download.pytorch.org/whl/$label"
        fi
        [ -s "$req" ] || continue
        log "$env: resolving $(wc -l < "$req") pin(s) on $index"
        PIP_CACHE_DIR="$CACHE/pipcache" "$prefix/bin/python" -m pip install --dry-run --ignore-installed --no-deps \
            --disable-pip-version-check -q --index-url "$index" -r "$req" --report "$rep" \
            > "$CACHE/reports/$env.$label.log" 2>&1 \
            || { tail -20 "$CACHE/reports/$env.$label.log" >&2; die "$env: pip could not resolve the pins on $index"; }
        reports+=("$rep")
    done
    python3 "$HERE/wheels.py" resolve "$env" "$LOCKS" "${reports[@]}"
    if [ "${CHECK:-0}" = 1 ]; then
        python3 "$HERE/wheels.py" check "$env" "$LOCKS" "$CACHE/wheelcache" --jobs "${JOBS:-16}"
    fi
done

# the global npm packages of the node env (the Claude CLI) and the USD tools venv
if [ -x "$CONDA_ROOT/envs/nodejs/bin/npm" ]; then
    PATH="$CONDA_ROOT/envs/nodejs/bin:$PATH" npm ls -g --depth=0 2>/dev/null \
        | grep -oE '[@a-z0-9/_.-]+@[0-9][^ ]*' | grep -vE '^(npm|corepack)@' > "$LOCKS/nodejs.npm-global.txt" || true
fi
log "capture done: $(ls "$LOCKS" | wc -l) file(s) in $LOCKS"
