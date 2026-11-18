#!/usr/bin/env bash
# STAC Build — CAPTURE: write, from the pod that works TODAY, everything a rebuild installs from.
#
#   bash scripts/rebuild/capture.sh                 the whole pod: every env, the hub caches, the inventory
#                                                   of the volume, the backup set
#   bash scripts/rebuild/capture.sh da3 voice       only these envs' locks (then the inventory and the set)
#   CHECK=1 bash scripts/rebuild/capture.sh         also DOWNLOAD every wheel and compare its files with the
#                                                   ones installed here (≈ 35 GB in the cache, ≈ 12 min): the
#                                                   proof that a rebuild gives the same env
#   SET=0 bash scripts/rebuild/capture.sh           the locks only, no backup set
#
# RUN IT AFTER ANY CHANGE OF THE POD — a `pip install`, a `conda install`, a new vendor clone, a new
# weight, a compilation, a new secret — then commit scripts/rebuild/ and COPY THE BACKUP SET OFF THE POD.
# `bash scripts/rebuild/drift.sh` says whether it is needed.
# It reads the pod and writes only scripts/rebuild/locks/, models.tsv's hub-cache rows and the set.
#
# USER 2026-11-18: "Prepara todo en .sh para que se creen automáticamente los entornos se descarguen
# todos los modelos y se vuelva a armar en un nuevo volumen en cuestión de poco tiempo … de forma
# completa y sin errores" / "debe quedar exactamente como está ahora … cada cosa que avancemos debemos
# actualizarlo para que siempre quede actualizado a la perfección".
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib.sh"

mkdir -p "$LOCKS" "$CACHE/reports" "$CACHE/wheelcache"          # (LOCKS, CACHE: lib.sh)
ENVS_ARG=("$@"); [ ${#ENVS_ARG[@]} -eq 0 ] && ENVS_ARG=("${STAC_ENVS[@]}")

# resolve_pins <env> <python with pip> — pip's own resolution of the env's pins into exact files
resolve_pins() {
    local env="$1" pip_python="$2" label req rep index reports=()
    # one index per group, so that a version is never taken from another index than its own: a local
    # label (+cu128) = PyTorch's index of that label; everything else = PyPI alone.
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
        PIP_CACHE_DIR="$CACHE/pipcache" "$pip_python" -m pip install --dry-run --ignore-installed --no-deps \
            --disable-pip-version-check -q --index-url "$index" -r "$req" --report "$rep" \
            > "$CACHE/reports/$env.$label.log" 2>&1 \
            || { tail -20 "$CACHE/reports/$env.$label.log" >&2; die "$env: pip could not resolve the pins on $index"; }
        reports+=("$rep")
    done
    python3 "$HERE/wheels.py" resolve "$env" "$LOCKS" "${reports[@]}"
    # the verdict of every file against the installed one: `identical` packages are downloaded by a
    # rebuild; anything else is archived as it is in the backup set (capture_env.py asis)
    python3 "$HERE/wheels.py" check "$env" "$LOCKS" "$CACHE/wheelcache" --jobs "${JOBS:-16}"
}

for env in "${ENVS_ARG[@]}"; do
    prefix="$CONDA_ROOT/envs/$env"
    [ -d "$prefix" ] || die "env $env does not exist in $CONDA_ROOT/envs"
    log "$env: conda's exact packages"
    "$CONDA_ROOT/bin/conda" list -p "$prefix" --explicit --md5 > "$LOCKS/$env.conda-explicit.txt"
    python3 "$HERE/audit_env_files.py" "$prefix" "$env" "$LOCKS"      # files no package owns; edited conda files
    [ -x "$prefix/bin/python" ] || { log "$env: no python — conda packages only"; continue; }
    "$prefix/bin/python" "$HERE/capture_env.py" "$env" "$LOCKS"
    # (the files and verdicts of an earlier capture stay valid while the pins are the same: no new download)
    if [ "${CHECK:-0}" != 1 ] && python3 "$HERE/wheels.py" current "$env" "$LOCKS"; then
        log "$env: the pins are the ones already resolved and checked"
    else
        resolve_pins "$env" "$prefix/bin/python"
    fi
    "$prefix/bin/python" "$HERE/capture_env.py" asis "$env" "$LOCKS"
    [ -s "$LOCKS/$env.modified.tsv" ] && warn "$env: files edited in place after their install — save them under overlays/$env/: $(cut -f4 "$LOCKS/$env.modified.tsv" | tr '\n' ' ')"
done

if [ $# -eq 0 ]; then
    # conda's own (base) env, as the pinned installer leaves it plus whatever was added since
    "$CONDA_ROOT/bin/conda" list -p "$CONDA_ROOT" --explicit --md5 > "$LOCKS/base.conda-explicit.txt"
    # the global npm packages of the node env (the assistant's CLI)
    if [ -x "$CONDA_ROOT/envs/nodejs/bin/npm" ]; then
        PATH="$CONDA_ROOT/envs/nodejs/bin:$PATH" npm ls -g --depth=0 2>/dev/null \
            | grep -oE '[@a-z0-9/_.-]+@[0-9][^ ]*' | grep -vE '^(npm|corepack)@' > "$LOCKS/nodejs.npm-global.txt" || true
    fi
    # the USD tools venv (python 3.11 of the system): its packages, read from its site-packages
    if [ -d "$WORKSPACE/usdtools/lib/python3.11/site-packages" ] && [ -x /usr/bin/python3.11 ]; then
        /usr/bin/python3.11 -c "
import sys, runpy
sys.path[:] = ['$WORKSPACE/usdtools/lib/python3.11/site-packages', '/usr/lib/python3.11', '/usr/lib/python3.11/lib-dynload', '$HERE']
sys.argv = ['capture_env.py', 'usdtools', '$LOCKS']
runpy.run_path('$HERE/capture_env.py', run_name='__main__')"
        grep -vE '^(pip|setuptools)==' "$LOCKS/usdtools.pip.txt" > "$LOCKS/usdtools.pip.txt.tmp" && mv "$LOCKS/usdtools.pip.txt.tmp" "$LOCKS/usdtools.pip.txt"
        rm -f "$LOCKS"/usdtools.{pip-url.txt,pip-local.tsv,modified.tsv,overlaps.tsv}
        CHECK=1 resolve_pins usdtools /usr/bin/python3.11
    fi
    # the hub caches' rows of the model manifest, from what the caches hold
    python3 "$HERE/capture_models.py"
fi

# the inventory of the whole volume and the backup set (what no public source can give back)
if [ "${SET:-1}" = 1 ]; then
    log "the inventory of the volume and the backup set (≈ 3 min)"
    python3 "$HERE/backup.py" make
else
    python3 "$HERE/inventory.py" scan
fi
rm -rf "$HERE/__pycache__"
log "capture done. NEXT: git add scripts/rebuild && git commit — and copy $WORKSPACE/stac_backup OFF the pod"
