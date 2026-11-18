#!/usr/bin/env bash
# STAC Build — what the rebuild's scripts share: the paths, the list of envs, logging, steps that
# are done once (a stamp per step: a second run resumes where the first stopped), checked downloads.
# Sourced, never run.

REBUILD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAC_ROOT="${STAC_ROOT:-$(cd "$REBUILD_DIR/../.." && pwd)}"
# The pod's layout is part of the product: config.yaml, the service scripts and the hot patches name
# these absolute paths. A rebuild puts everything back WHERE IT WAS.
WORKSPACE="${WORKSPACE:-/workspace}"
CONDA_ROOT="${CONDA_ROOT:-$WORKSPACE/miniforge3}"
HF_HOME="${HF_HOME:-$WORKSPACE/hf_cache}"
VENDOR="$STAC_ROOT/vendor"
LOCKS="$REBUILD_DIR/locks"
# downloads and build trees: on the VOLUME by default (a pod's own disk may be 20 GB; this needs ~60 GB
# and is deleted at the end). STAC_REBUILD_CACHE=/tmp/… puts it on the local disk when that is large enough.
CACHE="${STAC_REBUILD_CACHE:-$WORKSPACE/.stac_rebuild/cache}"
STATE="${STAC_REBUILD_STATE:-$WORKSPACE/.stac_rebuild}"        # stamps + logs: on the volume, they survive a restart
LOGS="$STATE/logs"

# every env of the pod, in the order they are built (the first six are the product's)
STAC_ENVS=(da3 mapanything semantic voice amb3r genrecon CloudComPy310 nodejs colmap gs)
# envs whose failure does not stop a rebuild (not used by the pipeline nor the app)
STAC_OPTIONAL_ENVS=(colmap gs)
# ENVS="da3 voice" rebuild.sh — only these envs (everything else of the run stays)
if [ -n "${ENVS:-}" ]; then read -ra STAC_ENVS <<<"$ENVS"; fi

export STAC_REBUILD_CACHE="$CACHE" STAC_ROOT REBUILD_DIR WORKSPACE CONDA_ROOT
export HF_HOME PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1 PYTHONUNBUFFERED=1
# the pod grants ~26 CPUs and shows the host's 252: an unbounded build or BLAS job is throttled to a crawl
export MAX_JOBS="${MAX_JOBS:-16}" MAKEFLAGS="${MAKEFLAGS:--j16}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"     # the A100 (sm_80)

log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '[%s] WARNING: %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die()  { printf '[%s] FAILED: %s\n' "$(date +%H:%M:%S)" "$*" >&2; exit 1; }

is_optional_env() { local e; for e in "${STAC_OPTIONAL_ENVS[@]}"; do [ "$e" = "$1" ] && return 0; done; return 1; }

# done <step> — was the step completed by an earlier run?   mark <step> — it is now.
step_done() { [ -f "$STATE/done/$1" ]; }
mark_done() { mkdir -p "$STATE/done"; date -u +%FT%TZ > "$STATE/done/$1"; }

# run_step <name> <function> [args…] — run once, in ITS OWN bash process (step.sh: `set -e` holds there —
# inside a function called from an `if` or a `||` bash ignores it, and a failed command would pass);
# its output goes to its own log; a failure names the log. FORCE=1 runs it again.
run_step() {
    local name="$1"; shift
    if step_done "$name" && [ "${FORCE:-0}" != 1 ]; then log "✓ $name (done $(cat "$STATE/done/$name"))"; return 0; fi
    mkdir -p "$LOGS"
    local t0=$SECONDS rc=0
    log "▶ $name"
    bash "$REBUILD_DIR/step.sh" "$@" > "$LOGS/$name.log" 2>&1 || rc=$?
    if [ $rc -eq 0 ]; then
        mark_done "$name"; log "✓ $name ($((SECONDS - t0)) s)"
    else
        rm -f "$STATE/done/$name"
        warn "$name failed — last lines of $LOGS/$name.log:"; tail -25 "$LOGS/$name.log" | sed 's/^/      /' >&2
        return 1
    fi
}

# fetch <url> <dest> [sha256] — download with resume and retries; with a sha256, a file that already
# matches is kept and a file that does not is an error (never a silent wrong weight).
fetch() {
    local url="$1" dest="$2" sha="${3:-}"
    if [ -f "$dest" ] && [ -n "$sha" ] && [ "$(sha256sum "$dest" | cut -d' ' -f1)" = "$sha" ]; then return 0; fi
    mkdir -p "$(dirname "$dest")"
    curl -fL --retry 8 --retry-delay 5 --connect-timeout 30 -C - -o "$dest.part" "$url" \
        || { rm -f "$dest.part"; curl -fL --retry 8 --retry-delay 5 --connect-timeout 30 -o "$dest.part" "$url"; }
    if [ -n "$sha" ]; then
        local got; got="$(sha256sum "$dest.part" | cut -d' ' -f1)"
        [ "$got" = "$sha" ] || { rm -f "$dest.part"; echo "sha256 of $url is $got, expected $sha" >&2; return 1; }
    fi
    mv "$dest.part" "$dest"
}

# activate <env> — conda's activation scripts (the compilers', gdal's) read unset variables: not under `set -u`
activate() { set +u; source "$CONDA_ROOT/etc/profile.d/conda.sh"; conda activate "$1"; set -u; }

# env_python <env> — the env's interpreter
env_python() { echo "$CONDA_ROOT/envs/$1/bin/python"; }

# the tokens: the pod's .env (git-ignored) — HF_TOKEN, GIT_USER, GIT_TOKEN, DINOV3_URL…
load_env() {
    local f
    for f in "$STAC_ROOT/.env" "$WORKSPACE/.env"; do
        if [ -f "$f" ]; then set -a; source "$f"; set +a; return 0; fi
    done
    return 1
}
