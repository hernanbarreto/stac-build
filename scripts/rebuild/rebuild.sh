#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════════════════════
# STAC Build — REBUILD THE POD ON AN EMPTY VOLUME
#
# USER 2026-11-18: "Prepara todo en .sh para que se creen automáticamente los entornos se descarguen
# todos los modelos y se vuelva a armar en un nuevo volumen en cuestión de poco tiempo … de forma
# completa y sin errores".
#
#   On the new pod (same template: RunPod PyTorch, Ubuntu 24.04, CUDA 12.8.1 *devel*; volume at /workspace):
#
#     1.  nano /workspace/.env                  # the four secrets — scripts/rebuild/env.template
#     2.  source /workspace/.env
#         git clone https://$GIT_USER:$GIT_TOKEN@github.com/hernanbarreto/stac-build.git /workspace/stac-build
#         git -C /workspace/stac-build remote set-url origin https://github.com/hernanbarreto/stac-build.git
#         cp /workspace/.env /workspace/stac-build/.env
#     3.  bash /workspace/stac-build/scripts/rebuild/rebuild.sh
#     4.  bash /workspace/stac-build/init_pod.sh        # the services, as on every boot
#
#   Run it again after any failure: every step that finished is skipped (stamps in
#   /workspace/.stac_rebuild/done, one log per step in /workspace/.stac_rebuild/logs).
#
#     rebuild.sh                 everything
#     rebuild.sh --list          the steps and their state
#     rebuild.sh --only STEP     one step (again)            e.g. --only pip_da3
#     rebuild.sh --verify        only check what is installed against the locks
#     KEEP_CACHE=1 rebuild.sh    do not delete the download cache at the end (~60 GB)
#
# What it restores: the ten conda envs (exact packages), every vendor tree at its commit with our
# patches, every compiled extension, every model weight (sha256-checked), the node packages, the
# octree converter, the small tools. What it does NOT: the sessions (server/projects), the users'
# database (server/data) — data, to be copied — and Tailscale's identity. See README.md here.
# ═══════════════════════════════════════════════════════════════════════════════════════════════
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib.sh"
source "$HERE/steps_vendors.sh"          # (git_credentials; the steps themselves run through step.sh)
source "$HERE/steps_backup.sh"           # (have_backup)

T0=$SECONDS
FAILED=()
mkdir -p "$STATE/done" "$LOGS" "$CACHE"

# required <name> <function> [args] — a failure is recorded and the run goes on with what does not
# depend on it; optional — a failure is a warning.
required() { run_step "$@" || FAILED+=("$1"); }
optional() { run_step "$@" || warn "$1 failed (optional — not needed by the pipeline nor the app)"; }

# in_background <name> <function> [args] — the same, detached; collect with wait_background
BG_PIDS=(); BG_NAMES=()
in_background() { ( run_step "$@" ) & BG_PIDS+=($!); BG_NAMES+=("$1"); }
wait_background() {
    local i
    for i in "${!BG_PIDS[@]}"; do
        if ! wait "${BG_PIDS[$i]}"; then
            if [[ " ${OPTIONAL_STEPS[*]:-} " == *" ${BG_NAMES[$i]} "* ]]; then warn "${BG_NAMES[$i]} failed (optional)"; else FAILED+=("${BG_NAMES[$i]}"); fi
        fi
    done
    BG_PIDS=(); BG_NAMES=()
}

list_steps() {
    local s
    for s in backup_check restore_private preflight system submodules vendor_clones vendor_binaries miniforge restore_asis models \
             $(printf 'conda_%s ' "${STAC_ENVS[@]}") fetch_wheels $(printf 'pip_%s ' "${STAC_ENVS[@]}") \
             extras_da3 extras_mapanything extras_semantic extras_voice extras_amb3r extras_genrecon extras_CloudComPy310 \
             extras_colmap extras_gs extras_nodejs onetbb potree usdtools remote_tools models_links convert_dinov3 restore_git verify; do
        if step_done "$s"; then printf '  ✓ %-24s %s\n' "$s" "$(cat "$STATE/done/$s")"; else printf '  · %-24s\n' "$s"; fi
    done
}

run_only() {
    local s="$1"; FORCE=1
    case "$s" in
        conda_*)   required "$s" step_conda_env "${s#conda_}" ;;
        pip_*)     required "$s" step_pip_env "${s#pip_}" ;;
        extras_da3|extras_mapanything|extras_amb3r|extras_genrecon|extras_nodejs) required "$s" "step_$s" ;;
        extras_*)  required "$s" step_extras_plain "${s#extras_}" ;;
        *)         required "$s" "step_$s" ;;
    esac
}

main() {
    log "STAC Build rebuild — logs in $LOGS"
    # ── 0. the backup set, when it is here: checked whole, and the secrets first (.env) ──────────
    if have_backup; then
        log "backup set: $BACKUP_SET — the compiled parts, the secrets and the local git work come from it, as they were"
        FORCE=1 run_step backup_check step_backup_check || die "the backup set is damaged — nothing was changed"
        required restore_private step_restore_private
    else
        warn "NO BACKUP SET at $BACKUP_SET: what was compiled on the old pod is compiled AGAIN (the same sources, other bytes); its secrets, local branches and stashes are NOT restored"
    fi
    load_env || warn "no .env found in $STAC_ROOT nor $WORKSPACE — the private repos and the gated weights will fail"
    git_credentials

    # ── A. the ground: tools, access, the code of others ─────────────────────────────────────────
    FORCE=1 run_step preflight step_preflight || die "preflight failed — nothing was changed"
    required system          step_system
    required submodules      step_submodules
    required vendor_clones   step_vendor_clones
    required vendor_binaries step_vendor_binaries
    required miniforge       step_miniforge
    have_backup && required restore_asis step_restore_asis        # (into the vendor trees just cloned, the caches, the interface)
    [ ${#FAILED[@]} -eq 0 ] || { summary; exit 1; }

    # ── B. in the background, all along: ~78 GB of weights, and every wheel of every env ─────────
    in_background models step_models
    MODELS_PID=${BG_PIDS[0]}; BG_PIDS=(); BG_NAMES=()
    ( run_step fetch_wheels step_fetch_wheels ) & WHEELS_PID=$!

    # ── C. the conda packages, one env after another (they share conda's package cache) ──────────
    local env
    for env in "${STAC_ENVS[@]}"; do
        if is_optional_env "$env"; then optional "conda_$env" step_conda_env "$env"; else required "conda_$env" step_conda_env "$env"; fi
    done
    wait "$WHEELS_PID" || FAILED+=(fetch_wheels)

    # ── D. the pip packages: the envs are independent, four at a time ───────────────────────────
    OPTIONAL_STEPS=(pip_colmap pip_gs)
    local running=0
    for env in "${STAC_ENVS[@]}"; do
        step_done "conda_$env" || continue
        in_background "pip_$env" step_pip_env "$env"
        running=$((running + 1))
        if [ $running -ge "${PIP_PARALLEL:-4}" ]; then wait_background; running=0; fi
    done
    wait_background

    # ── E. what is built from the vendor trees — one at a time (two envs build in vendor/VGGT-Long) ─
    step_done pip_da3          && required extras_da3          step_extras_da3
    step_done pip_mapanything  && required extras_mapanything  step_extras_mapanything
    step_done pip_semantic     && required extras_semantic     step_extras_plain semantic
    step_done pip_voice        && required extras_voice        step_extras_plain voice
    step_done pip_amb3r        && required extras_amb3r        step_extras_amb3r
    step_done pip_genrecon     && required extras_genrecon     step_extras_genrecon
    step_done pip_CloudComPy310 && required extras_CloudComPy310 step_extras_plain CloudComPy310
    step_done pip_colmap       && optional extras_colmap       step_extras_plain colmap
    step_done pip_gs           && optional extras_gs           step_extras_plain gs
    step_done conda_nodejs     && required extras_nodejs       step_extras_nodejs
    required onetbb       step_onetbb
    required potree       step_potree
    optional usdtools     step_usdtools
    optional remote_tools step_remote_tools

    # ── F. the weights are in: the links between them, the one conversion, the check of it all ───
    log "waiting for the weights (log: $LOGS/models.log)"
    wait "$MODELS_PID" || FAILED+=(models)
    required models_links   step_models_links
    step_done extras_genrecon && required convert_dinov3 step_convert_dinov3
    have_backup && required restore_git step_restore_git          # (every local branch, stash and worktree)
    FORCE=1 required verify step_verify
    summary
}

summary() {
    echo
    log "════ REBUILD: $(( (SECONDS - T0) / 60 )) min ════"
    list_steps
    if [ ${#FAILED[@]} -eq 0 ]; then
        if have_backup; then log "ALL DONE — with the backup set: the pod as it was. Next: bash $STAC_ROOT/init_pod.sh"
        else log "ALL DONE — WITHOUT the backup set: the compiled parts are new builds. Next: bash $STAC_ROOT/init_pod.sh"; fi
        if [ "${KEEP_CACHE:-0}" != 1 ]; then rm -rf "$CACHE"; "$CONDA_ROOT/bin/conda" clean --all -y -q >/dev/null 2>&1 || true; log "download cache removed"; fi
        cat "$LOGS/verify.log" 2>/dev/null | sed 's/^/   /'
        return 0
    fi
    warn "FAILED STEPS: ${FAILED[*]}"
    warn "each one's log: $LOGS/<step>.log — fix, then run rebuild.sh again (finished steps are skipped)"
    cat "$LOGS/verify.log" 2>/dev/null | sed 's/^/   /'
    return 1
}

case "${1:-}" in
    --list)   list_steps ;;
    --verify) load_env; FORCE=1 run_step verify step_verify; cat "$LOGS/verify.log" ;;
    --only)   [ -n "${2:-}" ] || die "--only needs a step name (rebuild.sh --list)"; load_env; git_credentials; run_only "$2"; [ ${#FAILED[@]} -eq 0 ] ;;
    "")       main ;;
    *)        sed -n '2,33p' "${BASH_SOURCE[0]}"; exit 2 ;;
esac
