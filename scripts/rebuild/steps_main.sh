#!/usr/bin/env bash
# STAC Build — REBUILD, the steps of the run itself: the preflight, the weights, the final check.
# Sourced by step.sh.

step_preflight() {
    echo "repo      $STAC_ROOT @ $(git -C "$STAC_ROOT" rev-parse --short HEAD)"
    echo "workspace $WORKSPACE   conda $CONDA_ROOT   HF_HOME $HF_HOME   cache $CACHE"
    [ "$STAC_ROOT" = /workspace/stac-build ] || echo "WARNING: the repo is not at /workspace/stac-build — config.yaml and the service scripts name that path"
    # the pod's template gives every shell these (the old pod's /etc/rp_environment): what is launched outside the
    # backend reads the SECOND hub cache through HF_HOME — another template, another cache
    local template_hf="/workspace/.cache/huggingface/"
    if [ -f /etc/rp_environment ] && ! grep -q "HF_HOME=\"$template_hf\"" /etc/rp_environment; then
        echo "WARNING: this pod's template does not set HF_HOME=$template_hf as the old one did (/etc/rp_environment) — a stage launched from a shell would look for its models elsewhere"
    fi
    local gpu; gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    case "$gpu" in *A100*) : ;; *) echo "WARNING: the card is '$gpu', the old pod's was an A100 (sm_80): the backup set's compiled CUDA extensions were built for it — run with COMPILE=1 to compile them here" ;; esac
    [ -n "${HF_TOKEN:-}" ] || { echo "HF_TOKEN is not set (.env): the gated weights (SAM 3 / 3.1, VGGT-Omega) cannot be fetched"; return 1; }
    local code
    code="$(curl -s -o /dev/null -w '%{http_code}' -I -L --max-time 30 -H "Authorization: Bearer $HF_TOKEN" \
            https://huggingface.co/facebook/sam3.1/resolve/daa63191845a41281374e725f4c9e51c7a824460/config.json)"
    [ "$code" = 200 ] || { echo "Hugging Face answered $code for the gated repo facebook/sam3.1: the token is wrong or has no access"; return 1; }
    echo "Hugging Face token: access to the gated repos"
    git ls-remote --exit-code https://github.com/hernanbarreto/Depth-Anything-3.git HEAD >/dev/null \
        || { echo "no access to the private fork hernanbarreto/Depth-Anything-3 (GIT_USER / GIT_TOKEN in .env)"; return 1; }
    echo "GitHub: access to the private fork"
    if [ -z "${DINOV3_URL:-}" ]; then
        echo "WARNING: DINOV3_URL is not set — Meta's two DINOv3 checkpoints will be reported missing"
    else
        code="$(python3 - <<'PY'
import os, subprocess, sys
sys.path.insert(0, os.environ["REBUILD_DIR"])
import models
url = models.dinov3_url("dinov3_vitl16/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
print(subprocess.run(["curl", "-s", "-o", "/dev/null", "-I", "-w", "%{http_code}", "--max-time", "30", url], capture_output=True, text=True).stdout)
PY
)"
        if [ "$code" = 200 ]; then echo "Meta's DINOv3 link: valid"
        else echo "WARNING: Meta's DINOv3 link answers $code (expired) — ask for a new one at https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/ and put it in .env as DINOV3_URL; everything else is rebuilt meanwhile"; fi
    fi
}

# MODEL_MATCH=text — only the weights whose path or source holds it (a partial rebuild)
step_models() { python3 "$REBUILD_DIR/models.py" fetch --jobs "${MODEL_JOBS:-6}" ${MODEL_MATCH:+--match "$MODEL_MATCH"}; }
step_models_links() { python3 "$REBUILD_DIR/models.py" fetch --only link,copy; }

step_verify() {
    local env rc=0
    for env in "${STAC_ENVS[@]}"; do
        [ -s "$LOCKS/$env.records.tsv" ] && [ -x "$(env_python "$env")" ] || continue
        "$(env_python "$env")" "$REBUILD_DIR/verify_env.py" "$env" "$LOCKS" || { is_optional_env "$env" || rc=1; }
    done
    python3 "$REBUILD_DIR/models.py" verify || rc=1
    if have_backup; then step_backup_verify || rc=1; fi
    python3 "$REBUILD_DIR/inventory.py" check || rc=1
    "$VENDOR/PotreeConverter/build/PotreeConverter" --help >/dev/null 2>&1 && echo "PotreeConverter: runs" || { echo "PotreeConverter: DOES NOT RUN"; rc=1; }
    "$VENDOR/KTX-Software-4.4.2-Linux-x86_64/bin/ktx" --version >/dev/null 2>&1 && echo "ktx: runs" || { echo "ktx: DOES NOT RUN"; rc=1; }
    ( cd "$STAC_ROOT/server" && PYTHONHASHSEED=0 "$(env_python da3)" -c "import main; print('backend: imports')" ) || { echo "backend: DOES NOT IMPORT"; rc=1; }
    return $rc
}

