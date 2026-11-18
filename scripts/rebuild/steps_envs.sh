#!/usr/bin/env bash
# STAC Build — REBUILD, the conda environments. Sourced by rebuild.sh.
#
# An env is rebuilt in three steps, each from a lock written by capture.sh on the pod that works:
#   conda_<env>   the exact conda packages (URL + md5 of every one: no solver runs)
#   pip_<env>     the exact wheel files (URL + sha256), installed offline with --no-deps; the few
#                 source distributions are compiled here; a package that overwrote another's files
#                 on the original pod is installed last here too
#   extras_<env>  what came from a directory of the pod: the editable vendor trees, the extensions
#                 compiled from pinned commits, the files patched in place (overlays)
# An env that already exists and was not built by this script is LEFT ALONE (REBUILD_EXISTING=1
# recreates it): running the rebuild on the live pod changes nothing.

# Miniforge3 26.3.2-2, the installer the pod was built with (it stays in /workspace, as on the pod)
MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/download/26.3.2-2/Miniforge3-26.3.2-2-Linux-x86_64.sh"
MINIFORGE_SHA256="42260ffe3830fb953d5eee1bbb32229ff06aa7c3833c1ed7a9a0420a95685d94"

step_miniforge() {
    if [ ! -x "$CONDA_ROOT/bin/conda" ]; then
        fetch "$MINIFORGE_URL" "$WORKSPACE/Miniforge3-Linux-x86_64.sh" "$MINIFORGE_SHA256"
        bash "$WORKSPACE/Miniforge3-Linux-x86_64.sh" -b -p "$CONDA_ROOT"
        touch "$CONDA_ROOT/.stac_rebuild"
    fi
    "$CONDA_ROOT/bin/conda" --version
    # the base env against its lock: the installer's own packages, unless something was added since
    local want got
    want="$(grep '^https' "$LOCKS/base.conda-explicit.txt" | sed 's/#.*//' | sort)"
    got="$("$CONDA_ROOT/bin/conda" list -p "$CONDA_ROOT" --explicit | grep '^https' | sort)"
    if [ "$want" != "$got" ]; then
        if [ -f "$CONDA_ROOT/.stac_rebuild" ]; then
            echo "the base env differs from its lock — installing the lock's packages"
            "$CONDA_ROOT/bin/conda" install -y -q -p "$CONDA_ROOT" --file "$LOCKS/base.conda-explicit.txt"
        else
            echo "the base env differs from its lock and is not this script's — left alone"
        fi
    else
        echo "base: $(echo "$got" | wc -l) conda packages, as locked"
    fi
}

# may this run build the env? yes when it does not exist, or when an earlier run of this script
# began it (its marker is inside the prefix); no when it is somebody else's.
env_is_ours() {
    local prefix="$CONDA_ROOT/envs/$1"
    [ ! -d "$prefix" ] || [ -f "$prefix/.stac_rebuild" ] || [ "${REBUILD_EXISTING:-0}" = 1 ]
}

step_conda_env() {
    local env="$1" prefix="$CONDA_ROOT/envs/$1" lock="$LOCKS/$1.conda-explicit.txt"
    [ -f "$lock" ] || { echo "no lock for $env: $lock"; return 1; }
    if ! env_is_ours "$env"; then echo "$prefix exists and is not this script's — left alone (REBUILD_EXISTING=1 to recreate)"; return 0; fi
    rm -rf "$prefix"
    "$CONDA_ROOT/bin/conda" create -y -q -p "$prefix" --file "$lock"
    touch "$prefix/.stac_rebuild"
    local want got
    want="$(grep -c '^https' "$lock")"
    got="$("$CONDA_ROOT/bin/conda" list -p "$prefix" --explicit | grep -c '^https')"
    [ "$want" = "$got" ] || { echo "$env holds $got conda packages, the lock $want"; return 1; }
    echo "$env: $got conda packages"
}

# every wheel of every env, downloaded ONCE into the content-addressed cache (the envs share most of
# the large ones) — one process, so no two downloads ever write the same file
step_fetch_wheels() {
    local env
    for env in "${STAC_ENVS[@]}"; do
        [ -s "$LOCKS/$env.wheels.tsv" ] || continue
        python3 "$REBUILD_DIR/wheels.py" fetch "$env" "$LOCKS" "$CACHE/wheelcache" "$CACHE/wheels/$env" --jobs "${JOBS:-16}"
    done
}

step_pip_env() {
    local env="$1" dir="$CACHE/wheels/$1" py
    py="$(env_python "$env")"
    [ -s "$LOCKS/$env.wheels.tsv" ] || { echo "$env: nothing from pip"; return 0; }
    if ! env_is_ours "$env"; then echo "$env is not this script's — left alone"; return 0; fi
    python3 "$REBUILD_DIR/wheels.py" fetch "$env" "$LOCKS" "$CACHE/wheelcache" "$dir" --jobs "${JOBS:-16}"
    awk -F'\t' '$6 == "wheel" {print $1 "==" $2}' "$LOCKS/$env.wheels.tsv" > "$dir/wheels.txt"
    # the wheels: exactly these files, no index asked, nothing resolved
    "$py" -m pip install --no-deps --no-index --find-links "$dir" -r "$dir/wheels.txt"
    # the source distributions: with the backup set their installed files come from it, as they were
    # (restore_env_archive, in the env's extras step); without it they are compiled here.
    local name version sha size url kind rest
    while IFS=$'\t' read -r name version sha size url kind rest; do
        [ "$kind" = sdist ] || continue
        if use_prebuilt; then echo "── $name $version: from the backup set (not compiled)"; continue; fi
        is_overlap_winner "$env" "$name" && continue          # (installed last, below)
        echo "── building $name $version from source"
        "$py" -m pip install --no-deps "$dir/$(basename "${url%%#*}")"
    done < "$LOCKS/$env.wheels.tsv"
    # wheels by URL (exact files, sha256 in the line) and git commits (built at install: with the backup
    # set their installed files come from it, as the source distributions' do)
    if [ -s "$LOCKS/$env.pip-url.txt" ]; then
        if use_prebuilt; then grep -v ' @ git+' "$LOCKS/$env.pip-url.txt" > "$dir/by-url.txt" || true
        else cp "$LOCKS/$env.pip-url.txt" "$dir/by-url.txt"; fi
        if [ -s "$dir/by-url.txt" ]; then "$py" -m pip install --no-deps -r "$dir/by-url.txt"; fi
    fi
    # the packages whose files won over another's on the original pod: installed again, last
    if [ -s "$LOCKS/$env.overlaps.tsv" ]; then
        local winner
        for winner in $(cut -f1 "$LOCKS/$env.overlaps.tsv" | sort -u); do
            while IFS=$'\t' read -r name version sha size url kind rest; do
                [ "$name" = "$winner" ] || continue
                if [ "$kind" = sdist ] && use_prebuilt; then continue; fi      # (its files come from the set, after this step)
                echo "── $name $version last (its files replace another package's, as on the original pod)"
                "$py" -m pip install --no-deps --force-reinstall --no-index --find-links "$dir" "$dir/$(basename "${url%%#*}")" \
                    || "$py" -m pip install --no-deps --force-reinstall "$dir/$(basename "${url%%#*}")"
            done < "$LOCKS/$env.wheels.tsv"
        done
    fi
    "$py" "$REBUILD_DIR/capture_env.py" scripts "$env" "$LOCKS"      # (python / python3.x in each launcher's first line)
    "$py" -m pip --version
}

is_overlap_winner() { [ -s "$LOCKS/$1.overlaps.tsv" ] && cut -f1 "$LOCKS/$1.overlaps.tsv" | grep -qxF "$2"; }

# the files edited in place inside an env (found by capture_env.py's audit): copied over the install
apply_overlays() {
    local env="$1" src="$REBUILD_DIR/overlays/$1" site
    [ -d "$src" ] || return 0
    site="$("$(env_python "$env")" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    cp -rv "$src/." "$site/"
}

# ── what came from a directory of the pod ─────────────────────────────────────────────────────
step_extras_da3() {
    env_is_ours da3 || { echo "da3 is not this script's — left alone"; return 0; }
    if use_prebuilt; then
        restore_env_archive da3; apply_overlays da3
        "$(env_python da3)" -c "import torch, depth_anything_3, sim3solve, nvdiffrast; print('da3 OK (from the set): torch', torch.__version__)"
        return 0
    fi
    local py; py="$(env_python da3)"
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
    "$py" -m pip install --no-deps -e "$VENDOR/depth-anything-3"
    "$py" -m pip install --no-deps --no-build-isolation -e "$VENDOR/nvdiffrast"        # (_nvdiffrast_c, CUDA)
    "$py" -m pip install --no-deps --no-build-isolation -e "$VENDOR/VGGT-Long"         # (sim3solve, cp310)
    apply_overlays da3
    "$py" -c "import torch, depth_anything_3, sim3solve, nvdiffrast; print('da3 OK: torch', torch.__version__)"
}

step_extras_mapanything() {
    env_is_ours mapanything || { echo "mapanything is not this script's — left alone"; return 0; }
    if use_prebuilt; then
        restore_env_archive mapanything; apply_overlays mapanything
        "$(env_python mapanything)" -c "import torch, mapanything, sim3solve, uniception; print('mapanything OK (from the set): torch', torch.__version__)"
        return 0
    fi
    local py; py="$(env_python mapanything)"
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
    "$py" -m pip install --no-deps -e "$VENDOR/MapAnything2"
    "$py" -m pip install --no-deps --no-build-isolation -e "$VENDOR/VGGT-Long"         # (sim3solve, cp312)
    apply_overlays mapanything           # uniception's DINOv2 from /workspace/dinov2_mapanything
    "$py" -c "import torch, mapanything, sim3solve, uniception; print('mapanything OK: torch', torch.__version__)"
}

step_extras_amb3r() {
    env_is_ours amb3r || { echo "amb3r is not this script's — left alone"; return 0; }
    if use_prebuilt; then
        restore_env_archive amb3r; apply_overlays amb3r
        "$(env_python amb3r)" -c "import torch, dpretrieval; print('amb3r OK (from the set): torch', torch.__version__)"
        return 0
    fi
    activate amb3r
    export JOBS="${MAX_JOBS:-16}"
    command -v nvcc >/dev/null || export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
    bash "$VENDOR/AMB3R-SLAM/thirdparty/build.sh"                                       # DBoW2, dpretrieval, ALIKED
    apply_overlays amb3r
    python -c "import torch, dpretrieval; print('amb3r OK: torch', torch.__version__)"
}

# GenRecon's extensions, each from the commit the pod's were built from (2026-10-16; the python trees
# of the installed packages were compared with these commits on 2026-11-18)
GENRECON_EXT=(
  "nvdiffrast|https://github.com/NVlabs/nvdiffrast.git|253ac4fcea7de5f396371124af597e6cc957bfae"
  "nvdiffrec|https://github.com/JeffreyXiang/nvdiffrec.git|b296927cc7fd01c2ac1087c8065c4d7248f72da4"
  "CuMesh|https://github.com/JeffreyXiang/CuMesh.git|12289e1062f0603f2f0d0771b02e1395d247f26f"
  "FlexGEMM|https://github.com/JeffreyXiang/FlexGEMM.git|6dd94a859c26ee8246888502eada3dd8ad85532e"
)

step_extras_genrecon() {
    env_is_ours genrecon || { echo "genrecon is not this script's — left alone"; return 0; }
    if use_prebuilt; then
        restore_env_archive genrecon; apply_overlays genrecon
        "$(env_python genrecon)" -c "import torch, flash_attn, nvdiffrast, cumesh, flex_gemm, o_voxel, PIL; print('genrecon OK (from the set): torch', torch.__version__, 'PIL', PIL.__version__)"
        return 0
    fi
    activate genrecon
    export CUDA_HOME="$CONDA_PREFIX"                    # the env's own CUDA 12.6 toolkit
    local ext="$CACHE/src/genrecon_ext" row name url commit
    mkdir -p "$ext"
    for row in "${GENRECON_EXT[@]}"; do
        IFS='|' read -r name url commit <<<"$row"
        clone_at "$ext/$name" "$url" "$commit"
        git -C "$ext/$name" submodule update --init --recursive
        pip install --no-deps --no-build-isolation "$ext/$name"
    done
    rm -rf "$ext/o-voxel" && cp -r "$VENDOR/GenRecon/o-voxel" "$ext/o-voxel"    # (GenRecon's own, its eigen a submodule)
    pip install --no-deps --no-build-isolation "$ext/o-voxel"
    apply_overlays genrecon
    python -c "import torch, flash_attn, nvdiffrast, cumesh, flex_gemm, o_voxel, PIL; print('genrecon OK: torch', torch.__version__, 'PIL', PIL.__version__)"
}

step_extras_plain() {     # an env with nothing of its own but possible overlays (semantic, voice, gs…)
    env_is_ours "$1" || { echo "$1 is not this script's — left alone"; return 0; }
    if use_prebuilt; then restore_env_archive "$1"; fi
    apply_overlays "$1"
    "$(env_python "$1")" -c "import sys; print('$1 OK: python', sys.version.split()[0])"
}

step_extras_nodejs() {
    local bin="$CONDA_ROOT/envs/nodejs/bin"
    export PATH="$bin:$PATH"
    node --version
    if env_is_ours nodejs && [ -s "$LOCKS/nodejs.npm-global.txt" ]; then
        # shellcheck disable=SC2046
        npm install -g --no-fund --no-audit $(cat "$LOCKS/nodejs.npm-global.txt")
    fi
    ( cd "$STAC_ROOT/ui" && npm ci --no-fund --no-audit && npm run provision:xr8 )      # the interface + static/xr8
    ( cd "$STAC_ROOT/tools/glb" && npm ci --no-fund --no-audit )                        # the mesh compressor / tiler
}

# the isolated venv of the AR export (server/tools/glb_to_usdz.py): the system's python 3.11 and five
# packages, the exact files of locks/usdtools.wheels.tsv.
# ⚠ ON THE POD OF 2026-11-18 THIS VENV DOES NOT WORK: it was made when /usr/bin/python3 was 3.11, the image
# moved it to 3.12, and its bin/python (→ python3) no longer sees lib/python3.11 ("No module named pxr").
# This step makes it with python3.11 by name, so the rebuilt one WORKS — the one difference from the pod
# as it is, declared to the user (his to decide whether the broken one is wanted back).
step_usdtools() {
    local dest="$WORKSPACE/usdtools" base=/usr/bin/python3.11 dir="$CACHE/wheels/usdtools"
    if [ -x "$dest/bin/python" ] && "$dest/bin/python" -c "import pxr, trimesh, pymeshlab" 2>/dev/null; then echo "usdtools present"; return 0; fi
    if [ -d "$dest" ] && [ ! -f "$dest/.stac_rebuild" ] && [ "${REBUILD_EXISTING:-0}" != 1 ]; then
        echo "$dest exists and is not this script's — left alone (it does not import pxr: REBUILD_EXISTING=1 remakes it)"; return 0
    fi
    [ -x "$base" ] || { echo "the image has no /usr/bin/python3.11 (the venv's python)"; return 1; }
    rm -rf "$dest"; "$base" -m venv "$dest"; touch "$dest/.stac_rebuild"
    python3 "$REBUILD_DIR/wheels.py" fetch usdtools "$LOCKS" "$CACHE/wheelcache" "$dir" --jobs 8
    "$dest/bin/python" -m pip install --no-input --no-deps --no-index --find-links "$dir" -r "$LOCKS/usdtools.pip.txt"
    "$dest/bin/python" -c "import pxr, trimesh, pymeshlab; print('usdtools OK')"
}
