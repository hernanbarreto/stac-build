#!/usr/bin/env bash
# STAC Build — REBUILD, the vendor trees: the submodules, the pinned clones, the commits of ours that
# exist on no remote (git bundles), the patches. Sourced by rebuild.sh.
#
# What is NOT here because it comes with the repo itself: vendor/MapAnything2, vendor/PotreeConverter
# (its source), vendor/cloudcompy, vendor/sam3.

# name | url | base commit (on the remote) | bundle branch or - | commit the tree must end at |
# patch applied on the work tree or - | clone its submodules (yes / no)
VENDOR_CLONES=(
  "sam31|https://github.com/facebookresearch/sam3.git|5dd401d1c5c1d5c3eedff06d41b77af824517619|stac-determinism|a3c52e5c5228f47d74f88a1c09ecfe9a54943663|-|no"
  "AMB3R-SLAM|https://github.com/HengyiWang/amb3r-slam.git|54184655c259078bd60bc5ddebc838d8f5acda55|stac|4de8946af3fac4584f8375b317bae2796687559b|-|no"
  "GenRecon|https://github.com/kasothaphie/GenRecon.git|eaf1468118d20469d17079a4a19737297d2ef87b|-|eaf1468118d20469d17079a4a19737297d2ef87b|server/tools/genrecon/genrecon_stac.patch|yes"
  "nvdiffrast|https://github.com/NVlabs/nvdiffrast.git|253ac4fcea7de5f396371124af597e6cc957bfae|-|253ac4fcea7de5f396371124af597e6cc957bfae|-|no"
  "vggt-omega|https://github.com/facebookresearch/vggt-omega.git|39a0cb8af88554f15ddcb5354cd52bde588fa014|-|39a0cb8af88554f15ddcb5354cd52bde588fa014|-|no"
  "r3d|https://github.com/facebookresearch/r3d.git|9669cacd799345933c6d0241e134cfe560f4600f|-|9669cacd799345933c6d0241e134cfe560f4600f|-|no"
  "mvs-texturing|https://github.com/nmoehrle/mvs-texturing.git|f3374298ac959cb5afe47a14e4d35d2ac7fbdbb1|-|f3374298ac959cb5afe47a14e4d35d2ac7fbdbb1|-|no"
  "oneTBB-src|https://github.com/uxlfoundation/oneTBB.git|e9af1a1b38b858e8eb41a866e1eed32811e0fb4c|-|e9af1a1b38b858e8eb41a866e1eed32811e0fb4c|-|no"
)

# the submodules of the repo: path | the commit the repo pins | the bundle branch that holds it when
# the remote does not (depth-anything-3's last commit was never pushed to the fork)
VENDOR_SUBMODULES=(
  "vendor/VGGT-Long|-"
  "vendor/depth-anything-3|stac-main"
)

# our commits on top of an upstream pin exist on no remote: the backup set carries them (set/git/, backup.py)
vendor_bundle() { echo "$REBUILD_DIR/set/git/stac-build__vendor__$1.bundle"; }

# git over HTTPS with the token of .env (the private repos: stac-build itself, the Depth-Anything-3
# fork) — the same helper init_pod.sh installs: the token never lands in a remote URL or .gitconfig.
git_credentials() {
    local envfile="$STAC_ROOT/.env"; [ -f "$envfile" ] || envfile="$WORKSPACE/.env"
    git config --global credential.helper \
        "!f() { test \"\$1\" = get || exit 0; . $envfile 2>/dev/null; echo \"username=\${GIT_USER}\"; echo \"password=\${GIT_TOKEN}\"; }; f"
    git config --global --add safe.directory '*'
}

step_submodules() {
    cd "$STAC_ROOT"
    git submodule sync --quiet
    local row path branch want got
    for row in "${VENDOR_SUBMODULES[@]}"; do
        IFS='|' read -r path branch <<<"$row"
        want="$(git ls-tree HEAD "$path" | awk '{print $3}')"
        [ -n "$want" ] || { echo "$path is not a submodule of this commit"; return 1; }
        # (--recursive: depth-anything-3 carries its own submodule, da3_streaming/loop_utils/salad)
        if ! git submodule update --init --recursive "$path"; then
            # the clone is there, the pinned commit is not on the remote: it is in the bundle
            [ "$branch" != - ] && [ -e "$path/.git" ] || { echo "$path: clone failed (GIT_TOKEN in .env? network?)"; return 1; }
            echo "$path: $want is not on the remote — taking it from the set's bundle"
            git -C "$path" fetch "$(vendor_bundle "$(basename "$path")")" "refs/heads/$branch:refs/remotes/bundle/$branch"
            git submodule update --init --recursive "$path"
        fi
        got="$(git -C "$path" rev-parse HEAD)"
        [ "$got" = "$want" ] || { echo "$path is at $got, the repo pins $want"; return 1; }
        echo "$path @ $got"
    done
    # MapAnything's DINOv2 is loaded from a LOCAL checkout of the hub code (the hot patch of uniception:
    # scripts/rebuild/overlays/mapanything): facebookresearch/dinov2 @ 9c7e324, hubconf.py + dinov2/
    local tmp="$CACHE/src/dinov2"
    clone_at "$tmp" https://github.com/facebookresearch/dinov2.git 9c7e3245797cf7be5b5729445a4af1272bd610df
    mkdir -p "$WORKSPACE/dinov2_mapanything"
    cp "$tmp/hubconf.py" "$WORKSPACE/dinov2_mapanything/hubconf.py"
    rm -rf "$WORKSPACE/dinov2_mapanything/dinov2" && cp -r "$tmp/dinov2" "$WORKSPACE/dinov2_mapanything/dinov2"
    echo "$WORKSPACE/dinov2_mapanything = facebookresearch/dinov2 @ 9c7e324"
}

# clone_at <dir> <url> <commit> — a clone whose HEAD is exactly that commit
clone_at() {
    local dest="$1" url="$2" commit="$3"
    if [ -e "$dest/.git" ] && [ "$(git -C "$dest" rev-parse HEAD 2>/dev/null)" = "$commit" ]; then echo "$dest @ $commit"; return 0; fi
    if [ ! -e "$dest/.git" ]; then
        rm -rf "$dest"; mkdir -p "$(dirname "$dest")"
        git clone --quiet --no-checkout "$url" "$dest"
    fi
    git -C "$dest" cat-file -e "$commit^{commit}" 2>/dev/null || git -C "$dest" fetch --quiet origin "$commit"
    git -C "$dest" checkout --quiet --detach "$commit"
    [ "$(git -C "$dest" rev-parse HEAD)" = "$commit" ] || { echo "$dest: could not reach $commit"; return 1; }
    echo "$dest @ $commit"
}

step_vendor_clones() {
    mkdir -p "$VENDOR"
    local row name url base branch final patch recursive dest
    for row in "${VENDOR_CLONES[@]}"; do
        IFS='|' read -r name url base branch final patch recursive <<<"$row"
        dest="$VENDOR/$name"
        if [ -e "$dest/.git" ] && [ "$(git -C "$dest" rev-parse HEAD 2>/dev/null)" = "$final" ]; then
            echo "$name already at $final"
        else
            clone_at "$dest" "$url" "$base"
            if [ "$branch" != - ]; then
                # our own commits on top of the upstream pin: they exist on no remote
                git -C "$dest" fetch --quiet "$(vendor_bundle "$name")" "refs/heads/$branch:refs/heads/$branch"
                git -C "$dest" checkout --quiet "$branch"
            fi
            [ "$(git -C "$dest" rev-parse HEAD)" = "$final" ] || { echo "$name is at $(git -C "$dest" rev-parse HEAD), expected $final"; return 1; }
            echo "$name @ $final"
        fi
        if [ "$recursive" = yes ]; then git -C "$dest" submodule update --init --recursive; fi
        if [ "$patch" != - ]; then
            if git -C "$dest" apply --check "$STAC_ROOT/$patch" 2>/dev/null; then
                git -C "$dest" apply "$STAC_ROOT/$patch"; echo "$name: $patch applied"
            elif git -C "$dest" apply --reverse --check "$STAC_ROOT/$patch" 2>/dev/null; then
                echo "$name: $patch already applied"
            else
                echo "$name: $patch neither applies nor is applied — the tree differs (git -C $dest status)"; return 1
            fi
        fi
    done
}

# prebuilt trees that are downloaded, not built
step_vendor_binaries() {
    # KTX-Software 4.4.2 (the tiler's texture encoder: tools/glb/tile_mesh.mjs)
    local ktx="$VENDOR/KTX-Software-4.4.2-Linux-x86_64"
    if [ ! -x "$ktx/bin/ktx" ]; then
        fetch https://github.com/KhronosGroup/KTX-Software/releases/download/v4.4.2/KTX-Software-4.4.2-Linux-x86_64.tar.bz2 \
            "$CACHE/dl/KTX-Software-4.4.2-Linux-x86_64.tar.bz2"
        tar -xjf "$CACHE/dl/KTX-Software-4.4.2-Linux-x86_64.tar.bz2" -C "$VENDOR"
    fi
    "$ktx/bin/ktx" --version
    # CloudComPy's binary of 2024-09-27 (python 3.10): the tree server/run_cloudcompy*.sh point at
    local cc="$VENDOR/CloudComPy310"
    if [ ! -f "$cc/lib/cloudcompare/_cloudComPy.cpython-310-x86_64-linux-gnu.so" ]; then
        fetch https://www.simulation.openfields.fr/phocadownload/CloudComPy_Conda310_Linux64_20240927.tgz \
            "$CACHE/dl/CloudComPy_Conda310_Linux64_20240927.tgz"
        tar -xzf "$CACHE/dl/CloudComPy_Conda310_Linux64_20240927.tgz" -C "$VENDOR"
    fi
    ls "$cc/lib/cloudcompare/_cloudComPy.cpython-310-x86_64-linux-gnu.so"
}
