#!/usr/bin/env bash
# STAC Build — REBUILD, what is compiled outside the conda envs and the small tools on the volume.
# Sourced by rebuild.sh.

# the build tools the base image may lack (today's — runpod pytorch, Ubuntu 24.04, CUDA 12.8.1 devel —
# has all of them; the RUNTIME packages of the desktop are init_pod.sh's, at every boot)
step_system() {
    local need=() tool
    for tool in git curl wget unzip bzip2 tar gcc g++ make cmake ffmpeg tmux openssl; do
        command -v "$tool" >/dev/null || need+=("$tool")
    done
    if [ ${#need[@]} -gt 0 ]; then
        echo "installing the missing tools: ${need[*]}"
        local pkgs=()
        for tool in "${need[@]}"; do
            case "$tool" in gcc|g++|make) pkgs+=(build-essential) ;; *) pkgs+=("$tool") ;; esac
        done
        apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends "${pkgs[@]}"
    fi
    gcc --version | head -1; cmake --version | head -1; git --version
    if command -v nvidia-smi >/dev/null; then nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    else echo "WARNING: no nvidia-smi — the CUDA extensions will not build and nothing will run"; fi
    if [ -x /usr/local/cuda/bin/nvcc ]; then /usr/local/cuda/bin/nvcc --version | tail -1
    else echo "WARNING: /usr/local/cuda/bin/nvcc is missing — sim3solve and nvdiffrast (envs da3, mapanything) need a CUDA *devel* image"; fi
}

# oneTBB from its pinned source, where PotreeConverter's build looks for it (vendor/oneTBB)
step_onetbb() {
    local src="$VENDOR/oneTBB-src" dst="$VENDOR/oneTBB"
    if [ -f "$dst/lib/cmake/TBB/TBBConfig.cmake" ]; then echo "oneTBB present"; return 0; fi
    cmake -S "$src" -B "$CACHE/build/oneTBB" -DCMAKE_BUILD_TYPE=Release -DTBB_TEST=OFF -DTBB_STRICT=OFF \
          -DCMAKE_INSTALL_PREFIX="$dst" -DCMAKE_INSTALL_LIBDIR=lib
    cmake --build "$CACHE/build/oneTBB" -j "${MAX_JOBS:-16}"
    cmake --install "$CACHE/build/oneTBB"
    ls "$dst/lib/cmake/TBB/TBBConfig.cmake"
}

# the octree converter: the vendored source WITH the STAC patches (it is tracked in the repo), built
# as on the pod — Release, the system compiler, oneTBB from vendor/
step_potree() {
    local src="$VENDOR/PotreeConverter" bin="$VENDOR/PotreeConverter/build/PotreeConverter"
    if [ -x "$bin" ] && "$bin" --help >/dev/null 2>&1; then echo "PotreeConverter present"; return 0; fi
    cmake -S "$src" -B "$src/build" -DCMAKE_BUILD_TYPE=Release -DTBB_DIR="$VENDOR/oneTBB/lib/cmake/TBB" \
          -DCMAKE_EXE_LINKER_FLAGS="-L$VENDOR/oneTBB/lib -Wl,-rpath,$VENDOR/oneTBB/lib"
    cmake --build "$src/build" -j "${MAX_JOBS:-16}"
    [ -x "$bin" ] || { echo "the build left no $bin"; return 1; }
    ls -la "$bin" "$src/build/liblaszip.so"
}

# remote access: the static binaries (the node's identity is NOT restored — `bash scripts/tailscale_up.sh`
# prints the link to authorise the new node)
step_remote_tools() {
    local ts="$WORKSPACE/tailscale"
    if [ ! -x "$ts/tailscaled" ]; then
        fetch https://pkgs.tailscale.com/stable/tailscale_1.86.2_amd64.tgz "$CACHE/dl/tailscale_1.86.2_amd64.tgz"
        mkdir -p "$ts" && tar -xzf "$CACHE/dl/tailscale_1.86.2_amd64.tgz" -C "$CACHE/dl"
        cp "$CACHE/dl/tailscale_1.86.2_amd64/tailscale" "$CACHE/dl/tailscale_1.86.2_amd64/tailscaled" "$ts/"
    fi
    if [ ! -x "$WORKSPACE/cloudflared" ]; then
        fetch https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 "$WORKSPACE/cloudflared"
        chmod +x "$WORKSPACE/cloudflared"
    fi
    "$ts/tailscale" version | head -1
}

# GenRecon's DINOv3 in the transformers layout, converted from Meta's ViT-L/16 checkpoint (the same
# conversion as on the pod; its result is checked against the manifest's sha256 by `models.py verify`)
step_convert_dinov3() {
    local out="$VENDOR/GenRecon/weights/dinov3-vitl16-hf/model.safetensors"
    local src="$VENDOR/pointdit/pretrained/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    if [ -f "$out" ]; then echo "present: $out"; return 0; fi
    [ -f "$src" ] || { echo "Meta's ViT-L/16 checkpoint is missing ($src): DINOV3_URL in .env has expired — ask Meta for a new link, then run rebuild.sh again"; return 1; }
    ( cd "$STAC_ROOT/server/tools/genrecon" && "$(env_python genrecon)" convert_dinov3.py )
    ls -la "$out"
}
