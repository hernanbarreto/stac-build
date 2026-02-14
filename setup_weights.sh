#!/bin/bash
# ===========================================================================
# STAC-Builder — Model Weights Downloader
# Downloads all required model weights into the weights/ directory.
# These files are gitignored and must be fetched before first run.
#
# Usage:
#   bash setup_weights.sh          # Download all weights
#   bash setup_weights.sh da3      # Download DA3 weights only
#   bash setup_weights.sh sam3     # Download SAM3 weights only
#   bash setup_weights.sh vlm      # Download InternVL3 weights only
# ===========================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS_DIR="${SCRIPT_DIR}/weights"

# ── Colors ───────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC} $1"; }
ok()    { echo -e "${GREEN}[OK]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }

# ── Check dependencies ──────────────────────────────────────────────
command -v python3 >/dev/null 2>&1 || { echo "python3 required"; exit 1; }

# ── DA3 weights (DINO-SALAD backbone) ────────────────────────────────
download_da3() {
    local DA3_WEIGHTS="${WEIGHTS_DIR}/da3"
    mkdir -p "${DA3_WEIGHTS}"

    # The DINO-SALAD checkpoint used by DA3-Streaming
    local SALAD_CKPT="${DA3_WEIGHTS}/dino_salad.ckpt"
    if [ -f "${SALAD_CKPT}" ]; then
        ok "DA3 DINO-SALAD weights already present"
    else
        info "Downloading DINO-SALAD checkpoint..."
        python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='Araachie/dino-salad',
    filename='dino_salad.ckpt',
    local_dir='${DA3_WEIGHTS}',
)
print('Done')
"
        ok "DA3 DINO-SALAD weights downloaded"
    fi

    # DA3 model weights are auto-downloaded by HuggingFace Hub on first run
    info "DA3 model (depth-anything/DA3-LARGE) will auto-download on first run via HF Hub"
}

# ── SAM3 weights ─────────────────────────────────────────────────────
download_sam3() {
    local SAM3_WEIGHTS="${WEIGHTS_DIR}/sam3"
    mkdir -p "${SAM3_WEIGHTS}"

    # SAM3 checkpoints
    local SAM3_CKPT="${SAM3_WEIGHTS}/sam3_hiera_large.pt"
    if [ -f "${SAM3_CKPT}" ]; then
        ok "SAM3 weights already present"
    else
        info "Downloading SAM3 checkpoint..."
        python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='facebook/sam3-hiera-large',
    filename='sam3_hiera_large.pt',
    local_dir='${SAM3_WEIGHTS}',
)
print('Done')
"
        ok "SAM3 weights downloaded"
    fi
}

# ── InternVL3 (scene analyzer) ───────────────────────────────────────
download_vlm() {
    local VLM_WEIGHTS="${WEIGHTS_DIR}/vlm"
    mkdir -p "${VLM_WEIGHTS}"

    info "InternVL3 model will auto-download on first run via HF Hub"
    info "Model: OpenGVLab/InternVL3-1B (or configured model)"
    info "Cache location: ~/.cache/huggingface/hub/"
    ok "No manual download needed for VLM"
}

# ── Main ─────────────────────────────────────────────────────────────
echo "============================================="
echo "  STAC-Builder — Model Weights Setup"
echo "============================================="
echo ""

TARGET="${1:-all}"

case "${TARGET}" in
    da3)
        download_da3
        ;;
    sam3)
        download_sam3
        ;;
    vlm)
        download_vlm
        ;;
    all)
        download_da3
        echo ""
        download_sam3
        echo ""
        download_vlm
        ;;
    *)
        echo "Usage: $0 [da3|sam3|vlm|all]"
        exit 1
        ;;
esac

echo ""
ok "Weight setup complete!"
echo ""
echo "Weights directory: ${WEIGHTS_DIR}"
du -sh "${WEIGHTS_DIR}"/* 2>/dev/null || true
