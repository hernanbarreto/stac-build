#!/bin/bash
# STAC-Builder — build the `voice` conda env: speech → text and text → speech, local, on the card.
#
#   bash scripts/build_voice_env.sh
#
# USER 2026-11-14: "incorpora herramientas para hablarle y que traduzca a texto y que responda con
# voz sintetica en los 3 idiomas del sistema, ingles, español y frances".
#
# Its own env on purpose: the backend's (da3) and the VLM's (semantic) are validated and their
# package versions enter the engines' identities — nothing is installed into them. Idempotent.
#
#   speech → text   openai/whisper-large-v3-turbo (MIT) through transformers, on torch
#   text → speech   hexgrad/Kokoro-82M (Apache-2.0): en / es / fr; its Spanish and French
#                   phonemes come from espeak-ng, shipped INSIDE the env (espeakng-loader) — an
#                   apt package would be lost at the pod's restart
#
# The weights go to the pod's one HF cache (/workspace/hf_cache) at the service's first start.
set -e
source "${CONDA_ROOT:-/workspace/miniforge3}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "voice"; then
    echo "[voice] env already exists — skipping"
    exit 0
fi

echo "[$(date +%T)] creating env voice (python 3.11)"
conda create -y -n voice python=3.11
conda activate voice

echo "[$(date +%T)] torch 2.10.0 + torchaudio (cu128, the pod's CUDA build)"
pip install --no-input --no-cache-dir "torch==2.10.0" "torchaudio==2.10.0" \
    --index-url https://download.pytorch.org/whl/cu128

echo "[$(date +%T)] speech → text (Whisper through transformers)"
pip install --no-input --no-cache-dir "transformers>=4.57,<5" accelerate

echo "[$(date +%T)] text → speech (Kokoro) + audio I/O"
pip install --no-input --no-cache-dir "kokoro>=0.9.4" "misaki[en]" soundfile espeakng-loader phonemizer-fork

echo "[$(date +%T)] the service's own deps"
pip install --no-input --no-cache-dir fastapi "uvicorn[standard]" python-multipart pyyaml numpy av

echo "[$(date +%T)] DONE. versions:"
python -c "import torch, transformers, kokoro; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__, 'kokoro', getattr(kokoro, '__version__', '?'))"
