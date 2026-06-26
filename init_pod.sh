#!/bin/bash

# Arranque del pod STAC-BUILD
# source /workspace/miniforge3/etc/profile.d/conda.sh
# bash /workspace/stac-build/init_pod.sh

# 1. tmux si falta
if ! command -v tmux &> /dev/null; then
    apt update && apt install -y tmux
fi

CONDA="source /workspace/miniforge3/etc/profile.d/conda.sh"

# 1b. Deps del compresor de GLB (tools/glb) — el backend lo invoca tras generar
#     el scene.glb del TSDF. Idempotente: solo instala si falta node_modules.
if [ ! -d /workspace/stac-build/tools/glb/node_modules ]; then
    bash -c "$CONDA && conda activate nodejs && cd /workspace/stac-build/tools/glb && npm install"
fi

# 2. BACKEND
tmux new-session -d -s backend
tmux send-keys -t backend "$CONDA && cd /workspace/stac-build && export CONDA_ROOT=/workspace/miniforge3 && bash scripts/start.sh" Enter

# 3. VITE
tmux new-session -d -s vite
tmux send-keys -t vite "$CONDA && conda activate nodejs && cd /workspace/stac-build/ui && npm run dev:web" Enter

# 4. CLAUDE (listo para entrar, sin lanzar claude solo)
tmux new-session -d -s claude
tmux send-keys -t claude "$CONDA && conda activate nodejs && cd /workspace/stac-build" Enter

echo "Listo. Sesiones: backend, vite, claude"
echo "  tmux attach -t backend   (ver server)"
echo "  tmux attach -t vite      (ver UI)"
echo "  tmux attach -t claude    (escribi 'claude' para iniciar)"