#!/bin/bash

# Arranque del pod STAC-BUILD
# source /workspace/miniforge3/etc/profile.d/conda.sh
# bash /workspace/stac-build/init_pod.sh

# 1. paquetes de sistema si faltan: tmux + stack de escritorio para Electron/noVNC
#    (libnss3 etc. son runtime de Chromium; sin ellos Electron no arranca — visto
#    en pod fresco 2026-07-09: "libnss3.so: cannot open shared object file")
if ! command -v tmux &> /dev/null || [ ! -d /usr/share/novnc ]; then
    apt update && apt install -y tmux novnc websockify \
        libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libgtk-3-0 \
        libgbm1 libasound2 libxdamage1 libxcomposite1 libxrandr2 libxkbcommon0
fi

CONDA="source /workspace/miniforge3/etc/profile.d/conda.sh"

# 1a. SECRETS — todos los tokens viven en .env (git-ignored), nunca acá.
#     Cada sesión tmux lo sourcea vía $ENV para heredar HF_TOKEN, GIT_*, etc.
ENV_FILE=/workspace/stac-build/.env
if [ -f "$ENV_FILE" ]; then
    ENV="source $ENV_FILE"
else
    ENV="true"
    echo "⚠️  $ENV_FILE no existe — sesiones SIN tokens (HF_TOKEN, GIT_TOKEN...)."
fi

# Git auth por HTTPS: helper que lee GIT_USER/GIT_TOKEN desde .env al momento
# de cada fetch/push (el token nunca queda en .gitconfig ni en el remote URL).
git config --global credential.helper \
    '!f() { test "$1" = get || exit 0; . /workspace/stac-build/.env 2>/dev/null; echo "username=${GIT_USER}"; echo "password=${GIT_TOKEN}"; }; f'

# 1b. Deps del compresor de GLB (tools/glb) — el backend lo invoca tras generar
#     el scene.glb del TSDF. Idempotente: solo instala si falta node_modules.
if [ ! -d /workspace/stac-build/tools/glb/node_modules ]; then
    bash -c "$CONDA && conda activate nodejs && cd /workspace/stac-build/tools/glb && npm install"
fi

# 2. BACKEND
tmux new-session -d -s backend
tmux send-keys -t backend "$CONDA && $ENV && cd /workspace/stac-build && export CONDA_ROOT=/workspace/miniforge3 && bash scripts/start.sh" Enter

# 3. VITE
tmux new-session -d -s vite
tmux send-keys -t vite "$CONDA && conda activate nodejs && cd /workspace/stac-build/ui && npm run dev:web" Enter

# 3a. ELECTRON DESKTOP (headless pod → viewable over noVNC on :6080/vnc.html).
#     Brings up Xvfb + x11vnc + noVNC, waits for Vite, then runs the app.
#     Flag: set START_ELECTRON=0 to skip (e.g. no display stack needed).
START_ELECTRON="${START_ELECTRON:-1}"
if [ "$START_ELECTRON" = "1" ]; then
    tmux new-session -d -s electron
    tmux send-keys -t electron "$CONDA && conda activate nodejs && cd /workspace/stac-build && bash scripts/launch_electron.sh" Enter
    echo "🖥  Electron desktop starting in tmux 'electron' → open http://<pod-host>:6080/vnc.html"
else
    echo "⏭  Electron desktop skipped (START_ELECTRON=0)."
fi

# 3b. SEMANTIC SERVICE (Phase 0 — persistent vLLM/Qwen3-VL endpoint shared by
#     every pipeline consumer). Flag: set START_SEMANTIC=0 to skip (e.g. when the
#     GPU is fully needed by a heavy reconstruction run — see convivencia notes).
START_SEMANTIC="${START_SEMANTIC:-1}"
if [ "$START_SEMANTIC" = "1" ]; then
    tmux new-session -d -s semantic
    tmux send-keys -t semantic "$CONDA && $ENV && cd /workspace/stac-build && bash scripts/serve_semantic.sh" Enter
    echo "🧠 Semantic service (Qwen3-VL/vLLM) starting in tmux 'semantic'."
    # Readiness gate: run the healthcheck in its own tmux window so the pod
    # boot is not blocked, but the result lands in the session (window
    # 'health' shows OK + VRAM, or NOT READY after service.startup_timeout_s).
    tmux new-window -t semantic -n health
    tmux send-keys -t semantic:health \
        "$CONDA && conda activate semantic && cd /workspace/stac-build/server && python -m semantic.healthcheck && echo '✅ semantic READY'" Enter
    echo "   Gate: tmux attach -t semantic (window 'health' → ✅ semantic READY)"
    echo "   Smoke test:  conda activate semantic && cd server && python -m semantic.smoke_test"
else
    echo "⏭  Semantic service skipped (START_SEMANTIC=0)."
fi

# 4. CLAUDE (listo para entrar, sin lanzar claude solo)
tmux new-session -d -s claude
tmux send-keys -t claude "$ENV" Enter
# claude siempre arranca con el modelo Fable en este pod
tmux send-keys -t claude "export ANTHROPIC_MODEL=claude-fable-5" Enter
tmux send-keys -t claude "$CONDA && conda activate nodejs && cd /workspace/stac-build" Enter

echo "Listo. Sesiones: backend, vite, electron, semantic, claude"
echo "  tmux attach -t backend   (ver server)"
echo "  tmux attach -t vite      (ver UI dev server)"
echo "  tmux attach -t electron  (ver la app Electron / noVNC en :6080/vnc.html)"
echo "  tmux attach -t claude    (escribi 'claude' para iniciar)"