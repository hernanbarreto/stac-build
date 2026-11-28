#!/bin/bash

# Arranque del pod STAC-BUILD
# source /workspace/miniforge3/etc/profile.d/conda.sh
# bash /workspace/stac-build/init_pod.sh
# variant launch <script src="https://launchar.app/sdk/v1?key=O0j6XZVvKo3Ns3GiqezRiAOBpVnW66px&redirect=true"></script>

# 0. THE WAY IN — SSH AND JUPYTERLAB (USER 2026-11-28: "El arranque automático (init_pod.sh) tiene que dejar el SSH
#    funcionando en cada pod nuevo, sin que yo haga nada"): first, and at EVERY run — it only sets what is missing,
#    so a run by hand on a started pod mends the access and touches nothing else. Lines "ssh: …", "jupyter: …".
bash /workspace/stac-build/scripts/pod_access.sh

# 0b. ONE START PER POD START. Everything below creates tmux sessions and TYPES their commands into them: run a
#    second time it would type into the live panes (the backend's among them). The pod starts by itself now
#    (scripts/pod_autostart.sh — USER 2026-11-20: "todo debe arrancar automaticamente al iniciar el pod"), so a
#    run by hand afterwards does nothing more. STAC_INIT_FORCE=1 runs it anyway.
if [ "${STAC_INIT_FORCE:-0}" != "1" ] && command -v tmux &> /dev/null && tmux has-session -t backend 2> /dev/null; then
    echo "init_pod: the services are already started (tmux session 'backend' exists) — nothing else done."
    echo "  (the backend alone: bash scripts/restart_backend.sh; this script anyway: STAC_INIT_FORCE=1)"
    exit 0
fi

# 1. paquetes de sistema si faltan: tmux + stack de escritorio para Electron/noVNC
#    (libnss3 etc. son runtime de Chromium; sin ellos Electron no arranca — visto
#    en pod fresco 2026-07-09: "libnss3.so: cannot open shared object file")
#    THE PACKAGES' NAMES ON TODAY'S IMAGE (Ubuntu 24.04): several were renamed with a "t64" suffix and `libasound2`
#    has no installation candidate there — that one name made apt refuse the WHOLE list at every start (found
#    2026-11-28, USER: "el pod debe arrancar siempre sin errores": novnc, websockify and libnss3 never installed,
#    Electron down with "libnss3.so: cannot open shared object file"); each name is installed in its t64 form where
#    that form exists. xvfb and x11vnc (launch_electron.sh's display and its VNC server) were missing from the list.
#    Only what is missing is installed; the result is one line, "packages: …".
#    The desktop's (Electron's) part only when it is asked for: START_ELECTRON=1 (section 3a).
START_PKGS="tmux"
[ "${START_ELECTRON:-0}" = "1" ] && START_PKGS="$START_PKGS xvfb x11vnc novnc websockify libnss3 libatk1.0-0 libatk-bridge2.0-0
            libcups2 libdrm2 libgtk-3-0 libgbm1 libasound2 libxdamage1 libxcomposite1 libxrandr2 libxkbcommon0"
pkg_installed() { dpkg-query -W -f='${Status}' "$1" 2> /dev/null | grep -q 'install ok installed'; }
pkgs_missing() {
    local p out=""
    for p in $START_PKGS; do pkg_installed "$p" || pkg_installed "${p}t64" || out="$out $p"; done
    echo "$out"
}
missing="$(pkgs_missing)"
if [ -n "$missing" ]; then
    APT_LOG=/tmp/stac_start_apt.log                  # apt's own output (the container's disk)
    apt-get update -qq > "$APT_LOG" 2>&1
    want=""
    for p in $missing; do
        if apt-cache show "${p}t64" > /dev/null 2>&1; then want="$want ${p}t64"; else want="$want $p"; fi
    done
    # shellcheck disable=SC2086
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq $want >> "$APT_LOG" 2>&1
    missing="$(pkgs_missing)"
    if [ -n "$missing" ]; then
        echo "packages: ERROR — not installed:$missing (apt's output: $APT_LOG, its end below)"
        tail -n 5 "$APT_LOG" | sed 's/^/packages:   /'
    else
        echo "packages: installed —$want"
    fi
else
    echo "packages: ok — all present"
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

# 2b. AR SERVER (phone XR surface — own process, decoupled from the backend;
#     published by tailscale serve --https=8443 with a valid cert)
tmux new-session -d -s arserver
tmux send-keys -t arserver "$CONDA && $ENV && cd /workspace/stac-build && bash scripts/serve_ar.sh" Enter

# 3. VITE
tmux new-session -d -s vite
tmux send-keys -t vite "$CONDA && conda activate nodejs && cd /workspace/stac-build/ui && npm run dev:web" Enter

# 3r. THE PAGE FROM ANY BROWSER, WITHOUT AN SSH TUNNEL (USER 2026-11-20: "la idea es que pueda verlo desde
#     cualquier navegador, sin necesidad de abrir la consola"): the gate + tailscaled + the funnel — a fixed
#     https address behind a secret link (scripts/remote_access.sh; the address and the link: `... link`).
#     Flag: START_REMOTE=0 to skip. Never fatal: the pod's services start whatever Tailscale answers.
START_REMOTE="${START_REMOTE:-1}"
if [ "$START_REMOTE" = "1" ]; then
    bash /workspace/stac-build/scripts/remote_access.sh start \
        || echo "⚠ remote access not up — see the lines above; again with: bash scripts/remote_access.sh start"
fi

# 3a. ELECTRON DESKTOP (headless pod → viewable over noVNC on :6080/vnc.html).
#     Brings up Xvfb + x11vnc + noVNC, waits for Vite, then runs the app.
#     OFF UNLESS ASKED (USER 2026-11-28: "si no hace falta para nada electron habria que matarlo"): the template exposes
#     no 6080 and the page is used from a browser (the tunnel, the Tailscale address); started, it drew the viewer in
#     software on ~4 cores and slowed GenRecon's CPU stage. START_ELECTRON=1 starts it (and installs its packages).
START_ELECTRON="${START_ELECTRON:-0}"
if [ "$START_ELECTRON" = "1" ]; then
    tmux new-session -d -s electron
    tmux send-keys -t electron "$CONDA && conda activate nodejs && cd /workspace/stac-build && bash scripts/launch_electron.sh" Enter
    echo "🖥  Electron desktop starting in tmux 'electron' → open http://<pod-host>:6080/vnc.html"
else
    echo "⏭  Electron desktop skipped (START_ELECTRON=0)."
fi

# 3b. SEMANTIC SERVICE (vLLM / Qwen3-VL, the chat's engine) AND THE VOICE ENGINES — STARTED BY THE BACKEND at its
#     own start (USER 2026-11-14: "tanto el motor de voz y habla del sistema como el chat vlm deberan iniciar cuando
#     levante el backend porque son el core del funcionamiento de stac build"), with the chat's own engine profile
#     (semantic.chat.engine), which this script cannot give: a vLLM started here would be adopted by the backend
#     with the stages' profile, about three times slower for the chat. So nothing is launched here by default.
#     semantic.chat_on_demand: true keeps the engine down until the chat is used (USER 2026-11-12).
#     START_SEMANTIC=1 forces a stand-alone vLLM in tmux 'semantic' (a machine with no backend).
if [ -z "${START_SEMANTIC:-}" ]; then
    START_SEMANTIC=0
    /workspace/miniforge3/envs/da3/bin/python - <<'PY' || echo "❌ init_pod: semantic.chat_on_demand could not be read from config.yaml"
import sys
import yaml
sem = (yaml.safe_load(open("/workspace/stac-build/server/config.yaml")) or {}).get("semantic") or {}
on_demand = sem.get("chat_on_demand")
if not isinstance(on_demand, bool):
    sys.exit("config.yaml: semantic.chat_on_demand is missing or not true / false")
print("🧠 Chat on demand (semantic.chat_on_demand: true): the engine loads when the chat is used." if on_demand
      else "🧠 The backend starts the chat's engine (and the voice engines) at its own start.")
PY
fi
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