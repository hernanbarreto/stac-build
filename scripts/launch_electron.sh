#!/usr/bin/env bash
# ===========================================================================
# STAC-Build — Electron desktop launcher (headless pod, viewable over noVNC)
# ---------------------------------------------------------------------------
# Brings up a virtual display + VNC bridge, waits for the Vite dev server,
# then launches the Electron app on that display. Designed to be run inside a
# tmux session (see init_pod.sh): the display-stack daemons background
# themselves and Electron runs in the foreground so the tmux window IS the app.
#
# View it: open http://<pod-host>:${NOVNC_PORT:-6080}/vnc.html in a browser.
#
# Prereqs (started elsewhere): backend on 8765, Vite on ${VITE_PORT}. Vite is
# owned by the `vite` tmux session — this script only WAITS for it, never
# starts a second copy.
#
# Env overrides (all optional):
#   DISPLAY_NUM=:99  SCREEN=1680x1050x24  VNC_PORT=5900  NOVNC_PORT=6080
#   VITE_PORT=5173   LOG_DIR=/tmp/stac-desktop
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC
# ===========================================================================
set -uo pipefail

DISPLAY_NUM="${DISPLAY_NUM:-:99}"
SCREEN="${SCREEN:-1680x1050x24}"
VNC_PORT="${VNC_PORT:-5900}"
NOVNC_PORT="${NOVNC_PORT:-6080}"
VITE_PORT="${VITE_PORT:-5173}"
# "localhost", NOT 127.0.0.1: the Vite dev server may bind IPv6-only ([::1]),
# where an IPv4-literal probe/URL never connects (seen on this pod: "Vite not
# reachable after 90s" while Vite was up on [::1]:5173).
VITE_URL="http://localhost:${VITE_PORT}"
LOG_DIR="${LOG_DIR:-/tmp/stac-desktop}"
UI="$(cd "$(dirname "${BASH_SOURCE[0]}")/../ui" && pwd)"
NOVNC_WEB="${NOVNC_WEB:-/usr/share/novnc}"

mkdir -p "${LOG_DIR}"
export DISPLAY="${DISPLAY_NUM}"

log() { echo "[electron] $*"; }

start_bg() {  # start_bg <name> <pgrep-pattern> <command...>
  local name="$1" pat="$2"; shift 2
  if pgrep -f "${pat}" >/dev/null 2>&1; then
    log "${name} already running — reusing"
  else
    log "starting ${name}"
    "$@" >"${LOG_DIR}/${name}.log" 2>&1 &
    sleep 1
  fi
}

# 1. virtual framebuffer
start_bg xvfb "Xvfb ${DISPLAY_NUM}" \
  Xvfb "${DISPLAY_NUM}" -screen 0 "${SCREEN}" -ac +extension GLX +render -noreset
sleep 1

# 2. lightweight WM (optional — helps window sizing; skip if absent)
if command -v fluxbox >/dev/null 2>&1; then
  start_bg fluxbox "fluxbox" fluxbox
fi

# 3. VNC server exposing the virtual display
start_bg x11vnc "x11vnc -display ${DISPLAY_NUM}" \
  x11vnc -display "${DISPLAY_NUM}" -forever -shared -nopw -rfbport "${VNC_PORT}" -noxdamage

# 4. noVNC (browser client) bridging NOVNC_PORT -> VNC_PORT
if [ -d "${NOVNC_WEB}" ]; then
  start_bg novnc "websockify.*${NOVNC_PORT}" \
    websockify --web "${NOVNC_WEB}" "${NOVNC_PORT}" "localhost:${VNC_PORT}"
else
  log "WARN: noVNC web root ${NOVNC_WEB} not found — VNC still on :${VNC_PORT}"
fi

# 5. wait for the Vite dev server (started by the `vite` tmux session)
log "waiting for Vite on ${VITE_URL} …"
up=0
for _ in $(seq 1 90); do
  if curl -s -m 2 "${VITE_URL}/" >/dev/null 2>&1; then up=1; break; fi
  sleep 1
done
[ "${up}" = 1 ] && log "Vite is up" || log "WARN: Vite not reachable after 90s — launching anyway"

# 6. Electron on the virtual display (foreground → keeps the tmux window alive)
cd "${UI}"
pkill -f "electron/dist/electron" 2>/dev/null && sleep 1 || true
log "launching Electron → ${VITE_URL} (noVNC: http://<host>:${NOVNC_PORT}/vnc.html)"
export VITE_DEV_SERVER_URL="${VITE_URL}"
exec node_modules/.bin/electron . \
  --no-sandbox --disable-gpu-sandbox \
  --ignore-certificate-errors \
  --enable-unsafe-swiftshader --use-gl=angle --use-angle=swiftshader \
  --disable-dev-shm-usage
