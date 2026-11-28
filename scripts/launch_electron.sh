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

# 4. noVNC (browser client) bridging NOVNC_PORT -> VNC_PORT. Running = its port listening: a process-name
#    pattern also matched any shell whose command line held those words (2026-11-28: "already running" with
#    nothing on 6080).
if [ -d "${NOVNC_WEB}" ]; then
  if [ -n "$(ss -ltnH "sport = :${NOVNC_PORT}" 2>/dev/null)" ]; then
    log "novnc already listening on ${NOVNC_PORT} — reusing"
  else
    log "starting novnc"
    websockify --web "${NOVNC_WEB}" "${NOVNC_PORT}" "localhost:${VNC_PORT}" >"${LOG_DIR}/novnc.log" 2>&1 &
    sleep 1
  fi
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

# 6. D-Bus: the container runs none, and Chromium logs "Failed to connect to the bus" for the system bus and for
#    the session bus (2026-11-28, USER: "el pod debe arrancar siempre sin errores"). The system bus is started
#    once (a pid file left by a dead one removed first); the session bus is Electron's own (dbus-run-session).
if [ ! -S /run/dbus/system_bus_socket ]; then
  mkdir -p /run/dbus
  if [ -f /run/dbus/pid ] && ! kill -0 "$(cat /run/dbus/pid 2>/dev/null)" 2>/dev/null; then rm -f /run/dbus/pid; fi
  dbus-daemon --system --fork >"${LOG_DIR}/dbus.log" 2>&1 || log "WARN: the system D-Bus did not start (${LOG_DIR}/dbus.log)"
fi

# 7. Electron on the virtual display (foreground → keeps the tmux window alive)
cd "${UI}"
pkill -f "electron/dist/electron" 2>/dev/null && sleep 1 || true
log "launching Electron → ${VITE_URL} (noVNC: http://<host>:${NOVNC_PORT}/vnc.html)"
export VITE_DEV_SERVER_URL="${VITE_URL}"
exec dbus-run-session -- node_modules/.bin/electron . \
  --no-sandbox --disable-gpu-sandbox \
  --ignore-certificate-errors \
  --enable-unsafe-swiftshader --use-gl=angle --use-angle=swiftshader \
  --disable-dev-shm-usage
