#!/bin/bash
# The rig that lets the page be LOOKED AT on the pod at every screen size without touching the live backend or the
# live dev server (USER 2026-11-20, to "¿me autorizás a abrir la página en un navegador dentro del pod?": "si por
# supuesto!"). What ui/scripts/responsive-shots.mjs needs running:
#
#   a PRIVATE WORKING COPY of the interface on the pod's local disk ($STAC_UI_WORK, /root/dev/ui-work): src, public,
#     scripts; node_modules a link to the real one; its vite.config.ts with its OWN dependency cache — the live dev
#     server's cache is never written. Changes are made and looked at THERE and reach the live tree in one batch.
#   the mock backend (scripts/mock-server.mjs, :8790) — NEVER the real one: opening a session there changes the
#     server's one session for everyone who has the page open.
#   a dev server of the copy pointed at the mock (:5199).
#   the test browser: Playwright's Chrome Headless Shell (the local disk: gone at a pod restart) and the two system
#     libraries it needs. It renders in SOFTWARE (responsive-shots.mjs checks nvidia-smi around its start).
#
#   bash ui/scripts/responsive-rig.sh start     make what is missing (never overwrites an existing copy), start both
#   bash ui/scripts/responsive-rig.sh status
#   bash ui/scripts/responsive-rig.sh diff      what the copy changes against the live tree
#   bash ui/scripts/responsive-rig.sh stop      the mock and the copy's dev server (the copy stays)
#   then:  cd $STAC_UI_WORK && taskset -c 0-3 nice -n 19 node scripts/responsive-shots.mjs <out> --sizes … --states …
set -e
UI=/workspace/stac-build/ui
W=${STAC_UI_WORK:-/root/dev/ui-work}
MOCK_PORT=8790
PAGE_PORT=5199
export PATH=/workspace/miniforge3/envs/nodejs/bin:$PATH

listening() { ss -ltn 2>/dev/null | grep -q ":$1 "; }
pid_on() { ss -ltnp 2>/dev/null | grep ":$1 " | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2; }

case "${1:-start}" in
    start)
        if [ ! -d "$W/src" ]; then
            echo "[rig] making the private copy in $W"
            mkdir -p "$W"
            rsync -a "$UI"/src "$UI"/public "$UI"/scripts "$UI"/index.html "$UI"/xr.html "$UI"/vite.config.ts \
                "$UI"/tsconfig.json "$UI"/tsconfig.node.json "$UI"/package.json "$UI"/.eslintrc.cjs "$W"/
            ln -sfn "$UI/node_modules" "$W/node_modules"
            # its own dependency cache, and leave to serve the packages of the real node_modules
            python3 - "$W/vite.config.ts" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
a = "  return {\n    plugins,\n"
b = "    server: {\n"
assert s.count(a) == 1 and s.count(b) == 1, "vite.config.ts changed: patch the copy's by hand (cacheDir, server.fs.allow)"
s = s.replace(a, a + "    cacheDir: path.join(__dirname, '.vite-cache'),\n")
s = s.replace(b, b + "      fs: { allow: [__dirname, '/workspace/stac-build/ui/node_modules'] },\n")
open(p, "w").write(s)
PY
        else
            echo "[rig] the private copy exists ($W) — left as it is"
        fi
        if ! ls /root/.cache/ms-playwright/chromium_headless_shell-* >/dev/null 2>&1; then
            echo "[rig] the test browser is not on this disk — downloading it (about 270 MB)"
            (cd "$UI" && npx --no-install playwright install --only-shell chromium)
        fi
        shell=$(ls -d /root/.cache/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell 2>/dev/null | tail -1)
        if [ -n "$shell" ] && ldd "$shell" 2>/dev/null | grep -q 'not found'; then
            echo "[rig] the test browser lacks system libraries — installing libnss3, libnspr4"
            DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libnss3 libnspr4 >/dev/null \
                || { apt-get update >/dev/null && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends libnss3 libnspr4 >/dev/null; }
        fi
        mkdir -p "$W/.run"
        if ! listening $MOCK_PORT; then
            (cd "$W" && setsid nohup nice -n 15 node scripts/mock-server.mjs $MOCK_PORT >"$W/.run/mock.log" 2>&1 </dev/null &)
        fi
        if ! listening $PAGE_PORT; then
            (cd "$W" && WEB_ONLY=true STAC_API_TARGET=http://localhost:$MOCK_PORT setsid nohup nice -n 15 \
                node node_modules/vite/bin/vite.js --port $PAGE_PORT --strictPort >"$W/.run/vite.log" 2>&1 </dev/null &)
        fi
        for _ in $(seq 1 20); do listening $MOCK_PORT && listening $PAGE_PORT && break; sleep 1; done
        "$0" status
        ;;
    status)
        [ -d "$W/src" ] && echo "copy: $W" || echo "copy: none"
        listening $MOCK_PORT && echo "mock backend: up (:$MOCK_PORT)" || echo "mock backend: DOWN"
        listening $PAGE_PORT && echo "the copy's dev server: up (http://localhost:$PAGE_PORT)" || echo "the copy's dev server: DOWN"
        ls -d /root/.cache/ms-playwright/chromium_headless_shell-* >/dev/null 2>&1 && echo "test browser: on this disk" || echo "test browser: not downloaded"
        ;;
    diff)
        [ -d "$W/src" ] || { echo "no copy"; exit 1; }
        diff -rq "$W/src" "$UI/src" | grep -v '__probe__' | sed -e "s#$W/##" -e "s#$UI/##" || true
        ;;
    stop)
        for port in $MOCK_PORT $PAGE_PORT; do
            p=$(pid_on $port); [ -n "$p" ] && kill "$p" && echo "[rig] stopped what listened on :$port (pid $p)"
        done
        ;;
    *)
        echo "usage: bash ui/scripts/responsive-rig.sh start | status | diff | stop" >&2
        exit 64
        ;;
esac
