#!/bin/bash
# STAC Build — the page from any browser, without an SSH tunnel (USER 2026-11-20: "la idea es que pueda verlo desde
# cualquier navegador, sin necesidad de abrir la consola" / "en la pc no puedo instalar tailscale porque es laboral" /
# "si, dale").
#
#   the pod ── tailscaled (userspace, its identity on the volume: a FIXED name whatever address the provider
#              gives the pod) ── Tailscale Funnel: https://<name>.<tailnet>.ts.net, a valid certificate, public
#           └─ THE GATE (scripts/remote/gate.nginx.conf.template, its own nginx on 127.0.0.1): nothing passes but a
#              browser that once opened the secret link — then the interface, with its own login as always.
#
#   bash scripts/remote_access.sh start     gate + tailscaled + funnel (idempotent; says what the user must do once)
#   bash scripts/remote_access.sh link      the secret link (open it once in each browser) — PRIVATE
#   bash scripts/remote_access.sh status    what is up
#   bash scripts/remote_access.sh rotate    a new key: every browser must open the new link
#   bash scripts/remote_access.sh gate      the gate alone, on 127.0.0.1 (nothing published)
#   bash scripts/remote_access.sh stop      the address stops answering (the gate and the funnel go; tailscaled stays)
#
# The key is STAC_GATE_KEY in .env (private, never committed); the rendered nginx file holds it and lives on the
# pod's LOCAL disk (/tmp/stac_gate, mode 600 — the volume shows every file world-readable), made again at every
# start. Tailnet-only access (a phone with the Tailscale app): scripts/tailscale_up.sh.
set -e
ROOT=/workspace/stac-build
TS_DIR=/workspace/tailscale
TS="$TS_DIR/tailscale --socket=$TS_DIR/tailscaled.sock"
RUN=$ROOT/logs/remote_gate
TMPD=/tmp/stac_gate
ENVF=$ROOT/.env
GATE_PORT=8770
BACKEND_PORT=8765
PAGE_PORT=5173
AR_PORT=8766
NGINX=/usr/sbin/nginx
CONF=$TMPD/gate.nginx.conf

key() { grep -E '^STAC_GATE_KEY=' "$ENVF" 2>/dev/null | tail -1 | cut -d= -f2-; }

new_key() {
    local k
    k=$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')
    [ ${#k} -eq 64 ] || { echo "[remote] could not draw a key" >&2; exit 1; }
    if grep -qE '^STAC_GATE_KEY=' "$ENVF" 2>/dev/null; then
        grep -vE '^STAC_GATE_KEY=' "$ENVF" > "$ENVF.gate.tmp"
        echo "STAC_GATE_KEY=$k" >> "$ENVF.gate.tmp"
        chmod --reference="$ENVF" "$ENVF.gate.tmp" 2>/dev/null || true
        mv "$ENVF.gate.tmp" "$ENVF"
    else
        echo "STAC_GATE_KEY=$k" >> "$ENVF"
    fi
}

# The gate's nginx master, or nothing. Its pid file lives on the volume and outlives the pod: on a new pod the
# number it holds can belong to another process (2026-11-24, reproduced: the start then "reloaded" — signalled
# that process, which died — and never started the gate). The number counts only when its process is this gate.
gate_pid() {
    local p
    p=$(cat "$RUN/nginx.pid" 2>/dev/null) || return 1
    [[ "$p" =~ ^[0-9]+$ ]] || return 1
    { tr '\0' ' ' < "/proc/$p/cmdline"; } 2>/dev/null | grep -qF -- "nginx -c $CONF -p $RUN" || return 1
    echo "$p"
}

gate_up() {
    [ -x "$NGINX" ] || { echo "[remote] $NGINX is not in this image — the gate needs nginx" >&2; exit 1; }
    [ -n "$(key)" ] || new_key
    mkdir -p "$RUN" "$TMPD"
    chmod 700 "$TMPD"
    ( umask 077
      sed -e "s#__RUN__#$RUN#g" -e "s#__TMP__#$TMPD#g" -e "s#__KEY__#$(key)#g" -e "s#__PORT__#$GATE_PORT#g" \
          -e "s#__BACKEND__#$BACKEND_PORT#g" -e "s#__PAGE__#$PAGE_PORT#g" \
          "$ROOT/scripts/remote/gate.nginx.conf.template" > "$CONF.new"
      mv "$CONF.new" "$CONF" )
    "$NGINX" -t -q -c "$CONF" -p "$RUN"
    if gate_pid >/dev/null; then
        "$NGINX" -s reload -c "$CONF" -p "$RUN"
        echo "[remote] gate reloaded (127.0.0.1:$GATE_PORT)"
    else
        rm -f "$RUN/nginx.pid"                  # a previous pod's: never signalled
        "$NGINX" -c "$CONF" -p "$RUN"
        echo "[remote] gate started (127.0.0.1:$GATE_PORT)"
    fi
}

gate_down() {
    if gate_pid >/dev/null; then
        "$NGINX" -s quit -c "$CONF" -p "$RUN" && echo "[remote] gate stopped"
    fi
}

daemon_up() {
    [ -x "$TS_DIR/tailscaled" ] || { echo "[remote] $TS_DIR/tailscaled is missing (scripts/rebuild restores it)" >&2; exit 1; }
    mkdir -p "$TS_DIR/state"
    if ! pgrep -f "tailscaled.*$TS_DIR/state" >/dev/null; then
        echo "[remote] starting tailscaled (userspace networking)"
        # its own session: a caller's time limit or closed terminal must not take the daemon with it
        setsid nohup "$TS_DIR/tailscaled" --tun=userspace-networking --statedir="$TS_DIR/state" \
            --socket="$TS_DIR/tailscaled.sock" >"$TS_DIR/tailscaled.log" 2>&1 </dev/null &
        sleep 5
    fi
    # `tailscale up` WAITS when the pod must be authorised again (it prints a link): never silently
    if ! timeout 20 $TS status >/dev/null 2>&1 || ! timeout 30 $TS up --hostname=stac-pod >"$TMPD/up.out" 2>&1; then
        cat "$TMPD/up.out" 2>/dev/null
        echo "[remote] THE POD MUST BE AUTHORISED ONCE: open the link above in a browser signed in to the Tailscale"
        echo "         account (or run: $TS up --hostname=stac-pod), then run this script again."
        exit 2
    fi
}

name() { $TS status --json 2>/dev/null | grep -o '"DNSName": *"[^"]*"' | head -1 | cut -d'"' -f4 | sed 's/\.$//'; }

case "${1:-start}" in
    start)
        gate_up
        daemon_up
        # the page, public, through the gate; the AR server as before (inside the tailnet only, its own port)
        # already published through the gate: nothing is set again (setting it again drops the address for a moment)
        st=$(timeout 15 $TS funnel status 2>/dev/null || true)
        if echo "$st" | grep -q "(Funnel on)" && echo "$st" | grep -q "proxy http://127.0.0.1:$GATE_PORT"; then
            echo "[remote] the page: https://$(name)/   (already published; the secret link: bash scripts/remote_access.sh link)"
            exit 0
        fi
        # inside the tailnet too the address answers through the gate (August's setup served the backend bare)
        timeout 20 $TS serve --bg "http://127.0.0.1:$GATE_PORT" >/dev/null 2>&1 || true
        # `tailscale funnel` WAITS while Funnel is not enabled for the tailnet (it prints the link that enables
        # it): its words are shown, never swallowed
        if ! timeout 30 $TS funnel --bg "http://127.0.0.1:$GATE_PORT" >"$TMPD/funnel.out" 2>&1; then
            cat "$TMPD/funnel.out"
            echo "[remote] THE FUNNEL IS NOT ON YET — open the link above once (it enables Funnel for this pod in the"
            echo "         tailnet), then run this script again. Until then the address answers inside the tailnet only."
            exit 3
        fi
        $TS serve --bg --https=8443 "http://127.0.0.1:$AR_PORT" >/dev/null 2>&1 || true
        echo "[remote] the page: https://$(name)/   (the secret link: bash scripts/remote_access.sh link)"
        ;;
    gate)                                   # the gate alone (render + start or reload): nothing published
        gate_up
        ;;
    link)
        [ -n "$(key)" ] || { echo "[remote] no key yet — run: bash scripts/remote_access.sh start" >&2; exit 1; }
        echo "https://$(name)/__gate?k=$(key)"
        ;;
    status)
        if gate_pid >/dev/null; then echo "gate: up (127.0.0.1:$GATE_PORT)"; else echo "gate: DOWN"; fi
        if pgrep -f "tailscaled.*$TS_DIR/state" >/dev/null; then echo "tailscaled: up — $(name)"; $TS funnel status 2>&1 | sed 's/^/   /'; else echo "tailscaled: DOWN"; fi
        ;;
    rotate)
        new_key
        gate_up
        echo "[remote] a new key: every browser must open the new link (bash scripts/remote_access.sh link)"
        ;;
    stop)
        $TS funnel --https=443 off >/dev/null 2>&1 || true
        gate_down
        ;;
    *)
        echo "usage: bash scripts/remote_access.sh start | link | status | rotate | stop" >&2
        exit 64
        ;;
esac
