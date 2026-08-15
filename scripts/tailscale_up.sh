#!/bin/bash
# STAC-Builder — Tailscale for the pod (phone/AR access without public ports).
#
# Userspace networking (RunPod containers have no /dev/net/tun) with state on
# the persistent volume, so the node identity survives pod restarts. After the
# one-time `tailscale up` browser auth, `tailscale serve` publishes the backend
# inside the tailnet with an automatically-provisioned VALID TLS certificate
# (https://<pod-hostname>.<tailnet>.ts.net) — which iOS requires for WebXR.
#
#   bash scripts/tailscale_up.sh          # start daemon + serve (prints auth URL
#                                         # on first run — open it, then re-run)
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC
set -e

TS_DIR="/workspace/tailscale"
BACKEND_PORT=8765          # the STAC backend (HTTPS, self-signed internally)

if [ ! -x "$TS_DIR/tailscaled" ]; then
    echo "tailscaled not found in $TS_DIR — download the static bundle first:" >&2
    echo "  curl -L https://pkgs.tailscale.com/stable/tailscale_1.86.2_amd64.tgz | tar xz" >&2
    exit 1
fi

mkdir -p "$TS_DIR/state"

# 1. daemon (idempotent)
if ! pgrep -f "tailscaled.*$TS_DIR/state" >/dev/null; then
    echo "[tailscale] starting tailscaled (userspace networking)"
    nohup "$TS_DIR/tailscaled" \
        --tun=userspace-networking \
        --statedir="$TS_DIR/state" \
        --socket="$TS_DIR/tailscaled.sock" \
        >"$TS_DIR/tailscaled.log" 2>&1 &
    sleep 3
fi

TS="$TS_DIR/tailscale --socket=$TS_DIR/tailscaled.sock"

# 2. bring the node up (prints a login URL on the first run)
if ! $TS status >/dev/null 2>&1; then
    echo "[tailscale] node needs auth — open the URL below in a browser logged"
    echo "            into your Tailscale account, then re-run this script:"
    $TS up --hostname=stac-pod
    exit 0
fi
$TS up --hostname=stac-pod >/dev/null 2>&1 || true

# 3. publish the backend inside the tailnet with a valid TLS cert.
#    The backend speaks HTTPS with a self-signed cert → https+insecure.
$TS serve --bg "https+insecure://localhost:${BACKEND_PORT}" >/dev/null
#    AR server (phone XR, own process, plain HTTP internally) on :8443.
$TS serve --bg --https=8443 "http://127.0.0.1:8766" >/dev/null
echo "[tailscale] serve configured:"
$TS serve status
echo
echo "[tailscale] phone URL: https://$($TS status --json | grep -o '"DNSName": *"[^"]*"' | head -1 | cut -d'"' -f4 | sed 's/\.$//')"
