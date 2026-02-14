#!/bin/bash
# ===========================================================================
# STAC-Builder — Docker Launcher (replaces scripts/start.sh for Docker mode)
#
# Usage:
#   bash scripts/docker-start.sh              # Start the server (production)
#   bash scripts/docker-start.sh --dev        # Dev mode (live code reload)
#   bash scripts/docker-start.sh --build      # Rebuild image, then start
#   bash scripts/docker-start.sh --build --dev # Rebuild + dev mode
#   bash scripts/docker-start.sh --stop       # Stop the running container
# ===========================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONTAINER_NAME="stac-builder"
IMAGE_NAME="stac-builder:latest"
PORT="${STAC_PORT:-8765}"
DEV_MODE=false

# ── Parse arguments ──────────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --dev)  DEV_MODE=true ;;
    esac
done

if [ "$DEV_MODE" = true ]; then
    MODE_LABEL="Docker — DEV 🔧"
else
    MODE_LABEL="Docker — Production"
fi

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     Ingerop IN3 - STAC-Builder [$MODE_LABEL]"                  
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# ── Handle --stop ────────────────────────────────────────────────────
if echo "$@" | grep -q '\-\-stop'; then
    echo "🛑 Stopping container..."
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
    echo "✅ Stopped"
    exit 0
fi

# ── Handle --build ───────────────────────────────────────────────────
if echo "$@" | grep -q '\-\-build'; then
    echo "🔨 Rebuilding Docker image..."
    docker build -t "$IMAGE_NAME" "$PROJECT_ROOT"
    echo ""
fi

# ── Check weights ────────────────────────────────────────────────────
if [ ! -d "${PROJECT_ROOT}/weights" ] || [ -z "$(ls -A ${PROJECT_ROOT}/weights 2>/dev/null)" ]; then
    echo "⚠️  No weights directory found."
    echo "   Run: bash setup_weights.sh"
    echo "   Continuing anyway (models may fail to load)..."
    echo ""
    mkdir -p "${PROJECT_ROOT}/weights"
fi

# ── Stop previous container if running ───────────────────────────────
docker stop "$CONTAINER_NAME" 2>/dev/null || true
docker rm "$CONTAINER_NAME" 2>/dev/null || true

# ── Windows network configuration (port proxy + firewall) ────────────
WSL_IP=$(ip addr show eth0 | grep 'inet ' | awk '{print $2}' | cut -d/ -f1)
if [ -z "$WSL_IP" ]; then
    WSL_IP=$(hostname -I | awk '{print $1}')
fi

WIN_IP=$(powershell.exe -Command "Get-NetIPAddress -AddressFamily IPv4 | Where-Object { \$_.InterfaceAlias -match 'Wi-Fi|Ethernet' -and \$_.PrefixOrigin -eq 'Dhcp' } | Select-Object -First 1 -ExpandProperty IPAddress" 2>/dev/null | tr -d '\r')

if [ -n "$WIN_IP" ]; then
    echo "🌐 Network config:"
    echo "   WSL IP:     $WSL_IP"
    echo "   Windows IP: $WIN_IP"
    powershell.exe -Command "netsh interface portproxy delete v4tov4 listenport=$PORT listenaddress=0.0.0.0 2>\$null; netsh interface portproxy add v4tov4 listenport=$PORT listenaddress=0.0.0.0 connectport=$PORT connectaddress=$WSL_IP" 2>/dev/null
    powershell.exe -Command "if (!(Get-NetFirewallRule -DisplayName 'STAC-Builder $PORT' -ErrorAction SilentlyContinue)) { New-NetFirewallRule -DisplayName 'STAC-Builder $PORT' -Direction Inbound -LocalPort $PORT -Protocol TCP -Action Allow }" 2>/dev/null
else
    WIN_IP="(not detected)"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Access Points:                                               ║"
echo "╠════════════════════════════════════════════════════════════════╣"
echo "║  📍 LOCAL:                                                     "
echo "║     https://localhost:$PORT/static/viewer.html                 "
echo "║     https://localhost:$PORT/static/camera.html                 "
echo "║  📱 EXTERNAL:                                                  "
echo "║     https://$WIN_IP:$PORT/static/viewer.html                   "
echo "║     https://$WIN_IP:$PORT/static/camera.html                   "
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# ── Build docker run command ─────────────────────────────────────────
DOCKER_ARGS=(
    --gpus all
    --name "$CONTAINER_NAME"
    -p "$PORT:$PORT"
    -e "STAC_PORT=$PORT"
    -v "${PROJECT_ROOT}/weights:/app/weights"
    -v "${PROJECT_ROOT}/server/scans:/app/server/scans"
    -v "${PROJECT_ROOT}/scripts/cert.pem:/app/certs/cert.pem"
    -v "${PROJECT_ROOT}/scripts/key.pem:/app/certs/key.pem"
    --rm
)

if [ "$DEV_MODE" = true ]; then
    echo "🔧 Dev mode: mounting server/, static/, config.yaml as live volumes"
    echo "   ✏️  Edit files → restart container (no rebuild needed)"
    echo ""
    DOCKER_ARGS+=(
        -v "${PROJECT_ROOT}/server:/app/server"
        -v "${PROJECT_ROOT}/static:/app/static"
    )
else
    echo "📦 Production mode: using code baked into image"
    echo "   ℹ️  Use --dev for live code editing"
    echo ""
fi

echo "🚀 Starting Docker container..."
exec docker run "${DOCKER_ARGS[@]}" "$IMAGE_NAME"
