#!/bin/bash
# STAC-Builder Startup Script
# Automatically selects conda environment based on slam_backend in config.yaml
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

set -e

# Configuration
USER_HOME="/home/hernan"
STAC_ROOT="$USER_HOME/stac-builder"
SERVER_DIR="$STAC_ROOT/server"
CONFIG_FILE="$SERVER_DIR/config.yaml"

# External project paths
DA3_PATH="$USER_HOME/Depth-Anything-3/src"
DA3_ROOT="$USER_HOME/Depth-Anything-3"
DA3_STREAMING_PATH="$DA3_ROOT/da3_streaming"
MAST3R_SLAM_PATH="$USER_HOME/mast3r_slam"
SAM3_PATH="$USER_HOME/sam3"

# Script and SSL paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CERT_FILE="$SCRIPT_DIR/cert.pem"
KEY_FILE="$SCRIPT_DIR/key.pem"

# ----------------------------------------------------------------------------
# Detect SLAM backend from config.yaml
# ----------------------------------------------------------------------------
get_slam_backend() {
    if [ -f "$CONFIG_FILE" ]; then
        # Extract slam_backend value using awk (handles quotes and CRLF)
        backend=$(awk -F: '/^slam_backend:/ {gsub(/[" \r]/, "", $2); print $2}' "$CONFIG_FILE")
        if [ -n "$backend" ]; then
            echo "$backend"
            return
        fi
    fi
    # Default to mast3r if not found
    echo "mast3r"
}

SLAM_BACKEND=$(get_slam_backend)

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     Ingerop IN3 - STAC-Builder - Site Scanner [HTTPS MODE]     ║"
echo "╠════════════════════════════════════════════════════════════════╣"
echo "║  SLAM Backend: $SLAM_BACKEND"
echo "╚════════════════════════════════════════════════════════════════╝"

# ----------------------------------------------------------------------------
# Activate appropriate conda environment
# ----------------------------------------------------------------------------
# Use miniforge3 path directly
CONDA_BASE="$USER_HOME/miniforge3"
source "$CONDA_BASE/etc/profile.d/conda.sh"

if [ "$SLAM_BACKEND" = "mast3r" ] || [ "$SLAM_BACKEND" = "hybrid" ]; then
    echo "🔧 Activating MASt3R-SLAM environment (backend: $SLAM_BACKEND)..."
    CONDA_ENV="mast3r-slam"
    # MASt3R-SLAM requires multiple paths for its dependencies
    MAST3R_THIRDPARTY="$MAST3R_SLAM_PATH/thirdparty/mast3r"
    DUST3R_PATH="$MAST3R_THIRDPARTY/dust3r"
    CROCO_PATH="$DUST3R_PATH/croco"
    export PYTHONPATH="$PYTHONPATH:$MAST3R_SLAM_PATH:$MAST3R_THIRDPARTY:$DUST3R_PATH:$CROCO_PATH:$SAM3_PATH"
    
    # Note: In hybrid mode, DA3 runs as a subprocess in its own 'da3' conda env.
    # No DA3 paths needed here — the subprocess sets its own PYTHONPATH.
else
    echo "🔧 Activating DA3 environment..."
    CONDA_ENV="da3"
    # NOTE: Do NOT add DA3_STREAMING_PATH here — it causes Python to find
    # da3_streaming.py (file) instead of da3_streaming/ (package directory)
    export PYTHONPATH="$PYTHONPATH:$DA3_PATH:$DA3_ROOT:$SAM3_PATH"
fi

conda activate "$CONDA_ENV"

echo ""
echo "📍 Environment: $CONDA_ENV"
echo "📍 Python: $(which python)"
echo "📍 PYTHONPATH: $PYTHONPATH"
echo "📍 SSL Certificate: $CERT_FILE"
echo ""

# ----------------------------------------------------------------------------
# Check SSL certificates
# ----------------------------------------------------------------------------
if [ ! -f "$CERT_FILE" ] || [ ! -f "$KEY_FILE" ]; then
    echo "⚠️  SSL certificates not found. Generating self-signed certificates..."
    openssl req -x509 -newkey rsa:4096 -keyout "$KEY_FILE" -out "$CERT_FILE" \
        -days 365 -nodes -subj "/CN=localhost"
    echo "✅ Certificates generated"
fi

# ----------------------------------------------------------------------------
# Network Configuration for External Devices
# ----------------------------------------------------------------------------
PORT=8765

echo ""
echo "🔄 Configuring network access..."

# Get WSL internal IP
WSL_IP=$(ip addr show eth0 | grep 'inet ' | awk '{print $2}' | cut -d/ -f1)
if [ -z "$WSL_IP" ]; then
    WSL_IP=$(hostname -I | awk '{print $1}')
fi
echo "   🔹 WSL IP: $WSL_IP"

# Get Windows public IP (for external devices)
WIN_IP=$(powershell.exe -Command "Get-NetIPAddress -AddressFamily IPv4 | Where-Object { \$_.InterfaceAlias -match 'Wi-Fi|Ethernet' -and \$_.PrefixOrigin -eq 'Dhcp' } | Select-Object -First 1 -ExpandProperty IPAddress" 2>/dev/null | tr -d '\r')

if [ -n "$WIN_IP" ]; then
    echo "   🔹 Windows IP: $WIN_IP"
    
    # Configure Windows port proxy (requires admin - may fail silently)
    echo "   🔹 Configuring port proxy..."
    powershell.exe -Command "netsh interface portproxy delete v4tov4 listenport=$PORT listenaddress=0.0.0.0 2>\$null; netsh interface portproxy add v4tov4 listenport=$PORT listenaddress=0.0.0.0 connectport=$PORT connectaddress=$WSL_IP" 2>/dev/null
    
    # Ensure firewall rule exists
    powershell.exe -Command "if (!(Get-NetFirewallRule -DisplayName 'STAC-Builder $PORT' -ErrorAction SilentlyContinue)) { New-NetFirewallRule -DisplayName 'STAC-Builder $PORT' -Direction Inbound -LocalPort $PORT -Protocol TCP -Action Allow }" 2>/dev/null
else
    WIN_IP="(not detected)"
    echo "   ⚠️  Could not detect Windows IP - external access may not work"
fi

# ----------------------------------------------------------------------------
# Navigate to server and start
# ----------------------------------------------------------------------------
cd "$SERVER_DIR"

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                        Access Points                               ║"
echo "╠════════════════════════════════════════════════════════════════════╣"
echo "║                                                                    ║"
echo "║  📍 LOCAL (same machine):                                          "
echo "║     https://localhost:$PORT/static/viewer.html                     "
echo "║     https://localhost:$PORT/static/camera.html                     "
echo "║                                                                    "
echo "║  � EXTERNAL (mobile/other devices on same network):               "
echo "║     https://$WIN_IP:$PORT/static/viewer.html"
echo "║     https://$WIN_IP:$PORT/static/camera.html"
echo "║                                                                    "
echo "║  📊 API Status: https://$WIN_IP:$PORT/slam/status"
echo "║                                                                    ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "🚀 Launching STAC Server..."
echo ""

# Run Uvicorn Server with SSL
# --reload disabled for GPU models (hot reload causes CUDA issues)
exec python -m uvicorn main:app --host 0.0.0.0 --port $PORT \
    --ssl-keyfile "$KEY_FILE" \
    --ssl-certfile "$CERT_FILE" \
    --ws-ping-interval 30 \
    --ws-ping-timeout 300