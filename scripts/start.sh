#!/bin/bash
# STAC-Builder Startup Script
# Uses MapAnything reconstruction backend
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

set -e

# Configuration
USER_HOME="/home/hernan"
STAC_ROOT="$USER_HOME/stac-builder"
SERVER_DIR="$STAC_ROOT/server"
CONFIG_FILE="$SERVER_DIR/config.yaml"

# External project paths
SAM3_PATH="$USER_HOME/sam3"

# Script and SSL paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CERT_FILE="$SCRIPT_DIR/cert.pem"
KEY_FILE="$SCRIPT_DIR/key.pem"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     Ingerop IN3 - STAC-Builder - Site Scanner [HTTPS MODE]     ║"
echo "╠════════════════════════════════════════════════════════════════╣"
echo "║  Backend: MapAnything"
echo "╚════════════════════════════════════════════════════════════════╝"

# ----------------------------------------------------------------------------
# Activate conda environment
# ----------------------------------------------------------------------------
CONDA_BASE="$USER_HOME/miniforge3"
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Conda environment (contains PyTorch, SAM3 deps, MapAnything deps)
CONDA_ENV="da3"
echo "🔧 Activating $CONDA_ENV environment..."
export PYTHONPATH="$PYTHONPATH:$SAM3_PATH"

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
echo "║  📱 EXTERNAL (mobile/other devices on same network):               "
echo "║     https://$WIN_IP:$PORT/static/viewer.html"
echo "║     https://$WIN_IP:$PORT/static/camera.html"
echo "║                                                                    "
echo "║  📊 API Status: https://$WIN_IP:$PORT/status"
echo "║                                                                    ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "🚀 Launching STAC Server..."

# Ensure external drive is mounted (WSL2 loses DrvFS mounts on reboot)
if grep -q "projects_dir" "$CONFIG_FILE" 2>/dev/null; then
    EXT_DRIVE="/mnt/e"
    if ! ls "$EXT_DRIVE" &>/dev/null; then
        echo "📀 Mounting external drive E: → $EXT_DRIVE ..."
        sudo mount -t drvfs E: "$EXT_DRIVE" 2>/dev/null && echo "   ✅ Drive mounted" || echo "   ⚠️  Could not mount E: — projects may not load"
    fi
fi
echo ""

# Run Uvicorn Server with SSL
# --reload disabled for GPU models (hot reload causes CUDA issues)
exec python -m uvicorn main:app --host 0.0.0.0 --port $PORT \
    --ssl-keyfile "$KEY_FILE" \
    --ssl-certfile "$CERT_FILE" \
    --ws-ping-interval 30 \
    --ws-ping-timeout 600