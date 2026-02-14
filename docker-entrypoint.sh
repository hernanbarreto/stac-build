#!/bin/bash
# ===========================================================================
# STAC-Builder — Docker Entrypoint
# Handles SSL certificate generation and server startup inside the container.
# ===========================================================================

set -e

PORT="${STAC_PORT:-8765}"
CERT_DIR="/app/certs"
CERT_FILE="$CERT_DIR/cert.pem"
KEY_FILE="$CERT_DIR/key.pem"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     Ingerop IN3 - STAC-Builder [Docker]                       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# ── Check weights ────────────────────────────────────────────────────
if [ ! -d "/app/weights" ] || [ -z "$(ls -A /app/weights 2>/dev/null)" ]; then
    echo "⚠️  No weights found at /app/weights"
    echo "   Mount weights: -v \$(pwd)/weights:/app/weights"
    echo "   Or run: bash setup_weights.sh"
    echo ""
fi

# ── SSL Certificates ────────────────────────────────────────────────
mkdir -p "$CERT_DIR"

if [ ! -f "$CERT_FILE" ] || [ ! -f "$KEY_FILE" ]; then
    echo "🔐 Generating self-signed SSL certificates..."
    openssl req -x509 -newkey rsa:4096 -keyout "$KEY_FILE" -out "$CERT_FILE" \
        -days 365 -nodes -subj "/CN=localhost" 2>/dev/null
    echo "✅ SSL certificates generated"
else
    echo "✅ Using existing SSL certificates"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Access Points:                                               ║"
echo "║  🔹 https://localhost:$PORT/static/viewer.html                ║"
echo "║  🔹 https://localhost:$PORT/static/camera.html                ║"
echo "║  📊 https://localhost:$PORT/slam/status                       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "🚀 Launching STAC Server on port $PORT..."
echo ""

cd /app/server

exec python3 -m uvicorn main:app --host 0.0.0.0 --port "$PORT" \
    --ssl-keyfile "$KEY_FILE" \
    --ssl-certfile "$CERT_FILE"
