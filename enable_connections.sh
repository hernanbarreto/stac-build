#!/bin/bash

# --- CONFIGURACIÓN ---
PORT=8765
# ---------------------

echo "🔄 Detectando IPs..."

# 1. Obtener la IP interna de WSL (limpiando la salida de ip addr)
WSL_IP=$(ip addr show eth0 | grep 'inet ' | awk '{print $2}' | cut -d/ -f1)

if [ -z "$WSL_IP" ]; then
    echo "❌ Error: No se pudo detectar la IP de WSL."
    exit 1
fi

echo "   🔹 IP interna WSL: $WSL_IP"

# 2. Obtener la IP de Windows (Usando powershell desde linux)
# Esto busca la IP preferida de la interfaz que tenga conexión a internet (Gateway por defecto)
WIN_IP=$(powershell.exe -Command "Get-NetIPAddress -AddressFamily IPv4 | Where-Object { \$_.InterfaceAlias -match 'Wi-Fi|Ethernet' -and \$_.PrefixOrigin -eq 'Dhcp' } | Select-Object -First 1 -ExpandProperty IPAddress" | tr -d '\r')

echo "   🔹 IP pública Windows: $WIN_IP"
echo ""
echo "🛠️  Configurando Windows (Requiere permisos de Admin)..."

# 3. Borrar regla anterior (para evitar basura) y crear la nueva
# Usamos powershell.exe para ejecutar los comandos de Windows
powershell.exe -Command "netsh interface portproxy delete v4tov4 listenport=$PORT listenaddress=0.0.0.0; netsh interface portproxy add v4tov4 listenport=$PORT listenaddress=0.0.0.0 connectport=$PORT connectaddress=$WSL_IP"

# 4. Asegurar firewall (esto puede dar error si ya existe, pero no afecta)
powershell.exe -Command "if (!(Get-NetFirewallRule -DisplayName 'WSL Server $PORT' -ErrorAction SilentlyContinue)) { New-NetFirewallRule -DisplayName 'WSL Server $PORT' -Direction Inbound -LocalPort $PORT -Protocol TCP -Action Allow }"

echo ""
echo "✅ ¡Listo! Dispositivo externo conectado."
echo "---------------------------------------------------"
echo "🌍 Accede desde tu celular/laptop a: http://$WIN_IP:$PORT"
echo "---------------------------------------------------"
