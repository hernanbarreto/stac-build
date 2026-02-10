#!/bin/bash
# ============================================================
# Real-time monitoring during DA3 processing
# Run in a SEPARATE terminal: bash monitor.sh
# ============================================================

while true; do
    clear
    echo "========== STAC SYSTEM MONITOR ($(date +%H:%M:%S)) =========="
    echo ""
    
    # Memory
    echo "── RAM ──"
    free -h | head -2
    echo ""
    
    # Swap usage
    SWAP_USED=$(free -m | grep Swap | awk '{print $3}')
    SWAP_TOTAL=$(free -m | grep Swap | awk '{print $2}')
    echo "Swap: ${SWAP_USED}MB / ${SWAP_TOTAL}MB"
    echo ""
    
    # GPU
    if command -v nvidia-smi &> /dev/null; then
        echo "── GPU ──"
        nvidia-smi --query-gpu=temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null
    fi
    echo ""
    
    # Top memory consumers
    echo "── Top 5 Memory Consumers ──"
    ps aux --sort=-%mem | head -6 | awk '{printf "%-6s %-6s %-6s %s\n", $2, $4"%", $6/1024"MB", $11}'
    echo ""
    
    # Warning thresholds
    MEM_AVAIL=$(cat /proc/meminfo | grep MemAvailable | awk '{print $2}')
    MEM_AVAIL_MB=$((MEM_AVAIL / 1024))
    if [ $MEM_AVAIL_MB -lt 1024 ]; then
        echo "⚠️  WARNING: Available RAM < 1GB ($MEM_AVAIL_MB MB)"
    elif [ $MEM_AVAIL_MB -lt 2048 ]; then
        echo "⚠️  CAUTION: Available RAM < 2GB ($MEM_AVAIL_MB MB)"
    else
        echo "✅ Available RAM: ${MEM_AVAIL_MB}MB"
    fi
    
    sleep 3
done
