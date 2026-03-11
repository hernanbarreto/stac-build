#!/bin/bash
# ============================================================
# WSL System Hardening for Heavy Processing (heavy processing)
# Run BEFORE starting the STAC server: sudo bash harden_wsl.sh
# ============================================================

echo "=== WSL System Hardening for Heavy Processing ==="

# 1. Increase swappiness (default 60 → 80)
# Higher value = kernel swaps out unused memory pages earlier,
# keeping more RAM free for active processing
echo 80 > /proc/sys/vm/swappiness
echo "✅ vm.swappiness = $(cat /proc/sys/vm/swappiness)"

# 2. Enable overcommit (allows allocations beyond physical RAM)
# 1 = "always overcommit" — lets the kernel use swap more aggressively
# Prevents OOM killer from killing processes prematurely
echo 1 > /proc/sys/vm/overcommit_memory
echo "✅ vm.overcommit_memory = $(cat /proc/sys/vm/overcommit_memory)"

# 3. Reduce vfs_cache_pressure (free up cached memory faster)
echo 50 > /proc/sys/vm/vfs_cache_pressure
echo "✅ vm.vfs_cache_pressure = $(cat /proc/sys/vm/vfs_cache_pressure)"

# 4. Drop filesystem caches to free RAM before processing
echo 3 > /proc/sys/vm/drop_caches
echo "✅ Dropped filesystem caches"

# 5. GPU Power Limit (reduce temperature under heavy load)
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi -pl 100 2>/dev/null && echo "✅ GPU power limit set to 100W (from 115W)" || echo "⚠️ GPU power limit not set (may need root)"
    nvidia-smi --query-gpu=temperature.gpu,power.draw,power.limit --format=csv,noheader
fi

echo ""
echo "=== Current Memory Status ==="
free -h
echo ""
echo "=== Swap ==="
swapon --show
echo ""
echo "Ready for heavy processing. Start the server now."
