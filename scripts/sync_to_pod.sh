#!/usr/bin/env bash
# Sincroniza el proyecto stac-builder a la pod de RunPod.
#
# Uso:
#   POD_HOST=213.173.107.104 POD_PORT=10099 ./scripts/sync_to_pod.sh
#
# La pod cambia de IP/puerto en cada reinicio. Copialos del dashboard
# (Connect → SSH over exposed TCP) y exportalos antes de correr el script.
#
# Sube:
#   - server/             (backend + datos de proyectos)
#   - vendor/MILo/        (repo MILo clonado)
# Excluye:
#   - ui/, .git/, __pycache__/, vendor pesados que ya corrieron upstream
# Destino: /workspace/stac-builder/  (volumen persistente de la pod)

set -e

POD_HOST="${POD_HOST:?POD_HOST no seteado. Ej: POD_HOST=213.173.107.104}"
POD_PORT="${POD_PORT:?POD_PORT no seteado. Ej: POD_PORT=10099}"
POD_USER="${POD_USER:-root}"
POD_KEY="${POD_KEY:-$HOME/.ssh/id_ed25519}"
POD_DEST="${POD_DEST:-/workspace/stac-builder}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EXCLUDE_FILE="$SCRIPT_DIR/.pod_rsync_exclude"

echo "[sync] origen   : $REPO_DIR"
echo "[sync] destino  : $POD_USER@$POD_HOST:$POD_DEST  (port $POD_PORT)"
echo "[sync] exclude  : $EXCLUDE_FILE"
echo

# Asegurar que existe el dir destino en la pod
ssh -p "$POD_PORT" -i "$POD_KEY" -o StrictHostKeyChecking=accept-new \
    "$POD_USER@$POD_HOST" "mkdir -p $POD_DEST"

RSYNC_OPTS=(-rltDvz --no-owner --no-group --no-perms --exclude-from="$EXCLUDE_FILE"
            -e "ssh -p $POD_PORT -i $POD_KEY -o StrictHostKeyChecking=accept-new")

# server/ → /workspace/stac-builder/server/
rsync "${RSYNC_OPTS[@]}" --delete \
      "$REPO_DIR/server/" \
      "$POD_USER@$POD_HOST:$POD_DEST/server/"

# vendor/{MILo, depth-anything-3, sam3, meshflow} → /workspace/stac-builder/vendor/<X>/
ssh -p "$POD_PORT" -i "$POD_KEY" -o StrictHostKeyChecking=accept-new \
    "$POD_USER@$POD_HOST" "mkdir -p $POD_DEST/vendor"
for vd in MILo depth-anything-3 sam3 meshflow; do
    if [[ -d "$REPO_DIR/vendor/$vd" ]]; then
        echo "[sync] vendor/$vd"
        rsync "${RSYNC_OPTS[@]}" --delete \
              "$REPO_DIR/vendor/$vd/" \
              "$POD_USER@$POD_HOST:$POD_DEST/vendor/$vd/"
    fi
done

echo
echo "[sync] OK"
