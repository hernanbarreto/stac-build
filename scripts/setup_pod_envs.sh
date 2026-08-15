#!/usr/bin/env bash
# Crea en la pod los conda envs necesarios para correr stac-build completo.
#
# Usa los YAMLs de docs/migration/ (que ya están preparados para portabilidad)
# en lugar de exportar los envs locales (que tienen torch compilado con la
# CUDA local y no sirve para la GPU de la pod).
#
# Uso (desde WSL):
#   POD_HOST=213.173.107.104 POD_PORT=10099 ./scripts/setup_pod_envs.sh

set -e

POD_HOST="${POD_HOST:?POD_HOST no seteado}"
POD_PORT="${POD_PORT:?POD_PORT no seteado}"
POD_USER="${POD_USER:-root}"
POD_KEY="${POD_KEY:-$HOME/.ssh/id_ed25519}"
POD_DEST="${POD_DEST:-/workspace/stac-builder}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MIGRATION_DIR="$REPO_DIR/docs/migration"

# Envs que vamos a crear (los YAMLs ya están en docs/migration/).
# NOTA: no existe env `sam3` — SAM 3.x corre in-process dentro del server (env
# `da3`, vendor/sam31 inyectado al sys.path por vendor_paths.py). Los envs
# `pgsr` (clon de da3 + extensiones CUDA compiladas) y `nodejs` se crean en
# bloques propios más abajo — no salen de un YAML.
ENVS=(stac-build da3 mapanything semantic CloudComPy310)

SSH="ssh -p $POD_PORT -i $POD_KEY -o StrictHostKeyChecking=accept-new"
SCP="scp -P $POD_PORT -i $POD_KEY -o StrictHostKeyChecking=accept-new"

# ── 1. Generar yamls limpios localmente (sin builds específicos del host) ──
STAGE_DIR="$SCRIPT_DIR/.pod_envs_stage"
mkdir -p "$STAGE_DIR"

echo "[envs] Preparando YAMLs para la pod en $STAGE_DIR"
for env in "${ENVS[@]}"; do
    src="$MIGRATION_DIR/environment_${env}.yml"
    dst="$STAGE_DIR/${env}.yml"
    if [[ ! -f "$src" ]]; then
        echo "  ⚠ no se encontró $src — salteando $env"
        continue
    fi
    # Sacar las builds (=algo_algo después del segundo =) para que conda resuelva
    # con paquetes compatibles con la pod (CUDA 12.4 en lugar de la CUDA local).
    sed -E 's/(=[0-9a-zA-Z._+-]+)=[^[:space:]]+$/\1/' "$src" > "$dst"
    echo "  - $env  ($(wc -l <"$dst") líneas)"
done

# MeshFlow: si no está en migration, lo exportamos desde local sin builds
if [[ ! -f "$STAGE_DIR/meshflow.yml" ]]; then
    if [[ -d "$HOME/miniforge3/envs/meshflow" ]]; then
        echo "  - meshflow (exportando desde local sin builds)"
        source "$HOME/miniforge3/etc/profile.d/conda.sh"
        conda env export -n meshflow --no-builds > "$STAGE_DIR/meshflow.yml"
    fi
fi

# ── 2. Subir a la pod ──
echo
echo "[envs] Subiendo YAMLs a la pod"
$SSH "$POD_USER@$POD_HOST" "mkdir -p $POD_DEST/envs_pod"
$SCP "$STAGE_DIR"/*.yml "$POD_USER@$POD_HOST:$POD_DEST/envs_pod/"

# ── 3. Instalar miniforge en la pod (volumen persistente) y crear envs ──
echo
echo "[envs] Instalando miniforge en la pod y creando envs (esto tarda ~30-60 min la primera vez)"
$SSH "$POD_USER@$POD_HOST" 'bash -s' <<'REMOTE_SCRIPT'
set -e
MINIFORGE_DIR="/workspace/miniforge3"

if [ ! -d "$MINIFORGE_DIR" ]; then
    echo "[pod] Instalando miniforge en $MINIFORGE_DIR"
    cd /tmp
    wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O miniforge.sh
    bash miniforge.sh -b -p "$MINIFORGE_DIR"
    rm -f miniforge.sh
else
    echo "[pod] miniforge ya instalado"
fi

source "$MINIFORGE_DIR/etc/profile.d/conda.sh"
cd /workspace/stac-builder

for yml in envs_pod/*.yml; do
    env_name=$(basename "$yml" .yml)
    if conda env list | awk '{print $1}' | grep -qx "$env_name"; then
        echo "[pod] env $env_name ya existe — skipping (correlo a mano con conda env update si querés actualizar)"
        continue
    fi
    echo "[pod] Creando env $env_name desde $yml"
    conda env create -n "$env_name" -f "$yml" || echo "  ⚠ falló create de $env_name"
done

echo
# ── Env pgsr: REQUERIDO por el backend DEFAULT (vggtomega_pgsr). Clon del env
# da3 + extensiones CUDA de vendor/pgsr compiladas (VENDORS.lock.md §2). Sin
# este env la etapa PGSR del pipeline default falla en un pod nuevo. ──
if conda env list | awk '{print $1}' | grep -qx "pgsr"; then
    echo "[pod] env pgsr ya existe — skipping"
elif [ ! -d /workspace/stac-builder/vendor/pgsr/submodules ]; then
    echo "  ⚠ vendor/pgsr no está provisionado (corré scripts/setup_vendors.sh) — salteando env pgsr"
else
    echo "[pod] Creando env pgsr (clon de da3) y compilando extensiones PGSR"
    conda create -n pgsr --clone da3 -y
    conda activate pgsr
    export TORCH_CUDA_ARCH_LIST="8.0;8.6"
    pip install /workspace/stac-builder/vendor/pgsr/submodules/diff-plane-rasterization \
        || echo "  ⚠ falló diff-plane-rasterization — compilar a mano"
    pip install /workspace/stac-builder/vendor/pgsr/submodules/simple-knn \
        || echo "  ⚠ falló simple-knn — compilar a mano"
    conda deactivate
fi

echo
# ── Env nodejs: Vite/Electron + tools/glb (compresión de mallas) ──
if conda env list | awk '{print $1}' | grep -qx "nodejs"; then
    echo "[pod] env nodejs ya existe — skipping"
else
    echo "[pod] Creando env nodejs"
    conda create -n nodejs -c conda-forge nodejs=20 -y || echo "  ⚠ falló create de nodejs"
fi

echo
# ── Env milo: usa install.py de vendor/MILo (compila rasterizers CUDA) ──
if [ -d /workspace/stac-builder/vendor/MILo ]; then
    if conda env list | awk '{print $1}' | grep -qx "milo"; then
        echo "[pod] env milo ya existe — skipping install (correlo a mano si querés rehacer)"
    else
        echo "[pod] Creando env milo (python 3.9) y corriendo install.py de MILo"
        conda create -n milo python=3.9 -y
        conda activate milo
        cd /workspace/stac-builder/vendor/MILo
        # La pod tiene CUDA 12.4; install.py default es 11.8, le pasamos 12.1 que es el más cercano soportado
        python install.py --cuda_version 12.1 || echo "  ⚠ install.py falló — revisar a mano"
        conda deactivate
    fi
fi

echo "[pod] Envs disponibles:"
conda env list
REMOTE_SCRIPT

echo
echo "[envs] OK. Próximo paso: reinstalar torch en cada env con la CUDA correcta de la pod."
