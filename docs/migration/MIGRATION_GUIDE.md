# STAC-Builder: Guía Definitiva de Migración a Linux Nativo

> [!IMPORTANT]
> **Guarda y comprime la carpeta `docs/migration/` y todo tu repositorio `stac-builder` en un pendrive o disco seguro antes de formatear.** 
> Asegúrate también de que tu disco externo (con las capturas y de proyectos `stac-projects`) no se borre.

## Paso 1: Instalación de Ubuntu (Recomendado 24.04 LTS o 22.04 LTS)

1. Instala el sistema operativo. Si tienes la opción en el instalador, marca la casilla **"Install third-party software for graphics and Wi-Fi hardware"**.
2. Una vez en tu escritorio Ubuntu, abre la terminal y actualiza el sistema:
```bash
sudo apt update && sudo apt upgrade -y
```

## Paso 2: Instalar los Drivers de NVIDIA (La Manera Fácil)

Con Ubuntu, los drivers de NVIDIA y CUDA son un solo comando:
```bash
sudo ubuntu-drivers autoinstall
```
*(Reinicia la computadora después de esto y comprueba que funciona abriendo la terminal y escribiendo `nvidia-smi`)*

## Paso 3: Instalar Dependencias del Sistema (Para C++ y Compilación)

Para que todo lo que armamos (PotreeConverter, CloudComPy en código fuente) siga funcionando, necesitas instalar estas librerías bases de Linux:

```bash
sudo apt install -y build-essential cmake git curl wget libssl-dev xorg-dev libgl1-mesa-dev unzip
```

También necesitas Node.js y npm instalados de forma nativa para levantar la interfaz gráfica (UI). Tu versión actual es Node 20. Instalalo así:
```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
```

## Paso 4: Instalar Conda (Miniforge3)
Como usabas `miniforge3` (optimizado y mucho mejor que anaconda base), hay que instalarlo exactamente igual:
```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
bash Miniforge3-Linux-x86_64.sh -b -p $HOME/miniforge3
source $HOME/miniforge3/bin/activate
conda init
```

## Paso 5: ¡Restaurar la Magia! (Entornos Conda)
Vuelve a volcar la carpeta completa de `stac-builder` en tu nuevo Linux y entra en la carpeta con la consola. Corre estos comandos para restaurar idénticamente todos los entornos a partir de los YAMLs que extrajimos:

```bash
cd /ruta/donde/hayas/dejado/stac-builder/docs/migration

# Recrear el servidor web (stac-build) y las dependencias base
conda env create -f environment_stac-build.yml

# Recrear DA3 y SLAM
conda env create -f environment_da3.yml

# Recrear CloudComPy para post-procesado C++
conda env create -f environment_CloudComPy310.yml

# Recrear SAM3 (Segmentación AI)
conda env create -f environment_sam3.yml

# Opcional (Si usas el fallback de MapAnything)
conda env create -f environment_mapanything.yml

# Recrear MeshFlow (mallas generativas por objeto — reemplazo de ShapeR)
conda env create -f environment_meshflow.yml
```

> [!NOTE]
> **MeshFlow** (Meta, CVPR 2026) reemplazó a ShapeR: consume la nube del
> segmento directamente (sin PKLs multi-vista, sin torchsparse, sin captions) y
> genera un GLB en ~12 s en la A100. Sus salidas son **assets visuales
> no-métricos** (`*_visual.glb`, `"metric": false` en meta.json). El env es solo
> el YAML + el checkpoint gated `facebook/meshflow` (HF_TOKEN) descargado a
> `vendor/meshflow/ckpt/meshflow/` (`config.yaml` + `model.pth`, 4.5 GB). Para
> condicionamiento por imagen de referencia hace falta además el checkpoint
> DINOv3 `dinov3_vitl16` en `/workspace/weights/dinov3/` (link firmado de Meta,
> expira a las ~48 h — la vía nube→malla no lo necesita).

Probablemente en `da3` y `sam3` Conda descargue temporalmente los runtimes de CUDA y un montón de *GBs* la primera vez, pero es todo automático. No tienes que configurar el path ni lidiar con librerías `nvccs` a mano.

## Paso 6: Compilar PotreeConverter C++ (Opcional)
Si por alguna razón al mudar el repositorio a Linux el binario `PotreeConverter` de la carpeta `vendor/` te da error de ejecución (a veces cambiar compilador de WSL a Linux puro rompe binarios C++), recompílalo usando el código fuente que ya tienes:
```bash
cd stac-builder/vendor/PotreeConverter
rm -rf build && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

¡Eso es todo! Reinstala los módulos de frontend (`cd ui && npm install`) y levanta el servidor normalmente. Tu nueva máquina volará.

## Paso 7: Arranque rápido (pod / remoto)

Para levantar todo de una en un pod nuevo (backend + UI + sesión claude, en tmux):

```bash
source /workspace/miniforge3/etc/profile.d/conda.sh
bash /workspace/stac-build/init_pod.sh
```

`init_pod.sh` crea tres sesiones tmux: `backend` (corre `scripts/start.sh` en el
env `da3`), `vite` (la UI) y `claude`. El backend escucha en el puerto 8765 (TLS
autofirmado).

> [!IMPORTANT]
> **Tokens y datos NO versionados.** Estos archivos están en `.gitignore` y hay
> que recrearlos/restaurarlos a mano en el pod nuevo:
> - `server/git.txt` — tokens de GitHub (`ghp_…`) y HuggingFace (`hf_…`). Recrealo con tus tokens.
> - `server/projects/**` — datos de escaneos/proyectos (GLB, PLY, npz). Restaurá desde tu backup.

## MeshFlow / TSDF

MeshFlow corre en el env `meshflow` (torch 2.8 cu126) lanzado on-demand por el
backend vía `server/run_meshflow.sh`; no requiere compilar extensiones CUDA.
El histórico de ShapeR (torchsparse/A100, checkpoints) quedó archivado en
`archive/` (env yml + pesos) y en la historia de git.
