# ShapeR + TSDF — Setup en pod/máquina remota (A100)

Guía para dejar funcionando el entorno **`shaper`** (reconstrucción 3D generativa
ShapeR) y las dependencias del backend en un pod nuevo. El paso difícil es
**compilar `torchsparse` para la A100 (compute capability 8.0 / `sm_80`)**: el
wheel publicado no trae kernels para esa arquitectura y el build estándar falla
por un flag `-L` vacío que emite el toolchain de conda. Acá está la receta que
funciona.

> [!IMPORTANT]
> Los tokens (GitHub `ghp_…`, HuggingFace `hf_…`) viven en `server/git.txt`, que
> está en `.gitignore` y **NO se versiona**. Recrealo a mano en el pod nuevo con
> tus propios tokens. Nunca lo commitees.

---

## 1. Entorno conda `shaper`

```bash
cd docs/migration
conda env create -f environment_shaper.yml      # python 3.10, torch 2.7.1+cu128, flash-attn, torch-cluster
conda activate shaper
```

El YAML lista `torchsparse==2.1.0`, pero **ese paquete NO se instala con pip a
secas** — hay que compilarlo desde el fork parcheado (sección 2). Si tras crear
el env `torchsparse` quedó instalado pero roto ("no kernel image is available
for execution on the device"), desinstalalo y recompilalo:

```bash
pip uninstall -y torchsparse
```

## 2. Compilar `torchsparse` para A100 (`sm_80`)

El fork: `nihalsid/torchsparse@20ccc92a3adceef1d88f63227b8e1d01b9c7ebcc`.

```bash
conda activate shaper
git clone https://github.com/nihalsid/torchsparse /tmp/ts_src
cd /tmp/ts_src
git checkout 20ccc92a3adceef1d88f63227b8e1d01b9c7ebcc
```

### 2a. Parche del `setup.py` (flag `-L` vacío)

El toolchain de conda (`x86_64-conda-linux-gnu-c++`) emite un `-L` (y a veces
`-I`) **sin argumento** al final del link, y `g++` aborta con *"missing argument
to '-L'"*. Insertá este monkeypatch **al principio** de `setup.py`, justo
después de los `import`:

```python
# --- Fix the broken link command -------------------------------------------
# The conda toolchain emits a bare "-L"/"-I" with no path at the end of the link
# command, which makes g++ fail. Strip any standalone "-L"/"-I" token from every
# compiler/linker invocation. The CUDA kernels compile fine for sm_80; this only
# cleans the empty flag.
import distutils.ccompiler as _ccomp
_orig_spawn = _ccomp.CCompiler.spawn
def _clean_spawn(self, cmd, *args, **kwargs):
    cmd = [c for c in cmd if c not in ("-L", "-I")]
    try:
        return _orig_spawn(self, cmd, *args, **kwargs)
    except TypeError:
        return _orig_spawn(self, cmd)
_ccomp.CCompiler.spawn = _clean_spawn
# ---------------------------------------------------------------------------
```

### 2b. Compilar e instalar

`TORCH_CUDA_ARCH_LIST=8.0` es lo que fuerza los kernels para A100 (sm_80):

```bash
conda activate shaper
export TORCH_CUDA_ARCH_LIST=8.0
export FORCE_CUDA=1
pip install --no-build-isolation /tmp/ts_src
```

### 2c. Dependencias que `--no-deps` saltea

ShapeR/torchsparse importan dos paquetes que el build no arrastra:

```bash
pip install rootpath backports.cached_property
```

### 2d. Verificación

```bash
python - <<'EOF'
import torch, torchsparse
from torchsparse import SparseTensor
import torchsparse.nn.functional as F
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
# smoke test de conv3d sparse en GPU
coords = torch.randint(0, 16, (100, 4), dtype=torch.int32).cuda()
feats = torch.randn(100, 8).cuda()
x = SparseTensor(feats, coords)
print("torchsparse OK:", x.F.shape, x.F.device)
EOF
```

Si imprime `torchsparse OK` sin "no kernel image", quedó bien.

---

## 3. Modelos / checkpoints a descargar

Requieren el token HF en `server/git.txt` (o `huggingface-cli login`).

| Modelo | Repo HF | Usado por | Tamaño aprox |
|---|---|---|---|
| **InternVL3-8B** | `OpenGVLab/InternVL3-8B` | autocaption de objetos (`scene_analyzer.py`) | ~16 GB |
| **ShapeR ckpts** | se bajan solos vía `setup_checkpoints()` a `vendor/ShapeR/checkpoints/` | ShapeR | varios GB |
| **SAM3** | `facebook/sam3` (+ `sam3.1` si se usa) | segmentación | ~3 GB |

> [!NOTE]
> El backend carga InternVL3 con `local_files_only=True` (`scene_analyzer.py`), así
> que **debe estar pre-descargado** o el autocaption falla y cae al fallback (usa
> el label del objeto como caption). Para descargarlo:
> ```bash
> conda activate da3
> huggingface-cli download OpenGVLab/InternVL3-8B
> ```
> Si no se quiere el autocaption, se usa caption manual desde la UI — no requiere
> este modelo.

---

## 4. Arranque del pod

`init_pod.sh` (en la raíz del repo) levanta todo en sesiones tmux:

```bash
source /workspace/miniforge3/etc/profile.d/conda.sh
bash /workspace/stac-build/init_pod.sh
# sesiones: backend (scripts/start.sh, env da3), vite (UI), claude
```

Notas:
- El **backend** corre en el env **`da3`** (`scripts/start.sh` → `conda activate da3`).
- **ShapeR** se lanza como subproceso vía `server/run_shaper.sh`, que activa el env
  `shaper` y usa `exec python` para heredar el PID + `PDEATHSIG` (así muere si se
  mata el backend). Autodetecta `CONDA_ROOT` entre varias rutas comunes.
- El **TSDF de escena** y el **crop por objeto** corren dentro del backend (env
  `da3`, que tiene open3d + trimesh).

---

## 5. Notas de GPU / rendimiento

- ShapeR y la rectificación de imágenes corren en **GPU** (A100). Si ves la etapa
  de rectificación al 100% de CPU, falta el fix de `vendor/ShapeR/preprocessing/helper.py`
  (mover el unproject/project de fisheye a `cuda`).
- `HF_HOME=/workspace/hf_cache` (persistente) — definido en `scripts/start.sh`.
- `PYTHONUNBUFFERED=1` está en `start.sh` para que los `print()` de los workers
  (`[Shape]`, `[TSDF]`) salgan en tiempo real al log.
