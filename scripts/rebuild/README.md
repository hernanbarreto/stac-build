# Rebuilding the pod on an empty volume

USER 2026-11-18 (RunPod announced a maintenance of the pod's server): *"Prepara todo en .sh
para que se creen automáticamente los entornos se descarguen todos los modelos y se vuelva a armar en un nuevo
volumen en cuestión de poco tiempo … de forma completa y sin errores"*.

Everything here was written FROM the pod that works and installs **the exact files it holds**, not "the latest".

## Run it

A pod of the same template (Ubuntu 24.04, CUDA 12.8.1 **devel**, python 3.11 in `/usr/bin`; the image is, by what
the pod shows, RunPod's PyTorch 2.8 / CUDA 12.8.1 one — copy its exact name from the pod's page before deleting
anything), the volume mounted at `/workspace`:

```bash
nano /workspace/.env                 # the four secrets — see env.template
source /workspace/.env
git clone https://$GIT_USER:$GIT_TOKEN@github.com/hernanbarreto/stac-build.git /workspace/stac-build
git -C /workspace/stac-build remote set-url origin https://github.com/hernanbarreto/stac-build.git
cp /workspace/.env /workspace/stac-build/.env
bash /workspace/stac-build/scripts/rebuild/rebuild.sh
bash /workspace/stac-build/init_pod.sh           # the services, as on every boot
```

About an hour (measured step by step on the pod's local disk, 2026-11-18: an env's conda packages 4–50 s, its pip
packages 80–160 s, the compiled extensions 0.5–7 min an env, the vendor clones 5 min; 78 GB of weights at the
pod's 100+ MB/s; a network volume writes slower). A failure stops only what depends on it; **run the same command
again** and every finished step is skipped. `rebuild.sh --list` shows the steps, `--only STEP` runs one,
`--verify` checks an installed pod against the locks. Logs: `/workspace/.stac_rebuild/logs/<step>.log`.

Running it on the pod that already works changes nothing: an env it did not build is left alone, a vendor tree at
its commit and a weight with the right sha256 are kept.

## What it restores

| | from | exactness |
|---|---|---|
| the ten conda envs | `locks/<env>.conda-explicit.txt` (URL + md5), `locks/<env>.wheels.tsv` (URL + sha256) | the same packages, the same files |
| what pip built from a directory | `steps_envs.sh` — the editable vendor trees, GenRecon's five extensions at pinned commits, AMB3R's native builds | the same sources; compiled again |
| a file patched inside an env | `overlays/<env>/` (uniception's DINOv2 loader in `mapanything`) | byte for byte |
| the vendor trees | `steps_vendors.sh` — commits; `bundles/` — our commits that are on no remote; the patches | the same commits and diffs |
| the model weights | `models.tsv` (77 entries, 78.5 GB: source, pinned revision, sha256, size) | sha256-checked |
| the octree converter, oneTBB, KTX, CloudComPy's binary, node packages, the USD venv, tailscale / cloudflared | `steps_builds.sh`, `steps_vendors.sh`, `steps_envs.sh` | built or downloaded again |

## What it does NOT restore — copy these yourself

- **The sessions**: `server/projects/` (videos, clouds, segmentations, meshes) and **the users' database**
  `server/data/` (stac.db, personnel, teams). Data, not software.
- **`.env`** (the tokens) and Tailscale's node identity (`bash scripts/tailscale_up.sh` authorises a new node).
- **Meta's two DINOv3 checkpoints when `DINOV3_URL` has expired** (it had on 2026-11-18): PointDiT and GenRecon
  need them. Ask Meta for a new link, put it in `.env`, run `rebuild.sh` again — or keep a copy of the two files
  (4.6 GB, `vendor/pointdit/pretrained/dinov3/`) and point `STAC_MIRROR` at a folder holding them named by their
  sha256.
- `texrecon` (`vendor/mvs-texturing`, kept "por las dudas"): cloned, not built — it was relinked by hand.

## What was checked on 2026-11-18, and what was not

Checked, in a scratch tree on the pod's local disk, never touching the live install:
- **every pip package of the nine python envs** (1,060) downloaded and compared with the installed one: the same
  files byte for byte, but for gsplat in the unused env `gs` (compiled on the pod; the rebuild takes PyPI's wheel);
- **every conda package** (1,080 URLs) still published; **every weight source** asked for its size and sha256
  (`models.py remote`): 65 of 67 give the manifest's file — the two that do not are DINOv3's (the expired link);
  the 77 entries verified against the pod's own files (`models.py verify`);
- **the envs `da3`, `mapanything`, `voice`, `amb3r`, `genrecon`, `colmap`, `nodejs` rebuilt from nothing** and
  verified (`verify_env.py`): every package identical or built; the backend imports from a fresh clone with the
  rebuilt `da3` (159 routes, as with the live env);
- **the twelve vendor trees** cloned from nothing: the same commits and local diffs; KTX's and CloudComPy's binary
  trees byte-identical; a fresh clone's submodule update FAILS on depth-anything-3 (its last commit is on no
  remote) and the bundle repairs it;
- **the rebuilt PotreeConverter** gives the same `octree.bin`, `hierarchy.bin` and `metadata.json` as the pod's on
  bufferStop's cloud (3.84 M points); **GenRecon's DINOv3 conversion** with the rebuilt env gives the same file;
  one GenRecon checkpoint (2.86 GB) fetched from TUM's server: identical;
- **`rebuild.sh` itself** end to end on a subset (two envs, one model): phases, background steps, stamps.

NOT checked: the whole run on a really empty volume; the envs `semantic`, `CloudComPy310` and `gs` were not
rebuilt (their packages were all checked, as above); the 78 GB were not downloaded again in full; nothing was
started on the card from a rebuilt env.

Known differences of a rebuilt pod: PotreeConverter's binary has another sha256 (same output) — it is part of
every octree's stamp, so a restored session's octree is converted again when asked; compiled extensions are
other builds of the same sources.

## Keeping it true

After any `pip install` / `conda install` in an env: `bash scripts/rebuild/capture.sh <env>` and commit
`locks/` (`CHECK=1` also downloads every wheel and compares it with the installed one, ~12 min for all). The
audit names a file edited in place inside an env (`locks/<env>.modified.tsv`): save it under `overlays/<env>/`.
A new weight is a new row of `models.tsv`; a new vendor tree a row of `steps_vendors.sh`. A commit made in a
vendor clone must be pushed to a remote of ours or bundled into `bundles/`.

The older scripts (`scripts/setup_vendors.sh`, `scripts/setup_pod_envs.sh`, `setup_weights.sh`, `export_envs.sh`,
`scripts/setup.sh`, `docs/migration/`) predate this and name removed envs and old pins: this directory replaces
them.
