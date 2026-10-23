# Vendor Inventory & Lockfile

The `vendor/` tree is **git-ignored** (large clones, compiled trees, gated
weights) — a fresh `git clone` lands **without** it. This file is the single
source of truth for what must live under `vendor/` and how each entry is
provisioned on a new machine.

**Restore the git-based vendors:** `bash scripts/setup_vendors.sh`
**List without touching anything:** `bash scripts/setup_vendors.sh --list`
**Model weights (separate):** `bash setup_weights.sh`

Runtime resolution of these paths is centralised in `server/vendor_paths.py`.

---

## 1. Git submodules — restored by `git submodule update --init --recursive`

Declared in `.gitmodules`, tracked as gitlinks.

| Dir | Upstream | Pin | Notes |
|-----|----------|-----|-------|
| `vendor/VGGT-Long` | `hernanbarreto/VGGT-Long` (STAC fork) | `01f6fa9` (branch `stac-g1-baseline`) | run-6 tree (ea178bc) + elastic taming + ownership backfill (+ hybrid DA3, config-off) |
| `vendor/depth-anything-3` | `hernanbarreto/Depth-Anything-3` (STAC fork, **PRIVATE**) | `52837d34` (`stac-main`) | cam-encoder pose conditioning + sky drop. Needs GitHub access to the private fork. |

`git clone --recursive https://github.com/hernanbarreto/stac-build.git` pulls
these in one step.

## 2. Pinned git clones — restored by `scripts/setup_vendors.sh`

Git-ignored plain clones (not submodules). The script clones each at the pinned
commit; already-present clones at the right commit are left untouched.

| Dir | Upstream | Pin | Used by |
|-----|----------|-----|---------|
| `vendor/r3d` | `facebookresearch/r3d` | `9669cacd` | **Reference only** — R3D was *ported* into `server/phase_r/` & `server/phase5_qa/`. Not imported at runtime; kept for provenance/diffing. |
| `vendor/sam31` | `facebookresearch/sam3` (`main`) | `5dd401d1` | SAM 3.1 Object Multiplex (`config.yaml models.segmentation.version: sam3.1`) + `server/patches/sam31_base_predictor_PATCHED.py` |
| `vendor/nvdiffrast` | `NVlabs/nvdiffrast` | `253ac4fc` | differentiable raster (texturing / render) |
| `vendor/mvs-texturing` | `nmoehrle/mvs-texturing` | `f3374298` | mesh texturing (C++, build from source) |
| `vendor/oneTBB-src` | `uxlfoundation/oneTBB` | `e9af1a1b` | TBB source → builds `vendor/oneTBB` |
| `vendor/vggt-omega` | `facebookresearch/vggt-omega` | `39a0cb8a` | optional VGGT-Ω backbone (weights below) |
| `vendor/mast3r` | `naver/mast3r` (+ submodule `naver/dust3r` branch `cvpr`, `1dce82d0`) | `f5209afc` | THE LOOP BRIDGE (audit 2026-10-23 phase 3d, `server/precision/loop_bridge.py`, `tracks.loop_bridge: mast3r`): wide-baseline matches across the two visits of every loop pair, verified by the session camera, entering F4 as queries on both sides. Checkpoint `MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth` (2.75 GB) fetched into torch hub's cache by `precision.tracks.verified_checkpoint`, sha256 `e28f91b4…8eb2` declared in config.yaml. **LICENCE: CC BY-NC-SA 4.0 (code and checkpoint, non-commercial)** — USER 2026-10-23: implemented regardless, the licence decided later; `tracks.loop_bridge: roma` is the permissive alternative. Python deps: `vendor/mast3r/requirements.txt` (einops, roma, huggingface_hub, torchvision) in the mapanything env. |
| `vendor/RoMa` | `Parskatt/RoMa` | `77f8d688` | the loop bridge's permissive backend (MIT; DINOv2 backbone Apache-2.0): `tracks.loop_bridge: roma` with its checkpoint URL + sha256 set in config.yaml. Not active by default. |
| `vendor/GenRecon` | `kasothaphie/GenRecon` (TUM, MIT) | `eaf1468` | generative scene meshing trial (USER 2026-10-16 "le damos, completo"): Trellis.2 fine-tuned per chunk, conditioned by posed images; env `genrecon` (py3.10, torch 2.6 cu126, flash-attn 2.7.3, nvdiffrast/nvdiffrec/CuMesh/FlexGEMM/o-voxel built), checkpoints `checkpoints/{sparse_structure,shape_slat,texture_slat}.pt` from kaldir.vc.cit.tum.de; input adapter in the session scratch (`work/<session>/colmap_mast3r` = our certified poses + cloud, Y-up → Z-up). NOT in the pipeline. **STAC patch** `server/tools/genrecon/genrecon_stac.patch` (2026-10-16 night, applied in-tree, re-apply with `git -C vendor/GenRecon apply`): `reconstruct_scene.py --coords_from_points/--coords_min_points/--coords_mode cloud|cloud+unseen` (the active voxels of every chunk from the session's PUBLISHED cloud; the model's sparse structure only where no keyframe saw through), `pipeline.run(coords_override, coords_hook)`, the chunker reads `levels.json` (the session's floor/ceiling levels instead of percentiles), the configs point at the pod's converted DINOv3. The clone is git-ignored; the patch + this row are its record. |
| ~~`vendor/openMVS`~~ | `cdcseacave/openMVS` (AGPL-3.0) | `610562a` | **REMOVED 2026-10-17 02:10** (USER: "todo lo que instalamos de colmap y openmvs se elimina" — the trial is closed, genrecon_photo won; `server/tools/openmvs/setup_env.sh` recreates env + clones + build if it is ever needed again). Was: **TRIAL 2026-10-17** (USER: "probar el pipeline completo en su mejor versión de OpenMVS para darle la oportunidad de competir contra GenRecon"): classical MVS meshing — DensifyPointCloud (CUDA) / ReconstructMesh / RefineMesh (CUDA) / TextureMesh — built in env `openmvs` (conda-forge Boost, Eigen, OpenCV, CGAL; `vendor/vcglib` headers, `cdcseacave/VCG`), scripts `server/tools/openmvs/` (export_colmap.py: our keyframes + certified poses + cloud with per-point visibility as a COLMAP text model; run_openmvs.sh full|cloud; publish_openmvs.py). AGPL is copyleft: a product decision before adoption. NOT in the pipeline. MASTER needs six libraries conda-forge lacks (rows below) and conda's ceres as the CPU build; the whole recipe is `server/tools/openmvs/setup_env.sh` (env `openmvs` 6 GB + binaries in `vendor/openMVS/_build/bin`, all git-ignored). Variants on pccr: `colmap` (COLMAP 4.2 SfM, no prior, Sim(3)-aligned by align_colmap.py), `full`, `cloud`. |
| ~~`vendor/vcglib`~~ | `cdcseacave/VCG` (GPL-3.0, headers) | `658ba36` | **REMOVED 2026-10-17 02:10** (USER: "todo lo que instalamos de colmap y openmvs se elimina" — the trial is closed, genrecon_photo won; `server/tools/openmvs/setup_env.sh` recreates env + clones + build if it is ever needed again). Was: OpenMVS's own VCG fork; headers only, nothing of ours links it. |
| ~~`vendor/halfmesh`~~ | `cdcseacave/halfmesh` (MIT) | `f218f7c` (v0.5.1) | **REMOVED 2026-10-17 02:10** (USER: "todo lo que instalamos de colmap y openmvs se elimina" — the trial is closed, genrecon_photo won; `server/tools/openmvs/setup_env.sh` recreates env + clones + build if it is ever needed again). Was: OpenMVS master's mesh library, the version its vcpkg port pins. |
| ~~`vendor/PoseLib`~~ | `cdcseacave/PoseLib` (BSD-3) | `ccdd2f6` | **REMOVED 2026-10-17 02:10** (USER: "todo lo que instalamos de colmap y openmvs se elimina" — the trial is closed, genrecon_photo won; `server/tools/openmvs/setup_env.sh` recreates env + clones + build if it is ever needed again). Was: the FORK OpenMVS's port pins (bearings API; upstream PoseLib and conda's 2.0.5 lack it). |
| ~~`vendor/TinyEXIF`~~ | `cdcseacave/TinyEXIF` (MIT) | `5445d04` | **REMOVED 2026-10-17 02:10** (USER: "todo lo que instalamos de colmap y openmvs se elimina" — the trial is closed, genrecon_photo won; `server/tools/openmvs/setup_env.sh` recreates env + clones + build if it is ever needed again). Was: OpenMVS SFM dependency. |
| ~~`vendor/TinyNPY`~~ | `cdcseacave/TinyNPY` (MIT) | `d36d3c3` | **REMOVED 2026-10-17 02:10** (USER: "todo lo que instalamos de colmap y openmvs se elimina" — the trial is closed, genrecon_photo won; `server/tools/openmvs/setup_env.sh` recreates env + clones + build if it is ever needed again). Was: OpenMVS SFM dependency. |
| ~~`vendor/tinyply`~~ | `ddiakopoulos/tinyply` (BSD-2 / public domain) | `c9bb690` | **REMOVED 2026-10-17 02:10** (USER: "todo lo que instalamos de colmap y openmvs se elimina" — the trial is closed, genrecon_photo won; `server/tools/openmvs/setup_env.sh` recreates env + clones + build if it is ever needed again). Was: halfmesh dependency. |
| ~~`vendor/thread-pool`~~ | `bshoshany/thread-pool` (MIT) | `bd4533f` | **REMOVED 2026-10-17 02:10** (USER: "todo lo que instalamos de colmap y openmvs se elimina" — the trial is closed, genrecon_photo won; `server/tools/openmvs/setup_env.sh` recreates env + clones + build if it is ever needed again). Was: one header (BS_thread_pool.hpp) copied into the env; halfmesh dependency. |
| ~~`vendor/ShapeR`~~ | — | — | **REMOVED 2026-10-17** (USER: "el botón meshing se va … eliminamos el entorno de shaper, eliminamos el shaper del código, y los otros que teníamos para mallado, el texrecon se queda por las dudas … el resto vuela todo"): per-object generative meshing (the Object button) — gone with the env `shaper`, the t5-v1_1-xl / CLIP / bert caches and `segmentation/shaper_export.py`. The scene mesh is GenRecon (`vendor/GenRecon`, workers/genrecon_worker.py) at the end of the segmentation chain. |
| ~~`vendor/point2cad`~~ | `prs-eth/point2cad` | — | **REMOVED 2026-10-16** (USER: "elimina p2c de todo el pipeline y borra el entorno, así nos sacamos uno con licencia de encima" — CC-BY-NC). The clone, the env `point2cad`, `segmentation/p2c_object.py`, the worker mode `p2c` and the Mesh stage are gone. |
| ~~`vendor/pgsr`~~ | — | — | **REMOVED 2026-10-17** (USER: "el botón meshing se va … eliminamos el entorno de shaper, eliminamos el shaper del código, y los otros que teníamos para mallado, el texrecon se queda por las dudas … el resto vuela todo"): the precision PGSR trainer and its per-object stage — gone with the env `pgsr`, `reconstruction/pgsr_*.py`, `workers/pgsr_worker.py` and the PGSR/TSDF pipeline stages. The scene mesh is GenRecon (`vendor/GenRecon`, workers/genrecon_worker.py) at the end of the segmentation chain. |

## 3. Non-git — weights / build trees / prebuilt (manual, documented)

Not clonable. Provision as noted; none is fetched by `git`.

| Dir | Type | How to provision |
|-----|------|------------------|
| `vendor/sam3` | code checkout, seg **ROLLBACK** baseline | Pinned to the **SAM 3.0 release** of `facebookresearch/sam3`. Weights: `bash setup_weights.sh sam3` → `weights/sam3`. NOT the default since 2026-08: the shipped default is SAM 3.1 (`vendor/sam31`, `config.yaml models.segmentation.version: sam3.1`); this checkout is the measured rollback path. |
| `vendor/cloudcompy` | compiled runtime tree (`bin/`,`lib/`) | Built install consumed by `vendor_paths.py`. `conda env create -f docs/migration/environment_CloudComPy310.yml`, then place the install here. |
| `vendor/CloudComPy310` | build/source tree | See `docs/migration/MIGRATION_GUIDE.md`. |
| `vendor/MapAnything2` | optional fallback | `conda env create -f docs/migration/environment_mapanything.yml`. |
| `vendor/PotreeConverter` | prebuilt binary + C++ source | Rebuild per `MIGRATION_GUIDE §6` (`cmake .. && make -j`) if the binary is missing/broken on a new arch. |
| `vendor/oneTBB` | compiled install | Built **from** `vendor/oneTBB-src` (`cmake` + `make install`). Platform-specific — never copy across archs. |
| `vendor/vggt-omega-weights` | gated HF weights (~4.3 GB) | Requires `HF_TOKEN`. See `setup_weights.sh` / README §weights. |

## 4. Removed / do-not-track

- `vendor/dinov3` — REAL CHECKOUT since 2026-09-04 (`facebookresearch/dinov3`,
  shallow clone): the dir was an EMPTY orphan gitlink; the 4-phase DINOv3 plan
  (feature scorer / pose_refine_fm / instance refine — server/reconstruction/
  dino_features.py) loads `dinov3.hub.backbones` DIRECTLY from it (hubconf
  drags torchmetrics deps the da3 env lacks). Weights: `weights/dinov3/`.
- `vendor/easy3d` — was an **orphan gitlink** (tracked mode
  160000 with no `.gitmodules` URL, empty on disk) that broke
  `git clone --recursive`. Untracked. DINOv3 weights (for meshflow reference
  conditioning) live under `weights/dinov3/`, not here.
- `vendor/DepthLM_Official`, `vendor/perception_models` — **dead `.gitmodules`
  entries** (declared but never tracked and absent on disk). Removed from
  `.gitmodules`. Re-add as real submodules if these deps come back.
- `vendor/pointdit` — SUBMODULE since 2026-10-04 (moved from third_party/ the same day: the repo's vendor convention) (`google-research/pointdit`, pinned
  `11f53a30aeb4d34d49d2beda655ee6f19b2ad7f8`, Apache-2.0): the monocular point-map DiT whose
  depth DETAIL refines the bent Omega maps inside f6_bend (claude_stac.txt 2026-10-04, flag
  `mono_detail.enabled`). Runs in env `da3` (torch 2.9.1 untouched; `torchmetrics` +
  `lightning-utilities` added with `--no-deps` for its torch.hub load of DINOv3). Untracked inside
  it: `third_party/dinov3` (clone of `facebookresearch/dinov3` @ `6876159a`, the hub code),
  `pretrained/pointdit{h,l}-512-mixdata-nodinov3-*.pth` (HF `haofeixu/pointdit`, sha256 prefixes
  cb01dd3b / 240c1a4f verified) and `pretrained/dinov3/dinov3_vit{h16plus,l16}_pretrain_lvd1689m-*.pth`
  (gated Meta weights, prefixes 7c1da9a5 / 8aa4cbdd verified, downloaded 2026-10-04 with the user's
  links). NOTE: `vendor/dinov3` and `weights/dinov3` named above are NOT on this pod any more
  (2026-10-04: neither directory exists); the DINOv3 weights live under PointDiT's `pretrained/dinov3`.

- `vendor/meshflow` — REMOVED 2026-10-04 (USER: "saca del vendor los pesos, del config y del código"): the
  checkout with its 4.3 GB `ckpt/`, the `meshflow:` config block, `run_meshflow_batch.py` / `run_meshflow.sh`,
  `segmentation/mesh_export.py` and the `engine: meshflow` branch. ShapeR is the one generative engine.
