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
| `vendor/VGGT-Long` | `hernanbarreto/VGGT-Long` (STAC fork) | `8dc61f2` | loop-closure + sky-removal + DA3-prior patches |
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
| `vendor/meshflow` | `facebookresearch/meshflow` | `55f56f60` | per-object generative meshes (replaced ShapeR). Needs gated ckpt `facebook/meshflow` → `vendor/meshflow/ckpt/meshflow/` (4.5 GB, HF_TOKEN) |
| `vendor/mvs-texturing` | `nmoehrle/mvs-texturing` | `f3374298` | mesh texturing (C++, build from source) |
| `vendor/oneTBB-src` | `uxlfoundation/oneTBB` | `e9af1a1b` | TBB source → builds `vendor/oneTBB` |
| `vendor/vggt-omega` | `facebookresearch/vggt-omega` | `39a0cb8a` | optional VGGT-Ω backbone (weights below) |
| `vendor/ShapeR` | `facebookresearch/ShapeR` | `d4402f55` | legacy per-object meshing (superseded by meshflow; kept for fallback) |

## 3. Non-git — weights / build trees / prebuilt (manual, documented)

Not clonable. Provision as noted; none is fetched by `git`.

| Dir | Type | How to provision |
|-----|------|------------------|
| `vendor/sam3` | code checkout, **DEFAULT** seg baseline | Pinned to the **SAM 3.0 release** of `facebookresearch/sam3`. Weights: `bash setup_weights.sh sam3` → `weights/sam3`. This is the default (`models.segmentation.version: sam3`). |
| `vendor/cloudcompy` | compiled runtime tree (`bin/`,`lib/`) | Built install consumed by `vendor_paths.py`. `conda env create -f docs/migration/environment_CloudComPy310.yml`, then place the install here. |
| `vendor/CloudComPy310` | build/source tree | See `docs/migration/MIGRATION_GUIDE.md`. |
| `vendor/MapAnything2` | optional fallback | `conda env create -f docs/migration/environment_mapanything.yml`. |
| `vendor/PotreeConverter` | prebuilt binary + C++ source | Rebuild per `MIGRATION_GUIDE §6` (`cmake .. && make -j`) if the binary is missing/broken on a new arch. |
| `vendor/oneTBB` | compiled install | Built **from** `vendor/oneTBB-src` (`cmake` + `make install`). Platform-specific — never copy across archs. |
| `vendor/vggt-omega-weights` | gated HF weights (~4.3 GB) | Requires `HF_TOKEN`. See `setup_weights.sh` / README §weights. |

## 4. Removed / do-not-track

- `vendor/dinov3`, `vendor/easy3d` — were **orphan gitlinks** (tracked mode
  160000 with no `.gitmodules` URL, empty on disk) that broke
  `git clone --recursive`. Untracked. DINOv3 weights (for meshflow reference
  conditioning) live under `weights/dinov3/`, not here.
- `vendor/DepthLM_Official`, `vendor/perception_models` — **dead `.gitmodules`
  entries** (declared but never tracked and absent on disk). Removed from
  `.gitmodules`. Re-add as real submodules if these deps come back.
