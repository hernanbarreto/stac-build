# The Gaussian-splat realism layer (branch `gs-layer`, 2026-10-31)

USER (2026-10-31): *"tenemos que tener el realismo de los GS con la capacidad de medir sobre malla
matemáticamente buena como la nuestra … en la etapa final, luego del mallado, con las mejores poses tras
correcciones, entrenar los GS y superponerlos. Sobre el modelo GS debe poder seleccionarse los objetos tal
como lo hacemos actualmente, incluir objetos mallados de otros lados (deberían venir con sus gaussians),
el humano, las pelotas, todo sobre la escena combinada GS + malla. La capa GS de momento es seleccionable
también, como la malla puede prenderse y apagarse. En un futuro debe ser propia junto con la malla el
entregable final. El nivel de detalle: lo mejor que podamos lograr con la tecnología actual."*

## The idea in one line

Two layers in ONE world frame: the certified cloud / mesh **measures**, the splats are **what is seen**.
A click on a splat selects the instance the segmentation named; a measurement goes to the mesh.

## What is built (code + tests, not yet run on a card)

| piece | where | what |
|---|---|---|
| config | `server/config.yaml` `gaussians:` + `pipeline.auto_gaussians` (false), `gaussians/config.py` | strict typed section: init (seed subsample), train (gsplat recipe, depth weight, exposure, held-out), export, label |
| training set | `gaussians/dataset.py` (env da3) | the SESSION as the chain left it: certified poses (`camera_poses.txt`), session camera (`camera.json` → native K + OPENCV distortion), the published cloud as SEEDS with colour, confidence and their CLOUD ROW, every k-th keyframe held out, the chain's per-keyframe depth (`segmentation.tsdf_export` loader, correction sidecar applied, the depth grid's own intrinsics) — written as plain arrays |
| trainer | `gaussians/train.py` (env `gs`: torch 2.8 cu128 + gsplat 1.5.3, Apache-2.0) | cameras FIXED; gaussians + one affine exposure per training keyframe; loss (1−w)·L1 + w·(1−SSIM) + depth L1 on the rendered expected depth where measured (normalised by the frame's median) + exposure pull to identity; gsplat default or MCMC densification; lens through gsplat's unscented projection; held-out PSNR/SSIM every `eval_every` and at the end; outputs `gaussians.ply` (3DGS), `gaussians.splat`, `seed_index.bin`, `report.json` |
| labels | `gaussians/label.py` | NO instance baked in (USER: the Segmentation Manager merges / creates / erases later): every gaussian carries its seed's cloud row (`seed_index.bin`, nearest seed within `label.nearest_radius_m`, NO_SEED beyond); the instance is `instance_ids.npy[row]` **live** |
| routes | `gaussians/api.py` (mounted in main.py) | `/api/gaussians/list/{sid}`, `/file/{sid}/{name}/{file}`, `/instances/{sid}/{name}` (uint32 per gaussian from the CURRENT instance_ids.npy; stale when the cloud's point count changed — said in the headers and the list) |
| worker / stage | `workers/gaussians_worker.py`, `pipeline_manager.StageId.GAUSSIANS` | after GENRECON, in SEGMENTATION_CHAIN and every cascade that invalidates the mesh; ON only with `pipeline.auto_gaussians` AND `gaussians.enabled` AND `auto_segment`; products `output/gaussians/<name>/` with `meta.json` (reconstruction id, epochs, cloud sha + point count) |
| tests | `tests/test_gaussians.py` | config strictness, PLY/.splat/seed-index records, the training set from a synthetic session (seeds are cloud points, deterministic, depth maps for training frames), live instances through the seed rows (merge, stale), the trainer's CPU parts (SSIM, PSNR, exposure, depth/grid pixel maps, SH), the worker's progress parsing, the stage wiring, the routes |

| viewer | `ui/src/components/GaussianLayer.ts`, Viewport handle (`reloadGaussians`, `setGaussiansVisible`, `setGaussianInstanceVisible`, `highlightGaussianInstance`, `refreshGaussianInstances`, `clearGaussians`), App, InstancesPanel ("Gaussian splats" section with on/off per model, stale badge, held-out PSNR, reload), AutosegmentDialog (the stage's checkbox, off by default), i18n en/es | Spark `@sparkjsdev/spark` 2.4.0 (MIT) inside the three.js scene: one `SparkRenderer`, one `SplatMesh` per model under the Potree octree group (the floor / display transform applies as to the cloud and the meshes), loaded when the cloud arrives and on demand. Per-splat control in an `RgbaArray` read by a dyno `objectModifier` in the splat shader: alpha = shown / hidden, red = highlighted — hiding an instance in the Instances list hides its splats, selecting one tints them; the instances come live from `/api/gaussians/instances`. Typecheck and `build:web` pass (the bundle grows by Spark) |

Not yet: the first pccr run (one GPU task at a time: after the audit and slam2 runs) and, with it, the
check by eye of the depth compositing of splats against meshes and of the display transform; a
`gaussians_ready` push so the list refreshes when the stage ends without reopening the session (today:
on session load and the section's reload button); external meshed objects sampled into gaussians.

## Decisions and their reasons

* **Poses and camera fixed.** If the splats moved the cameras the visual layer would part from the one
  that measures. gsplat's pose optimisation is never used.
* **Seeds from the certified cloud**, not from SfM points: 2 M measured seeds with colour and instance;
  densification only adds where the photometric error asks.
* **Depth supervision from the chain's maps** so the splats sit on the measured surface (a floater a few
  centimetres off the wall would break "I measure on the mesh, I see the splats").
* **Exposure per keyframe** (phone auto-exposure) — learned, pulled to identity; held-out frames are
  evaluated with identity (they have none).
* **Instances resolved live** through the seed row (see above) — the same `instance_ids.npy` the octree
  colours by; the Segmentation Manager's edits reach the splats with no retraining.
* **Licences**: gsplat Apache-2.0, Spark MIT. The Inria 3DGS reference code is non-commercial and is never
  vendored.
* **Determinism**: seeds fixed and recorded; gsplat's CUDA backward uses atomic adds — float-level, not
  bit-for-bit (declared in the report).

## How to run (when the card is free)

```
# the stage, through the pipeline: pipeline.auto_gaussians: true (or the Autosegment window) — or by hand:
/workspace/miniforge3/envs/da3/bin/python - <<'EOF'
from pathlib import Path; from workers import gaussians_worker  # builds the set, runs env gs, stamps meta
EOF
# env gs alone on a written training set:
/workspace/miniforge3/envs/gs/bin/python -m gaussians.train --trainset <out>/gaussians/gaussians/_trainset --out <out>/gaussians/gaussians --config <trainset>/train_config.json
```

Expected on pccr (295 keyframes, 464×832, A100): 15–25 min for 30 k iterations, 2–4 M gaussians, a
.splat of ~100 MB. The numbers in `report.json` (held-out PSNR / SSIM) are measured; whether it looks
right is Hernán's call in the viewer.
