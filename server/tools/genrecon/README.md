# GenRecon trial (2026-10-16, USER: "le damos dale, completo")

Generative scene meshing (TUM, MIT; vendor/GenRecon @ eaf1468): Trellis.2 fine-tuned per 3.7 m chunk,
conditioned by the session's keyframes, its CERTIFIED poses and its published cloud. NOT in the pipeline;
a trial published as a scene mesh (`output/tsdf/genrecon_scene/`) for the user's eye. Generative:
presence and shape, never a measurement.

- `setup_env.sh` — conda env `genrecon` (py3.10, torch 2.6 cu126, flash-attn 2.7.3, nvdiffrast, nvdiffrec,
  CuMesh, FlexGEMM, o-voxel; + matplotlib open3d scipy hf_transfer, libegl from conda-forge).
- `download_checkpoints.sh` — the 3 fine-tuned checkpoints (13.7 GB) from kaldir.vc.cit.tum.de, 32 ranged
  connections (0.25 MB/s per connection otherwise); laid out as training runs under
  `vendor/GenRecon/checkpoints/{ss_gen,shape_gen,tex_gen}/{config.json, ckpts/}` (the loader derives
  `<run>/config.json` from the checkpoint path).
- `convert_dinov3.py` — the pod's DINOv3 ViT-L/16 (Meta's .pth) → transformers layout
  (`vendor/GenRecon/weights/dinov3-vitl16-hf`), the official key mapping, validated against HF's own
  reference outputs (1e-3); the HF repo is gated and the account is not in its list. The five GenRecon
  configs point at the local folder.
- `adapter.py <session_dir> <work_dir>` — keyframes (symlinked `rgb/`), `camera.json` OPENCV intrinsics,
  epoch-N c2w poses as COLMAP w2c (`colmap_mast3r/images.txt`), the cloud as `points3D.txt`; the world
  turned Y-up → Z-up (the chunker reads the height along z). The baker writes glTF Y-up back in the
  session's frame (verified against the cloud's bounds) — no inverse transform needed.
- `run_session.sh` — stops vLLM the pipeline's way (workers.base.stop_semantic_service_verified), runs
  `reconstruct_scene.py --mode Iphone` (the README's smartphone settings, all keyframes), then
  `chunked_to_glb.py` (the full bake: 2 M faces + 4k textures per chunk ≈ 3 min per chunk — for the viewer
  use `--simplify_threshold 150000 --texture_size 2048`, 30 s per chunk), then `ensure_service` for vLLM.
- `publish.py` — copies `scene.glb` into `output/tsdf/<name>/` with its meta (provenance vlm_generated).

pccr 2026-10-16 (epoch 4, 295 keyframes, 600 k points, 25 chunks of 3.72 m): generation 15 min on the
A100 (VRAM peak 22 GB, the three stages sampled jointly for the whole scene: 12 steps each at 7.8 / 10.8 /
5.4 s per step), raw mesh 40.9 M faces; light bake 13 min → 3.6 M faces, 304 MB.

## The cloud decides presence (2026-10-16 night, USER: "lo ideal es que copie la nube")

Measured on pccr epoch 4 with `measure_mesh.py`: the vendor's own run put only 8 % of its mesh area within 10 cm
of the cloud — the chunker anchors each cube at the 0.2-percentile of z of a 600 k subsample (the entrance floor)
and the model, trained with the floor at the cube's base, paints the whole floor there (34 cm under the walk) and
everything standing on it. Now (vendor patch `genrecon_stac.patch`, applied by `setup_env.sh`):

- `adapter.py` writes `cloud_genrecon.npy` (the WHOLE published cloud in GenRecon's frame) and `levels.json`
  (lowest support level / highest ceiling level from the segmentation; the chunker reads it instead of percentiles).
- `run_cloud.sh <sid>:<scan> [name] [min pts/voxel = 10] [cloud | cloud+unseen]` → `reconstruct_scene.py
  --coords_from_points --coords_mode`: the active voxels of every chunk (32³, 11.6 cm; the decoder refines to
  7 mm inside) are the cloud's occupied voxels. `cloud` skips the sparse-structure stage (VRAM 8 GB). `cloud+unseen`
  generates it too and ADDS its voxels only where no keyframe saw through them (the cloud's z-buffer per keyframe,
  min-filtered, half res, one voxel of tolerance; the record `coords_from_points.json` counts generated / vetoed /
  added per chunk) — completion of what no camera saw, never where a camera saw empty space.
- `measure_mesh.py <glb>`: the one record to compare variants (mesh area within bands of the cloud, the cloud
  against the mesh surface, the cameras' height over the floor under them, the ceiling, named objects).
- `scene_texture.py <glb_in> <glb_out> [max views = 80]`: the texture by PROJECTING the keyframes with the certified
  poses (texrecon per tile, the views chosen by what they see), replacing the generated one; write the meta by hand.
- `snap_mesh.py`: vertices within the run's voxel moved onto the cloud's local surface; not applied (2 cm p50 = the
  cloud's own noise).

pccr results: mode cloud 95 % of the mesh area within 5 cm of the cloud, 87 % of the cloud within 5 cm of the
mesh, floor 0 cm, objects 1–3 cm (`output/tsdf/genrecon_cloud`, textured by projection `genrecon_cloud_tex`).
