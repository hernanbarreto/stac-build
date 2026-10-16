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
