# OpenMVS trial (2026-10-17) — classical MVS meshing measured against GenRecon

USER: "probar el pipeline completo en su mejor versión de OpenMVS para darle la oportunidad de competir contra
GenRecon" and "todos hablan de que la mejor combinación es colmap+openmvs … nadie en serio profesionalmente usa
omega + genrecon". NOT in the pipeline: a measured comparison on one session, published under the session's
`output/tsdf/` for his eye. OpenMVS is AGPL-3.0 (copyleft) — adopting it is a product decision, not a technical one.

## Variants (all on the same 295 keyframes of the session; pccr's video is a native 464×832 at 60 fps)

| name | poses | dense cloud | mesh |
|---|---|---|---|
| `openmvs_colmap` | COLMAP 4.2 SfM, no prior (PINHOLE, one camera, exhaustive matching), Sim(3)-aligned to the certified trajectory | OpenMVS DensifyPointCloud (CUDA, full res, every view, 2 geometric iterations) | ReconstructMesh → RefineMesh (CUDA) → TextureMesh |
| `openmvs_full` | our CERTIFIED poses (Ω + F0–F6 + closures) | OpenMVS DensifyPointCloud | same |
| `openmvs_cloud` | our certified poses | our CERTIFIED cloud (4 M of 19.4 M points, each with the keyframes that see it) | same |
| `genrecon_cloud` | our certified poses | our certified cloud decides presence | GenRecon (generated surface) — the product |
| `colmap_sparse` | COLMAP's own sparse points (49 k), aligned | — | a point GLB, for orientation only |

## Scripts

    bash setup_env.sh                                   # env `openmvs` + six source libs + OpenMVS master build (~25 min)
    bash run_colmap_sfm.sh <sid>:<scan>                 # the pure classical baseline: COLMAP SfM on the session's keyframes
    bash run_openmvs.sh  <sid>:<scan> colmap|full|cloud [name]   # OPENMVS_RESUME=1 keeps the work dir (dense maps reused)
    python ../genrecon/measure_mesh.py <glb> <session output dir> # the one record for every mesh (area within bands of
                                                                 # the cloud, cloud → mesh, camera height over the floor, objects)

`export_colmap.py` writes our keyframes + certified poses + cloud as a COLMAP text model (RAW frame, PINHOLE, every
point's track = the keyframes whose z-buffer sees it, measured keyframe first); `align_colmap.py` brings COLMAP's own
model into that frame by one Sim(3) over the camera centres and records the trajectories' disagreement;
`publish_openmvs.py` undoes the Z-up rotation of OpenMVS's glTF export (checked against `refined.ply`'s bounds) and
writes the mesh folder atomically.

## Gotchas met on the way (each cost real time — see CLAUDE.md 2026-10-17)

- COLMAP ≥ 3.12: `--SiftExtraction.use_gpu` → `--FeatureExtraction.use_gpu`, `--SiftMatching.*` → `--FeatureMatching.*`;
  the mapper may leave a 2-image fragment as model 0 (the script takes the model with the most registered images).
- OpenMVS MASTER is not the 2.3 release: nanoflann, ceres (the CPU conda build — the GPU one wants CUDA ≥ 12.9),
  tinyxml2, libheif, libjxl, tinygltf, fast_float, and from source TinyEXIF, TinyNPY, tinyply, the BS thread-pool
  header, halfmesh v0.5.1 (SHARED + PIC, or libMVS.so refuses its TLS relocation) and the PoseLib FORK
  `cdcseacave/PoseLib@ccdd2f6` (the bearings API); its `build/` folder holds its cmake modules (build in `_build/`);
  ceres's FindSuiteSparse needs OpenMP's C component (`enable_c.cmake` injected after `project()`); VCG is unused.
- A scene handed over in interface format (`scene.mvs`, `dense.mvs`) never gets a `mesh.mvs`: ReconstructMesh
  writes `mesh.ply` and the next stages take it with `--mesh-file`.
- The glTF export is rotated (x, y, z → x, z, −y) for a Y-up scene; `publish_openmvs.py` undoes it, verified.
