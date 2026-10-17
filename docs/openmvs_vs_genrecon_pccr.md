# OpenMVS vs GenRecon on pccr — measured 2026-10-17

USER: "hay que probar el pipeline completo en su mejor versión de OpenMVS para darle la oportunidad de competir
contra GenRecon" and "todos hablan de que la mejor combinación es colmap+openmvs … nadie en serio profesionalmente
usa omega + genrecon, eso me preocupa". Four meshes of the same session, measured with the same instrument
(`server/tools/genrecon/measure_mesh.py`), published under pccr's `output/tsdf/` for his eye. Scripts and the
environment recipe: `server/tools/openmvs/` (README there). Session: pccr 2026-08-31, 295 keyframes of a NATIVE
464×832 60 fps phone video — every method received exactly the same images.

## The four meshes

| | `genrecon_cloud` (product) | `openmvs_colmap` | `openmvs_full` | `openmvs_cloud` |
|---|---|---|---|---|
| poses | ours, certified | **COLMAP 4.2 SfM, no prior** | ours, certified | ours, certified |
| geometry source | our certified cloud decides presence, surface generated | OpenMVS dense MVS (its own) | OpenMVS dense MVS (its own) | **our certified cloud** (4 M pts with visibility) |
| surface | GenRecon (Trellis.2, generated) | Delaunay graph-cut → RefineMesh → TextureMesh | same | same |
| texture | generated | photos (texrecon-style atlas) | photos | photos |
| faces / area | 3.2 M / 332 m² | 382 k / 76 m² | 346 k / 69 m² | 1.29 M / 159 m² |
| mesh area within 5 cm of the cloud | **95 %** | 42 % | 84 % | **96 %** |
| cloud → mesh p50 / p90 | 2.0 / 5.3 cm | 9.8 / 35.8 cm | 4.7 / 35.0 cm | **1.5 / 4.3 cm** |
| cloud within 5 cm of the mesh | 87 % | 25 % | 52 % | **92 %** |
| camera height over the floor, mesh − cloud (p10/p90) | 0 cm (−5/+5) | −5 cm (−18/+9) | −2 cm (−5/+3) | 0 cm (−5/+8) |
| ceiling over the camera, mesh − cloud | −2 cm | 0 cm | −1 cm | 0 cm |
| door#25 p50 (within 5 cm) | 2.4 cm (93 %) | 12.1 cm (10 %) | 11.5 cm (30 %) | 1.3 cm (99 %) |
| wall#109 | 1.1 cm (99 %) | 8.4 cm (11 %) | 4.4 cm (55 %) | 1.2 cm (99 %) |
| pipe#141 | 3.3 cm (73 %) | 23.3 cm (4 %) | 7.1 cm (36 %) | 2.4 cm (81 %) |
| duct#273 | 2.4 cm (94 %) | 4.2 cm (60 %) | 3.1 cm (80 %) | 2.1 cm (88 %) |
| fire_extinguisher#338 | 2.4 cm (92 %) | 10.0 cm (13 %) | 3.8 cm (69 %) | 1.6 cm (98 %) |
| cabinet#411 | 1.2 cm (99 %) | 6.6 cm (29 %) | 1.9 cm (81 %) | 1.2 cm (99 %) |
| desk#599 | 1.4 cm (98 %) | 29.9 cm (3 %) | 53.2 cm (2 %, missing) | 1.8 cm (93 %) |
| work_surface#1108 | 2.5 cm (83 %) | 6.5 cm (36 %) | 2.1 cm (93 %) | 1.7 cm (97 %) |
| licence | MIT | AGPL-3.0 | AGPL-3.0 | AGPL-3.0 |

The reference of every row is OUR certified cloud, which is not ground truth: a row says how much a mesh agrees
with it, and the camera-height / ceiling rows are the internal checks that do not depend on it (the walk's cameras
over the floor the mesh puts under them). A mesh built FROM the cloud (`openmvs_cloud`, `genrecon_cloud`) agrees
with it by construction; the two MVS variants are the independent instruments.

## The two independent instruments: COLMAP's SfM and OpenMVS's dense MVS

- **COLMAP SfM (no prior)**: registered 295/295, 49,941 sparse points, 1.05 px reprojection, focal 374/376 px
  (ours 388/385). Aligned to the certified trajectory by one Sim(3) over the camera centres (`align_colmap.py`):
  rms 12.3 cm, median 6.6 cm, max 30.5 cm, rotation median 4.35°. That disagreement lands in its mesh as the
  floor's ±18 cm and the 6–30 cm at every object: on a 464×832 phone video of textureless walls the classical
  SfM drifts along the walk, and nothing in the classical chain closes it (no loops, no floor, no segmentation).
  Published for orientation: `colmap_sparse` (point GLB, 49,009 points inside the cloud's box).
- **OpenMVS dense MVS with OUR poses** (`openmvs_full`'s cloud, 1.18 M points): where it measures, it agrees
  with our cloud to **p50 0.6 cm, 95 % within 5 cm** — two instruments that share nothing but the poses
  (PatchMatch photoconsistency vs Ω + F0–F6) land on the same surfaces. What it does NOT measure is half the
  scene: only 51 % of our cloud has a dense point within 5 cm (72 % within 10 cm) — textureless walls, the
  floor's plain tiles, the desk (missing from the mesh, 53 cm to the nearest face). With COLMAP's poses the
  same MVS agrees to p50 5.1 cm and covers 27 %.

## Reading

1. The classical chain as everybody runs it (COLMAP + OpenMVS) is not competitive on THIS input: its poses
   drift (12 cm rms), its dense cloud covers a quarter of the scene, its mesh sits 6–30 cm off the objects and
   holds 76 m² of a 332 m² scene. The pipelines people quote it from run on DSLR photo sets, not on a 0.4 Mpx
   handheld video; the Falkingham test the user linked shows the same chain beaten by Metashape on photos.
2. Our reconstruction (Ω + F0–F6 + closures + floor) is confirmed by the classical MVS where the latter can
   see at all: 0.6 cm. The coverage of textureless surfaces is exactly what the learned depth adds.
3. The best MESH by these numbers is `openmvs_cloud`: our certified cloud + OpenMVS's Delaunay graph-cut,
   photometric refinement and photo texture — 1.5 cm p50, 92 % of the cloud within 5 cm, every object within
   1.2–2.4 cm, real textures. GenRecon in cloud mode is close behind (2.0 cm, 87 %, 1.1–3.3 cm) with a generated
   texture, a closed and complete surface (332 m², ceilings and hidden sides) and an MIT licence. OpenMVS's
   Delaunay leaves a few spurious large triangles (1 % of the area beyond 50 cm from the cloud; bounds reach
   y −2.96 under the floor) that a trim would remove.
4. Decision for the user: OpenMVS is AGPL-3.0 (copyleft — a product decision, not a technical one). The
   meshing stage of the pipeline stays GenRecon (`StageId.GENRECON`); `openmvs_cloud` exists as a measured
   alternative, selectable by hand (`run_openmvs.sh <sid>:<scan> cloud`), never run by the pipeline.

## Cost on the A100 80 GB (pccr, 295 keyframes)

| step | time |
|---|---|
| COLMAP SfM (GPU SIFT, exhaustive matching, mapper) | 12 min |
| OpenMVS DensifyPointCloud (CUDA, full res, 2 geometric iterations) | 1.3 min (COLMAP poses) / 4.3 min (ours, 4 M-point scene) |
| ReconstructMesh → RefineMesh (CUDA) → TextureMesh | 1.5 min (MVS clouds) / 9 min (our 4 M-point cloud: 2.6 + 4.6 + 2) |
| GenRecon, mode cloud | ≈ 15 min, VRAM 8 GB |

Intermediates: `vendor/openMVS/work/` (4.2 GB: COLMAP database and models, OpenMVS scenes, depth maps, dense
clouds, meshes) — deletable, every published mesh carries its record in its `*.meta.json`.
