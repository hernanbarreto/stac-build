#!/bin/bash
# GenRecon with the active voxels FROM THE CLOUD (USER 2026-10-16: "lo ideal es que copie la nube"): adapter →
# reconstruct_scene --coords_from_points (the sparse-structure stage skipped, the chunk cubes on the session's
# floor/ceiling levels) → light bake → publish. vLLM is NOT touched here: the caller frees the card.
#   bash run_cloud.sh <sid>:<scan> [mesh name, default genrecon_photo] [coords_min_points, default 10] [cloud | cloud+unseen | cloud+objects | cloud+hull | generative]
# THE PRODUCT (USER 2026-10-17: "genrecon_photo debe ser la única publicada final"): the generated mesh is published as <name>_raw
# into output/genrecon/ (kept for re-bakes, outside the viewer's list), the keyframes are baked into its own atlas
# (photo_albedo.py, material kept) and published as <name>.
# GENERATIVE (USER 2026-11-02: "veamos cómo sale generativa y no generativa, que aparezcan las dos variantes"): mode
# `generative` = GenRecon's own sparse structure — the model decides presence from the keyframes, the cloud only lays
# out the chunks — published as <name> with the texture the model generated, no photo bake ("en la generativa no,
# porque no va a ser exactamente igual").
# CUBES (USER 2026-11-02): GENRECON_CUBE_M > 0 = cubes of that side stacked from the lowest level to the cloud's top
# (genrecon_levels.stack_layers; the worker passes genrecon.cube_size_m), 0 = GenRecon's one layer floor to ceiling.
# TILES (USER 2026-11-02: "hay que implementar el tema de los tiles … para que pueda ser manejable una malla grande"):
# GENRECON_TILES=1 (the worker passes genrecon.tiles) cuts the published mesh into 3D Tiles beside it,
# output/tsdf/<name>/tiles/ (tools/glb/tile_mesh.mjs): the viewer streams them instead of the whole GLB. The old
# tileset leaves BEFORE the new mesh is published — a tileset never outlives the mesh it was cut from.
set -x
G=/workspace/stac-build/vendor/GenRecon; T=/workspace/stac-build/server/tools/genrecon
DA3=/workspace/miniforge3/envs/da3/bin/python; GR=/workspace/miniforge3/envs/genrecon/bin/python
NODE=/workspace/miniforge3/envs/nodejs/bin/node; GLB=/workspace/stac-build/tools/glb
export CUDA_HOME=/workspace/miniforge3/envs/genrecon OMP_NUM_THREADS=8 HF_HOME=${HF_HOME:-/workspace/hf_cache} HF_TOKEN=${HF_TOKEN:-$(cat ${HF_HOME:-/workspace/hf_cache}/token 2>/dev/null)}
spec=$1; name=${2:-genrecon_photo}; raw=${name}_raw; minpts=${3:-10}; mode=${4:-cloud}; sid=${spec%%:*}; scan=${spec##*:}
tag=cloud; [ "$mode" = "cloud+hull+unseen" ] && tag=full; [ "$mode" = "cloud+unseen" ] && tag=fill; [ "$mode" = "cloud+objects" ] && tag=obj; [ "$mode" = "cloud+hull" ] && tag=hull; [ "$mode" = "generative" ] && tag=gen
SES=/workspace/stac-build/server/projects/$sid/scans/$scan/src_default; W=$G/work/$sid; O=$G/work/${sid}_${tag}_out; OL=$G/work/${sid}_${tag}_out_light
echo "=== SESSION $sid ($scan) $(date +%H:%M:%S)"
tile_published() {
  [ "${GENRECON_TILES:-0}" = "1" ] || return 0
  local d=$SES/output/tsdf/$1
  (cd $GLB && $NODE tile_mesh.mjs $d/$1.glb $d/tiles --jobs 4) || { echo "tiles failed $sid"; exit 1; }
}
# GENRECON_CONF (USER 2026-11-05): the viewer's confidence slider for this session, 0..1 — GenRecon receives the points the
# viewer shows at that value (0 = every point); the cubes cover the whole of what is kept
CONF=${GENRECON_CONF:-0}
ADAPT="$SES $W 600000 ${GENRECON_CUBE_M:-0} $CONF $minpts ${GENRECON_SOLID_GROUND:?genrecon.solid_ground (the worker sets it)}"
EXTRA=""; HULL=0
if [ "$mode" = "cloud+hull" ] || [ "$mode" = "cloud+hull+unseen" ]; then
  # completion presence from the SAM3 masks (hull_points.py, run below after the adapter); the generator runs with
  # these extra points — in mode cloud (cloud+hull), or with the unseen hook on top (cloud+hull+unseen, USER 2026-11-08
  # "¿podremos combinarlos?": the hull's voxels join the cloud's before the hook, which then fills the holes the
  # cameras looked into)
  HULL=1
  EXTRA="--coords_extra_points $W/hull_points.npy"
  [ "$mode" = "cloud+hull" ] && mode=cloud || mode=cloud+unseen
fi
COORDS="--coords_from_points $W/cloud_genrecon.npy --coords_min_points $minpts --coords_mode $mode $EXTRA"
# USER 2026-11-12 ("la geometría debe ser correcta, exactamente igual a la nube lisa y correcta con detalle" / "el piso
# es una lámina así debe ser igual que las paredes"): GENRECON_SHAPE_FROM_CLOUD=1 (genrecon.shape_from_cloud) — the SHAPE
# is the cloud's, written in GenRecon's cells (one sheet per surface) and encoded by its shape encoder
# (tools/genrecon/cloud_shape.py); GENRECON_SHAPE_REFINE_T (genrecon.shape_refine_t) > 0 lets the shape flow continue
# from it, guided by the photos. Then the loose floating pieces go (tools/genrecon/drop_loose.py) and the bake keeps the
# single sheets: no global remesh (it closes every sheet into a shell), GenRecon's own smoothing (genrecon.smooth_iters)
if [ "${GENRECON_SHAPE_FROM_CLOUD:?genrecon.shape_from_cloud (the worker sets it)}" = "1" ] && [ "$mode" = "cloud" ]; then
  COORDS="$COORDS --shape_from_cloud --shape_refine_t ${GENRECON_SHAPE_REFINE_T:?genrecon.shape_refine_t (the worker sets it)}"
fi
NOTE="sparse structure FROM THE CLOUD (coords_from_points, min $minpts pts/voxel, mode $mode, the points at confidence >= $CONF of the viewer's slider), cubes over the whole cloud"
if [ "$mode" = "generative" ]; then
  COORDS=""
  NOTE="sparse structure GENERATED by the model from the keyframes (no voxel from the cloud, which lays out the chunks), chunk cubes on the session's floor/ceiling levels, the generated texture"
fi
RECON="reconstruct_scene.py --mode Iphone --path $W --output_path $O \
   --ss_ckpt $G/checkpoints/ss_gen/ckpts/sparse_structure.pt --shape_ckpt $G/checkpoints/shape_gen/ckpts/shape_slat.pt --tex_ckpt $G/checkpoints/tex_gen/ckpts/texture_slat.pt \
   --num_imgs_per_scene 999 --chunk_size_factor 1.08 --stat_std_ratio 3.0 --radius_nb_points 7 --radius_m 0.2 \
   --pipeline_config configs/pipelines/texture.json --proj_batch_voxels 2048 --colmap_subdir colmap_mast3r \
   $COORDS"
# RESUME AT THE MESH STEP (USER 2026-11-12: "la regla es que se continúa desde el punto de falla"; pccr 2026-08-24's mesh
# step failed after ~40 min of reconstruction and every launch began again at the adapter): the reconstruction's products
# carry the stamp of what they are made from (resume.py: the session files the adapter reads, the adapter's and the
# vendor's reconstruction code, this order). The same stamp and the products whole = the adapter and reconstruct_scene.py
# are skipped and the run continues HERE, the mesh step's own folder emptied (its chunk cache belongs to the mesh-step
# code that wrote it); any difference is named and the run starts at the adapter, as always. The hull modes never
# resume (their extra points come from the mask store, outside the stamp). GENRECON_DRY_RESUME=1: say whether this
# order would resume and stop — nothing is run or deleted.
RESUME=0
if [ $HULL -eq 0 ]; then
  (cd /workspace/stac-build/server && $DA3 $T/resume.py check $SES $W $O --conf $CONF --adapter "$ADAPT" --reconstruct "$RECON"); rc=$?
  if [ $rc -eq 0 ]; then RESUME=1; elif [ $rc -ne 3 ]; then echo "resume check failed $sid"; exit 1; fi
fi
if [ "${GENRECON_DRY_RESUME:-0}" = "1" ]; then echo "=== DRY RESUME $sid: resume=$RESUME (nothing run)"; exit 0; fi
if [ $RESUME -eq 1 ]; then
  echo "=== RESUME $sid at the mesh step $(date +%H:%M:%S)"
  rm -rf $OL
else
  (cd /workspace/stac-build/server && $DA3 $T/adapter.py $ADAPT) || { echo "adapter failed $sid"; exit 1; }
  if [ $HULL -eq 1 ]; then
    (cd /workspace/stac-build/server && $DA3 $T/hull_points.py $SES $W) || { echo "hull failed $sid"; exit 1; }
  fi
  rm -rf $O $OL
  # $T alone on GenRecon's path (never the server tree over the vendor's modules): inference/get_images.py imports
  # remap_gpu from it when the worker set STAC_UNDISTORT_DEVICE (genrecon.undistort_device)
  (cd $G && PYTHONPATH=$T${PYTHONPATH:+:$PYTHONPATH} $GR $RECON) || { echo "reconstruct failed $sid"; exit 1; }
  if [ $HULL -eq 0 ]; then
    (cd /workspace/stac-build/server && $DA3 $T/resume.py seal $SES $W $O) || { echo "resume stamp failed $sid"; exit 1; }
  fi
fi
INPUTS=$O/to_glb_inputs.pt; BAKE_FLAGS=""
if [ "$GENRECON_SHAPE_FROM_CLOUD" = "1" ] && [ "$mode" = "cloud" ]; then
  (cd /workspace/stac-build/server && $DA3 $T/drop_loose.py $O/to_glb_inputs.pt $W $O/to_glb_inputs_clean.pt) || { echo "drop loose failed $sid"; exit 1; }
  INPUTS=$O/to_glb_inputs_clean.pt
  BAKE_FLAGS="--skip_remesh --smooth_iters ${GENRECON_SMOOTH_ITERS:?genrecon.smooth_iters (the worker sets it)}"
fi
# GENRECON_SIMPLIFY 0 (USER 2026-11-05: "no simplificar cada cubo — la malla guarda todo el detalle, los tiles la muestran
# liviana"): no decimation target. GENRECON_BAKE_WORKERS > 1: the per-chunk bakes (UV atlas + texture, one CPU thread a
# chunk) in that many processes — the pre-bake meshes dumped, the workers bake them into the chunk cache, the merge
# resumes from it (bake_chunks_parallel.py: the loop of inference/chunked_glb.py, chunk for chunk)
SIMP=${GENRECON_SIMPLIFY:-150000}; [ "$SIMP" = "0" ] && SIMP=2000000000
NW=${GENRECON_BAKE_WORKERS:-1}
if [ "$NW" -gt 1 ]; then
  (cd $G && $GR chunked_to_glb.py --inputs $INPUTS --chunk_inputs $O/chunk_inputs.pt --output_dir $OL --dump_geometry_plys $BAKE_FLAGS) || { echo "dump failed $sid"; exit 1; }
  pids=""
  for k in $(seq 0 $((NW - 1))); do
    (cd $G && $GR $T/bake_chunks_parallel.py $INPUTS $OL $OL/chunks $k $NW $SIMP ${GENRECON_TEX:-2048}) &
    pids="$pids $!"
  done
  for p in $pids; do wait $p || { echo "bake worker failed $sid"; exit 1; }; done
fi
(cd $G && $GR chunked_to_glb.py --inputs $INPUTS --chunk_inputs $O/chunk_inputs.pt --output_dir $OL --simplify_threshold $SIMP --texture_size ${GENRECON_TEX:-2048} $BAKE_FLAGS) || { echo "bake failed $sid"; exit 1; }
rm -f $OL/chunk_*_geom.ply $OL/global_mesh.ply
rm -rf $SES/output/tsdf/$name/tiles $SES/output/tsdf/$name/tiles.part
if [ "$mode" = "generative" ]; then
  (cd /workspace/stac-build/server && $DA3 $T/publish.py $OL/scene.glb $SES $name $W "$NOTE") || { echo "publish failed $sid"; exit 1; }
  [ -f $SES/output/tsdf/$name/$name.glb ] || { echo "publish failed $sid"; exit 1; }
  tile_published $name
  echo "=== DONE $sid $(date +%H:%M:%S)"
  exit 0
fi
# the raw mesh never enters output/tsdf/ (the viewer loads every folder that appears there): output/genrecon/
rm -rf $SES/output/genrecon/$raw $SES/output/tsdf/$raw
(cd /workspace/stac-build/server && $DA3 $T/publish.py $OL/scene.glb $SES $raw $W "$NOTE" genrecon) || { echo "publish failed $sid"; exit 1; }
[ -f $SES/output/genrecon/$raw/$raw.glb ] || { echo "publish failed $sid"; exit 1; }
(cd /workspace/stac-build/server && $DA3 $T/photo_albedo.py $SES $SES/output/genrecon/$raw/$raw.glb $name ${GENRECON_TOPK:-3}) || { echo "bake failed $sid"; exit 1; }
tile_published $name
echo "=== DONE $sid $(date +%H:%M:%S)"
