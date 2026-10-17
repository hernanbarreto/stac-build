#!/bin/bash
# GenRecon with the active voxels FROM THE CLOUD (USER 2026-10-16: "lo ideal es que copie la nube"): adapter →
# reconstruct_scene --coords_from_points (the sparse-structure stage skipped, the chunk cubes on the session's
# floor/ceiling levels) → light bake → publish. vLLM is NOT touched here: the caller frees the card.
#   bash run_cloud.sh <sid>:<scan> [mesh name, default genrecon_photo] [coords_min_points, default 10] [cloud | cloud+unseen | cloud+objects | cloud+hull]
# THE PRODUCT (USER 2026-10-17: "genrecon_photo debe ser la única publicada final"): the generated mesh is published as <name>_raw,
# the keyframes are baked into its own atlas (photo_albedo.py, material kept) and published as <name>; the raw mesh moves to
# output/genrecon/ (kept for re-bakes, outside the viewer's list).
set -x
G=/workspace/stac-build/vendor/GenRecon; T=/workspace/stac-build/server/tools/genrecon
DA3=/workspace/miniforge3/envs/da3/bin/python; GR=/workspace/miniforge3/envs/genrecon/bin/python
export CUDA_HOME=/workspace/miniforge3/envs/genrecon OMP_NUM_THREADS=8 HF_HOME=${HF_HOME:-/workspace/hf_cache} HF_TOKEN=${HF_TOKEN:-$(cat ${HF_HOME:-/workspace/hf_cache}/token 2>/dev/null)}
spec=$1; name=${2:-genrecon_photo}; raw=${name}_raw; minpts=${3:-10}; mode=${4:-cloud}; sid=${spec%%:*}; scan=${spec##*:}
tag=cloud; [ "$mode" = "cloud+unseen" ] && tag=fill; [ "$mode" = "cloud+objects" ] && tag=obj; [ "$mode" = "cloud+hull" ] && tag=hull
SES=/workspace/stac-build/server/projects/$sid/scans/$scan/src_default; W=$G/work/$sid; O=$G/work/${sid}_${tag}_out; OL=$G/work/${sid}_${tag}_out_light
echo "=== SESSION $sid ($scan) $(date +%H:%M:%S)"
(cd /workspace/stac-build/server && $DA3 $T/adapter.py $SES $W 600000) || { echo "adapter failed $sid"; exit 1; }
EXTRA=""
if [ "$mode" = "cloud+hull" ]; then
  # completion presence from the SAM3 masks (hull_points.py); the generator runs in mode cloud with these extra points
  (cd /workspace/stac-build/server && $DA3 $T/hull_points.py $SES $W) || { echo "hull failed $sid"; exit 1; }
  EXTRA="--coords_extra_points $W/hull_points.npy"; mode=cloud
fi
rm -rf $O $OL
(cd $G && $GR reconstruct_scene.py --mode Iphone --path $W --output_path $O \
   --ss_ckpt $G/checkpoints/ss_gen/ckpts/sparse_structure.pt --shape_ckpt $G/checkpoints/shape_gen/ckpts/shape_slat.pt --tex_ckpt $G/checkpoints/tex_gen/ckpts/texture_slat.pt \
   --num_imgs_per_scene 999 --chunk_size_factor 1.08 --stat_std_ratio 3.0 --radius_nb_points 7 --radius_m 0.2 \
   --pipeline_config configs/pipelines/texture.json --proj_batch_voxels 2048 --colmap_subdir colmap_mast3r \
   --coords_from_points $W/cloud_genrecon.npy --coords_min_points $minpts --coords_mode $mode $EXTRA) || { echo "reconstruct failed $sid"; exit 1; }
(cd $G && $GR chunked_to_glb.py --inputs $O/to_glb_inputs.pt --chunk_inputs $O/chunk_inputs.pt --output_dir $OL --simplify_threshold ${GENRECON_SIMPLIFY:-150000} --texture_size ${GENRECON_TEX:-2048}) || { echo "bake failed $sid"; exit 1; }
(cd /workspace/stac-build/server && $DA3 $T/publish.py $OL/scene.glb $SES $raw $W "sparse structure FROM THE CLOUD (coords_from_points, min $minpts pts/voxel, mode $mode), chunk cubes on the session's floor/ceiling levels") || { echo "publish failed $sid"; exit 1; }
[ -f $SES/output/tsdf/$raw/$raw.glb ] || { echo "publish failed $sid"; exit 1; }
(cd /workspace/stac-build/server && $DA3 $T/photo_albedo.py $SES $SES/output/tsdf/$raw/$raw.glb $name ${GENRECON_TOPK:-3}) || { echo "bake failed $sid"; exit 1; }
mkdir -p $SES/output/genrecon && rm -rf $SES/output/genrecon/$raw && mv $SES/output/tsdf/$raw $SES/output/genrecon/
echo "=== DONE $sid $(date +%H:%M:%S)"
