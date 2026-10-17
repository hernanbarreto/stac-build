#!/bin/bash
# GenRecon over several sessions (USER 2026-10-16: "corré el resto de las sesiones"): vLLM handed over
# once, per session: adapter → reconstruct_scene → light bake (150 k faces / chunk, 2 k textures) → publish.
set -x
G=/workspace/stac-build/vendor/GenRecon; T=/workspace/stac-build/server/tools/genrecon
DA3=/workspace/miniforge3/envs/da3/bin/python; GR=/workspace/miniforge3/envs/genrecon/bin/python
export CUDA_HOME=/workspace/miniforge3/envs/genrecon OMP_NUM_THREADS=8 HF_HOME=${HF_HOME:-/workspace/hf_cache} HF_TOKEN=${HF_TOKEN:-$(cat ${HF_HOME:-/workspace/hf_cache}/token 2>/dev/null)}
cd /workspace/stac-build/server && $DA3 -c "
import sys; sys.path.insert(0, '/workspace/stac-build/server')
from workers.base import stop_semantic_service_verified
print(stop_semantic_service_verified(None, stage='GenRecon batch', log=print))" || exit 1
for spec in "$@"; do
  sid=${spec%%:*}; scan=${spec##*:}
  SES=/workspace/stac-build/server/projects/$sid/scans/$scan/src_default; W=$G/work/$sid; O=$G/work/${sid}_out; OL=$G/work/${sid}_out_light
  echo "=== SESSION $sid ($scan) $(date +%H:%M:%S)"
  (cd /workspace/stac-build/server && $DA3 $T/adapter.py $SES $W 600000) || { echo "adapter failed $sid"; continue; }
  rm -rf $O $OL
  (cd $G && $GR reconstruct_scene.py --mode Iphone --path $W --output_path $O \
     --ss_ckpt $G/checkpoints/ss_gen/ckpts/sparse_structure.pt --shape_ckpt $G/checkpoints/shape_gen/ckpts/shape_slat.pt --tex_ckpt $G/checkpoints/tex_gen/ckpts/texture_slat.pt \
     --num_imgs_per_scene 999 --chunk_size_factor 1.08 --stat_std_ratio 3.0 --radius_nb_points 7 --radius_m 0.2 \
     --pipeline_config configs/pipelines/texture.json --proj_batch_voxels 2048 --colmap_subdir colmap_mast3r) || { echo "reconstruct failed $sid"; continue; }
  (cd $G && $GR chunked_to_glb.py --inputs $O/to_glb_inputs.pt --chunk_inputs $O/chunk_inputs.pt --output_dir $OL --simplify_threshold 150000 --texture_size 2048) || { echo "bake failed $sid"; continue; }
  (cd /workspace/stac-build/server && $DA3 $T/publish.py $OL/scene.glb $SES genrecon_scene $W) || echo "publish failed $sid"
  echo "=== DONE $sid $(date +%H:%M:%S)"
done
cd /workspace/stac-build/server && $DA3 -c "
import sys; sys.path.insert(0, '/workspace/stac-build/server')
from semantic.service import ensure_service
print('vLLM restart:', ensure_service(timeout_s=0.0))" || true
echo "BATCH DONE"
