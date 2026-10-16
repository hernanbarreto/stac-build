#!/bin/bash
# GenRecon on pccr (epoch 4): the whole card (vLLM stopped the way the pipeline does), then the
# chunked GLB, then the scene back into our Y-up frame and published for the viewer.
set -x
G=/workspace/stac-build/vendor/GenRecon
W=$G/work/pccr; O=$G/work/pccr_out
DA3=/workspace/miniforge3/envs/da3/bin/python
GR=/workspace/miniforge3/envs/genrecon/bin/python
cd /workspace/stac-build/server && $DA3 -c "
import sys; sys.path.insert(0, '/workspace/stac-build/server')
from workers.base import stop_semantic_service_verified
print(stop_semantic_service_verified(None, stage='GenRecon on pccr', log=print))" || exit 1
nvidia-smi --query-gpu=memory.used --format=csv,noheader
cd $G && export CUDA_HOME=/workspace/miniforge3/envs/genrecon OMP_NUM_THREADS=8 HF_HOME=${HF_HOME:-/workspace/hf_cache} HF_TOKEN=${HF_TOKEN:-$(cat ${HF_HOME:-/workspace/hf_cache}/token 2>/dev/null)} && \
$GR reconstruct_scene.py --mode Iphone --path $W --output_path $O \
  --ss_ckpt $G/checkpoints/ss_gen/ckpts/sparse_structure.pt --shape_ckpt $G/checkpoints/shape_gen/ckpts/shape_slat.pt --tex_ckpt $G/checkpoints/tex_gen/ckpts/texture_slat.pt \
  --num_imgs_per_scene 999 --chunk_size_factor 1.08 --stat_std_ratio 3.0 --radius_nb_points 7 --radius_m 0.2 \
  --pipeline_config configs/pipelines/texture.json --proj_batch_voxels 2048 --colmap_subdir colmap_mast3r --save_imgs
RC=$?
echo "reconstruct exit $RC"
if [ $RC -eq 0 ]; then
  $GR chunked_to_glb.py --inputs $O/to_glb_inputs.pt --chunk_inputs $O/chunk_inputs.pt --output_dir $O
  echo "glb exit $?"
fi
# the chat service back (the pipeline's own path: ensure_service)
cd /workspace/stac-build/server && $DA3 -c "
import sys; sys.path.insert(0, '/workspace/stac-build/server')
from semantic.service import ensure_service
print('vLLM restart:', ensure_service(timeout_s=0.0))" || true
echo "RUN DONE"
