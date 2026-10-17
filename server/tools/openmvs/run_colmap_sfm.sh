#!/bin/bash
# The pure classical baseline (USER 2026-10-17: "todos hablan de que la mejor combinación es colmap+openmvs"):
# COLMAP 4.2 (CUDA) SfM on the session's keyframes — PINHOLE, one shared camera, exhaustive matching, mapper —
# then undistortion into an OpenMVS-ready workspace. Scale-free: aligned to our certified poses afterwards.
#   bash run_colmap_sfm.sh <sid>:<scan>
set -x
C="nice -n 10 taskset -c 0-15 /workspace/miniforge3/envs/colmap/bin/colmap"; V=/workspace/stac-build/vendor/openMVS
spec=$1; sid=${spec%%:*}; scan=${spec##*:}
SES=/workspace/stac-build/server/projects/$sid/scans/$scan/src_default; W=$V/work/${sid}_colmap
rm -rf $W; mkdir -p $W/images $W/sparse
# the same keyframes the pipeline used (every posed keyframe of the session)
/workspace/miniforge3/envs/da3/bin/python - <<PY
import sys; sys.path.insert(0, "/workspace/stac-build/server")
from pathlib import Path
from correction.session import load_session
s = load_session(Path("$SES") / "output"); n = 0
for f in s.frames:
    src = Path("$SES") / "frames" / f"{int(f):06d}.jpg"
    if src.exists(): (Path("$W") / "images" / src.name).symlink_to(src); n += 1
print(n, "keyframes linked")
PY
cd $W
$C feature_extractor --database_path db.db --image_path images --ImageReader.camera_model PINHOLE --ImageReader.single_camera 1 --FeatureExtraction.use_gpu 1 --SiftExtraction.max_num_features 8192 2>&1 | tail -3
$C exhaustive_matcher --database_path db.db --FeatureMatching.use_gpu 1 --FeatureMatching.guided_matching 1 2>&1 | tail -3
$C mapper --database_path db.db --image_path images --output_path sparse 2>&1 | tail -5
ls sparse
# the mapper may leave several models (fragments): the one that registered the most images is the reconstruction
best=""; nbest=0
for m in $(ls -d sparse/*/ 2>/dev/null); do
  n=$($C model_analyzer --path $m 2>&1 | grep -oE "Registered images: [0-9]+" | grep -oE "[0-9]+$"); echo "model $m: $n registered images"
  [ -n "$n" ] && [ "$n" -gt "$nbest" ] && { best=$m; nbest=$n; }
done
[ -n "$best" ] || { echo "no sparse model"; exit 1; }
$C model_analyzer --path $best 2>&1 | grep -E "Registered|Points|Observations|Mean track|Mean reprojection"
mkdir -p $W/sparse_txt; $C model_converter --input_path $best --output_path $W/sparse_txt --output_type TXT 2>&1 | tail -1
$C image_undistorter --image_path images --input_path $best --output_path $W/undistorted --output_type COLMAP 2>&1 | tail -2
$C model_converter --input_path $W/undistorted/sparse --output_path $W/undistorted/sparse --output_type TXT 2>&1 | tail -1
grep -c "" $W/undistorted/sparse/images.txt; head -4 $W/undistorted/sparse/cameras.txt
echo "=== COLMAP SFM DONE $sid $(date +%H:%M:%S)"
