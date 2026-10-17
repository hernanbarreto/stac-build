#!/bin/bash
# OpenMVS trial (USER 2026-10-17: "probar el pipeline completo en su mejor versión de OpenMVS para darle la
# oportunidad de competir contra GenRecon"). Three variants on one session, all published for the viewer and
# measured by server/tools/genrecon/measure_mesh.py:
#   full    its OWN dense cloud from the keyframes + our CERTIFIED poses (DensifyPointCloud, CUDA, full resolution,
#           every view, geometric consistency) → ReconstructMesh → RefineMesh (CUDA) → TextureMesh
#   cloud   our CERTIFIED cloud (with the keyframes that see every point) → ReconstructMesh → RefineMesh → TextureMesh
#   colmap  the pure classical chain: COLMAP's own SfM poses (run_colmap_sfm.sh, aligned to our frame by
#           align_colmap.py) → the same dense chain as `full`
#   bash run_openmvs.sh <sid>:<scan> [full|cloud|colmap] [mesh name]      (OPENMVS_RESUME=1 keeps the work dir: dense maps are reused)
set -x
V=/workspace/stac-build/vendor/openMVS; B=$V/_build/bin; E=/workspace/miniforge3/envs/openmvs; T=/workspace/stac-build/server/tools/openmvs
DA3=/workspace/miniforge3/envs/da3/bin/python
export LD_LIBRARY_PATH=$E/lib:$B:$LD_LIBRARY_PATH OMP_NUM_THREADS=16
spec=$1; variant=${2:-cloud}; name=${3:-openmvs_$variant}; sid=${spec%%:*}; scan=${spec##*:}
SES=/workspace/stac-build/server/projects/$sid/scans/$scan/src_default
echo "=== OPENMVS $variant $sid ($scan) $(date +%H:%M:%S)"
if [ "$variant" = "colmap" ]; then
  W=$V/work/${sid}_colmap; A=$W/aligned; O=$W/out_colmap
  [ -f $A/sparse/points3D.txt ] || (cd /workspace/stac-build/server && $DA3 $T/align_colmap.py $SES $W/undistorted/sparse $W/undistorted/images $A) || { echo "align failed"; exit 1; }
  W=$A
else
  W=$V/work/$sid; O=$W/out_$variant
  [ -f $W/sparse/points3D.txt ] || { (cd /workspace/stac-build/server && $DA3 $T/export_colmap.py $SES $W 4000000 8) && mv $W/colmap $W/sparse; } || { echo "export failed"; exit 1; }
fi
[ "${OPENMVS_RESUME:-0}" = 1 ] || rm -rf $O; mkdir -p $O; cd $O
[ -f scene.mvs ] || $B/InterfaceCOLMAP -i $W -o scene.mvs --image-folder $W/images -w $O 2>&1 | tail -5 || { echo "interface failed"; exit 1; }
if [ "$variant" = "full" ] || [ "$variant" = "colmap" ]; then
  # OpenMVS's own dense cloud at its best: full resolution, all views, geometric iterations, colours + normals
  [ -f dense.mvs ] || $B/DensifyPointCloud scene.mvs -o dense.mvs -w $O --resolution-level 0 --max-resolution 4096 --min-resolution 640 \
     --number-views 0 --number-views-fuse 3 --geometric-iters 2 --iters 4 --estimate-colors 2 --estimate-normals 2 \
     --filter-point-cloud 0 --postprocess-dmaps 1 --gpu-device 0 --max-threads 16 2>&1 | tail -8 || { echo "densify failed"; exit 1; }
  IN=dense.mvs
else
  IN=scene.mvs
fi
# the surface: visibility-aware Delaunay + graph cut; no decimation, spikes and spurious pieces removed, small holes closed.
# master keeps the scene in interface format and writes the MESH as a file: the next stages take it with --mesh-file
$B/ReconstructMesh $IN -o mesh.ply -w $O --decimate 1 --remove-spurious 20 --remove-spikes 1 --close-holes 30 --smooth 2 \
   --free-space-support 0 --quality-factor 1 --thickness-factor 1 --max-threads 16 2>&1 | tail -6 || { echo "reconstruct failed"; exit 1; }
# photometric refinement at full resolution on the card
$B/RefineMesh $IN -m mesh.ply -o refined.ply -w $O --resolution-level 0 --max-views 8 --scales 2 --scale-step 0.5 --decimate 1 \
   --close-holes 30 --ensure-edge-size 1 --max-face-area 64 --regularity-weight 0.2 --gpu-device 0 --max-threads 16 2>&1 | tail -6 || { echo "refine failed"; exit 1; }
# the photographic texture, global + local seam levelling, straight to GLB
$B/TextureMesh $IN -m refined.ply -o $name.glb -w $O --export-type glb --resolution-level 0 --global-seam-leveling 1 --local-seam-leveling 1 \
   --max-texture-size 8192 --empty-color 0 --max-threads 16 2>&1 | tail -6 || { echo "texture failed"; exit 1; }
ls -la $O/*.glb $O/*.ply 2>/dev/null | head
(cd /workspace/stac-build/server && $DA3 $T/publish_openmvs.py $O/$name.glb $SES $name "$variant") || { echo "publish failed"; exit 1; }
echo "=== DONE $variant $sid $(date +%H:%M:%S)"
