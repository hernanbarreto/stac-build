"""LoGeR poses + PointDiT depth maps + DA3 metric scale — a SEPARATE reconstruction path
(USER 2026-10-06). It runs beside the vggtomega backend and changes nothing of it.

    python -m loger_pdit.run --session <scan dir> [--stride N]

Steps (each resumable, each its own process so the GPU is freed between them):
  1. loger      LoGeR* on the keyframes in walk order → poses, local depth, confidence, K
  2. da3        DA3 metric depth per keyframe (isolated, per frame) → the metric scale
  3. pointdit   PointDiT per keyframe → depth maps up to an affine map
  4. fuse       one global metric scale, a per-frame affine fit of PointDiT to LoGeR's depth,
                unprojection with LoGeR's poses, voxel cleaning → cloud.ply
Outputs under <scan>/output/loger_pdit/.
"""
