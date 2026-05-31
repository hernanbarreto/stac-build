#!/usr/bin/env python3
"""Fast cleaned_cloud from existing chunks — voxel downsample only, NO CloudCompy.
Reads chunk_*.ply, voxel-downsamples each (1.5cm), merges, final voxel, writes
cleaned_cloud.ply. For viewing now; origins/SOR skipped (redo properly later)."""
import sys, glob
from pathlib import Path
import numpy as np
import open3d as o3d

out = Path(sys.argv[1])
V = float(sys.argv[2]) if len(sys.argv) > 2 else 0.015
chunks = sorted(out.glob("chunk_*.ply"))
print(f"[quick] {len(chunks)} chunks, voxel={V*1000:.0f}mm", flush=True)
pts, cols = [], []
for i, c in enumerate(chunks):
    pc = o3d.io.read_point_cloud(str(c))
    d = pc.voxel_down_sample(V)
    pts.append(np.asarray(d.points)); cols.append(np.asarray(d.colors))
    print(f"[quick] {i+1}/{len(chunks)} {c.name}: {len(d.points):,} pts", flush=True)
P = np.vstack(pts); C = np.vstack(cols)
print(f"[quick] merged {len(P):,} → final voxel...", flush=True)
m = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
m.colors = o3d.utility.Vector3dVector(C)
m = m.voxel_down_sample(V)
dst = out / "cleaned_cloud.ply"
o3d.io.write_point_cloud(str(dst), m)
print(f"[quick] DONE: {dst} = {len(m.points):,} pts", flush=True)
