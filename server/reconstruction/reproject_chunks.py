"""
Re-project the backbone chunk point clouds to the BA-refined poses.
================================================================================
The bundle adjustment (run_colmap_ba) refines camera_poses.txt, but the cloud is
built by CloudCompPy from the per-chunk PLYs whose points were baked at the ORIGINAL
poses. Without this step the cloud stays at old poses while the TSDF integrates at the
refined poses → cloud ≠ mesh. This transforms every chunk point by its source frame's
pose delta  (X' = refined_c2w[f] · old_c2w[f]⁻¹ · X)  so the cloud is rebuilt at the
refined poses → cloud and TSDF are consistent.

Runs AFTER run_colmap_ba (which left the old poses in camera_poses.txt.precolmapba)
and BEFORE CloudCompPy. Only the backbone chunks are touched (chunk_0..N); the LiDAR
(999) / dense-fusion (998) chunks use other pose sources and are skipped.

    python -m reconstruction.reproject_chunks --output-dir <out>
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict

import numpy as np

logger = logging.getLogger("ReprojChunks")


def _poses(path: Path, frames: Path) -> Dict[int, np.ndarray]:
    mats = [np.array([float(v) for v in ln.split()], np.float64).reshape(4, 4)
            for ln in path.read_text().splitlines() if len(ln.split()) == 16]
    nums = [int(float(x)) for x in frames.read_text().split()]
    if len(nums) != len(mats):
        return {}
    return {nums[i]: mats[i] for i in range(len(mats))}


def run(output_dir: Path) -> int:
    cf = output_dir / "camera_frames.txt"
    pp = output_dir / "camera_poses.txt"
    bak = output_dir / "camera_poses.txt.precolmapba"
    if not (cf.exists() and pp.exists() and bak.exists()):
        logger.warning("missing camera_poses.txt / .precolmapba / camera_frames.txt — "
                       "no BA backup to re-project against; skipping")
        return 0
    old = _poses(bak, cf)
    new = _poses(pp, cf)
    if not old or not new:
        logger.warning("poses/frames not aligned — skipping re-projection")
        return 0
    # per-frame delta = refined_c2w @ inv(old_c2w)
    delta: Dict[int, np.ndarray] = {}
    for f in old:
        if f in new:
            delta[f] = new[f] @ np.linalg.inv(old[f])

    from plyfile import PlyData, PlyElement
    n_chunks = 0
    for ply in sorted(output_dir.glob("chunk_*.ply")):
        stem = ply.stem
        if stem.startswith(("chunk_997", "chunk_998", "chunk_999")):
            continue  # densify-fillers (already at refined poses) / dense-fusion / LiDAR
        # NO FALLBACK: a backbone chunk MUST be re-projectable — missing origins, a length
        # mismatch or a read error means the cloud would stay at old poses (inconsistent with
        # the TSDF), so we raise instead of silently leaving it.
        orig = output_dir / f"{stem}_origins.npz"
        if not orig.exists():
            raise FileNotFoundError(f"{ply.name}: no {orig.name} — cannot re-project this chunk")
        fg = np.load(orig)["frame_global"].astype(np.int64)
        pd = PlyData.read(str(ply))
        v = pd["vertex"].data                          # structured array
        xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
        if len(fg) != len(xyz):
            raise ValueError(f"{ply.name}: origins ({len(fg)}) != points ({len(xyz)})")
        moved = 0
        for f in np.unique(fg):
            d = delta.get(int(f))
            if d is None:
                continue                               # a point from a frame the BA didn't touch
            m = fg == f
            xyz[m] = xyz[m] @ d[:3, :3].T + d[:3, 3]
            moved += int(m.sum())
        v["x"] = xyz[:, 0].astype(v["x"].dtype)
        v["y"] = xyz[:, 1].astype(v["y"].dtype)
        v["z"] = xyz[:, 2].astype(v["z"].dtype)
        PlyData([PlyElement.describe(v, "vertex")],
                text=False, byte_order="<").write(str(ply))
        n_chunks += 1
        logger.info(f"  re-projected {ply.name}: {moved:,}/{len(xyz):,} pts")
    logger.info(f"✅ re-projected {n_chunks} backbone chunks to refined poses "
                f"(cloud now consistent with the BA / TSDF)")
    return n_chunks


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    a = ap.parse_args()
    n = run(Path(a.output_dir))
    if n == 0:                            # NO FALLBACK: BA ran but no chunk was re-projected → fail
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
