"""Replay applied corrections from the ledger — bit-faithful, no slerp
recompute.

``python -m correction.replay --session <session_dir> --to-epoch N
[--cloud <ply>] [--poses <txt>] [--out <dir>]``

Rebuilds the cloud and poses at epoch N by applying the persisted
``corrections/epoch_<i>.npz`` transforms of N's ANCESTRY (a rejected run has no
npz and never gets an epoch; since 2026-09-16 a correction can also run on top
of a re-selected older epoch, which branches the history — replay follows the
`parent_epoch` line to N and ignores the epochs of another branch) starting
from an epoch-0 cloud/poses pair. Transforms are
keyed by REAL frame numbers, so the same ledger can be re-applied to a
re-reconstruction of the same scene whose keyframe set differs: keyframes are
matched by frame_global, unmatched frames inherit interpolation implicitly
through their stored per-keyframe entries.

This is also the verification path: the replay test asserts that replaying to
the current epoch reproduces the session's cloud and poses within float
tolerance.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

_SERVER_DIR = str(Path(__file__).resolve().parents[1])
if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)

from correction.ledger import load_epoch_npz  # noqa: E402
from correction.epoch import epoch_lineage  # noqa: E402
from correction.session import read_ply, read_poses, write_ply, \
    write_poses  # noqa: E402
from segmentation.session_io import _load_frame_index_map  # noqa: E402


def apply_epoch_to_arrays(xyz: np.ndarray, fg: np.ndarray,
                          poses: np.ndarray, frames: list,
                          npz: dict) -> None:
    """Apply one epoch's per-keyframe transform in place, re-keyed by real
    frame number (frame_global)."""
    kf_by_frame: Dict[int, int] = {int(f): i
                                   for i, f in enumerate(npz["frames"])}
    R_kf, t_kf, k_kf = npz["R_kf"], npz["t_kf"], npz["k_kf"]
    b_kf = (npz["b_kf"] if "b_kf" in getattr(npz, "files", npz.keys())
            else np.zeros(len(k_kf)))
    # cameras first? No: depth expansion needs the PRE-transform camera
    # centres, which are the current poses — transform points, then cameras.
    cam_center = {}
    cam_axis = {}
    for i, f in enumerate(frames):
        j = kf_by_frame.get(int(f))
        if j is not None:
            cam_center[int(f)] = poses[i][:3, 3].copy()
            cam_axis[int(f)] = poses[i][:3, 2].copy()
    for j_src, f in enumerate(npz["frames"]):
        sel = np.where(fg == int(f))[0]
        if not len(sel):
            continue
        kv = float(k_kf[j_src])
        bv = float(b_kf[j_src])
        if kv != 1.0 or bv != 0.0:
            cam = cam_center.get(int(f))
            if cam is None:
                raise RuntimeError(
                    f"replay: frame {f} carries a depth correction but the "
                    f"target session has no pose for it — cannot re-apply "
                    f"the depth expansion")
            if bv != 0.0:
                z = (xyz[sel] - cam) @ cam_axis[int(f)]
                zc = np.where(np.abs(z) > 1e-9, z, 1e-9)
                xyz[sel] = cam + (xyz[sel] - cam) * ((kv * z + bv) / zc)[:, None]
            else:
                xyz[sel] = cam + (xyz[sel] - cam) * kv
        xyz[sel] = xyz[sel] @ R_kf[j_src].T + t_kf[j_src]
    for i, f in enumerate(frames):
        j = kf_by_frame.get(int(f))
        if j is None:
            continue
        R, t = R_kf[j], t_kf[j]
        poses[i][:3, :3] = R @ poses[i][:3, :3]
        poses[i][:3, 3] = R @ poses[i][:3, 3] + t


def replay(session_dir: Path, to_epoch: int,
           cloud_path: Optional[Path] = None,
           poses_path: Optional[Path] = None,
           out_dir: Optional[Path] = None, log=print) -> dict:
    """Replay epochs 1..to_epoch onto an epoch-0 cloud/poses pair. By default
    reads the epoch-0 artifacts from ``out_dir`` inputs the caller provides;
    writes ``replay_cloud.ply`` + ``replay_camera_poses.txt`` into out_dir
    (default: <output>/replay/)."""
    session_dir = Path(session_dir)
    output_dir = session_dir if (session_dir / "corrections").exists() \
        else session_dir / "output"
    cloud_path = Path(cloud_path) if cloud_path else \
        output_dir / "cleaned_cloud.ply"
    poses_path = Path(poses_path) if poses_path else \
        output_dir / "camera_poses.txt"
    out_dir = Path(out_dir) if out_dir else output_dir / "replay"
    out_dir.mkdir(parents=True, exist_ok=True)

    header, data = read_ply(cloud_path)
    if "frame_global" not in (data.dtype.names or ()):
        raise RuntimeError(f"{cloud_path} lacks frame_global — replay needs "
                           f"provenance")
    xyz = np.stack([data["x"], data["y"], data["z"]],
                   axis=1).astype(np.float64)
    fg = data["frame_global"].astype(np.int64)
    poses = read_poses(poses_path)
    frames = _load_frame_index_map(output_dir)
    if not frames or len(frames) != len(poses):
        raise RuntimeError("camera_frames.txt does not match the pose file — "
                           "cannot key the replay by real frame numbers")

    lineage = epoch_lineage(output_dir, int(to_epoch))
    for ep in lineage[1:]:            # lineage[0] is epoch 0, the starting point
        npz = load_epoch_npz(output_dir, ep)
        apply_epoch_to_arrays(xyz, fg, poses, frames, npz)
        log(f"  replayed epoch {ep} "
            f"({int((npz['k_kf'] != 1.0).sum())} depth keyframe(s))")

    data_out = data.copy()
    data_out["x"] = xyz[:, 0].astype(data.dtype["x"])
    data_out["y"] = xyz[:, 1].astype(data.dtype["y"])
    data_out["z"] = xyz[:, 2].astype(data.dtype["z"])
    cloud_out = out_dir / "replay_cloud.ply"
    poses_out = out_dir / "replay_camera_poses.txt"
    write_ply(cloud_out, header, data_out)
    write_poses(poses_out, poses)
    log(f"  replay written: {cloud_out}, {poses_out}")
    return {"cloud": str(cloud_out), "poses": str(poses_out),
            "epochs_applied": len(lineage) - 1, "lineage": lineage}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", required=True,
                    help="session dir (or its output/ dir)")
    ap.add_argument("--to-epoch", type=int, required=True)
    ap.add_argument("--cloud", default=None,
                    help="epoch-0 cloud (default: the session's current "
                         "cleaned_cloud.ply — use an epoch-0 copy for a "
                         "true replay)")
    ap.add_argument("--poses", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    replay(Path(args.session), args.to_epoch,
           cloud_path=args.cloud, poses_path=args.poses, out_dir=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
