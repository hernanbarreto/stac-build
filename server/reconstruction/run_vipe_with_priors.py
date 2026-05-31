#!/usr/bin/env python3
"""
Run ViPE as the universal refinement engine, fed with EXTERNAL per-frame PRIORS
(metric depth + camera poses) — instead of pure RGB. ViPE natively consumes these
as constraints: METRIC_DEPTH enters the bundle adjustment as a per-frame depth
prior (anchors metric scale CONTINUOUSLY → kills scale drift on long runs) and POSE
seeds the trajectory (smooths chunk-handoff jumps via the global BA).

The priors are produced by a "source" step and are source-agnostic:
  - da3   : depth + poses extracted from DA3 streaming (NOT its chunks).
  - stray : depth (LiDAR) + poses (ARKit).  [future]

Prior layout (what this script reads), all keyed by ORIGINAL frame number:
  <priors>/depth/{frame:06d}.npy        float32 meters (any res; resized to frame)
  <priors>/poses.txt                    one row-major 4x4 c2w (16 floats) per line
  <priors>/frames.txt                   the frame number for each poses.txt line

We DO NOT patch the ViPE vendor: we build the stream with ViPE's own
AssignAttributesProcessor and only relax the one init assertion (METRIC_DEPTH) on
the pipeline INSTANCE at runtime. optimize_intrinsics is forced off (required by
ViPE when a metric-depth prior is present); GeoCalib still estimates intrinsics.

Runs in ViPE's .venv.

    vendor/vipe/.venv/bin/python server/reconstruction/run_vipe_with_priors.py \
        --frames-dir <frames> --priors <out>/vipe_priors \
        --output <out>/vipe_run --pipeline dav3
"""
import argparse
import types
from pathlib import Path

import cv2
import numpy as np
import torch

from vipe import make_pipeline
from vipe.config import parse_typed_config
from vipe.streams.base import (
    AssignAttributesProcessor,
    FrameAttribute,
    ProcessedVideoStream,
)
from vipe.streams.frame_dir_stream import FrameDirStream
from vipe.utils.geometry import se3_matrix_to_se3
from vipe.utils.logging import configure_logging


def _patched_add_init_processors(self, video_stream):
    """Identical to DefaultAnnotationPipeline._add_init_processors but WITHOUT the
    METRIC_DEPTH assertion — we deliberately inject a metric-depth prior. GeoCalib
    still runs (intrinsics), and POSE is allowed (it is not asserted upstream)."""
    from vipe.pipeline.default import GeoCalibIntrinsicsProcessor, TrackAnythingProcessor
    init_processors = []
    assert FrameAttribute.INTRINSICS not in video_stream.attributes()
    assert FrameAttribute.CAMERA_TYPE not in video_stream.attributes()
    assert FrameAttribute.INSTANCE not in video_stream.attributes()
    init_processors.append(GeoCalibIntrinsicsProcessor(video_stream, camera_type=self.camera_type))
    if self.init_cfg.instance is not None:
        init_processors.append(
            TrackAnythingProcessor(
                self.init_cfg.instance.phrases,
                add_sky=self.init_cfg.instance.add_sky,
                sam_run_gap=int(video_stream.fps() * self.init_cfg.instance.kf_gap_sec),
            )
        )
    return ProcessedVideoStream(video_stream, init_processors)


def _load_depth(p: Path) -> np.ndarray | None:
    if p.with_suffix(".npy").exists():
        return np.load(p.with_suffix(".npy")).astype(np.float32)
    if p.with_suffix(".npz").exists():
        z = np.load(p.with_suffix(".npz"))
        return z["depth"].astype(np.float32) if "depth" in z else None
    return None


def _build_priors(frames_dir: Path, priors: Path, H: int, W: int, inject_poses: bool):
    """Return (poses_list, depths_list) aligned to FrameDirStream's sorted order.
    Entries are None where a prior is missing. If inject_poses is False, poses stay
    None (monocular: ViPE computes poses freely; we only anchor metric via depth)."""
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
    files = sorted({f for e in exts for f in
                    list(frames_dir.glob(f"*{e}")) + list(frames_dir.glob(f"*{e.upper()}"))})

    # poses keyed by frame number
    pose_by_frame: dict[int, np.ndarray] = {}
    ptxt, ftxt = priors / "poses.txt", priors / "frames.txt"
    if ptxt.exists() and ftxt.exists():
        P = [np.array(list(map(float, ln.split())), np.float64).reshape(4, 4)
             for ln in ptxt.read_text().splitlines() if ln.strip()]
        F = [int(x) for x in ftxt.read_text().split()]
        pose_by_frame = {fr: P[i] for i, fr in enumerate(F) if i < len(P)}

    depth_dir = priors / "depth"
    poses, depths = [], []
    n_pose = n_depth = 0
    for f in files:
        fr = int(f.stem)
        c2w = pose_by_frame.get(fr)
        if inject_poses and c2w is not None:
            poses.append(se3_matrix_to_se3(torch.from_numpy(c2w).float()))
            n_pose += 1
        else:
            poses.append(None)
        d = _load_depth(depth_dir / f"{fr:06d}") if depth_dir.exists() else None
        if d is not None:
            if d.shape != (H, W):
                d = cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)
            d = np.where(np.isfinite(d) & (d > 0), d, 0.0).astype(np.float32)
            depths.append(torch.from_numpy(d).cuda())
            n_depth += 1
        else:
            depths.append(None)
    print(f"[vipe-priors] {len(files)} frames | poses={n_pose} "
          f"({'injected' if inject_poses else 'OFF — ViPE computes freely'}) | "
          f"depth={n_depth}", flush=True)
    return poses, depths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True, type=Path)
    ap.add_argument("--priors", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--pipeline", default="dav3")
    ap.add_argument("--inject-poses", action="store_true",
                    help="Inject DA3/ARKit poses as a prior (stray/lidar). Omit for "
                         "monocular (da3): only depth anchors metric, ViPE solves poses.")
    args = ap.parse_args()

    logger = configure_logging()
    overrides = [
        f"pipeline={args.pipeline}",
        f"pipeline.output.path={args.output}",
        "pipeline.output.save_artifacts=true",
        "pipeline.output.save_viz=false",
        "pipeline.slam.optimize_intrinsics=false",   # required when a metric-depth prior is present
        "streams=frame_dir_stream",
        f"streams.base_path={args.frames_dir}",
    ]
    cfg = parse_typed_config("default", hydra_args=overrides)
    pipeline = make_pipeline(cfg.pipeline)
    # relax ONLY the metric-depth init assertion (vendor untouched, instance-level)
    pipeline._add_init_processors = types.MethodType(_patched_add_init_processors, pipeline)

    base = FrameDirStream(args.frames_dir)
    H, W = base.frame_size()
    poses, depths = _build_priors(args.frames_dir, args.priors, H, W, args.inject_poses)

    attrs = {FrameAttribute.METRIC_DEPTH: depths}      # always anchor metric via depth
    if args.inject_poses:
        attrs[FrameAttribute.POSE] = poses             # only stray/lidar seeds the trajectory
    proc = AssignAttributesProcessor(attrs)
    stream = ProcessedVideoStream(base, [proc]).cache(desc="Reading frames + priors")

    logger.info(f"Running ViPE with priors (depth{'+pose' if args.inject_poses else ' only, poses free'}) "
                f"→ {args.output}")
    pipeline.run(stream)
    logger.info("ViPE (with priors) finished")


if __name__ == "__main__":
    import sys
    import traceback
    try:
        main()
    except Exception:
        print("[vipe-priors] FATAL — traceback follows:", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)
