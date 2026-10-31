"""The training set of the gaussians stage, built by the da3 worker from the SESSION (the chain's
own readers) and written as plain arrays the training env reads without the server's modules:

  * the keyframes (``camera_frames.txt``) with their native images, the certified poses
    (``camera_poses.txt``, c2w, the world of the published cloud) and the session camera
    (``camera.json`` → K in native pixels through the grid map, OPENCV k1 k2 p1 p2);
  * the SEEDS: a stable subsample of the published cloud (``cleaned_cloud.ply``) with colour,
    confidence and the projection's per-point instance id (``instance_ids.npy``, 0 = unsegmented);
    the initial scale of a seed is the mean distance to its nearest seeds;
  * the held-out keyframes (every k-th) — the report's PSNR / SSIM are measured on them;
  * the depth the chain measured per keyframe (the same loader the TSDF chain uses, the depth
    correction sidecar applied), on the record grid with its map to native pixels.

Nothing here decides geometry: the poses, the camera and the cloud are read as the chain left them.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from gaussians.config import GaussiansConfig
from gaussians.plyio import read_ply_vertices

TRAINSET_NAME = "trainset.npz"
META_NAME = "trainset.json"
DEPTH_DIRNAME = "depth"


class DatasetError(RuntimeError):
    pass


@dataclass
class TrainingSet:
    frames: np.ndarray                 # (N,) keyframe frame numbers, in camera_frames.txt order
    image_paths: List[str]             # (N,) the native keyframe images
    width: int
    height: int
    K: np.ndarray                      # (3,3) native pixels
    dist: np.ndarray                   # (4,) k1 k2 p1 p2 (OPENCV)
    c2w: np.ndarray                    # (N,4,4) certified poses, the cloud's world
    holdout: np.ndarray                # (N,) bool — held out of the training, measured
    seeds_xyz: np.ndarray              # (M,3) float32
    seeds_rgb: np.ndarray              # (M,3) uint8
    seeds_cloud_index: np.ndarray      # (M,) uint32 — the seed's row in the published cloud: the instance is looked up
                                       # LIVE in the cloud's instance_ids.npy (the Segmentation Manager's edits apply)
    seeds_instance: np.ndarray         # (M,) uint32 at build time, 0 = unsegmented (recorded, never baked in)
    seeds_scale_log: np.ndarray        # (M,3) float32 — log of the initial isotropic scale
    seeds_confidence: np.ndarray       # (M,) float32
    depth_grid: Optional[Tuple[int, int]]      # (h, w) of the depth maps, None = no depth supervision
    grid_map: Dict[str, float]         # record grid → native: u_n = (u_g - pad_left + 0.5)·scale_x - 0.5 + crop_x
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(len(self.frames))

    @property
    def n_seeds(self) -> int:
        return int(len(self.seeds_xyz))


def _frame_files(session_dir: Path, frames: np.ndarray) -> List[str]:
    fdir = session_dir / "frames"
    by_frame: Dict[int, str] = {}
    sel = fdir / "selected_frames.json"
    if sel.exists():
        try:
            doc = json.loads(sel.read_text())
            for kf in doc.get("keyframes") or []:
                if isinstance(kf, dict) and "frame" in kf and "file" in kf:
                    by_frame[int(kf["frame"])] = str(kf["file"])
        except (ValueError, TypeError):
            pass
    out = []
    for f in frames.tolist():
        name = by_frame.get(int(f), f"{int(f):06d}.jpg")
        p = fdir / name
        if not p.exists():
            raise DatasetError(f"keyframe {int(f)}: {p} is missing")
        out.append(str(p))
    return out


def camera_native(cam) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """K in native pixels, the distortion as recorded, the record-grid → native map."""
    from precision.camera import K_grid_to_native
    if str(cam.model) != "OPENCV" or len(cam.params) != 8:
        raise DatasetError(f"camera.json: model {cam.model!r} with {len(cam.params)} parameter(s); OPENCV (8) expected")
    fx, fy, cx, cy, k1, k2, p1, p2 = [float(v) for v in cam.params]
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], np.float64)
    Kn = K_grid_to_native(K, cam.omega_grid)
    g = cam.omega_grid
    gm = {"scale_x": float(g.scale_x), "scale_y": float(g.scale_y), "pad_left": float(g.pad_left),
          "pad_top": float(g.pad_top), "crop_x": float(g.crop_x), "crop_y": float(g.crop_y),
          "grid_w": int(g.w), "grid_h": int(g.h)}
    return Kn, np.array([k1, k2, p1, p2], np.float64), gm


def subsample_seeds(n_points: int, confidence: Optional[np.ndarray], min_confidence: float, max_points: int,
                    seed: int) -> np.ndarray:
    """A stable subsample: the points at or above the confidence floor, a seeded permutation, the
    first ``max_points``, in cloud order."""
    idx = np.arange(n_points)
    if confidence is not None:
        idx = idx[np.asarray(confidence) >= float(min_confidence)]
    if len(idx) > max_points:
        rng = np.random.default_rng(int(seed))
        idx = idx[rng.permutation(len(idx))[:max_points]]
    return np.sort(idx)


def initial_scales(xyz: np.ndarray, knn: int) -> np.ndarray:
    """log of the mean distance to the ``knn`` nearest seeds (isotropic), per seed."""
    from scipy.spatial import cKDTree
    n = len(xyz)
    if n < 2:
        return np.full((n, 3), np.log(1e-2), np.float32)
    k = min(int(knn) + 1, n)
    d, _ = cKDTree(xyz).query(xyz, k=k)
    mean_d = d[:, 1:].mean(axis=1) if k > 1 else np.full(n, 1e-2)
    s = np.log(np.maximum(mean_d, 1e-6)).astype(np.float32)
    return np.repeat(s[:, None], 3, axis=1)


def holdout_mask(n: int, every: int) -> np.ndarray:
    k = int(every)
    return (np.arange(n) % k) == (k - 1)


def _depth_npz_dir(output_dir: Path) -> Optional[Path]:
    """The per-keyframe npz directory the chain's loader reads (its own candidate order), for the
    intrinsics the files carry — the depth grid is DA3's, not the native frame."""
    for d in (output_dir / "da3_run" / "results_output", output_dir / "results_output",
              output_dir / "gaus_slam_run" / "results_output", output_dir / "omega_run" / "results_output"):
        if d.exists() and any(d.glob("frame_*.npz")):
            return d
    return None


def depth_intrinsics(npz_dir: Optional[Path], frame: int) -> Optional[np.ndarray]:
    if npz_dir is None:
        return None
    p = npz_dir / f"frame_{int(frame)}.npz"
    if not p.exists():
        return None
    with np.load(p) as z:
        if "intrinsics" in z.files:
            return np.asarray(z["intrinsics"], np.float64).reshape(3, 3)
    return None


def _depth_loader(output_dir: Path, log: Callable):
    """The chain's own per-keyframe depth (segmentation.tsdf_export: DA3 npy / results_output npz,
    the correction sidecar applied). None, said, when the session holds none."""
    from segmentation.tsdf_export import _resolve_da3_depth
    res = _resolve_da3_depth(output_dir)
    if res is None:
        log("[gaussians] no per-keyframe depth maps in this session — the splats train without depth supervision")
        return None, None
    loader, (h, w) = res
    return loader, (int(h), int(w))


def build_training_set(session_dir: Path, cfg: GaussiansConfig, *, log: Callable = print) -> Tuple[TrainingSet, Optional[Callable]]:
    """(the training set, the depth loader to write the maps with)."""
    from intake.quality import read_session_epochs
    from precision.camera import load_camera_json
    session_dir = Path(session_dir).resolve()
    out = session_dir / "output"
    cloud_p = out / "cleaned_cloud.ply"
    if not cloud_p.exists():
        raise DatasetError(f"{cloud_p} is missing — no published cloud to seed the gaussians")
    poses_p, frames_p = out / "camera_poses.txt", out / "camera_frames.txt"
    if not poses_p.exists() or not frames_p.exists():
        raise DatasetError("camera_poses.txt / camera_frames.txt are missing — no certified poses")
    frames = np.array([int(float(x)) for x in frames_p.read_text().split()], np.int64)
    c2w = np.loadtxt(str(poses_p)).reshape(-1, 4, 4)
    if len(c2w) != len(frames):
        raise DatasetError(f"camera_poses.txt ({len(c2w)}) and camera_frames.txt ({len(frames)}) disagree")
    cam = load_camera_json(out / "camera.json")
    K, dist, gm = camera_native(cam)
    images = _frame_files(session_dir, frames)
    t0 = time.time()
    v = read_ply_vertices(cloud_p)
    names = set(v.dtype.names)
    for need in ("x", "y", "z", "red", "green", "blue"):
        if need not in names:
            raise DatasetError(f"{cloud_p}: property {need!r} is missing")
    n = len(v)
    conf = np.asarray(v["confidence"], np.float32) if "confidence" in names else None
    idx = subsample_seeds(n, conf, cfg.init.min_confidence, cfg.init.max_points, cfg.init.seed)
    xyz = np.stack([v["x"][idx], v["y"][idx], v["z"][idx]], axis=1).astype(np.float32)
    rgb = np.stack([v["red"][idx], v["green"][idx], v["blue"][idx]], axis=1).astype(np.uint8)
    inst_p = out / "instance_ids.npy"
    inst_note = "no instance_ids.npy — every seed unsegmented (0)"
    inst_all: Optional[np.ndarray] = None
    if inst_p.exists():
        arr = np.load(inst_p)
        if len(arr) == n:
            inst_all = arr
            inst_note = f"instance_ids.npy: {int((arr > 0).sum()):,} of {n:,} points segmented, " \
                        f"{len(np.unique(arr[arr > 0]))} instances"
        else:
            inst_note = f"instance_ids.npy has {len(arr)} entries for {n} points — ignored (every seed 0)"
    inst = (inst_all[idx] if inst_all is not None else np.zeros(len(idx), np.uint32)).astype(np.uint32)
    scales = initial_scales(xyz, cfg.init.scale_knn)
    hold = holdout_mask(len(frames), cfg.train.holdout_every)
    loader, grid = _depth_loader(out, log)
    epochs = read_session_epochs(session_dir)
    from correction.epoch import RECONSTRUCTION_ID_KEY, reconstruction_id_or_none
    meta = {"session_dir": str(session_dir), "n_cloud_points": int(n), "n_seeds": int(len(idx)),
            "seed_subsample": {"max_points": cfg.init.max_points, "min_confidence": cfg.init.min_confidence,
                               "seed": cfg.init.seed, "kept_share": float(len(idx)) / float(max(n, 1))},
            "instances": inst_note, "camera": {"source": str(cam.source), "camera_epoch": int(cam.camera_epoch),
                                               "native": [int(cam.width), int(cam.height)], "K_native": K.tolist(),
                                               "dist": dist.tolist()},
            "depth": {"available": grid is not None, "grid_hw": list(grid) if grid else None},
            "holdout": {"every": cfg.train.holdout_every, "n_heldout": int(hold.sum()), "n_train": int((~hold).sum())},
            RECONSTRUCTION_ID_KEY: reconstruction_id_or_none(out), **epochs,
            "seconds_build": round(time.time() - t0, 2)}
    log(f"[gaussians] training set: {len(frames)} keyframes ({int(hold.sum())} held out), {len(idx):,} seeds of "
        f"{n:,} points ({inst_note}); depth {'on' if grid else 'OFF'}; camera fx {K[0, 0]:.1f} fy {K[1, 1]:.1f} "
        f"on {cam.width}x{cam.height}")
    ts = TrainingSet(frames=frames, image_paths=images, width=int(cam.width), height=int(cam.height), K=K, dist=dist,
                     c2w=np.asarray(c2w, np.float64), holdout=hold, seeds_xyz=xyz, seeds_rgb=rgb,
                     seeds_cloud_index=idx.astype(np.uint32), seeds_instance=inst,
                     seeds_scale_log=scales, seeds_confidence=(conf[idx] if conf is not None else np.ones(len(idx), np.float32)),
                     depth_grid=grid, grid_map=gm, meta=meta)
    return ts, loader


def write_trainset(ts: TrainingSet, out_dir: Path, depth_loader: Optional[Callable], *, log: Callable = print) -> Path:
    """``trainset.npz`` + ``trainset.json`` + ``depth/frame_<f>.npz`` (training frames only, float32
    depth and a validity mask on the record grid)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / TRAINSET_NAME, frames=ts.frames, holdout=ts.holdout, K=ts.K, dist=ts.dist, c2w=ts.c2w,
             seeds_xyz=ts.seeds_xyz, seeds_rgb=ts.seeds_rgb, seeds_cloud_index=ts.seeds_cloud_index,
             seeds_instance=ts.seeds_instance, seeds_scale_log=ts.seeds_scale_log, seeds_confidence=ts.seeds_confidence)
    n_depth = 0
    n_k = 0
    if depth_loader is not None and ts.depth_grid is not None:
        ddir = out_dir / DEPTH_DIRNAME
        ddir.mkdir(exist_ok=True)
        npz_dir = _depth_npz_dir(Path(ts.meta.get("session_dir", out_dir)) / "output") if ts.meta.get("session_dir") else None
        for i, f in enumerate(ts.frames.tolist()):
            if ts.holdout[i]:
                continue
            got = depth_loader(int(f))
            if got is None:
                continue
            depth, valid = got
            np.savez(ddir / f"frame_{int(f)}.npz", depth=np.asarray(depth, np.float32), valid=np.asarray(valid, bool))
            Kd = depth_intrinsics(npz_dir, int(f))
            if Kd is not None:
                np.save(ddir / f"K_{int(f)}.npy", Kd)       # the depth grid's own camera (gaussians.train maps native → depth)
                n_k += 1
            n_depth += 1
    meta = dict(ts.meta, image_paths=ts.image_paths, width=ts.width, height=ts.height, grid_map=ts.grid_map,
                depth_grid=list(ts.depth_grid) if ts.depth_grid else None, n_depth_maps=n_depth, n_depth_intrinsics=n_k)
    (out_dir / META_NAME).write_text(json.dumps(meta, indent=1, default=str))
    log(f"[gaussians] training set written to {out_dir} ({n_depth} depth map(s))")
    return out_dir


def read_trainset(out_dir: Path) -> TrainingSet:
    out_dir = Path(out_dir)
    z = np.load(out_dir / TRAINSET_NAME)
    meta = json.loads((out_dir / META_NAME).read_text())
    dg = meta.get("depth_grid")
    return TrainingSet(frames=z["frames"], image_paths=list(meta["image_paths"]), width=int(meta["width"]),
                       height=int(meta["height"]), K=z["K"], dist=z["dist"], c2w=z["c2w"], holdout=z["holdout"].astype(bool),
                       seeds_xyz=z["seeds_xyz"], seeds_rgb=z["seeds_rgb"], seeds_cloud_index=z["seeds_cloud_index"],
                       seeds_instance=z["seeds_instance"],
                       seeds_scale_log=z["seeds_scale_log"], seeds_confidence=z["seeds_confidence"],
                       depth_grid=(int(dg[0]), int(dg[1])) if dg else None, grid_map=dict(meta["grid_map"]), meta=meta)


def read_depth_map(out_dir: Path, frame: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    p = Path(out_dir) / DEPTH_DIRNAME / f"frame_{int(frame)}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return z["depth"], z["valid"]
