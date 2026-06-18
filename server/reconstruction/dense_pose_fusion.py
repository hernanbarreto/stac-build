#!/usr/bin/env python3
"""
Dense frame fusion via FRAME-TO-KEYFRAME registration — cloud-stage completeness.
================================================================================

PROBLEM. VGGT-Long estimates poses + builds the cloud from KEYFRAMES only (DINO
cosine), so the cloud inherits keyframe sparsity → inter-keyframe gaps the TSDF
(masked to the cloud) reproduces as holes. DA3 produced depth for EVERY blur-valid
frame, but those extra frames are LOW-PARALLAX by construction (that is exactly why
they were NOT chosen as keyframes). Feeding them back through a MULTI-VIEW model
(VGGT/MapAnything) is wrong: multi-view needs parallax to triangulate, so on
low-motion frames it degrades ("onion"/cebolla artifacts — confirmed empirically).

METHOD (standard dense-SLAM: ORB-SLAM2 non-keyframe tracking, BundleFusion, RGBiD-
SLAM). Do NOT triangulate the extra frames. LOCALIZE each one by FRAME-TO-MODEL
registration against the nearest already-solved keyframe: register the extra
frame's DA3 depth cloud to the keyframe's depth cloud (colored ICP = joint
photometric+geometric, robust on flat indoor walls), initialized at the keyframe's
global pose (motion is small → great init). The result is the extra frame's pose
DIRECTLY in the global, loop-closed frame — anchored to the keyframe, no drift, no
multi-view, no onion. Its DA3 depth is then back-projected to add cloud coverage.

The keyframe poses/cloud are NOT touched (VGGT-Long stays exactly as today). This
is a POSTERIOR, OPT-IN, NON-FATAL step that only ADDS inter-keyframe coverage.

Frame selection is TWO-TIER:
  • pose-keyframes  (DINO drop<0.98)  — anchors, unchanged (selected_frames.json)
  • fusion-frames   (denser, deduped) — blur-valid frames kept only when their DINO
    cosine to the last-kept fusion frame is < dedup_cosine, so a camera that filmed
    the same spot for minutes does NOT flood the set ("new coverage, not all frames").

CONFIDENCE FILTERING is IDENTICAL to the backend's cloud filter: the same
conf_threshold_coef gate (vendor formula+coefficient) is applied per frame, and the
surviving points carry their REAL confidence so CloudCompy's conf_min_norm gate then
applies to them exactly as to the rest of the cloud.

Output (consumed by CloudCompy, same convention as the LiDAR backend):
  output/chunk_998_densefusion.ply           (x,y,z,r,g,b,confidence)
  output/chunk_998_densefusion_origins.npz   (frame_global, pixel_row, pixel_col, confidence)
and APPENDS the registered poses to camera_poses.txt / camera_frames.txt so the TSDF
stage can use them too.

Runs in an env with open3d + torch (the reconstruction env). OPT-IN: only runs when
reconstruction.dense_fusion.enabled is true.

Authors: Hernán Barreto — Ingerop IN3
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_SERVER_DIR = Path(__file__).resolve().parent.parent
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

logger = logging.getLogger("DensePoseFusion")
logger.propagate = False


# ── Pose / sidecar I/O ──────────────────────────────────────────────────────

def _load_keyframe_poses(output_dir: Path) -> Tuple[Dict[int, np.ndarray], Path, Path]:
    """Global, loop-closed keyframe c2w poses keyed by REAL frame number. Reads
    camera_poses.txt (16 floats/line = 4x4 row-major c2w) + camera_frames.txt (real
    frame number/line). Returns ({frame:4x4}, poses_path, frames_path)."""
    for base in (output_dir / "da3_run", output_dir):
        pp, fp = base / "camera_poses.txt", base / "camera_frames.txt"
        if pp.exists() and fp.exists():
            mats = [np.array(list(map(float, ln.split())), dtype=np.float64).reshape(4, 4)
                    for ln in pp.read_text().splitlines() if len(ln.split()) == 16]
            nums = [int(x) for x in fp.read_text().split()]
            if len(mats) == len(nums) and mats:
                return {nums[i]: mats[i] for i in range(len(mats))}, pp, fp
    raise FileNotFoundError("camera_poses.txt + camera_frames.txt not found / mismatched")


# ── DA3 depth (the per-frame depth we register + fuse) ──────────────────────

def _da3_depth_loader(output_dir: Path):
    """loader(frame_num) → (depth HxW float32, K 3x3, conf HxW|None) from
    da3_run/results_output/frame_<n>.npz (keys: depth, intrinsics, conf)."""
    for d in (output_dir / "da3_run" / "results_output", output_dir / "results_output",
              output_dir / "gaus_slam_run" / "results_output"):
        if d.exists() and any(d.glob("frame_*.npz")):
            npz_dir = d
            break
    else:
        return None
    probe = np.load(str(next(iter(npz_dir.glob("frame_*.npz")))))
    if "depth" not in probe or "intrinsics" not in probe:
        return None
    h, w = probe["depth"].shape

    def _load(frame_num: int):
        p = npz_dir / f"frame_{frame_num}.npz"
        if not p.exists():
            return None
        z = np.load(str(p))
        if "depth" not in z or "intrinsics" not in z:
            return None
        K = np.asarray(z["intrinsics"], np.float64).reshape(3, 3)
        conf = z["conf"].astype(np.float32) if "conf" in z else None
        return z["depth"].astype(np.float32), K, conf

    return _load, (h, w)


def _conf_threshold_coef(config: dict) -> float:
    """The SAME conf_threshold_coef the reconstruction backend uses for the cloud's
    chunk PLYs (read from config — never hardcoded)."""
    recon = (config or {}).get("reconstruction", {}) or {}
    for sect in ("mapanything", "da3"):
        v = (recon.get(sect, {}) or {}).get("conf_threshold_coef")
        if v is not None:
            return float(v)
    return 0.75


def _conf_keep_mask(conf: np.ndarray, coef: float) -> np.ndarray:
    """EXACT replica of the backend's cloud-point confidence filter
    (vendor/VGGT-Long/vggt_long.py::_stac_write_origins):
        thr = mean(conf) * conf_threshold_coef ;  keep = (conf >= thr) & (conf > 1e-5)."""
    c = conf.astype(np.float32)
    thr = float(np.mean(c)) * float(coef)
    return (c >= thr) & (c > 1e-5)


# ── Per-frame point clouds (camera frame) ───────────────────────────────────

def _unproject(depth: np.ndarray, K: np.ndarray, dmin: float = 0.05, dmax: float = 1e9,
               stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """(xyz_cam (M,3), mask_flat_idx) for valid depth pixels (optionally strided)."""
    H, W = depth.shape
    vv, uu = np.indices((H, W))
    if stride > 1:
        vv, uu = vv[::stride, ::stride], uu[::stride, ::stride]
        d = depth[::stride, ::stride]
    else:
        d = depth
    m = (d > dmin) & (d < dmax)
    z = d[m].astype(np.float64)
    x = (uu[m] - K[0, 2]) * z / K[0, 0]
    y = (vv[m] - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], 1), (vv[m] * W + uu[m])


# ── Fusion-frame selection (dedup gate) ─────────────────────────────────────

def _build_fusion_frames(frames_dir: Path, kf_nums: set, dedup_cosine: float,
                         config: dict) -> List[int]:
    """The extra inter-keyframe frames to refine+fuse = DA3's DENSE set (da3_frames.json,
    DINO 0.99, the SINGLE source of truth selected upstream) MINUS the keyframes. Falls
    back to recomputing the DINO dedup only if da3_frames.json is missing."""
    da3p = frames_dir / "da3_frames.json"
    if da3p.exists():
        try:
            names = json.loads(da3p.read_text()).get("selected_files", [])
            dense = {int(Path(n).stem) for n in names}
            fusion = sorted(dense - set(kf_nums))
            logger.info(f"[dense-fusion] fusion-frames: {len(fusion)} extra = da3_frames.json "
                        f"({len(dense)}) minus {len(kf_nums)} keyframes")
            return fusion
        except Exception as e:
            logger.warning(f"[dense-fusion] da3_frames.json unreadable ({e}) — recomputing DINO")
    # Fallback: recompute the dedup from DINOv2 embeddings.
    from frames.selector import (_load_dino_model, _get_dino_transform,
                                 _extract_embeddings_batched, _cosine_similarity)
    fcfg = (config or {}).get("frame_selection", {})
    dev = fcfg.get("dino_device", "cuda")
    files = sorted(frames_dir.glob("*.jpg"))
    nums = [int(f.stem) for f in files]
    model = _load_dino_model(fcfg.get("dino_model", "dinov2_vitb14"), dev)
    tr = _get_dino_transform()
    embs = _extract_embeddings_batched(model, frames_dir, files, tr, dev,
                                       batch_size=int(fcfg.get("dino_batch_size", 64)))
    kept: List[int] = []
    last = None
    for n, e in zip(nums, embs):
        if n in kf_nums:
            kept.append(n); last = e; continue
        if last is None or _cosine_similarity(e, last) < dedup_cosine:
            kept.append(n); last = e
    fusion = sorted(set(kept) - set(kf_nums))
    logger.info(f"[dense-fusion] fusion-frames: {len(fusion)} extra (dedup<{dedup_cosine} fallback)")
    return fusion


# ── Frame-to-keyframe registration (colored ICP, low-parallax-safe) ─────────

def _to_o3d(xyz: np.ndarray, rgb01: Optional[np.ndarray]):
    import open3d as o3d
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(np.ascontiguousarray(xyz))
    if rgb01 is not None:
        p.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(rgb01))
    return p


def _register_to_keyframe(src_cam, tgt_world, init_c2w, voxels=(0.04, 0.02, 0.01)):
    """Multiscale COLORED ICP **on the GPU** (Open3D tensor `multi_scale_icp`, CUDA).
    src in camera frame, tgt in world; init = keyframe c2w. Returns (c2w 4x4, fitness).

    GPU-ONLY by design — there is NO CPU fallback: the legacy CPU colored ICP is
    unviably slow at this scale (validated: ~35 min for ~380 fillers). If CUDA is not
    available the run FAILS loudly so the environment gets fixed, never silently degrades."""
    import open3d as o3d
    import open3d.core as o3c
    if not o3d.core.cuda.is_available():
        raise RuntimeError("dense_fusion ICP requires CUDA (GPU). No GPU available — the CPU "
                           "colored ICP is unviably slow; fix the env, do NOT fall back to CPU.")
    dev = o3c.Device("CUDA:0")
    T = np.ascontiguousarray(np.asarray(init_c2w, np.float64))
    s = o3d.t.geometry.PointCloud.from_legacy(src_cam, o3c.float32, dev)
    t = o3d.t.geometry.PointCloud.from_legacy(tgt_world, o3c.float32, dev)
    t.estimate_normals(max_nn=30, radius=voxels[0] * 2)
    s.estimate_normals(max_nn=30, radius=voxels[0] * 2)
    vs = o3d.utility.DoubleVector(list(voxels))
    crit = [o3d.t.pipelines.registration.ICPConvergenceCriteria(1e-6, 1e-6, it)
            for it in (50, 30, 14)[:len(voxels)]]
    maxc = o3d.utility.DoubleVector([v * 1.4 for v in voxels])
    reg = o3d.t.pipelines.registration
    init = o3c.Tensor(T, o3c.float64)
    res = reg.multi_scale_icp(s, t, vs, crit, maxc, init,
                              reg.TransformationEstimationForColoredICP())
    return res.transformation.numpy(), float(res.fitness)


def _mapanything_hw(output_dir: Path):
    """MapAnything's depth resolution (H,W) = the KEYFRAME-cloud resolution. The DA3
    filler depth (a different, lower res) is resized to THIS before back-projection so
    the cloud has uniform density + consistent pixel traceability across keyframes and
    fillers. Returns (H,W) or None (then no resize → DA3 native res)."""
    run = output_dir / "maplong_run"
    # 1) any MapAnything chunk .npy → depth (S,H,W) (present right after MapAnything, when
    #    dense_fusion runs — before the optional aligned-chunk cleanup).
    for sub in ("_tmp_results_aligned", "_tmp_results_unaligned"):
        sd = run / sub
        if not sd.exists():
            continue
        for p in sorted(sd.glob("chunk_*.npy")):
            try:
                d = np.load(str(p), allow_pickle=True).item()
                a = np.asarray(d.get("depth"))
                while a.ndim > 3:
                    a = a[0]
                if a.ndim == 2:
                    a = a[None]
                return int(a.shape[-2]), int(a.shape[-1])
            except Exception:
                continue
    # 2) fallback: the keyframe cloud's pixel traceability max (chunk_*_origins.npz).
    for p in sorted(output_dir.glob("chunk_*_origins.npz")):
        try:
            z = np.load(p)
            return int(z["pixel_row"].max()) + 1, int(z["pixel_col"].max()) + 1
        except Exception:
            continue
    return None


def _resize_nn(a: np.ndarray, TH: int, TW: int) -> np.ndarray:
    """Nearest-neighbour resize (H,W)→(TH,TW). NN, not bilinear: depth/conf must NOT be
    blended across discontinuities. Dependency-free (pure index map)."""
    H, W = a.shape
    yi = np.clip((np.arange(TH) * H / TH).astype(np.int64), 0, H - 1)
    xi = np.clip((np.arange(TW) * W / TW).astype(np.int64), 0, W - 1)
    return a[yi][:, xi]


def densify_and_fuse(output_dir: Path, frames_dir: Path, config: dict) -> Optional[Path]:
    """Main entry — see module docstring."""
    import open3d as o3d
    from PIL import Image
    t0 = time.time()
    dcfg = (config or {}).get("reconstruction", {}).get("dense_fusion", {}) or {}
    dedup_cosine = float(dcfg.get("dedup_cosine", 0.99))
    min_fitness = float(dcfg.get("min_fitness", 0.3))   # reject a registration below this
    coef = _conf_threshold_coef(config)

    kf_poses, poses_path, frames_path = _load_keyframe_poses(output_dir)
    kf_nums = np.array(sorted(kf_poses))
    logger.info(f"[dense-fusion] {len(kf_nums)} keyframe anchors")

    da3 = _da3_depth_loader(output_dir)
    if da3 is None:
        logger.warning("[dense-fusion] no DA3 per-frame depth — abort"); return None
    depth_loader, _ = da3

    fusion = _build_fusion_frames(frames_dir, set(int(k) for k in kf_nums), dedup_cosine, config)
    if not fusion:
        logger.info("[dense-fusion] no extra fusion frames after dedup — nothing to do"); return None

    def _jpg(n: int) -> Path:
        return frames_dir / f"{n:06d}.jpg"

    def _rgb01(n: int, H: int, W: int):
        jpg = _jpg(n)
        if jpg.exists():
            return (np.asarray(Image.open(jpg).convert("RGB").resize((W, H), Image.BILINEAR),
                               np.float32) / 255.0)
        return None

    # Cache each keyframe's WORLD cloud (camera depth → world via its global pose) as
    # the ICP target. Built on demand for the nearest keyframe of each fusion frame.
    _kf_cache: Dict[int, object] = {}

    def _kf_target(kf: int):
        """Keyframe WORLD cloud + per-point NORMALS + RGB (xyz, normals, rgb) as float32
        numpy — the COLORED-ICP target. Normals (Open3D, one-time per keyframe) + colour
        feed the torch colored ICP. Cached."""
        if kf in _kf_cache:
            return _kf_cache[kf]
        d = depth_loader(kf)
        if d is None:
            _kf_cache[kf] = None; return None
        depth, K, _ = d
        xyz_cam, idx = _unproject(depth, K, stride=2)            # stride for ICP speed
        c2w = kf_poses[kf]
        world = (xyz_cam @ c2w[:3, :3].T + c2w[:3, 3]).astype(np.float32)
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(world)
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=30))
        H, W = depth.shape
        col = _rgb01(kf, H, W)
        rgb = (col.reshape(-1, 3)[idx] if col is not None
               else np.full((len(world), 3), 0.5, np.float32)).astype(np.float32)
        _kf_cache[kf] = (world, np.asarray(pcd.normals, np.float32), rgb)
        return _kf_cache[kf]

    from reconstruction.torch_icp import multiscale_icp_cuda
    dense_poses: Dict[int, np.ndarray] = {}
    n_ok = n_rej = 0
    logger.info(f"[dense-fusion] frame-to-keyframe COLORED ICP over {len(fusion)} fillers "
                f"(geometric+photometric on CUDA, pure torch — GPU-verified)…")
    for f in fusion:
        d = depth_loader(f)
        if d is None:
            continue
        depth, K, conf = d
        kf = int(kf_nums[int(np.argmin(np.abs(kf_nums - f)))])    # nearest keyframe
        tgt = _kf_target(kf)
        if tgt is None:
            continue
        tgt_xyz, tgt_n, tgt_rgb = tgt
        H, W = depth.shape
        xyz_cam, idx = _unproject(depth, K, stride=2)
        scol = _rgb01(f, H, W)
        src_rgb = (scol.reshape(-1, 3)[idx] if scol is not None
                   else np.full((len(xyz_cam), 3), 0.5, np.float32)).astype(np.float32)
        c2w, fit = multiscale_icp_cuda(xyz_cam.astype(np.float32), src_rgb,
                                       tgt_xyz, tgt_n, tgt_rgb, kf_poses[kf])
        if fit < min_fitness:
            n_rej += 1; continue                                  # bad registration → skip
        dense_poses[f] = c2w
        n_ok += 1
        if (n_ok + n_rej) % 25 == 0:
            logger.info(f"[dense-fusion] {n_ok+n_rej}/{len(fusion)} processed "
                        f"({n_ok} ok, {n_rej} rej) {time.time()-t0:.0f}s")
    logger.info(f"[dense-fusion] frame-to-keyframe ICP: {n_ok} registered, {n_rej} rejected "
                f"(fitness<{min_fitness}) in {time.time()-t0:.0f}s")
    if not dense_poses:
        return None

    # Back-project each registered frame's DA3 depth (FULL res) with its global pose,
    # applying the SAME confidence gate as the cloud; carry REAL conf for CloudCompy.
    logger.info(f"[dense-fusion] confidence gate = conf_threshold_coef={coef} (== backend), "
                "per frame; CloudCompy conf_min_norm applies after")
    # Unify the filler depth resolution to MapAnything's (the keyframe-cloud res, the
    # higher one) so the cloud has uniform density + consistent traceability.
    _tgt = _mapanything_hw(output_dir)
    if _tgt:
        logger.info(f"[dense-fusion] resizing DA3 filler depth → MapAnything res "
                    f"{_tgt[1]}x{_tgt[0]} (uniform cloud + consistent trace)")
    all_xyz, all_rgb, all_fg, all_pr, all_pc, all_cf = [], [], [], [], [], []
    for f, c2w in dense_poses.items():
        d = depth_loader(f)
        if d is None:
            continue
        depth, K, conf = d
        if conf is None or conf.shape != depth.shape:
            continue
        if _tgt and depth.shape != _tgt:
            TH, TW = _tgt
            H0, W0 = depth.shape
            depth = _resize_nn(depth, TH, TW)
            conf = _resize_nn(conf, TH, TW)
            K = np.asarray(K, np.float64).copy()
            K[0, 0] *= TW / W0; K[0, 2] *= TW / W0   # scale fx,cx to the new width
            K[1, 1] *= TH / H0; K[1, 2] *= TH / H0   # scale fy,cy to the new height
        H, W = depth.shape
        vv, uu = np.indices((H, W))
        m = (depth > 0.05) & _conf_keep_mask(conf, coef)
        if not m.any():
            continue
        z = depth[m].astype(np.float64)
        x = (uu[m] - K[0, 2]) * z / K[0, 0]
        y = (vv[m] - K[1, 2]) * z / K[1, 1]
        world = np.stack([x, y, z], 1) @ c2w[:3, :3].T + c2w[:3, 3]
        all_xyz.append(world.astype(np.float32))
        col = _rgb01(f, H, W)
        all_rgb.append((col[m] * 255).astype(np.uint8) if col is not None
                       else np.full((int(m.sum()), 3), 180, np.uint8))
        all_fg.append(np.full(int(m.sum()), f, np.float32))
        all_pr.append(vv[m].astype(np.float32))
        all_pc.append(uu[m].astype(np.float32))
        all_cf.append(conf[m].astype(np.float32))

    if not all_xyz:
        return None
    xyz = np.concatenate(all_xyz); rgb = np.concatenate(all_rgb)
    fg = np.concatenate(all_fg); pr = np.concatenate(all_pr)
    pc = np.concatenate(all_pc); conf = np.concatenate(all_cf)

    ply = output_dir / "chunk_998_densefusion.ply"
    np.savez_compressed(output_dir / "chunk_998_densefusion_origins.npz",
                        frame_global=fg, pixel_row=pr, pixel_col=pc, confidence=conf)
    dt = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                   ('r', 'u1'), ('g', 'u1'), ('b', 'u1'), ('confidence', '<f4')])
    vd = np.empty(len(xyz), dt)
    vd['x'], vd['y'], vd['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    vd['r'], vd['g'], vd['b'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    vd['confidence'] = conf
    with open(ply, 'wb') as fobj:
        fobj.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\n"
                   f"property float x\nproperty float y\nproperty float z\n"
                   f"property uchar red\nproperty uchar green\nproperty uchar blue\n"
                   f"property float confidence\nend_header\n".encode())
        vd.tofile(fobj)

    # Append the filler poses to EVERY camera_poses.txt / camera_frames.txt copy the consumers
    # read — in the SAME order — so they stay consistent. The TSDF reads output/ (or da3_run/),
    # but the UI camera-pose viz + the synced flythrough read maplong_run/camera_poses.txt FIRST
    # (main.py). If only one copy were updated, the UI would show 845 keyframes while the TSDF
    # used keyframes+fillers. Only append to a copy whose keyframe count still matches (845) so
    # we never double-append or desync. The UI pairs poses[i] ↔ frames[i] by line index.
    _sorted = sorted(dense_poses.items())
    _n_kf = len(kf_nums)
    _updated = []
    for _base in (output_dir, output_dir / "maplong_run", output_dir / "da3_run"):
        _pp = _base / "camera_poses.txt"
        if not _pp.exists():
            continue
        try:
            _have = sum(1 for ln in _pp.read_text().splitlines() if len(ln.split()) == 16)
        except Exception:
            continue
        if _have != _n_kf:          # already has fillers, or a different set → skip (no desync)
            continue
        with open(_pp, "a") as fp:
            for _f, _c2w in _sorted:
                fp.write(" ".join(f"{v:.8g}" for v in _c2w.reshape(-1)) + "\n")
        _fr = _base / "camera_frames.txt"
        if _fr.exists():
            with open(_fr, "a") as ff:
                for _f, _ in _sorted:
                    ff.write(f"{_f}\n")
        _updated.append(_pp.parent.name or "output")

    logger.info(f"[dense-fusion] ✅ {len(xyz):,} extra cloud pts from {len(dense_poses)} frames "
                f"→ {ply.name} (+ poses appended to: {', '.join(_updated)}) in {time.time()-t0:.0f}s")
    return ply


def main():
    ap = argparse.ArgumentParser(description="Dense frame fusion via frame-to-keyframe ICP")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    # The module logger has propagate=False, so basicConfig (root) never sees its records
    # → the subprocess was SILENT (no ICP progress). Attach a flushing handler DIRECTLY to
    # the module logger + line-buffer stdout so the parent forwards every line live.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    class _Flushing(logging.StreamHandler):
        def emit(self, record):
            super().emit(record)
            try:
                self.flush()
            except Exception:
                pass

    _h = _Flushing(sys.stdout)
    _h.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    logger.setLevel(logging.INFO)
    logger.addHandler(_h)
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
    cfg = {}
    if args.config and Path(args.config).exists():
        import yaml
        cfg = yaml.safe_load(open(args.config)) or {}
    p = densify_and_fuse(Path(args.output_dir), Path(args.frames_dir), cfg)
    print(f"[dense-fusion] result: {p}", flush=True)
    sys.exit(0 if p else 1)


if __name__ == "__main__":
    main()
