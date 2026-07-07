# STAC-Builder — Phase R writeback: close the loop back into the fusion.
#
# Phase R computes the inter-window pose corrections (R.4) and depth
# regularization (R.5), but the metrics/store alone do not change the final mesh.
# This module APPLIES the refinement to the artifacts the TSDF fusion actually
# reads, gated by the R.9 fail-safe A/B so a regressing refinement is never
# written:
#
#   * R.4 — compose the per-window Sim(3) corrections onto each keyframe's c2w and
#           write them to camera_poses.txt (same target as the COLMAP-BA
#           writeback, with a .prephaser backup). tsdf_export(use_refined_poses)
#           then consumes them automatically.
#   * R.5 — regularize whitelist-class depth toward the fitted plane inside each
#           instance mask and write the depth npz back (with a .prephaser backup).
#
# Both are reversible (backups) and no-ops when there is nothing to correct
# (single window → identity corrections; no whitelist instances → depth
# unchanged), so running them on a single-window scene is safe.
#
# PROVENANCE: ours; pose-file format mirrors reconstruction/run_colmap_ba.py.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from .residuals import Sim3


# ── R.4 pose writeback ──────────────────────────────────────────────
def compose_refined_poses(pose_map: dict[int, np.ndarray],
                          window_map: dict[int, str],
                          windows: list[str],
                          corrections: list[Sim3]) -> dict[int, np.ndarray]:
    """Apply the per-window Sim(3) correction to each frame's c2w.

    For a camera-to-world pose, the window correction M_w maps that window's
    world frame into the globally-consistent frame:
        R' = M_w.R @ R_c2w         (orientation)
        t' = M_w.s·(M_w.R @ t_c2w) + M_w.t   (camera centre, metric)
    Frames whose window has an identity correction are returned unchanged.
    """
    widx = {w: i for i, w in enumerate(windows)}
    out: dict[int, np.ndarray] = {}
    for fid, c2w in pose_map.items():
        c2w = np.asarray(c2w, float)
        w = window_map.get(fid, windows[0] if windows else "global")
        M = corrections[widx[w]] if w in widx and widx[w] < len(corrections) else Sim3.identity()
        R = M.R @ c2w[:3, :3]
        t = M.apply(c2w[:3, 3])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        out[fid] = T
    return out


def write_refined_camera_poses(output_dir, refined_c2w: dict[int, np.ndarray],
                               backup_suffix: str = ".prephaser") -> list[str]:
    """Write refined c2w into every camera_poses.txt copy, row-aligned with the
    matching camera_frames.txt (frame numbers). Backs each file up once. Mirrors
    reconstruction/run_colmap_ba.py::_writeback. Returns the files written."""
    output_dir = Path(output_dir)
    written: list[str] = []
    for base in (output_dir, output_dir / "omega_run", output_dir / "maplong_run",
                 output_dir / "da3_run"):
        pp, fp = base / "camera_poses.txt", base / "camera_frames.txt"
        if not pp.exists() or not fp.exists():
            continue
        plines = pp.read_text().splitlines()
        nums = [int(float(x)) for x in fp.read_text().split()]
        pose_lines = [ln for ln in plines if len(ln.split()) == 16]
        if len(pose_lines) != len(nums):
            continue  # poses/frames not aligned — skip this copy
        bak = pp.with_suffix(pp.suffix + backup_suffix)
        if not bak.exists():
            shutil.copy(pp, bak)
        out, pi = [], 0
        for ln in plines:
            if len(ln.split()) == 16:
                fn = nums[pi]; pi += 1
                if fn in refined_c2w:
                    ln = " ".join(f"{v:.8g}" for v in refined_c2w[fn].reshape(-1))
            out.append(ln)
        pp.write_text("\n".join(out) + "\n")
        written.append(str(pp))
    return written


# ── R.4 cloud writeback (chunks pre-merge / merged cloud via trace) ──
def apply_corrections_to_clouds(output_dir, windows: list, totals: list,
                                window_map: dict[int, str],
                                backup_suffix: str = ".prephaser") -> dict:
    """Apply the per-window Sim(3) corrections to the CLOUD geometry so the
    merge/viewer see the anchored result, not just the TSDF:

      * chunk_*.ply present (anchored order: Phase R runs BEFORE cloudcompy):
        each chunk is one window — transform its points by that window's
        correction, then the merge fuses already-corrected chunks.
      * chunks already merged+deleted (resumed/legacy sessions): transform
        cleaned_cloud.ply PER POINT via its frame_global traceability
        (point → source frame → window → correction). No re-reconstruction.

    Identity corrections are a no-op. Backups once (.prephaser). The Potree
    octree is invalidated when the merged cloud changes."""
    from reconstruction.surface_fit.consolidate import (_read_ply_structured,
                                                        _write_ply_structured)

    output_dir = Path(output_dir)
    widx = {w: i for i, w in enumerate(windows)}
    mats = {w: _sim3_matrix4(totals[widx[w]]) for w in windows}
    nontrivial = {w for w, M in mats.items()
                  if not np.allclose(M, np.eye(4), atol=1e-9)}
    if not nontrivial:
        return {"clouds": 0, "note": "identity corrections — clouds untouched"}

    def _transform(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
        return pts @ M[:3, :3].T + M[:3, 3]

    stats: dict = {"clouds": 0, "mode": None}
    chunks = sorted(output_dir.glob("chunk_*.ply"))
    if chunks:
        import re as _re
        stats["mode"] = "chunks"
        for ply in chunks:
            m = _re.match(r"chunk_(\d+)\.ply$", ply.name)
            if not m:
                continue          # chunk_999_lidar.ply etc. — not a window chunk
            k = int(m.group(1))
            if k >= 900:
                continue          # external complements use high indices
            w = f"w{k:03d}"
            if w not in nontrivial:
                continue
            loaded = _read_ply_structured(ply)
            if loaded is None:
                continue
            header, data = loaded
            if not {"x", "y", "z"} <= set(data.dtype.names or ()):
                continue
            bak = ply.with_suffix(ply.suffix + backup_suffix)
            if not bak.exists():
                shutil.copy(ply, bak)
            pts = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64)
            moved = _transform(pts, mats[w])
            out = data.copy()
            out["x"] = moved[:, 0].astype(data.dtype["x"])
            out["y"] = moved[:, 1].astype(data.dtype["y"])
            out["z"] = moved[:, 2].astype(data.dtype["z"])
            _write_ply_structured(ply, header, out)
            stats["clouds"] += 1
        return stats

    cloud = output_dir / "cleaned_cloud.ply"
    if not cloud.exists():
        return {"clouds": 0, "note": "no chunk PLYs and no cleaned cloud"}
    loaded = _read_ply_structured(cloud)
    if loaded is None:
        return {"clouds": 0, "note": "unsupported cleaned-cloud layout"}
    header, data = loaded
    names = set(data.dtype.names or ())
    if "frame_global" not in names or not {"x", "y", "z"} <= names:
        return {"clouds": 0, "note": "no frame_global traceability — cloud untouched"}
    stats["mode"] = "merged_cloud_trace"
    fg = np.asarray(data["frame_global"], np.int64)
    default_w = windows[0] if windows else "global"
    uniq = np.unique(fg)
    frame_to_w = {int(f): window_map.get(int(f), default_w) for f in uniq}
    pts = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64)
    touched = False
    for w in nontrivial:
        sel_frames = np.array([f for f, ww in frame_to_w.items() if ww == w])
        if not len(sel_frames):
            continue
        sel = np.isin(fg, sel_frames)
        if sel.any():
            pts[sel] = _transform(pts[sel], mats[w])
            touched = True
    if touched:
        bak = cloud.with_suffix(cloud.suffix + backup_suffix)
        if not bak.exists():
            shutil.copy(cloud, bak)
        out = data.copy()
        out["x"] = pts[:, 0].astype(data.dtype["x"])
        out["y"] = pts[:, 1].astype(data.dtype["y"])
        out["z"] = pts[:, 2].astype(data.dtype["z"])
        _write_ply_structured(cloud, header, out)
        stats["clouds"] = 1
        # the octree no longer matches the corrected cloud — drop it so the
        # cloudcompy light-resume / lazy viewer conversion rebuilds it
        potree = output_dir / "potree"
        if potree.is_dir():
            shutil.rmtree(potree, ignore_errors=True)
            stats["potree_invalidated"] = True
    return stats


def _sim3_matrix4(M: Sim3) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = M.s * M.R
    T[:3, 3] = M.t
    return T


# ── R.5 depth regularization writeback ──────────────────────────────
def apply_depth_regularization(output_dir, store, session_dir, config: dict | None = None,
                               backup_suffix: str = ".prephaser") -> dict:
    """Regularize whitelist-class depth toward the per-instance fitted plane and
    write the depth npz back (with a backup). No-op for instances that are not
    whitelisted or lack enough points. Returns a small stats dict."""
    from .depth_regularization import (fit_plane, plane_world_to_camera,
                                       regularize_depth_to_plane)
    from .build_instances import _scaled_K, _frame_size
    from segmentation.session_io import _load_camera_source

    output_dir = Path(output_dir)
    session_dir = Path(session_dir)
    cfg = (config or {}).get("phase_r", {}) if config else {}
    whitelist = set(cfg.get("depth_reg_whitelist",
                            ["wall", "slab", "column", "vault", "platform", "floor"]))
    weight = float(cfg.get("depth_reg_weight", 0.5))

    # locate depth dir + provider
    depth_base = None
    depth_rel = None
    for c in ["omega_run/results_output", "da3_run/results_output", "results_output"]:
        if (output_dir / c).is_dir():
            depth_base = output_dir / c
            depth_rel = c
            break
    if depth_base is None:
        return {"applied": 0, "reason": "no depth dir"}

    # omega npz depth is up-to-scale while store points (→ fitted planes) are
    # metric; convert the metric plane prediction back to the npz's native
    # units so the blend never mixes scales (see InstanceStoreBuilder.
    # _metric_depth_scale for the full story).
    depth_scale = 1.0
    if not depth_rel.startswith("da3_run"):
        marker = output_dir / ".metric_scale_applied"
        if marker.exists():
            try:
                depth_scale = float(marker.read_text().strip().split("=")[-1])
            except Exception:
                depth_scale = 1.0

    import json
    seg = json.load(open(output_dir / "segmentation.json"))
    id_to = {i["id"]: i for i in seg.get("instances", [])}
    npz = np.load(output_dir / seg.get("mask_file", "seg_masks.npz"))
    cam = _load_camera_source(session_dir, output_dir)
    frames_dir = session_dir / "frames"

    n_applied = 0
    for inst in store.list_instances():
        iid = inst["instance_id"]
        label = inst["label"]
        # R.5 eligibility — Phase 2's refined class updates it for the next
        # loop iteration (spec): a classification row is authoritative over
        # the raw prompting label; label substring match is the fallback.
        cls = store.get_classification(iid)
        if cls is not None:
            if not cls.get("whitelist_eligible", False):
                continue
            label = cls.get("class_final") or label
        elif not any(w in label for w in whitelist):
            continue
        pts = store.get_points(iid)
        if pts is None or len(pts) < 30:
            continue
        n, d = fit_plane(pts)
        plane = (n, d)
        # per masklet frame, regularize masked depth in place
        for m in store.get_masklets(iid):
            key = m["mask_key"]
            if key not in npz.files:
                continue
            fid = m["frame_id"]
            p = depth_base / f"frame_{fid}.npz"
            if not p.exists() or cam is None or fid not in cam.pose_map:
                continue
            arr = dict(np.load(p))
            if "depth" not in arr:
                continue
            depth = arr["depth"].astype(np.float32)
            mask = npz[key]
            K = cam.K_for(fid)
            if K is None:
                continue
            fw, fh = _frame_size(frames_dir, fid, mask.shape)
            dh, dw = depth.shape[:2]
            Kd = _scaled_K(np.asarray(K, float), (fw, fh), (dw, dh))
            # the plane is fit on WORLD points — move it into THIS camera's
            # frame (and native depth units) before predicting per-pixel depth
            cam_plane = plane_world_to_camera(plane, np.asarray(cam.pose_map[fid], float))
            if depth_scale != 1.0:
                cam_plane = (cam_plane[0], cam_plane[1] / depth_scale)
            reg = regularize_depth_to_plane(depth, mask, cam_plane, Kd, weight=weight,
                                            label=label, whitelist=whitelist)
            if np.shares_memory(reg, depth) and np.array_equal(reg, depth):
                continue
            bak = p.with_suffix(p.suffix + backup_suffix)
            if not bak.exists():
                shutil.copy(p, bak)
            arr["depth"] = reg.astype(np.float32)
            np.savez(p, **arr)
            n_applied += 1
    return {"applied": n_applied, "whitelist": sorted(whitelist), "weight": weight}
