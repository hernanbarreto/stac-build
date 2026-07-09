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

import logging
import shutil
from pathlib import Path

import numpy as np

from .residuals import Sim3

# Forwarded to the pipe (server.log + UI) by the phase_r worker — every
# writeback action must be reconstructible from the log.
logger = logging.getLogger("PhaseR")


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
        logger.info(f"[R.4 writeback] refined poses → {pp} "
                    f"({len(refined_c2w)} frames, backup {bak.name})")
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
        # Windows named after chunks (w000, w001, …) → each chunk transforms
        # rigidly as one unit. Any OTHER window naming (e.g. the single-window
        # pseudo-split s000, s001, …) → the chunk is transformed PER POINT via
        # its origins sidecar (chunk_NNN_origins.npz: frame_global per point).
        chunk_named = all(_re.match(r"w\d{3}$", str(w)) for w in windows)
        stats["mode"] = "chunks" if chunk_named else "chunks_per_point"
        for ply in chunks:
            m = _re.match(r"chunk_(\d+)\.ply$", ply.name)
            if not m:
                continue          # chunk_999_lidar.ply etc. — not a window chunk
            k = int(m.group(1))
            if k >= 900:
                continue          # external complements use high indices
            loaded = _read_ply_structured(ply)
            if loaded is None:
                continue
            header, data = loaded
            if not {"x", "y", "z"} <= set(data.dtype.names or ()):
                continue
            pts = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64)
            touched = False
            if chunk_named:
                w = f"w{k:03d}"
                if w not in nontrivial:
                    continue
                pts = _transform(pts, mats[w])
                touched = True
            else:
                org = ply.with_name(f"{ply.stem}_origins.npz")
                if not org.exists():
                    logger.warning(f"[R.4 writeback] {ply.name}: window ids are not "
                                   f"chunk-aligned and no origins sidecar — SKIPPED "
                                   f"(cannot trace points to windows)")
                    stats.setdefault("skipped_no_origins", []).append(ply.name)
                    continue
                fg = np.load(org)["frame_global"].astype(np.int64)
                if len(fg) != len(data):
                    logger.warning(f"[R.4 writeback] {ply.name}: origins rows "
                                   f"({len(fg)}) ≠ vertices ({len(data)}) — SKIPPED")
                    stats.setdefault("skipped_misaligned", []).append(ply.name)
                    continue
                default_w = windows[0] if windows else "global"
                for w in nontrivial:
                    sel_frames = np.array(
                        [f for f in np.unique(fg)
                         if window_map.get(int(f), default_w) == w], dtype=np.int64)
                    if not len(sel_frames):
                        continue
                    sel = np.isin(fg, sel_frames)
                    if sel.any():
                        pts[sel] = _transform(pts[sel], mats[w])
                        touched = True
            if not touched:
                continue
            bak = ply.with_suffix(ply.suffix + backup_suffix)
            if not bak.exists():
                shutil.copy(ply, bak)
            out = data.copy()
            out["x"] = pts[:, 0].astype(data.dtype["x"])
            out["y"] = pts[:, 1].astype(data.dtype["y"])
            out["z"] = pts[:, 2].astype(data.dtype["z"])
            _write_ply_structured(ply, header, out)
            stats["clouds"] += 1
            logger.info(f"[R.4 writeback] corrected {ply.name} "
                        f"({stats['mode']}, backup {bak.name})")
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


# ── R.4 aligned-depth scale writeback ───────────────────────────────
def apply_corrections_to_aligned_depth(output_dir, windows: list, totals: list,
                                       window_map: dict[int, str],
                                       backup_suffix: str = ".prephaser") -> dict:
    """A window Sim(3) correction with s ≠ 1 rescales that window's world — its
    frames' CAMERA-FRAME depth must scale by s too, or the TSDF integrates depth
    inconsistent with the corrected poses/cloud. Scales the aligned chunk depth
    (maplong_run/_tmp_results_aligned) per frame by its window's s. No-op when
    every correction scale is ~1 (the usual case: priors pull s toward 1)."""
    output_dir = Path(output_dir)
    widx = {w: i for i, w in enumerate(windows)}
    scales = {w: float(totals[widx[w]].s) for w in windows}
    scaled_windows = {w: s for w, s in scales.items() if abs(s - 1.0) > 1e-6}
    if not scaled_windows:
        return {"frames": 0, "note": "all correction scales ≈ 1 — depth untouched"}
    locate = _aligned_chunk_locator(output_dir)
    if locate is None:
        return {"frames": 0, "note": "no aligned chunk depth store"}
    by_chunk: dict = {}
    for fid, w in window_map.items():
        s = scaled_windows.get(w)
        if s is None:
            continue
        loc = locate(fid)
        if loc is not None:
            by_chunk.setdefault(loc[0], []).append((loc[1], fid, s))
    n_frames = 0
    for cpath, items in sorted(by_chunk.items()):
        try:
            data = np.load(str(cpath), allow_pickle=True).item()
        except Exception:
            continue
        if not isinstance(data, dict) or data.get("depth") is None:
            continue
        depth_arr = np.array(data["depth"])
        stack = _depth_stack_view(depth_arr)
        changed = False
        for local, fid, s in items:
            if local >= stack.shape[0]:
                continue
            stack[local] = (stack[local].astype(np.float32) * s).astype(stack.dtype)
            changed = True
            n_frames += 1
        if changed:
            bak = cpath.with_suffix(cpath.suffix + backup_suffix)
            if not bak.exists():
                shutil.copy(cpath, bak)
            data["depth"] = depth_arr
            np.save(str(cpath), data)
            logger.info(f"[R.4 writeback] scaled aligned depth in {cpath.name} "
                        f"(windows with s≠1: { {w: round(s, 5) for w, s in scaled_windows.items()} })")
    return {"frames": n_frames, "windows_scaled": scaled_windows}


# ── R.5 depth regularization writeback ──────────────────────────────
def _resize_nearest(a: np.ndarray, hw: tuple) -> np.ndarray:
    """Nearest-neighbour resize of a per-pixel array (mask/conf) to (H, W)."""
    H, W = hw
    a = np.asarray(a)
    if a.shape[:2] == (H, W):
        return a
    yi = (np.arange(H) * a.shape[0] / H).astype(int).clip(0, a.shape[0] - 1)
    xi = (np.arange(W) * a.shape[1] / W).astype(int).clip(0, a.shape[1] - 1)
    return a[yi][:, xi]


def _erode_mask(mask: np.ndarray, px: int) -> np.ndarray:
    """R.2/R.5 aggressive erosion: only mask INTERIORS feed the regularization
    (our depth is DA3/omega, not a sensor — borders bleed across silhouettes)."""
    m = np.asarray(mask) > 0
    if px <= 0:
        return m
    try:
        from scipy.ndimage import binary_erosion
        return binary_erosion(m, iterations=int(px))
    except Exception:
        return m


def _aligned_chunk_locator(output_dir: Path):
    """frame number → (chunk_*.npy path, local index) for the ALIGNED chunk depth
    (maplong_run/_tmp_results_aligned) — the store the TSDF actually integrates
    with depth_source mapanything/auto. Mirrors the TSDF loader's resolution
    (frame_list.json position + chunk_step, chunk clamped to the last one so the
    overlap tail resolves). Returns None when the store is absent."""
    run_dir = output_dir / "maplong_run"
    chunks = run_dir / "_tmp_results_aligned"
    if not chunks.exists():
        chunks = run_dir / "_tmp_results_unaligned"   # legacy fallback
    flp = run_dir / "frame_list.json"
    if not flp.exists():
        flp = output_dir / "frame_list.json"
    if not chunks.exists() or not flp.exists():
        return None
    import json as _json
    import re as _re
    try:
        names = _json.loads(flp.read_text())
    except Exception:
        return None
    frame_to_pos: dict = {}
    for pos, n in enumerate(names):
        m = _re.search(r"(\d+)", str(n))
        if m:
            frame_to_pos.setdefault(int(m.group(1)), pos)
    if not frame_to_pos:
        return None
    chunk_step = None
    for mp in sorted(output_dir.glob("chunk_*_meta.json")):
        try:
            chunk_step = int(_json.load(open(mp)).get("chunk_step"))
            if chunk_step:
                break
        except Exception:
            continue
    if not chunk_step:
        try:
            import yaml as _yaml
            for cfg_name in ("vggt_long_config.yaml", "vggt_omega_config.yaml"):
                p = run_dir / cfg_name
                if p.exists():
                    c = _yaml.safe_load(open(p))
                    chunk_step = int(c["Model"]["chunk_size"]) - int(c["Model"]["overlap"])
                    break
        except Exception:
            chunk_step = None
    if not chunk_step or chunk_step < 1:
        return None
    n_chunks = 0
    while (chunks / f"chunk_{n_chunks}.npy").exists():
        n_chunks += 1
    if n_chunks == 0:
        return None

    def locate(fid: int):
        pos = frame_to_pos.get(int(fid))
        if pos is None:
            return None
        K = min(pos // chunk_step, n_chunks - 1)
        return chunks / f"chunk_{K}.npy", pos - K * chunk_step

    return locate


def _depth_stack_view(arr: np.ndarray) -> np.ndarray:
    """View the chunk 'depth' array as a (n, H, W) stack (matches the TSDF loader)."""
    a = arr
    while a.ndim > 3:
        a = a[0]
    if a.ndim == 2:
        a = a[None]
    return a


def apply_depth_regularization(output_dir, store, session_dir, config: dict | None = None,
                               backup_suffix: str = ".prephaser") -> dict:
    """Regularize whitelist-class depth toward the per-instance fitted plane and
    write it back (with backups). Two targets, same edits:
      1. the per-frame npz store (omega_run/da3_run/results_output) — consumed by
         the DA3-based TSDF depth sources and any per-frame reader;
      2. the ALIGNED CHUNK depth (maplong_run/_tmp_results_aligned/chunk_*.npy) —
         what the TSDF actually integrates with depth_source mapanything/auto.
         Units there are METRIC (apply_scale already scaled them), the same as the
         store planes → no unit conversion.
    No-op for instances that are not whitelisted or lack enough points. Returns a
    small stats dict."""
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
    erosion_px = int(cfg.get("depth_reg_erosion_px", cfg.get("mask_erosion_px", 8)))

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
    edits: dict = {}   # fid → [(eroded mask, world plane, label)] for the chunk pass
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
            if cam is None or fid not in cam.pose_map:
                continue
            mask_e = _erode_mask(npz[key], erosion_px)   # interiors only (R.2/R.5)
            if not mask_e.any():
                continue
            edits.setdefault(fid, []).append((mask_e, plane, label))
            p = depth_base / f"frame_{fid}.npz"
            if not p.exists():
                continue
            arr = dict(np.load(p))
            if "depth" not in arr:
                continue
            depth = arr["depth"].astype(np.float32)
            K = cam.K_for(fid)
            if K is None:
                continue
            fw, fh = _frame_size(frames_dir, fid, mask_e.shape)
            dh, dw = depth.shape[:2]
            Kd = _scaled_K(np.asarray(K, float), (fw, fh), (dw, dh))
            # the plane is fit on WORLD points — move it into THIS camera's
            # frame (and native depth units) before predicting per-pixel depth
            cam_plane = plane_world_to_camera(plane, np.asarray(cam.pose_map[fid], float))
            if depth_scale != 1.0:
                cam_plane = (cam_plane[0], cam_plane[1] / depth_scale)
            reg = regularize_depth_to_plane(depth, _resize_nearest(mask_e, (dh, dw)),
                                            cam_plane, Kd, weight=weight,
                                            label=label, whitelist=whitelist)
            if np.shares_memory(reg, depth) and np.array_equal(reg, depth):
                continue
            bak = p.with_suffix(p.suffix + backup_suffix)
            if not bak.exists():
                shutil.copy(p, bak)
            arr["depth"] = reg.astype(np.float32)
            np.savez(p, **arr)
            n_applied += 1

    # ── 2) the ALIGNED chunk depth the TSDF integrates (metric units) ──
    n_chunk_frames = _regularize_aligned_chunks(
        output_dir, edits, cam, frames_dir, whitelist, weight, backup_suffix)
    logger.info(f"[R.5 writeback] depth regularization: {n_applied} per-frame npz edits "
                f"({depth_rel}), {n_chunk_frames} aligned-chunk frame edits, "
                f"erosion {erosion_px}px, weight {weight}, whitelist {sorted(whitelist)}")
    return {"applied": n_applied, "chunk_frames": n_chunk_frames,
            "whitelist": sorted(whitelist), "weight": weight,
            "erosion_px": erosion_px}


def _regularize_aligned_chunks(output_dir: Path, edits: dict, cam, frames_dir: Path,
                               whitelist: set, weight: float,
                               backup_suffix: str = ".prephaser") -> int:
    """Apply the collected per-frame plane edits to the aligned chunk depth
    (maplong_run/_tmp_results_aligned/chunk_*.npy). Metric units end to end
    (chunks were scaled by scale_align.apply_scale; the store planes are metric),
    so — unlike the up-to-scale omega npz — NO depth_scale conversion. Backs each
    chunk file up once; edits are grouped per chunk so each npy loads once."""
    from .depth_regularization import plane_world_to_camera, regularize_depth_to_plane
    from .build_instances import _scaled_K, _frame_size

    if not edits or cam is None:
        return 0
    locate = _aligned_chunk_locator(output_dir)
    if locate is None:
        return 0
    by_chunk: dict = {}
    for fid, lst in edits.items():
        loc = locate(fid)
        if loc is None or fid not in cam.pose_map:
            continue
        by_chunk.setdefault(loc[0], []).append((loc[1], fid, lst))

    n_frames = 0
    for cpath, items in sorted(by_chunk.items()):
        try:
            data = np.load(str(cpath), allow_pickle=True).item()
        except Exception:
            continue
        if not isinstance(data, dict) or data.get("depth") is None:
            continue
        depth_arr = np.array(data["depth"])            # owned copy, original nesting
        stack = _depth_stack_view(depth_arr)           # (n, H, W) view into the copy
        dh, dw = int(stack.shape[-2]), int(stack.shape[-1])
        ki = data.get("intrinsic", data.get("intrinsics"))
        if ki is not None:
            ki = np.asarray(ki, dtype=np.float64)
            while ki.ndim > 3:
                ki = ki[0]
        changed = False
        for local, fid, lst in items:
            if local >= stack.shape[0]:
                continue
            # intrinsics: per-frame from the chunk (already at the chunk depth grid);
            # fall back to the session camera scaled to the grid.
            if ki is not None and ki.ndim == 3 and local < ki.shape[0]:
                Kd = ki[local].reshape(3, 3)
            elif ki is not None and ki.shape == (3, 3):
                Kd = ki
            else:
                K = cam.K_for(fid)
                if K is None:
                    continue
                fw, fh = _frame_size(frames_dir, fid, lst[0][0].shape)
                Kd = _scaled_K(np.asarray(K, float), (fw, fh), (dw, dh))
            d_slice = stack[local].astype(np.float32)
            orig = d_slice.copy()
            for mask_e, plane, label in lst:
                cam_plane = plane_world_to_camera(plane, np.asarray(cam.pose_map[fid], float))
                d_slice = regularize_depth_to_plane(
                    d_slice, _resize_nearest(mask_e, (dh, dw)), cam_plane, Kd,
                    weight=weight, label=label, whitelist=whitelist).astype(np.float32)
            if np.array_equal(d_slice, orig):
                continue
            stack[local] = d_slice.astype(stack.dtype)
            changed = True
            n_frames += 1
        if changed:
            bak = cpath.with_suffix(cpath.suffix + backup_suffix)
            if not bak.exists():
                shutil.copy(cpath, bak)
            data["depth"] = depth_arr
            np.save(str(cpath), data)
    return n_frames
