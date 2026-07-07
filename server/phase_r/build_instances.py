# STAC-Builder — Phase R: build the canonical instance store from masklets.
#
# R.1 + R.2 + R.3 + R.7 + R.8 wired end-to-end: SAM3 masklets (Manager format) +
# BA poses + DA3 depth -> per-instance world points (eroded mask interiors, KNN
# filtered) -> gravity-aligned OBB -> onion metric -> canonical store. Dynamic
# classes (person / mobile equipment / train) are flagged excluded from pose &
# fusion (R.7) and the excluded pixel budget is logged.
#
# This is the single-reconstruction path; the cross-window Sim(3) residuals (R.4)
# and plurality vote across windows plug in on top of the same store.
#
# PROVENANCE: ours, using R3D-ported geometry.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .geometry import fit_gravity_aligned_obb, filter_outliers_knn, lift_masked_depth_to_world
from .instance_store import InstanceStore
from .onion import detect_onion


@dataclass
class BuildStats:
    n_instances: int = 0
    n_with_points: int = 0
    n_with_obb: int = 0
    n_onion: int = 0
    n_dynamic_excluded: int = 0
    dynamic_pixel_fraction: float = 0.0
    per_label: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (f"instances={self.n_instances} with_points={self.n_with_points} "
                f"with_obb={self.n_with_obb} onion={self.n_onion} "
                f"dynamic_excluded={self.n_dynamic_excluded} "
                f"dynamic_px_frac={self.dynamic_pixel_fraction:.4f}")


def _scaled_K(K: np.ndarray, from_wh: tuple[int, int], to_wh: tuple[int, int]) -> np.ndarray:
    """Scale intrinsics from one resolution to another (width,height)."""
    (fw, fh), (tw, th) = from_wh, to_wh
    sx, sy = tw / fw, th / fh
    Ks = K.copy().astype(float)
    Ks[0, 0] *= sx; Ks[0, 2] *= sx
    Ks[1, 1] *= sy; Ks[1, 2] *= sy
    return Ks


class InstanceStoreBuilder:
    def __init__(self, session_dir, output_dir, store_path, config: dict | None = None,
                 dynamic_labels: set[str] | None = None):
        self.session_dir = Path(session_dir)
        self.output_dir = Path(output_dir)
        self.frames_dir = self.session_dir / "frames"
        self.store_path = str(store_path)
        cfg = (config or {}).get("phase_r", {}) if config else {}
        self.erosion_px = cfg.get("mask_erosion_px", 8)
        self.depth_stride = cfg.get("depth_stride", 4)
        self.knn_k = cfg.get("knn_k", 6)
        self.max_depth = cfg.get("max_depth_m", None)
        self.min_points = cfg.get("min_points", 30)
        self.vote_max_views = cfg.get("vote_max_views", 20)
        self.obb_trim_percent = float(cfg.get("obb_trim_percent", 1.0))
        # R.7 dynamic classes excluded from pose & fusion
        self.dynamic_labels = dynamic_labels or self._default_dynamic()

    def _default_dynamic(self) -> set[str]:
        try:
            from segmentation.autoprompt.vocabulary import load_vocabulary
            dl = set(load_vocabulary().dynamic_labels())
        except Exception:
            dl = set()
        return dl | {"train", "person", "equipment", "vehicle", "tram"}

    # ── inputs ──────────────────────────────────────────────────────
    def _load_segmentation(self):
        seg = json.load(open(self.output_dir / "segmentation.json"))
        id_to = {i["id"]: i for i in seg.get("instances", [])}
        npz = np.load(self.output_dir / seg.get("mask_file", "seg_masks.npz"))
        masks: dict[int, dict[int, str]] = {}
        for key in npz.files:
            if not key.startswith("f") or "_o" not in key:
                continue
            fpart, opart = key[1:].split("_o")
            fid, oid = int(fpart), int(opart)
            masks.setdefault(oid, {})[fid] = key
        return id_to, npz, masks, seg.get("scene_type")

    def _dubious_labels(self) -> set[str]:
        """Labels the auto-prompter flagged as all-dubious (below the
        confidence gate). Spec item 5: they vote in Phase R but generate NO
        pose residuals until validated (status='dubious' in the store)."""
        p = self.output_dir / "vlm_analysis.json"
        if not p.exists():
            return set()
        try:
            return set(json.load(open(p)).get("dubious_labels", []))
        except Exception:
            return set()

    def _metric_depth_scale(self, depth_dir_rel: str) -> float:
        """Scale to bring per-frame depth into the METRIC pose frame.

        scale_align.apply_scale() multiplies the chunk clouds and camera centres
        by s (marker `.metric_scale_applied`) but NEVER the raw omega
        `results_output/frame_*.npz` depth — it stays up-to-scale. Lifting that
        depth with the metric poses put every instance at ~1/s of its true size
        (volumes wrong by up to s³) — the root cause of wrong assistant
        measurements. DA3 depth (`da3_run/`) is metric already and is untouched."""
        if depth_dir_rel.startswith("da3_run"):
            return 1.0
        marker = self.output_dir / ".metric_scale_applied"
        if not marker.exists():
            return 1.0
        try:
            return float(marker.read_text().strip().split("=")[-1])
        except Exception:
            return 1.0

    def _depth_provider(self):
        for c in ["omega_run/results_output", "da3_run/results_output", "results_output"]:
            d = self.output_dir / c
            if d.is_dir():
                s = self._metric_depth_scale(c)

                def prov(fid, _d=d, _s=s):
                    p = _d / f"frame_{fid}.npz"
                    if not p.exists():
                        return None
                    arr = np.load(p)
                    # tolerate archives without a literal "depth" key (same
                    # fallback as the autoprompt depth provider)
                    key = "depth" if "depth" in arr else (arr.files[0] if arr.files else None)
                    if key is None:
                        return None
                    return arr[key].astype(np.float32) * np.float32(_s)
                return prov
        return lambda fid: None

    # ── build ───────────────────────────────────────────────────────
    def build(self, gravity: np.ndarray | None = None) -> BuildStats:
        from segmentation.session_io import _load_camera_source
        id_to, npz, masks, scene_type = self._load_segmentation()
        cam = _load_camera_source(self.session_dir, self.output_dir)
        if cam is None:
            raise RuntimeError("no camera source (BA poses) — cannot lift depth")
        depth_of = self._depth_provider()
        dubious_labels = self._dubious_labels()
        seg_obbs = load_seg_display_obbs(self.output_dir)

        store = InstanceStore(self.store_path)
        if scene_type:
            store.set_meta("scene_type", scene_type)
        # camera trajectory (raw frame) — feeds the R3D ego tools
        # (get_my_position / get_distance_from_me) in Phase 5
        fids = sorted(cam.pose_map)
        if len(fids) > 100:
            fids = [fids[i] for i in np.linspace(0, len(fids) - 1, 100).astype(int)]
        store.set_meta("camera_positions", json.dumps(
            {int(f): np.asarray(cam.pose_map[f], float)[:3, 3].round(5).tolist()
             for f in fids}))
        marker = self.output_dir / ".metric_scale_applied"
        if marker.exists():
            store.set_meta("metric_scale", marker.read_text().strip().split("=")[-1])
        stats = BuildStats()

        total_px = 0.0
        dyn_px = 0.0
        for oid, frames in masks.items():
            meta = id_to.get(oid, {})
            label = meta.get("label", f"obj{oid}")
            iid = meta.get("instance_id", oid)
            is_dynamic = any(d in label for d in self.dynamic_labels)
            stats.n_instances += 1
            stats.per_label[label] = stats.per_label.get(label, 0) + 1

            # pool lifted points across member frames (+ source frame per point,
            # so R.2/R.3 can split per window through the frame->window map)
            pooled = []
            pooled_fids = []
            member_frames = sorted(frames.keys())
            for fid in member_frames:
                mask = npz[frames[fid]]
                total_px += float((mask > 0).sum())
                if is_dynamic:
                    dyn_px += float((mask > 0).sum())
                    continue  # R.7: dynamic excluded from geometry
                if fid not in cam.pose_map:
                    continue
                depth = depth_of(fid)
                if depth is None:
                    continue
                K = cam.K_for(fid)
                if K is None:
                    continue
                # frame image size (mask ref) vs depth size -> scale K to depth res
                fw, fh = _frame_size(self.frames_dir, fid, mask.shape)
                dh, dw = depth.shape[:2]
                Kd = _scaled_K(np.asarray(K, float), (fw, fh), (dw, dh))
                pts = lift_masked_depth_to_world(
                    depth, mask, Kd, cam.pose_map[fid],
                    stride=self.depth_stride, erosion_px=self.erosion_px,
                    max_depth=self.max_depth)
                if len(pts):
                    pooled.append(pts)
                    pooled_fids.append(np.full(len(pts), fid, np.int32))

            status = "proposed"
            if label in dubious_labels:
                status = "dubious"   # votes, but no pose residuals (spec item 5)
            if is_dynamic:
                status = "dynamic_excluded"
                stats.n_dynamic_excluded += 1
            store.upsert_instance(
                iid, label, known=bool(meta.get("known", False)),
                source="autoprompt", confidence=float(meta.get("confidence", 0.0)),
                status=status, n_views=len(member_frames),
                first_frame=member_frames[0], last_frame=member_frames[-1],
                scene_type=scene_type)
            for fid in member_frames:
                store.add_masklet_ref(iid, fid, frames[fid],
                                      box_xywh=_mask_bbox_xywh(npz[frames[fid]]))

            if is_dynamic or not pooled:
                continue
            allpts = np.vstack(pooled)
            allfids = np.concatenate(pooled_fids)
            keep = filter_outliers_knn(allpts, k=self.knn_k)
            allpts, allfids = allpts[keep], allfids[keep]
            if len(allpts) < self.min_points:
                continue
            store.set_points(iid, allpts, frame_ids=allfids)
            stats.n_with_points += 1
            # canonical OBB: the segmentation OBB (cleaned cloud, what the
            # viewer draws) when present; robust trimmed fit on lifted points
            # otherwise
            if iid in seg_obbs:
                T, aabb = seg_obbs[iid]
                obb = (T, aabb, T[:3, 3])
            else:
                obb = fit_gravity_aligned_obb(allpts, gravity=gravity,
                                              trim_percent=self.obb_trim_percent)
            if obb is not None:
                T, aabb, pos = obb
                store.set_obb(iid, T, aabb, pos, window_id="global",
                              gravity=gravity, n_points=len(allpts))
                stats.n_with_obb += 1
                on = detect_onion(allpts, T, aabb, min_points=self.min_points)
                store.set_onion_metric(iid, on.bimodal, on.separation_m, on.bic_delta)
                if on.bimodal:
                    stats.n_onion += 1

        stats.dynamic_pixel_fraction = (dyn_px / total_px) if total_px else 0.0
        store.set_meta("dynamic_pixel_fraction", str(stats.dynamic_pixel_fraction))

        # R.2 — multi-view plurality vote + entropy (baseline, pre-refinement)
        try:
            views = build_vote_views(self.session_dir, self.output_dir,
                                     dynamic_labels=self.dynamic_labels,
                                     max_views=self.vote_max_views)
            run_scene_vote(store, views)
        except Exception as e:  # vote is a health metric — never break the build
            store.set_meta("vote_error", str(e))

        store.close()
        return stats


def load_seg_display_obbs(output_dir) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Canonical OBBs from segmentation_result.json, converted DISPLAY -> RAW
    store frame (inverse floor_transform).

    These boxes are fit on the CLEANED merged cloud — exactly the geometry the
    viewer renders — while the depth-lifted store points are noisier (thicker
    boxes, shifted centres). Preferring them as the instance's global OBB makes
    the assistant's highlight coincide EXACTLY with the box the user already
    sees, and the measurements quote the cleaned geometry.
    Returns {instance_id: (T_raw 4x4, aabb_raw 6)}."""
    import json
    output_dir = Path(output_dir)
    p = output_dir / "segmentation_result.json"
    if not p.exists():
        return {}
    try:
        seg = json.load(open(p))
    except Exception:
        return {}
    # display = M · raw  (M = [s·R | t] from floor_transform.npz)
    Minv = np.eye(4)
    s = 1.0
    ft = output_dir / "floor_transform.npz"
    if ft.exists():
        try:
            d = np.load(ft)
            s = float(d["s"])
            R = np.asarray(d["R"], float)
            t = np.asarray(d["t"], float)
            M = np.eye(4); M[:3, :3] = s * R; M[:3, 3] = t
            Minv = np.linalg.inv(M)
        except Exception:
            Minv = np.eye(4); s = 1.0
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    R_minv = Minv[:3, :3] * s  # rotation part of M⁻¹ (undo the 1/s factor)
    for inst in seg.get("instances", []):
        obb = inst.get("obb") or {}
        if not obb.get("center") or not obb.get("half_extents"):
            continue
        c = np.asarray(obb["center"], float)
        h = np.asarray(obb["half_extents"], float)
        Rd = np.asarray(obb.get("rotation", np.eye(3)), float)
        T = np.eye(4)
        T[:3, :3] = R_minv @ Rd
        T[:3, 3] = (Minv @ np.append(c, 1.0))[:3]
        h_raw = h / s
        aabb = np.array([-h_raw[0], h_raw[0], -h_raw[1], h_raw[1], -h_raw[2], h_raw[2]])
        iid = inst.get("instance_id", inst.get("id"))
        if iid is not None:
            out[int(iid)] = (T, aabb)
    return out


# ── R.2 shared vote machinery (baseline in build; re-vote in Phase R loop) ──

def build_vote_views(session_dir, output_dir, dynamic_labels: set[str] | None = None,
                     max_views: int = 20) -> list:
    """Build the multi-view vote views (c2w + K at mask resolution + all
    instances' masks per frame). Dynamic instances (R.7) never vote."""
    import json
    from segmentation.session_io import _load_camera_source
    from .vote import View

    session_dir, output_dir = Path(session_dir), Path(output_dir)
    seg = json.load(open(output_dir / "segmentation.json"))
    id_to = {i["id"]: i for i in seg.get("instances", [])}
    npz = np.load(output_dir / seg.get("mask_file", "seg_masks.npz"))
    cam = _load_camera_source(session_dir, output_dir)
    if cam is None:
        return []
    dyn = dynamic_labels or set()

    # frame -> {instance_id: mask_key}, skipping dynamic voters
    per_frame: dict[int, dict[int, str]] = {}
    for key in npz.files:
        if not key.startswith("f") or "_o" not in key:
            continue
        fpart, opart = key[1:].split("_o")
        fid, oid = int(fpart), int(opart)
        meta = id_to.get(oid, {})
        if any(d in meta.get("label", "") for d in dyn):
            continue
        per_frame.setdefault(fid, {})[meta.get("instance_id", oid)] = key

    fids = sorted(f for f in per_frame if f in cam.pose_map)
    if len(fids) > max_views:  # sample evenly to bound the vote cost
        fids = [fids[i] for i in np.linspace(0, len(fids) - 1, max_views).astype(int)]

    views: list[View] = []
    frames_dir = session_dir / "frames"
    for fid in fids:
        K = cam.K_for(fid)
        if K is None:
            continue
        masks = {iid: (npz[k] > 0) for iid, k in per_frame[fid].items()}
        mh, mw = next(iter(masks.values())).shape[:2]
        fw, fh = _frame_size(frames_dir, fid, (mh, mw))
        Km = _scaled_K(np.asarray(K, float), (fw, fh), (mw, mh))
        views.append(View(np.asarray(cam.pose_map[fid], float), Km, masks, (mw, mh),
                          fid=fid))
    return views


def run_scene_vote(store: InstanceStore, views: list, corrections: dict | None = None,
                   cell_m: float = 1.0, persist: bool = True) -> dict:
    """Vote every instance's points against all views (R.2). `corrections` maps
    window_id -> 4x4 similarity applied to BOTH that window's points and views
    (re-vote step of the refinement loop; see PhaseRPipeline). Persists per-point
    entropy/share, per-instance aggregates and per-region entropy when `persist`.
    Returns {"mean_entropy", "n_points"} over the whole scene."""
    from .vote import vote_points_batch, region_entropy

    if not views:
        return {"mean_entropy": 0.0, "n_points": 0}
    all_pts, all_ents = [], []
    mean_sum, n_total = 0.0, 0
    for inst in store.list_instances():
        if inst["status"] == "dynamic_excluded":
            continue
        iid = inst["instance_id"]
        pts = store.get_points(iid)
        if pts is None or not len(pts):
            continue
        if corrections:
            pts = _apply_window_corrections(store, iid, pts, corrections)
        _a, ents, shares = vote_points_batch(pts, views)
        if persist:
            store.set_vote_metrics(iid, float(ents.mean()), float(ents.max()), len(pts))
            store.set_point_votes(iid, ents, shares)
        all_pts.append(pts)
        all_ents.append(ents)
        mean_sum += float(ents.sum())
        n_total += len(pts)
    if persist and all_pts:
        for key, agg in region_entropy(np.vstack(all_pts), np.concatenate(all_ents),
                                       cell_m=cell_m).items():
            store.set_vote_region(key, agg["mean_entropy"], agg["n_points"])
    return {"mean_entropy": (mean_sum / n_total) if n_total else 0.0,
            "n_points": n_total}


def _apply_window_corrections(store: InstanceStore, iid: int, pts: np.ndarray,
                              corrections: dict) -> np.ndarray:
    """Apply per-window 4x4 similarity corrections to an instance's points using
    the per-point source frame (via the corrections' embedded window_map)."""
    window_map = corrections.get("__window_map__", {})
    fids = store.get_point_frames(iid)
    if fids is None or not window_map:
        return pts
    out = pts.copy()
    for w, M in corrections.items():
        if w == "__window_map__":
            continue
        sel = np.array([window_map.get(int(f)) == w for f in fids])
        if sel.any():
            ph = np.concatenate([out[sel], np.ones((int(sel.sum()), 1))], axis=1)
            out[sel] = (np.asarray(M) @ ph.T).T[:, :3]
    return out


def _mask_bbox_xywh(mask) -> np.ndarray | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    return np.array([x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h], np.float32)


def _frame_size(frames_dir: Path, fid: int, mask_shape) -> tuple[int, int]:
    p = frames_dir / f"{fid:06d}.jpg"
    if p.exists():
        try:
            from PIL import Image
            with Image.open(p) as im:
                return im.size  # (w, h)
        except Exception:
            pass
    h, w = mask_shape[:2]
    return (w, h)
