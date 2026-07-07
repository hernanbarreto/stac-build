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

    def _depth_provider(self):
        for c in ["omega_run/results_output", "da3_run/results_output", "results_output"]:
            d = self.output_dir / c
            if d.is_dir():
                def prov(fid, _d=d):
                    p = _d / f"frame_{fid}.npz"
                    if not p.exists():
                        return None
                    arr = np.load(p)
                    return arr["depth"].astype(np.float32) if "depth" in arr else None
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

        store = InstanceStore(self.store_path)
        if scene_type:
            store.set_meta("scene_type", scene_type)
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

            # pool lifted points across member frames
            pooled = []
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

            status = "proposed"
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
                store.add_masklet_ref(iid, fid, frames[fid])

            if is_dynamic or not pooled:
                continue
            allpts = np.vstack(pooled)
            keep = filter_outliers_knn(allpts, k=self.knn_k)
            allpts = allpts[keep]
            if len(allpts) < self.min_points:
                continue
            store.set_points(iid, allpts)
            stats.n_with_points += 1
            obb = fit_gravity_aligned_obb(allpts, gravity=gravity)
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
        store.close()
        return stats


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
