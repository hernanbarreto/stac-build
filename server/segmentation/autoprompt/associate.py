# STAC-Builder — Auto-prompter: geometric temporal association (Phase 1).
#
# Same physical object detected in several keyframes -> ONE instance seeded with
# prompts in multiple frames. Association is GEOMETRIC, driven by the BA poses:
# a detection's box is lifted to 3D (via depth if available, else a planar
# assumption at a representative depth) and reprojected into the other keyframes;
# links are formed by reprojected-box IoU + same label.
#
# INVIOLABLE RULE (spec R.1): association is by GEOMETRY, never by appearance.
# In a tunnel/station identical columns recur every N metres; an appearance match
# between geometrically-disconnected detections creates false loop closures
# (perceptual aliasing) that destroy the reconstruction. There is deliberately NO
# visual-similarity path in this module.
#
# PROVENANCE: ours. Uses the BA poses from session_io (our reconstruction).
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .detector import Detection

# depth_provider(frame_id) -> (depth HxW float32 metres, K 3x3) or None
DepthProvider = Callable[[int], "tuple[np.ndarray, np.ndarray] | None"]


@dataclass
class Member:
    frame_id: int
    box: tuple[float, float, float, float]  # normalized xyxy
    confidence: float


@dataclass
class Instance:
    instance_id: int
    label: str
    members: list[Member] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        # Aggregate confidence = best single-view evidence; multi-view support
        # is a separate, stronger signal recorded in n_views.
        return max((m.confidence for m in self.members), default=0.0)

    @property
    def n_views(self) -> int:
        return len(self.members)

    def best_member(self) -> Member:
        # Largest, most confident view — the crop used for Phase-2 classification.
        def area(m: Member) -> float:
            x1, y1, x2, y2 = m.box
            return (x2 - x1) * (y2 - y1)
        return max(self.members, key=lambda m: area(m) * (0.5 + m.confidence))

    def to_dict(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "n_views": self.n_views,
            "members": [
                {"frame_id": m.frame_id, "box": [round(c, 5) for c in m.box],
                 "confidence": round(m.confidence, 4)}
                for m in self.members
            ],
            "origin": "vlm_proposed",
        }


# ── geometry ────────────────────────────────────────────────────────

def _box_corners_3d(
    box: tuple[float, float, float, float], K: np.ndarray, img_wh: tuple[int, int],
    depth_center: float,
) -> np.ndarray:
    """Lift a normalized box to 4 3D corners in the CAMERA frame, assuming a
    fronto-parallel rectangle at `depth_center` (metres). Returns (4,3)."""
    w, h = img_wh
    x1, y1, x2, y2 = box
    px = np.array([x1, x2, x2, x1]) * w
    py = np.array([y1, y1, y2, y2]) * h
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z = depth_center
    X = (px - cx) / fx * z
    Y = (py - cy) / fy * z
    Z = np.full(4, z)
    return np.stack([X, Y, Z], axis=1)  # (4,3)


def _project_box(
    corners_cam_i: np.ndarray, c2w_i: np.ndarray, c2w_j: np.ndarray,
    K_j: np.ndarray, img_wh_j: tuple[int, int],
) -> tuple[float, float, float, float] | None:
    """Project camera-i 3D corners into frame j; return normalized xyxy box."""
    w, h = img_wh_j
    corners_h = np.concatenate([corners_cam_i, np.ones((4, 1))], axis=1)  # (4,4)
    world = (c2w_i @ corners_h.T).T                                        # (4,4)
    cam_j = (np.linalg.inv(c2w_j) @ world.T).T                            # (4,4)
    Z = cam_j[:, 2]
    if np.any(Z <= 1e-6):
        return None  # behind camera j
    fx, fy = K_j[0, 0], K_j[1, 1]
    cx, cy = K_j[0, 2], K_j[1, 2]
    u = fx * cam_j[:, 0] / Z + cx
    v = fy * cam_j[:, 1] / Z + cy
    x1, x2 = u.min() / w, u.max() / w
    y1, y2 = v.min() / h, v.max() / h
    return (x1, y1, x2, y2)


def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


class _UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


# ── main entry ──────────────────────────────────────────────────────

def associate_detections(
    detections_by_frame: dict[int, list[Detection]],
    pose_map: dict[int, np.ndarray] | None,
    K_for: Callable[[int], "np.ndarray | None"] | None,
    img_wh_for: Callable[[int], tuple[int, int]],
    depth_provider: DepthProvider | None = None,
    iou_threshold: float = 0.3,
    assumed_depth_m: float = 4.0,
    start_instance_id: int = 1,
) -> list[Instance]:
    """Group per-frame detections into geometrically-consistent instances.

    If poses/K are available, links two same-label detections in different
    frames when each reprojects onto the other with IoU > threshold. Without
    poses (fallback), each detection becomes its own instance — SAM3 tracking
    still merges within a window; we NEVER link by appearance.
    """
    # Flatten to an indexed list.
    flat: list[tuple[int, Detection]] = []
    for fid, dets in detections_by_frame.items():
        for d in dets:
            flat.append((fid, d))
    n = len(flat)
    uf = _UnionFind(n)

    have_geometry = pose_map is not None and K_for is not None and len(pose_map) > 0

    if have_geometry:
        # Precompute camera-frame 3D corners per detection (depth at box center).
        corners: list[np.ndarray | None] = []
        for fid, d in flat:
            K = K_for(fid)
            if K is None or fid not in pose_map:
                corners.append(None)
                continue
            wh = img_wh_for(fid)
            zc = _center_depth(d, depth_provider, fid, wh, assumed_depth_m)
            corners.append(_box_corners_3d(d.box, K, wh, zc))

        for i in range(n):
            fi, di = flat[i]
            if corners[i] is None or fi not in pose_map:
                continue
            for j in range(i + 1, n):
                fj, dj = flat[j]
                if fi == fj or di.label != dj.label:
                    continue
                if corners[j] is None or fj not in pose_map:
                    continue
                # project i -> j and j -> i, require both consistent
                pj = _project_box(corners[i], pose_map[fi], pose_map[fj], K_for(fj), img_wh_for(fj))
                pi = _project_box(corners[j], pose_map[fj], pose_map[fi], K_for(fi), img_wh_for(fi))
                if pj is None or pi is None:
                    continue
                if _iou(pj, dj.box) >= iou_threshold and _iou(pi, di.box) >= iou_threshold:
                    uf.union(i, j)

    # Build instances from union-find roots.
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)

    instances: list[Instance] = []
    iid = start_instance_id
    for _, idxs in sorted(groups.items()):
        fid0, d0 = flat[idxs[0]]
        inst = Instance(instance_id=iid, label=d0.label)
        for k in idxs:
            fid, d = flat[k]
            inst.members.append(Member(frame_id=fid, box=d.box, confidence=d.confidence))
        inst.members.sort(key=lambda m: m.frame_id)
        instances.append(inst)
        iid += 1
    return instances


def _center_depth(
    d: Detection, depth_provider: DepthProvider | None, fid: int,
    img_wh: tuple[int, int], assumed_depth_m: float,
) -> float:
    if depth_provider is None:
        return assumed_depth_m
    got = depth_provider(fid)
    if got is None:
        return assumed_depth_m
    depth, _K = got
    h, w = depth.shape[:2]
    x1, y1, x2, y2 = d.box
    cx = int(np.clip((x1 + x2) / 2 * w, 0, w - 1))
    cy = int(np.clip((y1 + y2) / 2 * h, 0, h - 1))
    # median over a small window, ignoring invalid depths
    x0, xe = max(0, cx - 3), min(w, cx + 4)
    y0, ye = max(0, cy - 3), min(h, cy + 4)
    patch = depth[y0:ye, x0:xe]
    valid = patch[(patch > 0) & np.isfinite(patch)]
    if valid.size == 0:
        return assumed_depth_m
    return float(np.median(valid))
