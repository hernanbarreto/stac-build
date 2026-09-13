"""Synthetic ground-truth generator for the metric-cloud tests (claude_stac.txt
§12) — no GPU, no model: a corridor loop around a block, analytic depth
rendering against planes / cylinders / boxes, chunk predictions with injected
per-chunk Sim3 drift, loop bridges with their own gauge, DA3-like anchors,
synthetic SAM3 masks, movable objects, floaters and inconsistent pixels.

Conventions (identical to the production data):
  * world Y-up, floor at y = 0; cameras OpenCV (+X right, +Y down, +Z forward);
  * a chunk dict = {'world_points' (S,H,W,3), 'world_points_conf' (S,H,W),
    'depth' (S,H,W), 'extrinsic' (S,4,4) c2w, 'intrinsic' (S,3,3),
    'images' (S,3,H,W)};
  * masks keyed ``f<frame>_o<oid>`` with oid = instance_id − 1 on the
    prediction grid; per-point provenance (frame_global, pixel_row, pixel_col).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ── primitives ──────────────────────────────────────────────────────────────

@dataclass
class Plane:
    p0: np.ndarray
    n: np.ndarray
    u: np.ndarray
    v: np.ndarray
    hu: float
    hv: float
    oid: int
    label: str


@dataclass
class Cylinder:
    c: np.ndarray        # point on the (vertical) axis, y = base
    r: float
    h: float
    oid: int
    label: str


@dataclass
class Box:
    lo: np.ndarray
    hi: np.ndarray
    oid: int
    label: str


@dataclass
class Scene:
    prims: List[object] = field(default_factory=list)
    labels: Dict[int, str] = field(default_factory=dict)

    def add(self, p):
        self.prims.append(p)
        self.labels[p.oid] = p.label
        return p


def _unit(v):
    v = np.asarray(v, np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def wall(scene: Scene, p0, n, u, hu, hv, oid, label="wall", y_center=1.5):
    p0 = np.asarray(p0, np.float64)
    p0[1] = y_center
    n = _unit(n)
    u = _unit(u)
    v = _unit(np.cross(n, u))
    return scene.add(Plane(p0, n, u, v, hu, hv, oid, label))


def corridor_loop_scene(ceiling_h: float = 3.0) -> Scene:
    """Outer room 24 x 16 m (x ∈ ±12, z ∈ ±8) with an inner block 12 x 6 m
    (x ∈ ±6, z ∈ ±3): a 5 m wide corridor loop. Floor + ceiling, four outer
    walls, four block walls, columns along the outer walls, two boxes."""
    sc = Scene()
    # floor / ceiling
    sc.add(Plane(np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
                 np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), 12.0, 8.0, 1, "floor"))
    sc.add(Plane(np.array([0.0, ceiling_h, 0.0]), np.array([0.0, -1.0, 0.0]),
                 np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), 12.0, 8.0, 2, "ceiling"))
    # outer walls (normals inward)
    wall(sc, [12.0, 0, 0], [-1, 0, 0], [0, 0, 1], 8.0, ceiling_h / 2, 3)
    wall(sc, [-12.0, 0, 0], [1, 0, 0], [0, 0, 1], 8.0, ceiling_h / 2, 4)
    wall(sc, [0, 0, 8.0], [0, 0, -1], [1, 0, 0], 12.0, ceiling_h / 2, 5)
    wall(sc, [0, 0, -8.0], [0, 0, 1], [1, 0, 0], 12.0, ceiling_h / 2, 6)
    # block walls (normals outward, toward the corridor)
    wall(sc, [6.0, 0, 0], [1, 0, 0], [0, 0, 1], 3.0, ceiling_h / 2, 7)
    wall(sc, [-6.0, 0, 0], [-1, 0, 0], [0, 0, 1], 3.0, ceiling_h / 2, 8)
    wall(sc, [0, 0, 3.0], [0, 0, 1], [1, 0, 0], 6.0, ceiling_h / 2, 9)
    wall(sc, [0, 0, -3.0], [0, 0, -1], [1, 0, 0], 6.0, ceiling_h / 2, 10)
    # columns along the long outer walls
    oid = 11
    for x in (-8.0, -4.0, 0.0, 4.0, 8.0):
        for z in (7.2, -7.2):
            sc.add(Cylinder(np.array([x, 0.0, z]), 0.3, ceiling_h, oid, "column"))
            oid += 1
    # movable boxes
    sc.add(Box(np.array([9.0, 0.0, -6.0]), np.array([10.0, 0.8, -5.0]), oid, "box"))
    oid += 1
    sc.add(Box(np.array([-9.5, 0.0, 5.5]), np.array([-8.5, 0.6, 6.5]), oid, "box"))
    return sc


def corridor_scene(length_m: float = 40.0, width_m: float = 4.0, ceiling_h: float = 3.0,
                   step_at_x: Optional[float] = None, step_h: float = 0.30) -> Scene:
    """A straight corridor along +x (x ∈ [0, L], z ∈ ±w/2): floor (optionally
    with a REAL step of ``step_h`` beyond ``step_at_x``), ceiling, two long
    walls, columns every 8 m on the +z wall, beams across the ceiling."""
    sc = Scene()
    hw = width_m / 2.0
    if step_at_x is None:
        sc.add(Plane(np.array([length_m / 2, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
                     np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), length_m / 2, hw, 1, "floor"))
    else:
        sc.add(Plane(np.array([step_at_x / 2, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
                     np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), step_at_x / 2, hw, 1, "floor"))
        sc.add(Plane(np.array([(length_m + step_at_x) / 2, step_h, 0.0]), np.array([0.0, 1.0, 0.0]),
                     np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]),
                     (length_m - step_at_x) / 2, hw, 1, "floor"))
        sc.add(Plane(np.array([step_at_x, step_h / 2, 0.0]), np.array([-1.0, 0.0, 0.0]),
                     np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]), hw, step_h / 2, 20, "riser"))
    sc.add(Plane(np.array([length_m / 2, ceiling_h, 0.0]), np.array([0.0, -1.0, 0.0]),
                 np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), length_m / 2, hw, 2, "ceiling"))
    wall(sc, [length_m / 2, 0, hw], [0, 0, -1], [1, 0, 0], length_m / 2, ceiling_h / 2, 3)
    wall(sc, [length_m / 2, 0, -hw], [0, 0, 1], [1, 0, 0], length_m / 2, ceiling_h / 2, 4)
    wall(sc, [length_m, 0, 0], [-1, 0, 0], [0, 0, 1], hw, ceiling_h / 2, 5)
    wall(sc, [0.0, 0, 0], [1, 0, 0], [0, 0, 1], hw, ceiling_h / 2, 6)
    oid = 7
    for x in np.arange(4.0, length_m - 2.0, 8.0):
        sc.add(Cylinder(np.array([x, 0.0, hw - 0.4]), 0.25, ceiling_h, oid, "column"))
        oid += 1
    for x in np.arange(6.0, length_m - 2.0, 8.0):
        sc.add(Box(np.array([x - 0.15, ceiling_h - 0.4, -hw]), np.array([x + 0.15, ceiling_h, hw]),
                   oid, "beam"))
        oid += 1
    return sc


def out_and_back_trajectory(n_kf: int, length_m: float = 40.0, y: float = 1.5,
                            lane_z: float = 0.5, yaw_offset_deg: float = 35.0) -> np.ndarray:
    """Walk +x along z = −lane, turn, walk back along z = +lane: the walk
    returns to its start (one loop at the end) and every wall/floor/column is
    seen twice from the two lanes. The camera looks ahead-and-toward the far
    wall of its lane."""
    half = n_kf // 2
    poses = []
    for k in range(n_kf):
        if k < half:
            x = 1.0 + (length_m - 2.0) * k / max(half - 1, 1)
            pos, fwd = (x, y, -lane_z), np.array([1.0, 0.0, 0.0])
            a = np.radians(yaw_offset_deg)
        else:
            x = (length_m - 1.0) - (length_m - 2.0) * (k - half) / max(n_kf - half - 1, 1)
            pos, fwd = (x, y, lane_z), np.array([-1.0, 0.0, 0.0])
            a = np.radians(yaw_offset_deg)
        Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        poses.append(look_c2w(pos, Ry @ fwd))
    return np.stack(poses)


# ── cameras ─────────────────────────────────────────────────────────────────

def intrinsics(H: int, W: int, fov_deg: float = 70.0) -> np.ndarray:
    f = 0.5 * W / np.tan(np.radians(fov_deg) / 2)
    return np.array([[f, 0.0, W / 2.0], [0.0, f, H / 2.0], [0.0, 0.0, 1.0]])


def look_c2w(pos, forward, down=(0.0, -1.0, 0.0)) -> np.ndarray:
    f = _unit(forward)
    d = _unit(down)
    r = _unit(np.cross(d, f))
    d = _unit(np.cross(f, r))
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2] = r, d, f
    M[:3, 3] = np.asarray(pos, np.float64)
    return M


def loop_trajectory(n_kf: int, hx: float = 9.0, hz: float = 5.5, y: float = 1.5,
                    yaw_offset_deg: float = 35.0, extra_laps: float = 0.12) -> np.ndarray:
    """Rectangular walk around the block, returning past the start (the loop
    closure) — `extra_laps` fraction of a lap beyond one full lap. The camera
    looks toward the OUTER wall, yaw_offset from the motion direction."""
    P = 2 * (2 * hx + 2 * hz)
    s = np.linspace(0.0, P * (1.0 + extra_laps), n_kf, endpoint=False)
    poses = []
    for si in s:
        d = si % P
        if d < 2 * hx:                       # +x edge at z = -hz
            pos, fwd = (-hx + d, y, -hz), (1.0, 0.0, 0.0)
        elif d < 2 * hx + 2 * hz:            # +z edge at x = +hx
            pos, fwd = (hx, y, -hz + (d - 2 * hx)), (0.0, 0.0, 1.0)
        elif d < 4 * hx + 2 * hz:            # -x edge at z = +hz
            pos, fwd = (hx - (d - 2 * hx - 2 * hz), y, hz), (-1.0, 0.0, 0.0)
        else:                                # -z edge at x = -hx
            pos, fwd = (-hx, y, hz - (d - 4 * hx - 2 * hz)), (0.0, 0.0, -1.0)
        fwd = np.asarray(fwd)
        # rotate the view toward the outer wall (right-hand side of the walk)
        a = np.radians(yaw_offset_deg)
        Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        poses.append(look_c2w(pos, Ry @ fwd))
    return np.stack(poses)


# ── rendering ───────────────────────────────────────────────────────────────

def render(scene: Scene, c2w: np.ndarray, K: np.ndarray, H: int, W: int,
           max_depth: float = 40.0):
    """(depth (H,W), oid (H,W) int, world points (H,W,3)) by analytic ray
    casting. depth = camera-z; NaN / 0 where nothing is hit."""
    vv, uu = np.meshgrid(np.arange(H, dtype=np.float64), np.arange(W, dtype=np.float64),
                         indexing="ij")
    dir_cam = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1],
                        np.ones_like(uu)], axis=-1).reshape(-1, 3)
    R, o = c2w[:3, :3], c2w[:3, 3]
    d = dir_cam @ R.T                       # unnormalised: t == camera z
    best = np.full(len(d), np.inf)
    oid = np.zeros(len(d), np.int32)
    for p in scene.prims:
        if isinstance(p, Plane):
            den = d @ p.n
            with np.errstate(divide="ignore", invalid="ignore"):
                t = ((p.p0 - o) @ p.n) / den
            hit = np.isfinite(t) & (t > 1e-3) & (t < max_depth)
            tt = np.where(hit, t, 0.0)
            X = o + tt[:, None] * d
            du = (X - p.p0) @ p.u
            dv = (X - p.p0) @ p.v
            hit &= (np.abs(du) <= p.hu) & (np.abs(dv) <= p.hv)
        elif isinstance(p, Cylinder):
            ox, oz = o[0] - p.c[0], o[2] - p.c[2]
            a = d[:, 0] ** 2 + d[:, 2] ** 2
            b = 2 * (ox * d[:, 0] + oz * d[:, 2])
            c = ox * ox + oz * oz - p.r * p.r
            disc = b * b - 4 * a * c
            hit = disc > 0
            t = np.full(len(d), np.inf)
            sq = np.sqrt(np.where(hit, disc, 0.0))
            with np.errstate(divide="ignore", invalid="ignore"):
                t1 = (-b - sq) / (2 * a)
                t2 = (-b + sq) / (2 * a)
            t = np.where(t1 > 1e-3, t1, t2)
            hit &= (t > 1e-3) & (t < max_depth)
            y = o[1] + t * d[:, 1]
            hit &= (y >= p.c[1]) & (y <= p.c[1] + p.h)
        else:  # Box, slab method
            with np.errstate(divide="ignore", invalid="ignore"):
                t0 = (p.lo - o) / d
                t1 = (p.hi - o) / d
            tmin = np.maximum.reduce(np.minimum(t0, t1), axis=1)
            tmax = np.minimum.reduce(np.maximum(t0, t1), axis=1)
            hit = (tmax >= tmin) & (tmax > 1e-3)
            t = np.where(tmin > 1e-3, tmin, tmax)
            hit &= (t < max_depth)
        closer = hit & (t < best)
        best[closer] = t[closer]
        oid[closer] = p.oid
    depth = np.where(np.isfinite(best), best, 0.0).reshape(H, W)
    X = (o + best[:, None] * d)
    X[~np.isfinite(best)] = 0.0
    return depth, oid.reshape(H, W), X.reshape(H, W, 3)


# ── session ─────────────────────────────────────────────────────────────────

@dataclass
class Session:
    scene: Scene
    poses: np.ndarray          # (N,4,4) GT c2w
    K: np.ndarray
    H: int
    W: int
    depth: np.ndarray          # (N,H,W)
    oid: np.ndarray            # (N,H,W)
    points: np.ndarray         # (N,H,W,3) GT world
    frame_numbers: np.ndarray  # real frame numbers (N,)

    @property
    def n_kf(self):
        return len(self.poses)


def make_session(n_kf: int = 150, H: int = 48, W: int = 64, seed: int = 0,
                 scene: Optional[Scene] = None, frame_step: int = 30,
                 poses: Optional[np.ndarray] = None) -> Session:
    scene = scene or corridor_loop_scene()
    poses = loop_trajectory(n_kf) if poses is None else np.asarray(poses, np.float64)
    n_kf = len(poses)
    K = intrinsics(H, W)
    depth = np.zeros((n_kf, H, W), np.float64)
    oid = np.zeros((n_kf, H, W), np.int32)
    pts = np.zeros((n_kf, H, W, 3), np.float64)
    for i in range(n_kf):
        depth[i], oid[i], pts[i] = render(scene, poses[i], K, H, W)
    return Session(scene, poses, K, H, W, depth, oid, pts,
                   frame_numbers=np.arange(n_kf) * frame_step)


# ── Sim3 helpers ────────────────────────────────────────────────────────────

def yaw_R(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def sim3_apply(X, s, R, t):
    return s * (np.asarray(X) @ np.asarray(R).T) + np.asarray(t)


def sim3_pose(c2w, s, R, t):
    M = np.asarray(c2w, np.float64).copy()
    M[:3, :3] = R @ M[:3, :3]
    M[:3, 3] = s * (R @ M[:3, 3]) + t
    return M


def chunk_ranges(n: int, size: int, overlap: int) -> List[Tuple[int, int]]:
    if n <= size:
        return [(0, n)]
    step = size - overlap
    out, start = [], 0
    while start < n:
        end = min(start + size, n)
        out.append((start, end))
        if end >= n:
            break
        start += step
    return out


def drift_field(n_kf: int, yaw_deg_per_kf: float = 0.0, t_per_kf=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Accumulated per-keyframe rigid drift D_g = Exp(g·ξ) (N,4,4) — the
    feed-forward error that grows along the walk and that no per-chunk rigid
    seam can remove (the keyframe graph's job)."""
    D = np.zeros((n_kf, 4, 4))
    for g in range(n_kf):
        D[g] = np.eye(4)
        D[g, :3, :3] = yaw_R(yaw_deg_per_kf * g)
        D[g, :3, 3] = np.asarray(t_per_kf, np.float64) * g
    return D


def chain_drift(poses: np.ndarray, steps: np.ndarray) -> np.ndarray:
    """Realistic odometric drift: a small LOCAL perturbation Exp(steps[g]) of
    every relative motion, integrated along the chain (the feed-forward error
    of one step is small; the far-field displacement is the accumulated
    rotation times the lever arm). Returns D (N,4,4) with T'_g = D_g · T_g."""
    import sys
    from pathlib import Path as _P
    vendor = _P(__file__).resolve().parents[2] / "vendor" / "VGGT-Long"
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    from loop_utils.lie import se3_exp, se3_inv
    P = np.asarray(poses, np.float64)
    n = len(P)
    T = [P[0].copy()]
    for g in range(1, n):
        Z = se3_inv(P[g - 1]) @ P[g]
        T.append(T[-1] @ Z @ se3_exp(np.asarray(steps[g - 1], np.float64)))
    return np.stack([T[g] @ se3_inv(P[g]) for g in range(n)])


def make_chunks(sess: Session, chunk_size: int = 60, overlap: int = 30,
                scale_err: Sequence[float] = None, yaw_err_deg: Sequence[float] = None,
                t_err: Sequence[Sequence[float]] = None, depth_noise_rel: float = 0.003,
                seed: int = 0, drift: Optional[np.ndarray] = None):
    """Chunk prediction dicts in each chunk's OWN gauge: GT geometry under an
    injected Sim3 error E_k = (s_k, R_k, t_k), plus multiplicative depth
    noise. ``drift`` (N,4,4): per-keyframe accumulated rigid drift applied to
    the frame's points AND camera before the chunk gauge (shared frames carry
    the same drift in both chunks, so exact seams stay exact while the walk
    as a whole is bent). Returns (chunks, chunk_indices, errors)."""
    rng = np.random.default_rng(seed)
    ci = chunk_ranges(sess.n_kf, chunk_size, overlap)
    n = len(ci)
    scale_err = list(scale_err) if scale_err is not None else [1.0] * n
    yaw_err_deg = list(yaw_err_deg) if yaw_err_deg is not None else [0.0] * n
    t_err = list(t_err) if t_err is not None else [(0.0, 0.0, 0.0)] * n
    chunks, errors = [], []
    for k, (a, b) in enumerate(ci):
        s, R, t = float(scale_err[k]), yaw_R(float(yaw_err_deg[k])), np.asarray(t_err[k], np.float64)
        errors.append((s, R, t))
        S = b - a
        wp = np.zeros((S, sess.H, sess.W, 3), np.float32)
        dep = np.zeros((S, sess.H, sess.W), np.float32)
        conf = np.zeros((S, sess.H, sess.W), np.float32)
        ext = np.zeros((S, 4, 4), np.float64)
        Ks = np.tile(sess.K[None], (S, 1, 1))
        for li, g in enumerate(range(a, b)):
            valid = sess.depth[g] > 0
            noise = 1.0 + depth_noise_rel * rng.standard_normal(sess.depth[g].shape)
            z = sess.depth[g] * noise
            cam = sess.poses[g]
            # re-lift the noisy depth along the GT rays, then the chunk gauge
            vv, uu = np.meshgrid(np.arange(sess.H), np.arange(sess.W), indexing="ij")
            dcam = np.stack([(uu - sess.K[0, 2]) / sess.K[0, 0],
                             (vv - sess.K[1, 2]) / sess.K[1, 1], np.ones_like(uu)], -1)
            P = cam[:3, 3] + (dcam * z[..., None]) @ cam[:3, :3].T
            if drift is not None:
                Dg = drift[g]
                P = P @ Dg[:3, :3].T + Dg[:3, 3]
                cam = Dg @ cam
            P = sim3_apply(P.reshape(-1, 3), s, R, t).reshape(sess.H, sess.W, 3)
            wp[li] = np.where(valid[..., None], P, 0.0)
            dep[li] = np.where(valid, s * z, 0.0)
            conf[li] = np.where(valid, 1.0, 0.0)
            ext[li] = sim3_pose(cam, s, R, t)
        chunks.append({"world_points": wp, "world_points_conf": conf, "depth": dep,
                       "extrinsic": ext, "intrinsic": Ks,
                       "images": np.zeros((S, 3, sess.H, sess.W), np.float32), "mask": None})
    return chunks, ci, errors


def make_bridge(sess: Session, item, s_L: float = 1.0, yaw_L_deg: float = 0.0,
                t_L=(0.0, 0.0, 0.0), depth_noise_rel: float = 0.003, seed: int = 1,
                corrupt_b: float = 0.0):
    """Bridge prediction over windows item[1] and item[3] in its OWN gauge
    E_L. ``corrupt_b`` adds a non-rigid perturbation (metres, per pixel) to the
    b-window — a broken/false bridge the geometric check must reject."""
    rng = np.random.default_rng(seed)
    (a0, a1), (b0, b1) = item[1], item[3]
    frames = list(range(a0, a1)) + list(range(b0, b1))
    S = len(frames)
    R_L, t_L = yaw_R(yaw_L_deg), np.asarray(t_L, np.float64)
    wp = np.zeros((S, sess.H, sess.W, 3), np.float32)
    dep = np.zeros((S, sess.H, sess.W), np.float32)
    conf = np.zeros((S, sess.H, sess.W), np.float32)
    ext = np.zeros((S, 4, 4), np.float64)
    for li, g in enumerate(frames):
        valid = sess.depth[g] > 0
        z = sess.depth[g] * (1.0 + depth_noise_rel * rng.standard_normal(sess.depth[g].shape))
        cam = sess.poses[g]
        vv, uu = np.meshgrid(np.arange(sess.H), np.arange(sess.W), indexing="ij")
        dcam = np.stack([(uu - sess.K[0, 2]) / sess.K[0, 0],
                         (vv - sess.K[1, 2]) / sess.K[1, 1], np.ones_like(uu)], -1)
        P = cam[:3, 3] + (dcam * z[..., None]) @ cam[:3, :3].T
        if corrupt_b > 0 and li >= (a1 - a0):
            P = P + corrupt_b * rng.standard_normal(P.shape)
        P = sim3_apply(P.reshape(-1, 3), s_L, R_L, t_L).reshape(sess.H, sess.W, 3)
        wp[li] = np.where(valid[..., None], P, 0.0)
        dep[li] = np.where(valid, s_L * z, 0.0)
        conf[li] = np.where(valid, 1.0, 0.0)
        ext[li] = sim3_pose(cam, s_L, R_L, t_L)
    return {"world_points": wp, "world_points_conf": conf, "depth": dep, "extrinsic": ext,
            "intrinsic": np.tile(sess.K[None], (S, 1, 1)),
            "images": np.zeros((S, 3, sess.H, sess.W), np.float32), "mask": None}


def write_anchors(sess: Session, anchor_dir: Path, frames: Sequence[int],
                  noise_rel: float = 0.08, seed: int = 2) -> None:
    """DA3-like metric depth npz for the given keyframes: GT depth with a
    per-frame multiplicative bias (monocular scale jitter)."""
    rng = np.random.default_rng(seed)
    anchor_dir = Path(anchor_dir)
    anchor_dir.mkdir(parents=True, exist_ok=True)
    for g in frames:
        bias = 1.0 + noise_rel * rng.standard_normal()
        d = (sess.depth[g] * bias).astype(np.float32)
        np.savez_compressed(anchor_dir / f"frame_{int(sess.frame_numbers[g])}.npz",
                            depth=d, conf=np.where(d > 0, 1.0, 0.0).astype(np.float32))


# ── config for the fork (Model.loops / Model.scale / metric_lock) ───────────

def fork_loops_cfg(stac_server_dir: Optional[str] = None, **over) -> dict:
    d = {"anchors_per_bridge": 3, "max_edge_sigma_m": 0.05, "max_residual_m": 0.10,
         "min_correspondences": 500, "corr_per_frame": 2000, "fit_sample": 200000,
         "scale_tol_log": 0.05, "scale_break_sigma_factor": 4.0, "starved_sigma_m": 0.30,
         "ambiguous_sigma_factor": 3.0, "attention_verify": False, "attention_min_score": 0.5,
         "movable_labels": ["box", "person"], "min_shared_structural_labels": 1,
         "intra_chunk_loops": True, "bridge_extra_frames": 0,
         "spatial": {"drift_floor_m": 0.30, "drift_rate_m_per_m": 0.013,
                     "drift_floor_deg": 2.0, "drift_rate_deg_per_m": 0.10,
                     "identity_reject_factor": 3.0, "frustum_margin_px": 4.0, "occlusion_tol_m": 0.30,
                     "min_depth_m": 0.3, "max_depth_m": 15.0, "min_frustum_frames": 2,
                     "frustum_window_kf": 3, "frustum_points": 800, "min_visible_frac": 0.30,
                     "size_tol": 0.15, "repetitive_labels": ["column"],
                     "min_context_instances": 1, "corridor_width_m": 3.0,
                     "corridor_min_frustum_frames": 4, "dims_pct_lo": 5.0, "dims_pct_hi": 95.0,
                     "same_surface_angle_deg": 10.0, "same_surface_offset_m": 0.10,
                     "same_surface_planar_ratio": 0.05, "same_surface_axis_ratio": 0.15},
         "stac_server_dir": stac_server_dir}
    d.update(over)
    return d


def fork_scale_cfg(**over) -> dict:
    d = {"sigma_loop": 0.005, "sigma_vio": 0.03, "sigma_regulated": 0.01,
         "verify_max_dev": 0.10, "vio_segment_s": 5.0, "vio_min_segments_chunk": 3,
         "vio_min_seg_disp_m": 0.5, "break_localisation_gap": 2.0}
    d.update(over)
    return d


def fork_graph_cfg(**over) -> dict:
    d = {"sigma_odo_intra_m": 0.01, "sigma_odo_intra_deg": 0.2, "loop_sigma_rot_deg": 1.0,
         "sigma_gravity_deg": 2.0, "huber_delta_m": 0.10, "huber_delta_deg": 2.0,
         "dense_max_unknowns": 12000, "lambda_init": 1e-4, "lambda_max": 1e12,
         "lm_diag_floor": 1e-9, "tol": 1e-8, "rel_tol": 1e-6, "max_iters": 50,
         "pcg_tol": 1e-10, "pcg_max_iters": 2000, "min_loop_gain": 0.5,
         "max_seam_degradation_m": 0.005, "holdout_offsets": [4, 10], "holdout_stride": 3,
         "holdout_samples": 4000, "holdout_max_nn_m": 0.10, "run_without_loops": False}
    d.update(over)
    return d


def fork_authority_cfg(**over) -> dict:
    d = {"saturation_warn": 0.8, "pose_graph_max_m": 1.0, "pose_graph_max_deg": 5.0,
         "scale_graph_max_log": 0.10, "depth_graph_max_log_a": 0.10,
         "depth_graph_max_b_m": 0.50, "intra_chunk_max_m": 0.15}
    d.update(over)
    return d


def structural_cfg(**over) -> dict:
    d = {"floor_datum": {"enabled": True, "sigma_angle_deg": 1.0, "sigma_offset_m": 0.02,
                         "max_tilt_deg": 10.0, "reference_span_kf": 15, "step_demote_m": 0.15, "low_band_pct": 5.0,
                         "band_m": 0.5, "min_points": 200, "ransac_tol_m": 0.02,
                         "ransac_iters": 200},
         "wall_planarity": {"enabled": True, "wall_tol_m": 0.02, "sigma_angle_deg": 1.0,
                            "sigma_offset_m": 0.02, "min_span_m": 4.0, "min_points_per_kf": 100,
                            "labels": ["wall"], "reference_span_kf": 15,
                            "planar_ratio": 0.05, "min_patch_extent_m": 1.0},
         "column_vertical": {"enabled": True, "sigma_deg": 1.0, "labels": ["column"],
                             "min_points_per_kf": 50, "axis_ratio": 0.15},
         "repeated_parallel": {"enabled": True, "sigma_deg": 1.0, "labels": ["beam"],
                               "min_points_per_kf": 50, "axis_ratio": 0.15},
         "regulated_dims": []}
    for k, val in over.items():
        node = d
        parts = k.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = val
    return d


def raw_server_cfg(**over) -> dict:
    """The server-side config dict for reconstruction.loops.config (mirrors
    server/config.yaml; every key present)."""
    lp = fork_loops_cfg()
    sp = lp.pop("spatial")
    lp.pop("stac_server_dir")
    lp.pop("bridge_extra_frames")
    raw = {
        "correction_graph": {"loop": lp, "graph": fork_graph_cfg()},
        "authority": fork_authority_cfg(),
        "structural": structural_cfg(),
        "certify": {"ensemble_offset_frames": 0, "keep_aligned_chunks": True},
        "loops": {"min_gap_keyframes": 30, "duplicate_min_sep_m": 0.20, "dbscan_eps_m": 0.15,
                  "dbscan_min_samples": 20, "cluster_min_points": 300, "bridge_extra_frames": 4,
                  "coverage_radius_m": 5.0, "min_coverage": 0.5,
                  "spatial": sp,
                  "semantic": {"enabled": False, "max_tokens": 256, "crops_per_instance": 1,
                               "default_class": "structural"}},
        "scale": fork_scale_cfg(),
    }
    for dotted, val in over.items():
        node = raw
        parts = dotted.split(".")
        for p in parts[:-1]:
            node = node[p]
        if val is None:
            node.pop(parts[-1], None)
        else:
            node[parts[-1]] = val
    return raw


# ── an on-disk session (for the instance detector and the fork integration) ─

def write_session_dir(root: Path, sess: Session, instances: Dict[int, dict],
                      point_stride: int = 1, drift_by_kf: Optional[np.ndarray] = None,
                      write_masks: bool = True) -> Path:
    """Write the production session layout: frames/ (empty jpg placeholders +
    frame_quality.json), output/cleaned_cloud.ply (+ provenance),
    camera_poses.txt, camera_frames.txt, intrinsic.txt, frame_list.json,
    segmentation_result.json, segmentation.json, seg_masks.npz.

    instances: {instance_id: {"label": str, "oids": [scene oids]}} — the
    SAM3 identity → which scene primitives it covers (a fused identity lists
    two distant columns). ``drift_by_kf`` (N,4,4) optional per-keyframe rigid
    error applied to points AND poses (the accumulated drift of the walk)."""
    root = Path(root)
    frames_dir = root / "frames"
    out = root / "output"
    frames_dir.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    N, H, W = sess.n_kf, sess.H, sess.W
    xyz, fg, pr, pc, conf, oid_pts = [], [], [], [], [], []
    poses = sess.poses.copy()
    for g in range(N):
        valid = sess.depth[g] > 0
        rr, cc = np.nonzero(valid)
        rr, cc = rr[::point_stride], cc[::point_stride]
        P = sess.points[g][rr, cc]
        if drift_by_kf is not None:
            D = drift_by_kf[g]
            P = P @ D[:3, :3].T + D[:3, 3]
            poses[g] = D @ sess.poses[g]
        xyz.append(P)
        fg.append(np.full(len(rr), int(sess.frame_numbers[g]), np.int32))
        pr.append(rr.astype(np.int16))
        pc.append(cc.astype(np.int16))
        conf.append(np.ones(len(rr), np.float32))
        oid_pts.append(sess.oid[g][rr, cc])
    xyz = np.concatenate(xyz).astype(np.float32)
    fg, pr, pc = np.concatenate(fg), np.concatenate(pr), np.concatenate(pc)
    conf, oid_pts = np.concatenate(conf), np.concatenate(oid_pts)
    n = len(xyz)
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"),
              ("blue", "u1"), ("confidence", "<f4"), ("frame_global", "<i4"),
              ("pixel_row", "<i2"), ("pixel_col", "<i2")]
    packed = np.empty(n, np.dtype(fields))
    packed["x"], packed["y"], packed["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    packed["red"] = packed["green"] = packed["blue"] = 128
    packed["confidence"], packed["frame_global"] = conf, fg
    packed["pixel_row"], packed["pixel_col"] = pr, pc
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}",
              "property float x", "property float y", "property float z",
              "property uchar red", "property uchar green", "property uchar blue",
              "property float confidence", "property int frame_global",
              "property short pixel_row", "property short pixel_col", "end_header"]
    with open(out / "cleaned_cloud.ply", "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        packed.tofile(f)
    (out / "camera_poses.txt").write_text("\n".join(
        " ".join(f"{x:.9f}" for x in P.reshape(-1)) for P in poses) + "\n")
    (out / "camera_frames.txt").write_text("\n".join(str(int(x)) for x in sess.frame_numbers) + "\n")
    K = sess.K
    (out / "intrinsic.txt").write_text("\n".join(
        f"{K[0,0]} {K[1,1]} {K[0,2]} {K[1,2]}" for _ in range(N)) + "\n")
    names = [f"{int(x):06d}.jpg" for x in sess.frame_numbers]
    (out / "frame_list.json").write_text(json.dumps(names))
    (frames_dir / "frame_quality.json").write_text(json.dumps(
        {"frames": [{"file": nm, "fft_score": 1.0, "valid": True, "inter_frame_diff": 1.0}
                    for nm in names]}))
    (frames_dir / "selected_frames.json").write_text(json.dumps(
        {"version": "2.0", "method": "synthetic", "total_frames": N, "selected_count": N,
         "selected_files": names}))
    # segmentation
    res_insts, seg_insts, masks = [], [], {}
    masks["scaled_res"] = np.array([H, W], np.int32)
    masks["frames"] = sess.frame_numbers.astype(np.int32)
    masks["obj_ids"] = np.array([int(i) - 1 for i in instances], np.int32)
    for iid, spec in instances.items():
        sel = np.isin(oid_pts, np.asarray(spec["oids"]))
        gi = np.flatnonzero(sel)
        res_insts.append({"id": int(iid), "instance_id": int(iid), "label": spec["label"],
                          "color": [200, 100, 50], "total_points": int(len(gi)),
                          "globalIndices": gi.tolist()})
        seg_insts.append({"id": int(iid) - 1, "instance_id": int(iid), "label": spec["label"],
                          "color": [200, 100, 50]})
        if write_masks:
            for g in range(N):
                m = np.isin(sess.oid[g], np.asarray(spec["oids"])).astype(np.uint8)
                if m.any():
                    masks[f"f{int(sess.frame_numbers[g])}_o{int(iid) - 1}"] = m
    (out / "segmentation_result.json").write_text(json.dumps(
        {"type": "segmentation", "version": "3.0", "cloud_source": "cleaned_cloud",
         "total_points": n, "segmented_points": sum(i["total_points"] for i in res_insts),
         "coverage": 0.0, "resolution": {"scaled": [H, W], "original": [H, W]},
         "instances": res_insts}))
    (out / "segmentation.json").write_text(json.dumps(
        {"version": "3.0", "prompt": "synthetic", "prompts": [],
         "resolution": {"scaled": [H, W], "original": [H, W]}, "instances": seg_insts,
         "mask_file": "seg_masks.npz"}))
    np.savez_compressed(out / "seg_masks.npz", **masks)
    (out / ".orientation_applied").write_text("synthetic\n")
    return root
