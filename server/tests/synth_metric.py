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
         "ambiguous_sigma_factor": 3.0, "nonstructural_sigma_factor": 2.0,
         "movable_labels": ["box", "person"], "min_shared_structural_labels": 1,
         "intra_chunk_loops": True, "bridge_extra_frames": 0,
         "spatial": {"min_walk_m": 5.0, "drift_floor_m": 0.30, "drift_rate_m_per_m": 0.013,
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
         # mirrors config.yaml: the odometry σ is measured from the drift the
         # loops see, and loops covering the same stretch must agree
         "odo_sigma_from_drift": False, "odo_sigma_min_m": 0.01, "odo_sigma_max_m": 0.50,
         "outlier_overlap_frac": 0.60, "outlier_mad_k": 3.0,
         "outlier_mad_floor_m_per_m": 0.005, "outlier_max_sigma_factor": 10.0,
         "drift_model": True, "drift_degree": 2, "drift_iters": 25, "drift_max_step": 1.0,
         "huber_delta_m": 0.10, "huber_delta_deg": 2.0,
         "dense_max_unknowns": 12000, "lambda_init": 1e-4, "lambda_max": 1e12,
         "lm_diag_floor": 1e-9, "tol": 1e-8, "rel_tol": 1e-6, "max_iters": 50,
         "pcg_tol": 1e-10, "pcg_max_iters": 2000, "min_loop_gain": 0.5,
         "max_seam_degradation_m": 0.005, "gate_mode": "advisory",
         "holdout_offsets": [4, 10], "holdout_stride": 3,
         "holdout_samples": 4000, "holdout_max_nn_m": 0.10}
    d.update(over)
    return d


def fork_authority_cfg(**over) -> dict:
    d = {"saturation_warn": 0.8, "pose_graph_max_m": 1.0, "pose_graph_max_deg": 5.0,
         "scale_graph_max_log": 0.10, "depth_graph_max_log_a": 0.10,
         "depth_graph_max_b_m": 0.50, "intra_chunk_max_m": 0.15}
    d.update(over)
    return d


def structural_cfg(**over) -> dict:
    d = {"wall_planarity": {"enabled": True, "wall_tol_m": 0.02, "sigma_angle_deg": 1.0,
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
    lp.pop("nonstructural_sigma_factor")      # lives in loops.semantic server-side
    raw = {
        "correction_graph": {"loop": lp, "graph": fork_graph_cfg()},
        "authority": fork_authority_cfg(),
        "structural": structural_cfg(),
        "certify": certify_cfg(),
        "witness": witness_cfg(),
        "loops": {"min_gap_keyframes": 30, "duplicate_min_sep_m": 0.20, "dbscan_eps_m": 0.15,
                  "dbscan_min_samples": 20, "cluster_min_points": 300, "bridge_extra_frames": 4,
                  "coverage_radius_m": 5.0, "min_coverage": 0.5,
                  "spatial": sp,
                  "semantic": {"enabled": False, "max_tokens": 256, "crops_per_instance": 1,
                               "default_class": "structural", "nonstructural_sigma_factor": 2.0},
                  "reprojection": {"enabled": True, "dilate_px": 3, "max_frames": 8,
                                   "min_self_recall": 0.40, "min_cross_recall": 0.40,
                                   "min_agreeing_frac": 0.60},
                  "salad": {"similarity_threshold": 0.65, "top_k": 5, "min_gap_keyframes": 11,
                            "min_gap_frac": 0.10, "nms_threshold": 3,
                            "image_size": [336, 336], "batch_size": 32}},
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
    # the DA3 anchors (scale_align's diagnostics): every keyframe an anchor
    # agreeing with the applied metre — the absolute reference the post-hoc
    # scale graph checks a chunk against (a scale error injected later shows
    # up as an agreement ratio ≠ 1 once the sidecar is regenerated)
    (out / "scale_diagnostics.json").write_text(json.dumps(
        {"version": 2, "s_applied": 1.0, "source": "synthetic",
         "anchors": {"count": int(N), "mad_rel": 0.0, "spread": 0.0,
                     "frames": [{"num": int(sess.frame_numbers[g]), "s_f": 1.0, "n_px": int(H * W)}
                                for g in range(N)], "s_over_walk": []},
         "scale_confidence": 1.0}, indent=1))
    return root


# ── F3: witnesses, depth by tracks, certification (claude_stac.txt §6, §8–§10) ──

def witness_cfg(**over) -> dict:
    """Mirror of config.yaml ``witness:`` (every key present)."""
    d = {"n_neighbors": 4, "tau_rel": 0.02, "cpu_threads": 8, "at_merge": True, "mask_erosion_px": 1,
         "occlusion_tol_rel": 0.05,
         "rules": {"verified_min_mv_votes": 2, "verified_max_mask_conflicts": 1, "conflict_min": 1},
         "clean_statuses": ["verified", "unobserved"],
         "mls_excluded_statuses": ["mask_conflict", "single_witness"],
         "tracks": {"enabled": True, "python": "python", "win": 24, "stride": 12, "loop_window": 8,
                    "min_views": 2, "reproj_max_px": 2.0, "sigma_rel": 0.01, "min_obs_per_frame": 20,
                    "depth_edge_tol_rel": 0.05},
         "contours": {"enabled": True, "min_gradient": 20.0, "samples_per_instance": 64,
                      "search_rel": 0.15, "search_steps": 31, "sigma_rel": 0.02},
         "depth": {"pair_offsets": [1, 2, 3, 5, 8], "pair_samples": 8000, "holdout_fraction": 0.25,
                   "min_pairs": 10, "improve": 0.8, "bound": 5.0, "zref_m": 5.0, "scale_only": False,
                   "pair_sigma_floor_rel": 0.002, "refine_iters": 2, "prior_sigma_rel": 0.005,
                   "pair_scatter_clip_sigma": 5.0}}
    d.update(over)
    return d


def certify_cfg(**over) -> dict:
    """Mirror of config.yaml ``certify:``."""
    d = {"ensemble_offset_frames": 0, "keep_aligned_chunks": True,
         "max_iters": 3, "eps": 0.05, "regression_eps": 0.01, "auto_after_segmentation": True,
         "objective_weights": {"loop_residual_m": 1.0, "seam_residual_m": 1.0, "closure_m": 1.0,
                               "depth_disagreement_frac": 5.0, "duplicates": 0.1},
         "gates": {"mode": "advisory", "max_seam_degradation_m": 0.005,
                   "max_loop_residual_increase_m": 0.005,
                   "max_depth_disagreement_increase": 0.001, "max_verified_drop_frac": 0.05,
                   "duplicates_must_not_increase": True},
         "scale": {"sigma_loop_min_log": 0.01, "sigma_seam_log": 0.02, "sigma_anchor_log": 0.03,
                   "min_copy_points": 300, "max_copy_residual_m": 0.05, "icp_iters": 30,
                   "icp_trim": 0.8, "max_correction_log": 0.2},
         "visit_loops": {"sigma_floor_m": 0.01, "unobserved_sigma_m": 5.0, "unobserved_sigma_deg": 30.0,
                         "window_kf": 15},
         "known_answer": {"chunk": "last", "yaw_deg": 1.0, "t_m": 0.20, "scale": 1.03, "tol_t_m": 0.05,
                          "tol_deg": 0.30, "tol_scale": 0.01},
         "envelope": {"levels_t_m": [0.1, 0.2, 0.4, 0.8, 1.6], "levels_scale_pct": [1, 2, 5, 10, 20],
                      "loop_densities": [1.0, 0.5, 0.25]},
         "determinism": {"tol_m": 1.0e-4, "tol_frac": 0.01, "seed": 0}}
    d.update(over)
    return d


def copy_session(sess: Session) -> Session:
    """A deep copy of the arrays (fixtures are shared; injections never leak)."""
    return Session(sess.scene, sess.poses.copy(), sess.K.copy(), sess.H, sess.W,
                   sess.depth.copy(), sess.oid.copy(), sess.points.copy(),
                   sess.frame_numbers.copy())


def _pixel_dirs(sess: Session, rows, cols) -> np.ndarray:
    K = sess.K
    return np.stack([(np.asarray(cols) - K[0, 2]) / K[0, 0],
                     (np.asarray(rows) - K[1, 2]) / K[1, 1], np.ones(len(rows))], -1)


def inject_floaters(sess: Session, n: int = 200, seed: int = 0, factor=(0.35, 0.7),
                    margin: int = 4, oids: Optional[Sequence[int]] = None
                    ) -> Tuple[Session, np.ndarray]:
    """``n`` random valid pixels of the depth maps become FLOATERS: the depth
    turns into factor × the surface depth (a blob in mid-air on the ray) and
    the point moves with it — cloud and per-frame depth stay consistent, the
    way a real floater is born. ``oids`` restricts the pixels to those scene
    objects. Returns (session copy, (n,3) [kf, row, col])."""
    s = copy_session(sess)
    rng = np.random.default_rng(seed)
    picks = []
    while len(picks) < n:
        g = int(rng.integers(0, s.n_kf))
        r = int(rng.integers(margin, s.H - margin)); c = int(rng.integers(margin, s.W - margin))
        if s.depth[g, r, c] <= 0:
            continue
        if oids is not None and int(s.oid[g, r, c]) not in set(int(o) for o in oids):
            continue
        f = float(rng.uniform(*factor))
        z = float(s.depth[g, r, c]) * f
        d = _pixel_dirs(s, [r], [c])[0]
        T = s.poses[g]
        s.depth[g, r, c] = z
        s.points[g, r, c] = T[:3, 3] + (T[:3, :3] @ d) * z
        picks.append((g, r, c))
    return s, np.asarray(picks, np.int64)


def apply_depth_affine(sess: Session, affine_by_kf: Dict[int, Tuple[float, float]]) -> Session:
    """z' = a·z + b on a keyframe's depth map and its points along their rays
    (a per-frame depth-field error, consistent between cloud and depth)."""
    s = copy_session(sess)
    for g, (a, b) in affine_by_kf.items():
        valid = s.depth[g] > 0
        z = s.depth[g]
        z2 = np.where(valid, a * z + b, 0.0)
        T = s.poses[g]
        ratio = np.where(valid, z2 / np.where(valid, z, 1.0), 1.0)
        s.points[g] = T[:3, 3] + (s.points[g] - T[:3, 3]) * ratio[..., None]
        s.depth[g] = z2
    return s


def write_aligned_chunks(root: Path, sess: Session, chunk_size: int = 60, overlap: int = 30,
                         drift_by_kf: Optional[np.ndarray] = None) -> Path:
    """The per-keyframe depth store the production witnesses read:
    output/maplong_run/_tmp_results_aligned/chunk_K.npy dicts (depth,
    intrinsic, world_points, world_points_conf, extrinsic = world w2c 3x4),
    chunk_plan.json, chunk_000_meta.json (chunk_step) and maplong_run/
    frame_list.json — from the session's (possibly corrupted) depth maps."""
    root = Path(root)
    out = root / "output"
    run = out / "maplong_run"
    al = run / "_tmp_results_aligned"
    al.mkdir(parents=True, exist_ok=True)
    ci = chunk_ranges(sess.n_kf, chunk_size, overlap)
    for k, (a, b) in enumerate(ci):
        S = b - a
        dep = np.zeros((S, sess.H, sess.W), np.float32)
        wp = np.zeros((S, sess.H, sess.W, 3), np.float32)
        conf = np.zeros((S, sess.H, sess.W), np.float32)
        ext = np.zeros((S, 3, 4), np.float64)
        for li, g in enumerate(range(a, b)):
            valid = sess.depth[g] > 0
            P = sess.points[g]
            pose = sess.poses[g]
            if drift_by_kf is not None:
                D = drift_by_kf[g]
                P = P @ D[:3, :3].T + D[:3, 3]
                pose = D @ pose
            dep[li] = np.where(valid, sess.depth[g], 0.0)
            wp[li] = np.where(valid[..., None], P, 0.0)
            conf[li] = valid.astype(np.float32)
            ext[li] = np.linalg.inv(pose)[:3]
        np.save(al / f"chunk_{k}.npy", {"depth": dep, "intrinsic": np.tile(sess.K[None], (S, 1, 1)),
                                        "world_points": wp, "world_points_conf": conf,
                                        "extrinsic": ext})
    names = [f"{int(x):06d}.jpg" for x in sess.frame_numbers]
    (run / "frame_list.json").write_text(json.dumps(names))
    (out / "chunk_plan.json").write_text(json.dumps(
        {"version": 1, "phase": "synthetic", "n_keyframes": int(sess.n_kf),
         "chunk_size": int(chunk_size), "overlap": int(overlap),
         "chunk_ranges": [[int(a), int(b)] for a, b in ci], "walk_m": None}, indent=1))
    (out / "chunk_000_meta.json").write_text(json.dumps(
        {"chunk_id": 0, "source_chunk": 0, "n_points": 0, "chunk_step": int(chunk_size - overlap)}))
    return al


def write_images(root: Path, sess: Session, seed: int = 0) -> Path:
    """frames/<name>.jpg: one flat colour per scene object (sharp contours at
    every object boundary — the contour witness's input)."""
    import cv2
    root = Path(root)
    fd = root / "frames"
    fd.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_oid = int(sess.oid.max()) + 1
    lut = rng.integers(40, 230, size=(n_oid + 1, 3)).astype(np.uint8)
    lut[0] = 0
    for g in range(sess.n_kf):
        img = lut[np.clip(sess.oid[g], 0, n_oid)]
        cv2.imwrite(str(fd / f"{int(sess.frame_numbers[g]):06d}.jpg"), img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 98])
    return fd


def synthetic_tracks(root: Path, sess: Session, win: int = 24, stride: int = 12,
                     loop_pairs: Sequence[Tuple[int, int]] = (), per_frame: int = 40,
                     loop_window: int = 8, seed: int = 0) -> Path:
    """output/ba_run/tracks.npz in vggt_tracks' layout from EXACT scene
    correspondences: query pixels of a window's first keyframe, their GT 3-D
    point projected into every keyframe of the window where it is visible
    (depth test). Loop pairs get their own window (both sides)."""
    root = Path(root)
    rng = np.random.default_rng(seed)
    n = sess.n_kf
    windows = []
    s = 0
    while True:
        e = min(s + win, n)
        windows.append(list(range(s, e)))
        if e >= n:
            break
        s += stride
    for i, j in loop_pairs:
        h = loop_window // 2
        wi = [k for k in range(max(0, i - h), min(n, i + h + 1))]
        wj = [k for k in range(max(0, j - h), min(n, j + h + 1))]
        windows.append(sorted(set(wi + wj)))
    K = sess.K
    t_id, t_frame, t_uv = [], [], []
    q_frame, q_uv = [], []
    tid = 0
    for wnd in windows:
        q = wnd[0]
        valid = np.argwhere(sess.depth[q] > 0)
        if len(valid) == 0:
            continue
        pick = valid[rng.choice(len(valid), min(per_frame, len(valid)), replace=False)]
        for r, c in pick:
            P = sess.points[q, r, c]
            obs = []
            for f in wnd:
                w2c = np.linalg.inv(sess.poses[f])
                X = w2c[:3, :3] @ P + w2c[:3, 3]
                if X[2] <= 0.05:
                    continue
                u = K[0, 0] * X[0] / X[2] + K[0, 2]
                v = K[1, 1] * X[1] / X[2] + K[1, 2]
                ui, vi = int(round(u)), int(round(v))
                if not (0 <= ui < sess.W and 0 <= vi < sess.H):
                    continue
                d = sess.depth[f, vi, ui]
                if d <= 0 or abs(d - X[2]) > 0.01 * X[2]:
                    continue      # occluded (or off the surface) in this view
                obs.append((f, u, v))
            if len(obs) < 2:
                continue
            for f, u, v in obs:
                t_id.append(tid); t_frame.append(int(sess.frame_numbers[f])); t_uv.append((u, v))
            q_frame.append(int(sess.frame_numbers[q])); q_uv.append((float(c), float(r)))
            tid += 1
    ba = root / "output" / "ba_run"
    ba.mkdir(parents=True, exist_ok=True)
    p = ba / "tracks.npz"
    np.savez_compressed(p, obs_track=np.asarray(t_id, np.int64), obs_frame=np.asarray(t_frame, np.int64),
                        obs_uv=np.asarray(t_uv, np.float32).reshape(-1, 2),
                        obs_vis=np.ones(len(t_id), np.float32), obs_score=np.ones(len(t_id), np.float32),
                        track_query_frame=np.asarray(q_frame, np.int64),
                        track_query_uv=np.asarray(q_uv, np.float32).reshape(-1, 2),
                        meta=np.asarray([sess.H, sess.W], np.int64))
    return p


def leak_mask(out_dir: Path, from_iid: int, to_iid: int, px: int = 20) -> int:
    """Corrupt seg_masks.npz the way a SAM3 leak does: instance ``from_iid``'s
    mask grows by ``px`` over ``to_iid``'s pixels and ``to_iid`` loses them,
    in every keyframe where both appear. Returns the number of leaked
    pixels."""
    from scipy.ndimage import binary_dilation
    p = Path(out_dir) / "seg_masks.npz"
    z = np.load(p)
    masks = {k: z[k] for k in z.files}
    a_suf, b_suf = f"_o{int(from_iid) - 1}", f"_o{int(to_iid) - 1}"
    leaked = 0
    for key in list(masks):
        if not key.endswith(a_suf) or not key.startswith("f"):
            continue
        kb = key[:key.index("_o")] + b_suf
        if kb not in masks:
            continue
        A = masks[key].astype(bool); B = masks[kb].astype(bool)
        grown = binary_dilation(A, iterations=int(px)) & B
        masks[key] = (A | grown).astype(np.uint8)
        masks[kb] = (B & ~grown).astype(np.uint8)
        leaked += int(grown.sum())
    np.savez_compressed(p, **masks)
    return leaked


# ── adversarial scenes and trajectories (§10.12) ─────────────────────────────

def hall_scene(nx: int = 4, nz: int = 3, pitch_m: float = 6.0, ceiling_h: float = 4.0,
               radius: float = 0.3) -> Scene:
    """A symmetric hall: a floor, a ceiling, four walls and a REGULAR grid of
    identical columns — every column looks like every other one (the
    identity trap of §4.5)."""
    sc = Scene()
    hx, hz = nx * pitch_m / 2.0 + 3.0, nz * pitch_m / 2.0 + 3.0
    sc.add(Plane(np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
                 np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), hx, hz, 1, "floor"))
    sc.add(Plane(np.array([0.0, ceiling_h, 0.0]), np.array([0.0, -1.0, 0.0]),
                 np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), hx, hz, 2, "ceiling"))
    wall(sc, [hx, 0, 0], [-1, 0, 0], [0, 0, 1], hz, ceiling_h / 2, 3)
    wall(sc, [-hx, 0, 0], [1, 0, 0], [0, 0, 1], hz, ceiling_h / 2, 4)
    wall(sc, [0, 0, hz], [0, 0, -1], [1, 0, 0], hx, ceiling_h / 2, 5)
    wall(sc, [0, 0, -hz], [0, 0, 1], [1, 0, 0], hx, ceiling_h / 2, 6)
    oid = 7
    for i in range(nx):
        for j in range(nz):
            x = -(nx - 1) * pitch_m / 2.0 + i * pitch_m
            z = -(nz - 1) * pitch_m / 2.0 + j * pitch_m
            sc.add(Cylinder(np.array([x, 0.0, z]), radius, ceiling_h, oid, "column"))
            oid += 1
    return sc


def hall_trajectory(n_kf: int, nx: int = 4, nz: int = 3, pitch_m: float = 6.0, y: float = 1.5,
                    yaw_offset_deg: float = 30.0) -> np.ndarray:
    """A serpentine through the column grid, returning past the start."""
    xs = [-(nx - 1) * pitch_m / 2.0 - pitch_m / 2.0 + i * pitch_m for i in range(nx + 1)]
    z0, z1 = -(nz - 1) * pitch_m / 2.0 - pitch_m / 2.0, (nz - 1) * pitch_m / 2.0 + pitch_m / 2.0
    way = []
    for i, x in enumerate(xs):
        way.append((x, z0 if i % 2 == 0 else z1))
        way.append((x, z1 if i % 2 == 0 else z0))
    way.append(way[0])
    seg = []
    for a, b in zip(way[:-1], way[1:]):
        seg.append((np.asarray(a, np.float64), np.asarray(b, np.float64)))
    total = sum(np.linalg.norm(b - a) for a, b in seg)
    s = np.linspace(0.0, total, n_kf, endpoint=False)
    poses = []
    a_ = np.radians(yaw_offset_deg)
    Ry = np.array([[np.cos(a_), 0, np.sin(a_)], [0, 1, 0], [-np.sin(a_), 0, np.cos(a_)]])
    for si in s:
        d = si
        for a, b in seg:
            L = np.linalg.norm(b - a)
            if d <= L:
                p = a + (b - a) * (d / L)
                fwd = np.array([(b - a)[0], 0.0, (b - a)[1]]) / L
                poses.append(look_c2w((p[0], y, p[1]), Ry @ fwd))
                break
            d -= L
    return np.stack(poses)


def rotation_only_trajectory(n_kf: int, pos=(0.0, 1.5, -5.5), sweep_deg: float = 120.0) -> np.ndarray:
    """A tripod-like stretch: the camera turns in place (no parallax)."""
    poses = []
    for k in range(n_kf):
        a = np.radians(-sweep_deg / 2 + sweep_deg * k / max(n_kf - 1, 1))
        poses.append(look_c2w(pos, (np.sin(a), 0.0, np.cos(a))))
    return np.stack(poses)


def static_trajectory(n_kf: int, pos=(0.0, 1.5, -5.5), forward=(1.0, 0.0, 0.0)) -> np.ndarray:
    """A still stretch: identical poses (the operator stopped)."""
    return np.stack([look_c2w(pos, forward) for _ in range(n_kf)])


def with_stretch(poses: np.ndarray, extra: np.ndarray, at: int) -> np.ndarray:
    """Insert ``extra`` poses into a trajectory at index ``at``."""
    return np.concatenate([poses[:at], extra, poses[at:]])
