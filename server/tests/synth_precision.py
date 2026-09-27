"""Shared synthetic scene for the precision-pipeline tests (claude_stac.txt §5)
— numpy + cv2 only, no GPU, no model.

A room of textured planes with textured boxes on the floor, a Brown–Conrady
camera (the OPENCV model of ``precision.camera``), a vectorised numpy raycast
renderer that returns RGB, ground-truth z-depth, world normals, per-pixel quad
ids and a dynamic-object mask, pose sequences (translation / pure rotation /
still / arc), configurable pose perturbations, a scale drift along the walk,
and a writer that lays the frames out exactly as a session does
(``<session>/frames/<frame:06d>.jpg`` at native resolution + an empty
``output/``).

Conventions (identical to the production data and to ``precision.camera``)
  * world Y-up, floor at y = 0; cameras OpenCV (+x right, +y down, +z forward);
    poses are c2w 4×4 float64;
  * a pixel (row r, col c) has continuous coordinates (x, y) = (c, r): integer
    coordinates sit on pixel centres;
  * ``Render.depth`` is the camera-z of the surface seen through each pixel
    centre (0 where nothing is hit); ``Render.quad_id`` indexes
    ``Scene.all_quads()`` (−1 where nothing is hit, and under a dynamic blob);
  * distortion is applied where a real lens applies it: every DISTORTED pixel
    centre is undistorted to its ray through ``precision.camera.undistort_points``
    on the synthetic camera, so renderer and pipeline share one arithmetic.

Everything here is deterministic given its seeds. It is a test fixture, not a
stage: it measures nothing and decides nothing.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:      # importable outside pytest too
    sys.path.insert(0, str(_SERVER_DIR))


def _cv2():
    import cv2   # local import: keep the module importable where cv2 is absent
    return cv2


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n == 0.0:
        raise ValueError("cannot normalise a zero vector")
    return v / n


def _rot_y(deg: float) -> np.ndarray:
    """Rotation about the world up axis (+y). Positive angles turn +z toward +x."""
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rodrigues(rotvec: np.ndarray) -> np.ndarray:
    """Rotation matrix of an axis-angle vector (radians)."""
    rotvec = np.asarray(rotvec, dtype=np.float64)
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.eye(3)
    k = rotvec / theta
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


# ── camera ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BrownCamera:
    """Pinhole + Brown–Conrady lens in NATIVE pixels: the OPENCV model
    [fx, fy, cx, cy, k1, k2, p1, p2] of ``precision.camera``."""
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0

    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
                        dtype=np.float64)

    def dist(self) -> np.ndarray:
        """cv2 distCoeffs order (k1, k2, p1, p2)."""
        return np.array([self.k1, self.k2, self.p1, self.p2], dtype=np.float64)

    def params(self) -> Tuple[float, ...]:
        return (float(self.fx), float(self.fy), float(self.cx), float(self.cy),
                float(self.k1), float(self.k2), float(self.p1), float(self.p2))

    def to_precision_camera(self, omega_grid=None):
        """The same camera as a ``precision.camera.CameraModel`` (source
        "synthetic", camera_epoch 0). ``omega_grid`` defaults to the grid the
        production preprocessing (balanced, 512) would produce for this frame
        size; the mask grid is the full-frame resize to that same shape, as
        ``build_session_camera`` assumes for a session without masks."""
        from precision import camera as C
        g = omega_grid if omega_grid is not None else \
            C.omega_grid_for(self.width, self.height, "balanced", 512)
        mask_grid = C.mask_grid_for(self.width, self.height, (g.h, g.w))
        return C.CameraModel(width=int(self.width), height=int(self.height), params=self.params(),
                             source="synthetic", camera_epoch=0, omega_grid=g, mask_grid=mask_grid)


def default_camera(width: int = 320, height: int = 240, fov_deg: float = 70.0,
                   distortion: bool = False) -> BrownCamera:
    """Square-pixel camera with the given horizontal field of view and the
    optical centre on the image centre (pixel-centre convention). With
    ``distortion`` a visible, not extreme, Brown lens is added."""
    f = 0.5 * width / np.tan(np.radians(fov_deg) / 2.0)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    if distortion:
        return BrownCamera(width, height, f, f, cx, cy, k1=-0.12, k2=0.03, p1=0.001, p2=-0.0005)
    return BrownCamera(width, height, f, f, cx, cy)


# ── geometry ─────────────────────────────────────────────────────────────

@dataclass
class TexturedQuad:
    """A finite rectangle origin + a·edge_u + b·edge_v, a, b ∈ [0, 1], carrying
    a texture whose columns follow edge_u and rows follow edge_v. The texture
    is uint8 (Th, Tw) grey or (Th, Tw, 3) RGB."""
    origin: np.ndarray
    edge_u: np.ndarray
    edge_v: np.ndarray
    texture: np.ndarray
    name: str
    label: str

    def __post_init__(self):
        self.origin = np.asarray(self.origin, dtype=np.float64).reshape(3)
        self.edge_u = np.asarray(self.edge_u, dtype=np.float64).reshape(3)
        self.edge_v = np.asarray(self.edge_v, dtype=np.float64).reshape(3)
        self.texture = np.asarray(self.texture)
        if self.texture.dtype != np.uint8 or self.texture.ndim not in (2, 3) or \
                (self.texture.ndim == 3 and self.texture.shape[2] != 3):
            raise ValueError(f"quad {self.name!r}: texture must be uint8 (T,T) or (T,T,3), "
                             f"got {self.texture.dtype} {self.texture.shape}")
        if np.linalg.norm(np.cross(self.edge_u, self.edge_v)) == 0.0:
            raise ValueError(f"quad {self.name!r}: edge_u and edge_v are parallel")

    def normal(self) -> np.ndarray:
        """Unit normal of the winding (edge_u × edge_v); the renderer orients
        it toward the camera."""
        return _unit(np.cross(self.edge_u, self.edge_v))

    def corners(self) -> np.ndarray:
        o, u, v = self.origin, self.edge_u, self.edge_v
        return np.stack([o, o + u, o + u + v, o + v])


@dataclass
class Box:
    """Axis-aligned box rotated by ``yaw_deg`` about +y, expanded into six
    TexturedQuads (``<name>_bottom/top/front/back/left/right``) sharing one
    texture and label."""
    center: np.ndarray
    size: np.ndarray
    yaw_deg: float
    texture: np.ndarray
    name: str
    label: str

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=np.float64).reshape(3)
        self.size = np.asarray(self.size, dtype=np.float64).reshape(3)
        if np.any(self.size <= 0.0):
            raise ValueError(f"box {self.name!r}: size must be positive, got {self.size}")
        self._faces: Optional[Tuple[tuple, List[TexturedQuad]]] = None

    def _key(self) -> tuple:
        return (tuple(self.center.tolist()), tuple(self.size.tolist()), float(self.yaw_deg),
                id(self.texture), self.name, self.label)

    def quads(self) -> List[TexturedQuad]:
        """The six faces; rebuilt only when a field changed (the renderer
        expands every box on every frame)."""
        key = self._key()
        if self._faces is not None and self._faces[0] == key:
            return list(self._faces[1])
        faces = self._build_faces()
        self._faces = (key, faces)
        return list(faces)

    def _build_faces(self) -> List[TexturedQuad]:
        R = _rot_y(self.yaw_deg)
        ex, ey, ez = R[:, 0], R[:, 1], R[:, 2]
        sx, sy, sz = self.size
        hx, hy, hz = ex * (sx / 2.0), ey * (sy / 2.0), ez * (sz / 2.0)
        c = self.center
        lo = c - hx - hy - hz                      # the (−x, −y, −z) corner
        faces = [
            ("bottom", lo, ex * sx, ez * sz),
            ("top", lo + 2 * hy, ex * sx, ez * sz),
            ("front", lo, ex * sx, ey * sy),          # z = −sz/2 in the box frame
            ("back", lo + 2 * hz, ex * sx, ey * sy),
            ("left", lo, ez * sz, ey * sy),           # x = −sx/2 in the box frame
            ("right", lo + 2 * hx, ez * sz, ey * sy),
        ]
        return [TexturedQuad(o, u, v, self.texture, f"{self.name}_{f}", self.label)
                for f, o, u, v in faces]


@dataclass
class Scene:
    """Quads plus boxes. ``all_quads()`` is the flat list the renderer casts
    against and the space ``Render.quad_id`` indexes."""
    quads: List[TexturedQuad] = field(default_factory=list)
    name: str = "scene"
    boxes: List[Box] = field(default_factory=list)

    def all_quads(self) -> List[TexturedQuad]:
        out: List[TexturedQuad] = []
        for q in self.quads:                 # a Box dropped into quads is expanded too
            out.extend(q.quads() if isinstance(q, Box) else [q])
        for b in self.boxes:
            out.extend(b.quads())
        return out

    def quad_index(self, name: str) -> int:
        """Index of the quad called ``name`` in ``all_quads()``."""
        for i, q in enumerate(self.all_quads()):
            if q.name == name:
                return i
        raise KeyError(f"scene {self.name!r} has no quad named {name!r}")


def make_texture(kind: str, size: int = 256, seed: int = 0, contrast: float = 1.0) -> np.ndarray:
    """uint8 (size, size) grey texture centred on 128 with amplitude
    127·contrast: 'noise' (three octaves of smooth random texture — gradients
    at every pyramid level a tracker uses), 'checker' (8×8 cells), 'stripes'
    (16 vertical stripes — the aperture-problem texture), 'flat' (constant)."""
    if size < 2:
        raise ValueError(f"texture size must be ≥ 2, got {size}")
    if kind == "flat":
        return np.full((size, size), 128, dtype=np.uint8)
    if kind == "checker":
        cells = max(1, size // 8)
        yy, xx = np.mgrid[0:size, 0:size]
        pattern = (((yy // cells) + (xx // cells)) % 2).astype(np.float64) * 2.0 - 1.0
    elif kind == "stripes":
        period = max(2, size // 16)
        xx = np.arange(size)
        pattern = np.tile(((xx // (period // 2 if period >= 2 else 1)) % 2).astype(np.float64)
                          * 2.0 - 1.0, (size, 1))
    elif kind == "noise":
        cv2 = _cv2()
        rng = np.random.default_rng(seed)
        pattern = np.zeros((size, size), dtype=np.float64)
        for octave, weight in ((1, 1.0), (4, 1.0), (16, 1.0)):
            small = max(2, size // octave)
            layer = rng.standard_normal((small, small))
            if small != size:
                layer = cv2.resize(layer, (size, size), interpolation=cv2.INTER_LINEAR)
            pattern += weight * layer
        pattern /= max(float(np.abs(pattern).max()), 1e-12)
    else:
        raise ValueError(f"unknown texture kind {kind!r} (noise | checker | stripes | flat)")
    img = 128.0 + 127.0 * float(contrast) * pattern
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def room_scene(seed: int = 0, width_m: float = 6.0, depth_m: float = 8.0, height_m: float = 3.0,
               boxes: int = 2, flat_wall: bool = False) -> Scene:
    """A closed room: floor at y = 0 spanning x ∈ [−w/2, w/2], z ∈ [0, depth];
    walls 'wall_back' (z = depth), 'wall_front' (z = 0), 'wall_left' (x = −w/2),
    'wall_right' (x = +w/2); 'ceiling' at y = height; ``boxes`` textured boxes
    resting on the floor in the far half of the room. A camera at
    ``room_start_pose`` looks along +z at the back wall. ``flat_wall`` gives
    the back wall a textureless ('flat') texture."""
    hw = width_m / 2.0
    tex = lambda k, s, c=1.0: make_texture(k, 256, seed=seed + s, contrast=c)   # noqa: E731
    quads = [
        TexturedQuad((-hw, 0.0, 0.0), (width_m, 0.0, 0.0), (0.0, 0.0, depth_m),
                     tex("noise", 1), "floor", "floor"),
        TexturedQuad((-hw, 0.0, depth_m), (width_m, 0.0, 0.0), (0.0, height_m, 0.0),
                     tex("flat", 2) if flat_wall else tex("noise", 2), "wall_back", "wall"),
        TexturedQuad((-hw, 0.0, 0.0), (0.0, 0.0, depth_m), (0.0, height_m, 0.0),
                     tex("noise", 3), "wall_left", "wall"),
        TexturedQuad((hw, 0.0, 0.0), (0.0, 0.0, depth_m), (0.0, height_m, 0.0),
                     tex("noise", 4), "wall_right", "wall"),
        TexturedQuad((-hw, 0.0, 0.0), (width_m, 0.0, 0.0), (0.0, height_m, 0.0),
                     tex("noise", 5), "wall_front", "wall"),
        TexturedQuad((-hw, height_m, 0.0), (width_m, 0.0, 0.0), (0.0, 0.0, depth_m),
                     tex("noise", 6, 0.5), "ceiling", "ceiling"),
    ]
    rng = np.random.default_rng(seed)
    placed: List[Box] = []
    tries = 0
    while len(placed) < boxes and tries < 1000:
        tries += 1
        size = rng.uniform(0.4, 1.0, size=3)
        x = rng.uniform(-hw + 1.0, hw - 1.0)
        z = rng.uniform(0.35 * depth_m, 0.8 * depth_m)
        if any(np.hypot(x - b.center[0], z - b.center[2]) < 1.5 for b in placed):
            continue
        kind = "checker" if len(placed) % 2 == 0 else "stripes"
        placed.append(Box((x, size[1] / 2.0, z), size, float(rng.uniform(0.0, 90.0)),
                          make_texture(kind, 256, seed=seed + 10 + len(placed), contrast=0.8),
                          f"box{len(placed)}", "box"))
    if len(placed) < boxes:
        raise RuntimeError(f"room_scene: could not place {boxes} boxes without overlap in a "
                           f"{width_m}x{depth_m} m room")
    return Scene(quads=quads, name=f"room_{seed}", boxes=placed)


def room_start_pose(eye_height_m: float = 1.5, z_m: float = 1.0, x_m: float = 0.0) -> np.ndarray:
    """A camera inside ``room_scene`` looking along +z at the back wall."""
    return look_at_pose((x_m, eye_height_m, z_m), (x_m, eye_height_m, z_m + 1.0))


# ── camera model: rays and projection ────────────────────────────────────

_RAY_CACHE: Dict[BrownCamera, np.ndarray] = {}
_SOLVER: Dict[str, object] = {}


def undistort_solver() -> Dict[str, object]:
    """The point-undistortion solver settings production uses
    (``reconstruction.precision.camera.undistort_max_iter`` /
    ``undistort_eps_px`` of server/config.yaml, read once), so renderer and
    pipeline share one arithmetic."""
    if not _SOLVER:
        import yaml
        from precision import camera as C
        from precision.config import load_precision_config
        with open(_SERVER_DIR / "config.yaml") as f:
            _SOLVER.update(C.undistort_solver(load_precision_config(yaml.safe_load(f)).camera))
    return dict(_SOLVER)


def camera_rays(cam: BrownCamera) -> np.ndarray:
    """(H·W, 3) camera-frame directions with z = 1 for every DISTORTED pixel
    centre, row-major: the pixel's undistorted position under the same K
    (``precision.camera.undistort_points``, iterated to convergence) turned
    into a normalised ray. Cached per camera."""
    hit = _RAY_CACHE.get(cam)
    if hit is not None:
        return hit
    from precision import camera as C
    H, W = cam.height, cam.width
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    uv = np.stack([uu.ravel(), vv.ravel()], axis=1)
    und = C.undistort_points(uv, cam.to_precision_camera(), **undistort_solver())
    x = (und[:, 0] - cam.cx) / cam.fx
    y = (und[:, 1] - cam.cy) / cam.fy
    rays = np.stack([x, y, np.ones_like(x)], axis=1)
    _RAY_CACHE[cam] = rays
    return rays


def project(cam: BrownCamera, c2w: np.ndarray,
            xyz_world: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The forward model world → camera → normalised → Brown distortion →
    pixels: (uv_distorted (N, 2), z (N,) camera depth, in_front (N,) bool).
    Points at z ≤ 0 get in_front False (their uv is not meaningful)."""
    c2w = np.asarray(c2w, dtype=np.float64)
    X = np.asarray(xyz_world, dtype=np.float64).reshape(-1, 3)
    R, o = c2w[:3, :3], c2w[:3, 3]
    Xc = (X - o) @ R                       # R^T (X − o)
    z = Xc[:, 2]
    in_front = z > 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(z != 0.0, Xc[:, 0] / z, np.nan)
        y = np.where(z != 0.0, Xc[:, 1] / z, np.nan)
    r2 = x * x + y * y
    radial = 1.0 + cam.k1 * r2 + cam.k2 * r2 * r2
    xd = x * radial + 2.0 * cam.p1 * x * y + cam.p2 * (r2 + 2.0 * x * x)
    yd = y * radial + cam.p1 * (r2 + 2.0 * y * y) + 2.0 * cam.p2 * x * y
    uv = np.stack([cam.fx * xd + cam.cx, cam.fy * yd + cam.cy], axis=1)
    return uv, z, in_front


def unproject(cam: BrownCamera, c2w: np.ndarray, uv_distorted: np.ndarray,
              z: np.ndarray) -> np.ndarray:
    """Inverse of ``project`` for points with known camera depth ``z``:
    distorted pixels → undistorted (``precision.camera``) → ray × z → world."""
    from precision import camera as C
    c2w = np.asarray(c2w, dtype=np.float64)
    uv = np.asarray(uv_distorted, dtype=np.float64).reshape(-1, 2)
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    und = C.undistort_points(uv, cam.to_precision_camera(), **undistort_solver())
    d = np.stack([(und[:, 0] - cam.cx) / cam.fx, (und[:, 1] - cam.cy) / cam.fy,
                  np.ones(len(uv))], axis=1)
    return (d * z[:, None]) @ c2w[:3, :3].T + c2w[:3, 3]


# ── rendering ────────────────────────────────────────────────────────────

@dataclass
class MovingBlob:
    """A flat-coloured disc moving over the image at ``velocity_px`` per video
    frame, standing at ``depth_m`` in front of the camera: the synthetic
    dynamic object (a person crossing, a hand). Drawn last; it occludes what
    is deeper and is hidden by what is nearer."""
    center_px0: np.ndarray
    velocity_px: np.ndarray
    radius_px: float
    depth_m: float
    color: np.ndarray

    def __post_init__(self):
        self.center_px0 = np.asarray(self.center_px0, dtype=np.float64).reshape(2)
        self.velocity_px = np.asarray(self.velocity_px, dtype=np.float64).reshape(2)
        self.color = np.asarray(self.color, dtype=np.float64).reshape(3)
        if self.radius_px <= 0.0 or self.depth_m <= 0.0:
            raise ValueError("MovingBlob needs radius_px > 0 and depth_m > 0")

    def center_at(self, frame_index: int) -> np.ndarray:
        return self.center_px0 + self.velocity_px * float(frame_index)


@dataclass
class Render:
    rgb: np.ndarray            # uint8 (H, W, 3)
    depth: np.ndarray          # float32 (H, W) camera-z, 0 where nothing is hit
    normal: np.ndarray         # float32 (H, W, 3) world normals facing the camera (0 when no hit)
    quad_id: np.ndarray        # int32 (H, W) index into Scene.all_quads(), −1 none / blob
    dynamic_mask: np.ndarray   # bool (H, W) True under a MovingBlob


def _bilinear(tex: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bilinear lookup of a uint8 (Th, Tw) or (Th, Tw, 3) texture at quad
    coordinates a (columns) and b (rows) in [0, 1]; texel centres at integer
    coordinates. Returns (M, 1) or (M, 3) float64 — grey broadcasts to RGB
    at assignment, so a grey texture costs one channel of gathers."""
    Th, Tw = tex.shape[:2]
    flat = tex.reshape(Th * Tw, -1)                  # (Th·Tw, C) view
    x = np.clip(a, 0.0, 1.0) * (Tw - 1)
    y = np.clip(b, 0.0, 1.0) * (Th - 1)
    x0 = np.minimum(np.floor(x).astype(np.int64), Tw - 1)
    y0 = np.minimum(np.floor(y).astype(np.int64), Th - 1)
    x1 = np.minimum(x0 + 1, Tw - 1)
    y1 = np.minimum(y0 + 1, Th - 1)
    wx = (x - x0)[:, None]
    wy = (y - y0)[:, None]
    r0, r1 = y0 * Tw, y1 * Tw
    return (np.take(flat, r0 + x0, axis=0) * ((1.0 - wx) * (1.0 - wy))
            + np.take(flat, r0 + x1, axis=0) * (wx * (1.0 - wy))
            + np.take(flat, r1 + x0, axis=0) * ((1.0 - wx) * wy)
            + np.take(flat, r1 + x1, axis=0) * (wx * wy))


# Speed knob, not a decision: a quad whose projected bounding box covers more
# than this share of the image is intersected against every pixel; a smaller
# one only against the pixels inside its box. Both paths compute the same
# hits — the culling is exact (see _plan_quads).
_LOCAL_QUAD_MAX_AREA_FRAC = 0.5


def _plan_quads(quads: List[TexturedQuad], rays: np.ndarray, R: np.ndarray, o: np.ndarray,
                tol: float) -> Tuple[List[int], List[Tuple[int, np.ndarray]]]:
    """Split the quads into the FULL group (intersected against all pixels)
    and LOCAL ones with their candidate pixels. A planar convex quad whose
    four corners lie in front of the camera projects to a convex polygon, so
    the bounding box of its projected corners in UNDISTORTED normalised
    coordinates contains every pixel ray that can hit it; the ray grid holds
    exactly those coordinates per pixel, so the cull is conservative and
    exact. A quad with a corner at or behind the camera goes to the full
    group; one whose box misses the image entirely is dropped (no pixel can
    see it)."""
    xn, yn = rays[:, 0], rays[:, 1]
    x_lo, x_hi, y_lo, y_hi = xn.min(), xn.max(), yn.min(), yn.max()
    image_area = max((x_hi - x_lo) * (y_hi - y_lo), 1e-12)
    full: List[int] = []
    local: List[Tuple[int, np.ndarray]] = []
    for i, q in enumerate(quads):
        Xc = (q.corners() - o) @ R                       # corners in the camera frame
        if np.any(Xc[:, 2] <= tol):
            full.append(i)
            continue
        px, py = Xc[:, 0] / Xc[:, 2], Xc[:, 1] / Xc[:, 2]
        bx0, bx1 = max(px.min(), x_lo), min(px.max(), x_hi)
        by0, by1 = max(py.min(), y_lo), min(py.max(), y_hi)
        if bx0 > bx1 or by0 > by1:
            continue
        if (bx1 - bx0) * (by1 - by0) / image_area > _LOCAL_QUAD_MAX_AREA_FRAC:
            full.append(i)
            continue
        cand = np.flatnonzero((xn >= bx0) & (xn <= bx1) & (yn >= by0) & (yn <= by1))
        if cand.size:
            local.append((i, cand))
    return full, local


def _intersect(d: np.ndarray, o: np.ndarray, quads: Sequence[TexturedQuad],
               tol: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rays ``d`` (n, 3) from ``o`` against every quad at once: a ray hits quad
    q at t = ((origin_q − o)·n_q) / (d·n_q) and lands at quad coordinates
    a = ((o − origin_q)·eu_q + t·(d·eu_q)) / |eu_q|² (b alike), so three
    (n, 3) × (3, Q) products cover every ray and quad. Returns (t, a, b,
    inside) as (n, Q) arrays; ``inside`` = positive hit within the rectangle."""
    origins = np.stack([q.origin for q in quads])
    EU = np.stack([q.edge_u for q in quads])
    EV = np.stack([q.edge_v for q in quads])
    N = np.cross(EU, EV)
    rel0 = o[None, :] - origins                                   # (Q, 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (-np.einsum("qi,qi->q", rel0, N))[None, :] / (d @ N.T)
        a = (np.einsum("qi,qi->q", rel0, EU)[None, :] + t * (d @ EU.T)) / \
            np.einsum("qi,qi->q", EU, EU)[None, :]
        b = (np.einsum("qi,qi->q", rel0, EV)[None, :] + t * (d @ EV.T)) / \
            np.einsum("qi,qi->q", EV, EV)[None, :]
    inside = (np.isfinite(t) & (t > tol) & (a >= -tol) & (a <= 1.0 + tol)
              & (b >= -tol) & (b <= 1.0 + tol))
    return t, a, b, inside


def render(scene: Scene, cam: BrownCamera, c2w: np.ndarray, *, light: float = 1.0,
           dynamic: Optional[List[MovingBlob]] = None, noise_sigma: float = 0.0, seed: int = 0,
           frame_index: int = 0) -> Render:
    """Raycast the scene through every distorted pixel centre of ``cam`` from
    pose ``c2w``: vectorised ray/plane intersection + in-rectangle test per
    quad, the nearest positive hit wins (coincident surfaces resolve
    deterministically: the full-group quad, else the lower index — do not
    build visible coplanar overlaps); colour = bilinear texture × ``light``;
    world normals face the camera; dynamic blobs (at their position for
    ``frame_index``) are drawn last with their own depth; Gaussian noise of
    ``noise_sigma`` (grey levels, seeded) is added before quantisation.
    Deterministic for equal arguments."""
    c2w = np.asarray(c2w, dtype=np.float64)
    if c2w.shape != (4, 4) or not np.all(np.isfinite(c2w)):
        raise ValueError(f"c2w must be a finite 4x4 matrix, got shape {c2w.shape}")
    quads = scene.all_quads()
    if not quads:
        raise ValueError(f"scene {scene.name!r} has no quads to render")
    H, W = cam.height, cam.width
    n_px = H * W
    R, o = c2w[:3, :3], c2w[:3, 3]
    rays = camera_rays(cam)
    d = rays @ R.T                                   # world directions, camera-z = t
    tol = 1e-9                                       # shared edges leave no crack

    best = np.full(n_px, np.inf)
    qid = np.full(n_px, -1, dtype=np.int32)
    a_best = np.zeros(n_px)
    b_best = np.zeros(n_px)
    full, local = _plan_quads(quads, rays, R, o, tol)
    if full:
        t, a, b, inside = _intersect(d, o, [quads[i] for i in full], tol)
        t_in = np.where(inside, t, np.inf)
        col = np.argmin(t_in, axis=1)
        rows = np.arange(n_px)
        bt = t_in[rows, col]
        hit = np.isfinite(bt)
        best[hit] = bt[hit]
        qid[hit] = np.asarray(full, dtype=np.int32)[col[hit]]
        a_best[hit] = a[rows, col][hit]
        b_best[hit] = b[rows, col][hit]
    for i, cand in local:
        t, a, b, inside = _intersect(d[cand], o, [quads[i]], tol)
        upd = inside[:, 0] & (t[:, 0] < best[cand])
        if not np.any(upd):
            continue
        idx = cand[upd]
        best[idx] = t[upd, 0]
        qid[idx] = i
        a_best[idx] = a[upd, 0]
        b_best[idx] = b[upd, 0]

    rgb = np.zeros((n_px, 3), dtype=np.float64)
    normal = np.zeros((n_px, 3), dtype=np.float64)
    hit_idx = np.flatnonzero(qid >= 0)
    if hit_idx.size:
        order = hit_idx[np.argsort(qid[hit_idx], kind="stable")]     # pixels grouped by quad
        bounds = np.searchsorted(qid[order], np.arange(len(quads) + 1))
        for i, q in enumerate(quads):
            s, e = bounds[i], bounds[i + 1]
            if e > s:
                sel = order[s:e]
                rgb[sel] = _bilinear(q.texture, a_best[sel], b_best[sel])
        n_unit = np.stack([q.normal() for q in quads])
        nq = n_unit[qid[hit_idx]]
        facing = -np.sign(np.einsum("ij,ij->i", d[hit_idx], nq))       # toward the camera
        normal[hit_idx] = nq * facing[:, None]

    depth = np.where(np.isfinite(best), best, 0.0)
    dyn = np.zeros(n_px, dtype=bool)
    if dynamic:
        uu, vv = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
        uu, vv = uu.ravel(), vv.ravel()
        facing_cam = -R[:, 2]                         # a disc facing the camera
        for blob in dynamic:
            cu, cv = blob.center_at(frame_index)
            disc = (uu - cu) ** 2 + (vv - cv) ** 2 <= blob.radius_px ** 2
            visible = disc & ((depth == 0.0) | (blob.depth_m < depth))
            if not np.any(visible):
                continue
            rgb[visible] = blob.color
            depth[visible] = blob.depth_m
            qid[visible] = -1
            normal[visible] = facing_cam
            dyn[visible] = True

    rgb *= float(light)
    if noise_sigma > 0.0:
        rng = np.random.default_rng(seed)
        rgb += rng.normal(0.0, noise_sigma, size=rgb.shape)
    rgb = np.clip(np.round(rgb), 0, 255).astype(np.uint8)
    return Render(rgb=rgb.reshape(H, W, 3), depth=depth.astype(np.float32).reshape(H, W),
                  normal=normal.astype(np.float32).reshape(H, W, 3),
                  quad_id=qid.reshape(H, W), dynamic_mask=dyn.reshape(H, W))


# ── poses ────────────────────────────────────────────────────────────────

def look_at_pose(eye: Sequence[float], target: Sequence[float],
                 up: Sequence[float] = (0.0, 1.0, 0.0)) -> np.ndarray:
    """c2w of an OpenCV camera at ``eye`` looking at ``target`` with the
    world ``up`` pointing to the top of the image (camera −y)."""
    eye = np.asarray(eye, dtype=np.float64)
    f = _unit(np.asarray(target, dtype=np.float64) - eye)
    down = -_unit(up)
    r = _unit(np.cross(down, f))
    down = _unit(np.cross(f, r))
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2] = r, down, f
    M[:3, 3] = eye
    return M


_DIRECTIONS = {"right": (0, 1.0), "left": (0, -1.0), "down": (1, 1.0), "up": (1, -1.0),
               "forward": (2, 1.0), "back": (2, -1.0)}


def _direction_world(start: np.ndarray, direction) -> np.ndarray:
    if isinstance(direction, str):
        if direction not in _DIRECTIONS:
            raise ValueError(f"direction must be one of {sorted(_DIRECTIONS)} or a 3-vector, "
                             f"got {direction!r}")
        axis, sign = _DIRECTIONS[direction]
        return _unit(start[:3, axis] * sign)
    return _unit(direction)


def walk_poses(kind: str, n: int, *, start: np.ndarray, step_m: float = 0.05,
               yaw_deg_per_frame: float = 1.0, radius_m: float = 2.0, seed: int = 0,
               direction: Union[str, Sequence[float]] = "forward") -> np.ndarray:
    """(n, 4, 4) c2w sequence starting at ``start``:
    'translate' — the centre advances ``step_m`` per frame along ``direction``
    (a camera axis name, default the camera's forward, or a world 3-vector),
    rotation fixed; 'rotate' — centre fixed, yaw about world +y by
    ``yaw_deg_per_frame`` per frame; 'still' — every pose equals ``start``;
    'arc' — the centre follows a horizontal circle of ``radius_m`` with
    ``step_m`` of arc per frame and the camera turns with the tangent.
    All four kinds are deterministic; ``seed`` is accepted for signature
    stability and is not used by any of them (randomness belongs to
    ``perturb_poses``)."""
    start = np.asarray(start, dtype=np.float64)
    if start.shape != (4, 4):
        raise ValueError(f"start must be a 4x4 c2w, got shape {start.shape}")
    if n < 1:
        raise ValueError(f"n must be ≥ 1, got {n}")
    poses = np.repeat(start[None], n, axis=0).copy()
    k = np.arange(n, dtype=np.float64)
    if kind == "still":
        return poses
    if kind == "translate":
        dw = _direction_world(start, direction)
        poses[:, :3, 3] = start[:3, 3] + (k * step_m)[:, None] * dw
        return poses
    if kind == "rotate":
        for i in range(n):
            poses[i, :3, :3] = _rot_y(i * yaw_deg_per_frame) @ start[:3, :3]
        return poses
    if kind == "arc":
        if radius_m <= 0.0:
            raise ValueError(f"radius_m must be positive, got {radius_m}")
        f0 = start[:3, 2].copy()
        f0[1] = 0.0
        f0 = _unit(f0)                               # horizontal heading
        s = _rot_y(90.0) @ f0                        # toward the centre of curvature
        centre = start[:3, 3] + radius_m * s
        theta_deg = np.degrees(k * step_m / radius_m)
        for i in range(n):
            Rk = _rot_y(theta_deg[i])
            poses[i, :3, :3] = Rk @ start[:3, :3]
            poses[i, :3, 3] = centre - radius_m * (Rk @ s)
        return poses
    raise ValueError(f"unknown walk kind {kind!r} (translate | rotate | still | arc)")


def chainage(c2w: np.ndarray) -> np.ndarray:
    """(n,) cumulative |Δcentre| along the sequence; chainage[0] = 0."""
    c2w = np.asarray(c2w, dtype=np.float64)
    centres = c2w[:, :3, 3]
    steps = np.linalg.norm(np.diff(centres, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def perturb_poses(c2w: np.ndarray, *, rot_deg_sigma: float = 0.0, trans_sigma_m: float = 0.0,
                  drift_per_m: float = 0.0, seed: int = 0,
                  drift_dir: Optional[Sequence[float]] = None) -> np.ndarray:
    """Random per-pose noise (rotation vector with ``rot_deg_sigma`` per axis,
    applied in the camera frame; translation ``trans_sigma_m`` per axis) plus
    a systematic drift ``drift_per_m × chainage`` along ``drift_dir`` (a world
    vector; default the walk's net direction first → last centre, no drift on
    a walk that does not move). Zero sigmas and drift return an exact copy."""
    c2w = np.asarray(c2w, dtype=np.float64)
    out = c2w.copy()
    n = len(out)
    rng = np.random.default_rng(seed)
    if rot_deg_sigma > 0.0:
        rotvecs = np.radians(rng.normal(0.0, rot_deg_sigma, size=(n, 3)))
        for i in range(n):
            out[i, :3, :3] = out[i, :3, :3] @ _rodrigues(rotvecs[i])
    if trans_sigma_m > 0.0:
        out[:, :3, 3] += rng.normal(0.0, trans_sigma_m, size=(n, 3))
    if drift_per_m != 0.0:
        ch = chainage(c2w)
        if drift_dir is not None:
            dw = _unit(drift_dir)
        else:
            net = c2w[-1, :3, 3] - c2w[0, :3, 3]
            dw = _unit(net) if np.linalg.norm(net) > 0.0 else np.zeros(3)
        out[:, :3, 3] += (drift_per_m * ch)[:, None] * dw[None, :]
    return out


def scale_drift_along_walk(depths: List[np.ndarray], chainage: np.ndarray,
                           eps_per_m: float) -> List[np.ndarray]:
    """depth_k × (1 + eps_per_m · chainage_k): the injected F2 error, a depth
    scale growing with the distance walked, about each keyframe's own camera.
    ``chainage`` is the (n,) array of the function of that name (the
    parameter shadows it here on purpose: the design fixes the name).
    Zeros (no hit) stay zero; dtype is preserved."""
    ch = np.asarray(chainage, dtype=np.float64).reshape(-1)
    if len(depths) != len(ch):
        raise ValueError(f"{len(depths)} depth maps but {len(ch)} chainage values")
    return [(np.asarray(dk) * (1.0 + eps_per_m * ck)).astype(np.asarray(dk).dtype)
            for dk, ck in zip(depths, ch)]


# ── session layout ───────────────────────────────────────────────────────

def write_frames(frames_dir: os.PathLike, renders: List[Render],
                 frame_numbers: Optional[List[int]] = None, quality: int = 95) -> List[str]:
    """frames/<frame:06d>.jpg (cv2, JPEG quality ``quality``) for every render;
    ``frame_numbers`` default 0..n−1 and may be strided (a subsampled video).
    Returns the basenames in frame order."""
    cv2 = _cv2()
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    if frame_numbers is None:
        frame_numbers = list(range(len(renders)))
    if len(frame_numbers) != len(renders):
        raise ValueError(f"{len(renders)} renders but {len(frame_numbers)} frame numbers")
    if len(set(frame_numbers)) != len(frame_numbers):
        raise ValueError("frame_numbers must be unique")
    names: List[str] = []
    for r, fn in zip(renders, frame_numbers):
        name = f"{int(fn):06d}.jpg"
        bgr = cv2.cvtColor(np.ascontiguousarray(r.rgb), cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(frames_dir / name), bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)]):
            raise RuntimeError(f"cv2.imwrite failed for {frames_dir / name}")
        names.append(name)
    return names


@dataclass
class SynthSession:
    session_dir: Path
    frames_dir: Path
    output_dir: Path
    cam: BrownCamera
    poses: np.ndarray                 # (n, 4, 4) c2w ground truth
    frame_numbers: List[int]          # video frame index of every render / file
    renders: List[Render]
    scene: Scene
    files: List[str] = field(default_factory=list)      # basenames, frame order
    light: List[float] = field(default_factory=list)    # per-frame light used

    def frame_path(self, frame: int) -> Path:
        return self.frames_dir / f"{int(frame):06d}.jpg"


GT_NAME = "synth_gt.npz"


def write_session(tmp_dir: os.PathLike, scene: Scene, cam: BrownCamera, poses: np.ndarray, *,
                  frame_numbers: Optional[List[int]] = None,
                  light: Optional[Union[float, Sequence[float]]] = None,
                  dynamic: Optional[List[MovingBlob]] = None, noise_sigma: float = 0.0,
                  seed: int = 0) -> SynthSession:
    """Render every pose and lay the result out as a session: ``tmp_dir`` is
    the session directory, frames go to ``frames/<frame:06d>.jpg`` at the
    camera's native size, ``output/`` is created empty for later stages, and
    ``synth_gt.npz`` (poses, K, dist, frame_numbers, light) sits at the
    session root so a test that re-opens the directory can read the truth.
    ``light`` is a scalar or one value per frame; blobs move by their velocity
    per VIDEO frame (``frame_numbers``); the noise seed is ``seed + i``."""
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"poses must be (n, 4, 4), got {poses.shape}")
    n = len(poses)
    if frame_numbers is None:
        frame_numbers = list(range(n))
    frame_numbers = [int(f) for f in frame_numbers]
    if len(frame_numbers) != n:
        raise ValueError(f"{n} poses but {len(frame_numbers)} frame numbers")
    if light is None:
        lights = [1.0] * n
    elif np.ndim(light) == 0:
        lights = [float(light)] * n
    else:
        lights = [float(v) for v in light]
        if len(lights) != n:
            raise ValueError(f"{n} poses but {len(lights)} light values")
    session_dir = Path(tmp_dir)
    frames_dir = session_dir / "frames"
    output_dir = session_dir / "output"
    frames_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    renders = [render(scene, cam, poses[i], light=lights[i], dynamic=dynamic,
                      noise_sigma=noise_sigma, seed=seed + i, frame_index=frame_numbers[i])
               for i in range(n)]
    files = write_frames(frames_dir, renders, frame_numbers)
    np.savez(session_dir / GT_NAME, poses=poses, K=cam.K(), dist=cam.dist(),
             width=np.int64(cam.width), height=np.int64(cam.height),
             frame_numbers=np.asarray(frame_numbers, dtype=np.int64),
             light=np.asarray(lights, dtype=np.float64))
    return SynthSession(session_dir=session_dir, frames_dir=frames_dir, output_dir=output_dir,
                        cam=cam, poses=poses, frame_numbers=frame_numbers, renders=renders,
                        scene=scene, files=files, light=lights)
