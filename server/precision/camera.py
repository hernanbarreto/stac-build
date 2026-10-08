"""F0 — session camera model and EXACT grid-to-native-pixel mappings.

The phone is ONE camera. Every 2-D observation of the precision pipeline is
stored in pixels of the NATIVE frame (the video's own resolution), and every
model grid — VGGT-Ω's depth/conf grid, the SAM3 mask grid — is related to it
by the exact affine map the model's own preprocessing induced.

Conventions (hold everywhere in this package)
  * A pixel (row r, col c) of ANY grid has continuous coordinates (x, y) =
    (c, r): integer coordinates sit on pixel centres. The repo already works
    this way (the omega adapter unprojects ``arange`` grids; omega's K carries
    cx = W/2, cy = H/2 on its own grid).
  * A model grid is produced from the native frame by
        centre-crop to a supported aspect ratio  ->  resize to (w, h)
        ->  (only for mixed batches) centre-pad to a common size,
    exactly ``vggt_omega.utils.load_fn.load_and_preprocess_images``. Resizing
    (PIL / cv2, align_corners = False) maps pixel CENTRES linearly, so
        x_native = (x_grid - pad_left + 0.5) * scale_x - 0.5 + crop_x
        scale_x  = crop_w / content_w           (content = grid minus padding)
    and the same for y. ``tests/test_precision_camera.py`` checks this map
    against the vendor function itself.
  * Camera model OPENCV: params = [fx, fy, cx, cy, k1, k2, p1, p2] in native
    pixels (Brown–Conrady, the cv2 / COLMAP OPENCV convention). Distortion
    starts at 0 and is refined by F5 (``camera_epoch`` counts the refinements).

Nothing here decides anything: the only bounds it reads (``fx_spread_warn_pct``,
``aspect_tol``) come from ``reconstruction.precision.camera`` and produce
advisory flags, never a rejection.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

# 1 / Φ⁻¹(3/4): the MAD → σ conversion, a property of the normal distribution
_MAD_TO_SIGMA = 1.0 / 0.6744897501960817

# Values of the VENDOR's preprocessing (vggt_omega/utils/load_fn.py), replicated
# so this module stays importable without torch. They are properties of the
# model's input contract, not decisions of ours; the test asserts equality with
# the vendor function for both modes over random native resolutions.
_OMEGA_MIN_ASPECT = 0.5
_OMEGA_MAX_ASPECT = 2.0
OMEGA_PATCH_SIZE = 16
OMEGA_MODES = ("balanced", "max_size")


class CameraError(RuntimeError):
    """Raised when the session camera cannot be established (with the reason)."""


class NativeResolutionError(CameraError):
    """frames/ is not at the native resolution of the source video."""


# ── grid maps ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GridMap:
    """Exact affine relation between a model grid and the native frame.

    ``w, h``: grid size as the model sees it (padding included);
    ``content_w/h``: the resized crop before padding; ``pad_left/top``: padding
    offsets in grid pixels; ``crop_x/y/w/h``: the native crop window;
    ``native_w/h``: the frame. ``scale_x/y`` = native pixels per grid pixel.
    """
    name: str
    w: int
    h: int
    content_w: int
    content_h: int
    pad_left: int
    pad_top: int
    crop_x: int
    crop_y: int
    crop_w: int
    crop_h: int
    native_w: int
    native_h: int

    @property
    def scale_x(self) -> float:
        return self.crop_w / float(self.content_w)

    @property
    def scale_y(self) -> float:
        return self.crop_h / float(self.content_h)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["scale_x"] = self.scale_x
        d["scale_y"] = self.scale_y
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GridMap":
        keys = ("name", "w", "h", "content_w", "content_h", "pad_left", "pad_top",
                "crop_x", "crop_y", "crop_w", "crop_h", "native_w", "native_h")
        missing = [k for k in keys if k not in d]
        if missing:
            raise CameraError(f"grid map is missing keys {missing}")
        return cls(**{k: (d[k] if k == "name" else int(d[k])) for k in keys})


def _crop_to_supported_aspect(native_w: int, native_h: int) -> Tuple[int, int, int, int]:
    """(crop_x, crop_y, crop_w, crop_h) — vendor ``_crop_to_supported_aspect_ratio``."""
    width, height = int(native_w), int(native_h)
    aspect = height / max(width, 1)
    if aspect < _OMEGA_MIN_ASPECT:
        crop_w = min(width, max(1, int(round(height / _OMEGA_MIN_ASPECT))))
        left = max((width - crop_w) // 2, 0)
        return left, 0, crop_w, height
    if aspect > _OMEGA_MAX_ASPECT:
        crop_h = min(height, max(1, int(round(width * _OMEGA_MAX_ASPECT))))
        top = max((height - crop_h) // 2, 0)
        return 0, top, width, crop_h
    return 0, 0, width, height


def _balanced_target(aspect: float, resolution: int, patch: int) -> Tuple[int, int]:
    token_number = (resolution // patch) ** 2
    w_patches = np.sqrt(token_number / aspect)
    h_patches = token_number / w_patches
    w_patches = max(1, int(np.round(w_patches)))
    h_patches = max(1, int(np.round(h_patches)))
    return h_patches * patch, w_patches * patch


def _round_to_patch(value: float, patch: int) -> int:
    return max(patch, int(np.round(float(value) / patch)) * patch)


def _max_size_target(aspect: float, resolution: int, patch: int) -> Tuple[int, int]:
    if aspect >= 1.0:
        return resolution, _round_to_patch(resolution / aspect, patch)
    return _round_to_patch(resolution * aspect, patch), resolution


def omega_grid_for(native_w: int, native_h: int, mode: str, resolution: int,
                   patch_size: int = OMEGA_PATCH_SIZE, name: str = "omega") -> GridMap:
    """The grid ``load_and_preprocess_images(mode, image_resolution)`` produces
    for a native frame, with its exact crop. No padding: one video, one shape."""
    if mode not in OMEGA_MODES:
        raise CameraError(f"omega preprocessing mode must be one of {OMEGA_MODES}, got {mode!r}")
    if resolution <= 0 or resolution % patch_size != 0:
        raise CameraError(f"omega resolution {resolution} must be a positive multiple of {patch_size}")
    cx, cy, cw, ch = _crop_to_supported_aspect(native_w, native_h)
    aspect = ch / max(cw, 1)
    if mode == "balanced":
        th, tw = _balanced_target(aspect, resolution, patch_size)
    else:
        th, tw = _max_size_target(aspect, resolution, patch_size)
    return GridMap(name=name, w=int(tw), h=int(th), content_w=int(tw), content_h=int(th),
                   pad_left=0, pad_top=0, crop_x=cx, crop_y=cy, crop_w=cw, crop_h=ch,
                   native_w=int(native_w), native_h=int(native_h))


def grid_full_frame_resize(native_w: int, native_h: int, grid_w: int, grid_h: int,
                           name: str) -> GridMap:
    """A grid produced by resizing the WHOLE native frame to (grid_w, grid_h)."""
    return GridMap(name=name, w=int(grid_w), h=int(grid_h), content_w=int(grid_w),
                   content_h=int(grid_h), pad_left=0, pad_top=0, crop_x=0, crop_y=0,
                   crop_w=int(native_w), crop_h=int(native_h),
                   native_w=int(native_w), native_h=int(native_h))


def grid_like(reference: GridMap, grid_w: int, grid_h: int, name: str) -> GridMap:
    """A grid that shares ``reference``'s native crop but has its own size
    (e.g. a mask grid derived from the omega crop at another resolution)."""
    return replace(reference, name=name, w=int(grid_w), h=int(grid_h),
                   content_w=int(grid_w), content_h=int(grid_h), pad_left=0, pad_top=0)


def grid_to_native(uv: np.ndarray, g: GridMap) -> np.ndarray:
    """(..., 2) grid pixel coordinates (x = col, y = row) → native pixels."""
    uv = np.asarray(uv, dtype=np.float64)
    out = np.empty_like(uv)
    out[..., 0] = (uv[..., 0] - g.pad_left + 0.5) * g.scale_x - 0.5 + g.crop_x
    out[..., 1] = (uv[..., 1] - g.pad_top + 0.5) * g.scale_y - 0.5 + g.crop_y
    return out


def native_to_grid(uv: np.ndarray, g: GridMap) -> np.ndarray:
    """(..., 2) native pixel coordinates → grid pixels (exact inverse)."""
    uv = np.asarray(uv, dtype=np.float64)
    out = np.empty_like(uv)
    out[..., 0] = (uv[..., 0] - g.crop_x + 0.5) / g.scale_x - 0.5 + g.pad_left
    out[..., 1] = (uv[..., 1] - g.crop_y + 0.5) / g.scale_y - 0.5 + g.pad_top
    return out


def K_grid_to_native(K: np.ndarray, g: GridMap) -> np.ndarray:
    """A 3×3 K expressed on the grid → the same camera in native pixels."""
    K = np.asarray(K, dtype=np.float64)
    out = K.copy()
    out[0, 0] = K[0, 0] * g.scale_x
    out[1, 1] = K[1, 1] * g.scale_y
    out[0, 1] = K[0, 1] * g.scale_x
    out[0, 2] = (K[0, 2] - g.pad_left + 0.5) * g.scale_x - 0.5 + g.crop_x
    out[1, 2] = (K[1, 2] - g.pad_top + 0.5) * g.scale_y - 0.5 + g.crop_y
    return out


def K_native_to_grid(K: np.ndarray, g: GridMap) -> np.ndarray:
    """A 3×3 K in native pixels → the same camera on the grid (exact inverse)."""
    K = np.asarray(K, dtype=np.float64)
    out = K.copy()
    out[0, 0] = K[0, 0] / g.scale_x
    out[1, 1] = K[1, 1] / g.scale_y
    out[0, 1] = K[0, 1] / g.scale_x
    out[0, 2] = (K[0, 2] - g.crop_x + 0.5) / g.scale_x - 0.5 + g.pad_left
    out[1, 2] = (K[1, 2] - g.crop_y + 0.5) / g.scale_y - 0.5 + g.pad_top
    return out


def grid_valid_mask(uv_native: np.ndarray, g: GridMap) -> np.ndarray:
    """Which native points fall inside the grid's crop (visible to the model)."""
    uv = np.asarray(uv_native, dtype=np.float64)
    x, y = uv[..., 0], uv[..., 1]
    return ((x >= g.crop_x - 0.5) & (x <= g.crop_x + g.crop_w - 0.5) &
            (y >= g.crop_y - 0.5) & (y <= g.crop_y + g.crop_h - 0.5))


# ── the session camera ───────────────────────────────────────────────────

CAMERA_MODEL = "OPENCV"
CAMERA_PARAM_NAMES = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2")


@dataclass(frozen=True)
class CameraModel:
    width: int
    height: int
    params: Tuple[float, ...]          # fx, fy, cx, cy, k1, k2, p1, p2
    source: str                        # omega | stray | refine(F5) | synthetic
    camera_epoch: int
    omega_grid: GridMap
    mask_grid: Optional[GridMap] = None
    model: str = CAMERA_MODEL
    report: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.model != CAMERA_MODEL:
            raise CameraError(f"camera model must be {CAMERA_MODEL}, got {self.model!r}")
        if len(self.params) != len(CAMERA_PARAM_NAMES):
            raise CameraError(f"{CAMERA_MODEL} needs {len(CAMERA_PARAM_NAMES)} params "
                              f"{CAMERA_PARAM_NAMES}, got {len(self.params)}")
        object.__setattr__(self, "params", tuple(float(p) for p in self.params))

    # accessors
    @property
    def fx(self) -> float: return self.params[0]
    @property
    def fy(self) -> float: return self.params[1]
    @property
    def cx(self) -> float: return self.params[2]
    @property
    def cy(self) -> float: return self.params[3]

    def K(self) -> np.ndarray:
        fx, fy, cx, cy = self.params[:4]
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    def dist(self) -> np.ndarray:
        """cv2 distCoeffs order (k1, k2, p1, p2)."""
        return np.asarray(self.params[4:8], dtype=np.float64)

    def K_omega(self) -> np.ndarray:
        return K_native_to_grid(self.K(), self.omega_grid)

    def with_params(self, params: Sequence[float], source: str, camera_epoch: int,
                    report: Optional[Dict[str, Any]] = None) -> "CameraModel":
        return replace(self, params=tuple(float(p) for p in params), source=source,
                       camera_epoch=int(camera_epoch), report=dict(report or {}))

    def to_dict(self, geometry_epoch: Optional[int] = None) -> Dict[str, Any]:
        d = {
            "version": 1,
            "provenance": "tool_measured",
            "width": int(self.width), "height": int(self.height),
            "model": self.model,
            "param_names": list(CAMERA_PARAM_NAMES),
            "params": [float(p) for p in self.params],
            "source": self.source,
            "camera_epoch": int(self.camera_epoch),
            "omega_grid": self.omega_grid.to_dict(),
            "mask_grid": self.mask_grid.to_dict() if self.mask_grid is not None else None,
            "report": self.report,
        }
        if geometry_epoch is not None:
            d["geometry_epoch"] = int(geometry_epoch)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CameraModel":
        for k in ("width", "height", "model", "params", "source", "camera_epoch", "omega_grid"):
            if k not in d:
                raise CameraError(f"camera.json is missing key {k!r}")
        return cls(width=int(d["width"]), height=int(d["height"]), model=str(d["model"]),
                   params=tuple(float(p) for p in d["params"]), source=str(d["source"]),
                   camera_epoch=int(d["camera_epoch"]),
                   omega_grid=GridMap.from_dict(d["omega_grid"]),
                   mask_grid=(GridMap.from_dict(d["mask_grid"]) if d.get("mask_grid") else None),
                   report=dict(d.get("report") or {}))


def save_camera_json(path: os.PathLike, cam: CameraModel, geometry_epoch: Optional[int]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(cam.to_dict(geometry_epoch), f, indent=1)
    os.replace(tmp, path)
    return path


def load_camera_json(path: os.PathLike) -> CameraModel:
    path = Path(path)
    if not path.exists():
        raise CameraError(f"{path} does not exist — run precision.camera first")
    with open(path) as f:
        return CameraModel.from_dict(json.load(f))


# ── mappings bound to a camera ───────────────────────────────────────────

def omega_px_to_native(uv: np.ndarray, cam: CameraModel) -> np.ndarray:
    return grid_to_native(uv, cam.omega_grid)


def native_to_omega_px(uv: np.ndarray, cam: CameraModel) -> np.ndarray:
    return native_to_grid(uv, cam.omega_grid)


def mask_px_to_native(uv: np.ndarray, cam: CameraModel) -> np.ndarray:
    if cam.mask_grid is None:
        raise CameraError("camera.json carries no mask_grid — the session has no SAM3 mask grid yet")
    return grid_to_native(uv, cam.mask_grid)


def native_to_mask_px(uv: np.ndarray, cam: CameraModel) -> np.ndarray:
    if cam.mask_grid is None:
        raise CameraError("camera.json carries no mask_grid — the session has no SAM3 mask grid yet")
    return native_to_grid(uv, cam.mask_grid)


# ── distortion (Brown–Conrady via cv2) ───────────────────────────────────

def _cv2():
    import cv2  # local import: keep the module importable where cv2 is absent
    return cv2


def _check_solver(max_iter, eps_px, roundtrip_ulps) -> None:
    for name, v, integer in (("undistort_max_iter", max_iter, True),
                             ("undistort_eps_px", eps_px, False),
                             ("undistort_roundtrip_ulps", roundtrip_ulps, True)):
        ok = not isinstance(v, bool) and (isinstance(v, (int, np.integer)) if integer
                                          else isinstance(v, (int, float, np.floating)))
        if not ok or not (v > 0):
            raise CameraError(f"the point undistortion needs a positive "
                              f"{'integer ' if integer else ''}{name} "
                              f"(reconstruction.precision.camera.{name}), got {v!r}")


def _undistort_verified(uv: np.ndarray, K: np.ndarray, dist: np.ndarray, P: Optional[np.ndarray],
                        max_iter: int, eps_px: float, roundtrip_ulps: int) -> np.ndarray:
    """cv2.undistortPointsIter (output normalised when ``P`` is None, else pixels under
    ``P`` = K), then the proof it converged: cv2 stops at ``max_iter`` or ``eps_px`` and
    reports neither, so every point is RE-DISTORTED through the lens and must land within
    eps_px + roundtrip_ulps × ulp of where it was observed — eps_px is the solver's own
    stop, the ulp term (ulp of the camera's largest pixel magnitude) the float64 resolution
    of re-evaluating the round trip in another order of operations. A point that does not
    is not undistorted: the call fails naming how many and how far."""
    _check_solver(max_iter, eps_px, roundtrip_ulps)
    cv2 = _cv2()
    uv = np.asarray(uv, dtype=np.float64)
    pts = uv.reshape(-1, 1, 2)
    if pts.shape[0] == 0:
        return uv.copy()
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, int(max_iter), float(eps_px))
    out = cv2.undistortPointsIter(pts, K, dist, None, P, crit).reshape(-1, 2)
    xn = out if P is None else np.stack([(out[:, 0] - K[0, 2]) / K[0, 0],
                                         (out[:, 1] - K[1, 2]) / K[1, 1]], axis=1)
    obj = np.stack([xn[:, 0], xn[:, 1], np.ones(len(xn))], axis=1).reshape(-1, 1, 3)
    back, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), K, dist)
    err = np.linalg.norm(back.reshape(-1, 2) - pts.reshape(-1, 2), axis=1)
    mag = max(float(np.abs(pts).max()), abs(float(K[0, 0])), abs(float(K[1, 1])),
              abs(float(K[0, 2])), abs(float(K[1, 2])))
    tol = float(eps_px) + int(roundtrip_ulps) * float(np.spacing(mag))
    bad = ~(err <= tol)                                 # NaN is not converged either
    if bad.any():
        raise CameraError(
            f"point undistortion did not converge for {int(bad.sum())} of {len(err)} point(s): "
            f"re-distorted they land up to {float(np.nanmax(np.where(bad, err, 0.0))):.3g} px "
            f"(NaN: {int(np.isnan(err).sum())}) from where they were observed, tolerance "
            f"{tol:.3g} px (undistort_max_iter {int(max_iter)}; dist {np.asarray(dist).tolist()})")
    return out.reshape(uv.shape)


def undistort_points(uv: np.ndarray, cam: CameraModel, *, max_iter: int,
                     eps_px: float, roundtrip_ulps: int) -> np.ndarray:
    """Distorted native pixels → undistorted native pixels under the SAME K
    (the undistorted image plane of ``undistort_maps``). Iterated to
    convergence, not cv2's default 5 steps: the solver stops after
    ``max_iter`` iterations or when a point's lens reprojection moves it less
    than ``eps_px`` native px — ``reconstruction.precision.camera.
    undistort_max_iter`` / ``undistort_eps_px`` (:func:`undistort_solver`
    reads them from a loaded config) — and every point is then VERIFIED by
    re-distorting it (``roundtrip_ulps``): one that did not converge fails the
    call, it is never returned."""
    _check_solver(max_iter, eps_px, roundtrip_ulps)
    uv = np.asarray(uv, dtype=np.float64)
    if not np.any(cam.dist()):
        return uv.copy()
    return _undistort_verified(uv, cam.K(), cam.dist(), cam.K(), max_iter, eps_px, roundtrip_ulps)


def undistort_normalized(uv: np.ndarray, K: np.ndarray, dist: Sequence[float], *, max_iter: int,
                         eps_px: float, roundtrip_ulps: int) -> np.ndarray:
    """Distorted native pixels → normalised image coordinates (x/z, y/z) under
    (K, dist) — the same solver and the same verification as
    :func:`undistort_points` (the refinement's rungs carry their own camera)."""
    uv = np.asarray(uv, dtype=np.float64)
    return _undistort_verified(uv.reshape(-1, 2), np.asarray(K, np.float64),
                               np.asarray(dist, np.float64), None, max_iter, eps_px,
                               roundtrip_ulps).reshape(uv.shape)


def undistort_solver(camera_cfg) -> dict:
    """``{"max_iter", "eps_px", "roundtrip_ulps"}`` for :func:`undistort_points`
    from a loaded ``precision.config.CameraConfig``."""
    return {"max_iter": int(camera_cfg.undistort_max_iter),
            "eps_px": float(camera_cfg.undistort_eps_px),
            "roundtrip_ulps": int(camera_cfg.undistort_roundtrip_ulps)}


def distort_points(uv_undist: np.ndarray, cam: CameraModel) -> np.ndarray:
    """Undistorted native pixels (same K) → where the lens puts them."""
    cv2 = _cv2()
    uv = np.asarray(uv_undist, dtype=np.float64)
    shp = uv.shape
    if not np.any(cam.dist()):
        return uv.copy()
    x = (uv[..., 0].reshape(-1) - cam.cx) / cam.fx
    y = (uv[..., 1].reshape(-1) - cam.cy) / cam.fy
    obj = np.stack([x, y, np.ones_like(x)], axis=1).reshape(-1, 1, 3)
    proj, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), cam.K(), cam.dist())
    return proj.reshape(shp)


def undistort_maps(cam: CameraModel) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(map_x, map_y, K_new) for ``cv2.remap`` — the undistorted image keeps the
    session K (K_new = K) so undistorted observations need no second camera."""
    cv2 = _cv2()
    K = cam.K()
    m1, m2 = cv2.initUndistortRectifyMap(K, cam.dist(), None, K, (cam.width, cam.height),
                                         cv2.CV_32FC1)
    return m1, m2, K


# ── initialisation ───────────────────────────────────────────────────────

def _robust_spread_pct(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    med = float(np.median(x))
    if med == 0.0 or x.size < 2:
        return 0.0
    mad = float(np.median(np.abs(x - med)))
    return 100.0 * mad * _MAD_TO_SIGMA / abs(med)


def omega_intrinsics_to_rows(intr: np.ndarray) -> np.ndarray:
    """(N,3,3) K matrices or (N,4) [fx fy cx cy] rows → (N,4) rows."""
    a = np.asarray(intr, dtype=np.float64)
    if a.ndim == 3 and a.shape[1:] == (3, 3):
        return np.stack([a[:, 0, 0], a[:, 1, 1], a[:, 0, 2], a[:, 1, 2]], axis=1)
    if a.ndim == 2 and a.shape[1] >= 4:
        return a[:, :4]
    raise CameraError(f"omega intrinsics must be (N,3,3) or (N,4), got {a.shape}")


def camera_from_omega(intr_grid: np.ndarray, grid: GridMap, fx_spread_warn_pct: float,
                      mask_grid: Optional[GridMap] = None, camera_epoch: int = 0,
                      source: str = "omega") -> CameraModel:
    """Session camera from omega's per-frame K on its grid: each K is carried to
    native pixels through the exact grid map, the session takes the robust
    median, and the per-frame focal SPREAD is reported (advisory flag when it
    exceeds ``fx_spread_warn_pct``). Distortion starts at 0 (F5 refines)."""
    rows = omega_intrinsics_to_rows(intr_grid)
    if rows.shape[0] == 0:
        raise CameraError("no omega intrinsics to initialise the camera from")
    nat = np.empty_like(rows)
    for i, (fx, fy, cx, cy) in enumerate(rows):
        Kn = K_grid_to_native(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]), grid)
        nat[i] = (Kn[0, 0], Kn[1, 1], Kn[0, 2], Kn[1, 2])
    med = np.median(nat, axis=0)
    fx_spread = _robust_spread_pct(nat[:, 0])
    fy_spread = _robust_spread_pct(nat[:, 1])
    report = {
        "n_frames": int(rows.shape[0]),
        "fx_spread_pct": float(fx_spread),
        "fy_spread_pct": float(fy_spread),
        "fx_spread_warn_pct": float(fx_spread_warn_pct),
        "fx_spread_warning": bool(max(fx_spread, fy_spread) > fx_spread_warn_pct),
        "fx_native_min": float(nat[:, 0].min()), "fx_native_max": float(nat[:, 0].max()),
        "pixel_aspect_fx_over_fy": float(med[0] / med[1]) if med[1] else None,
        "omega_grid_fx_median": float(np.median(rows[:, 0])),
        "omega_grid_fy_median": float(np.median(rows[:, 1])),
    }
    params = (float(med[0]), float(med[1]), float(med[2]), float(med[3]), 0.0, 0.0, 0.0, 0.0)
    return CameraModel(width=grid.native_w, height=grid.native_h, params=params, source=source,
                       camera_epoch=int(camera_epoch), omega_grid=grid, mask_grid=mask_grid,
                       report=report)


def rescale_K(K: np.ndarray, from_wh: Tuple[int, int], to_wh: Tuple[int, int],
              aspect_tol: float) -> np.ndarray:
    """Carry a K between two full-frame resolutions of the SAME sensor
    (pure resize, no crop). Refuses when the aspect ratios disagree beyond
    ``aspect_tol`` — that would be a crop nobody declared."""
    fw, fh = from_wh
    tw, th = to_wh
    a_from = fh / float(fw)
    a_to = th / float(tw)
    if abs(a_from - a_to) / a_from > aspect_tol:
        raise CameraError(
            f"cannot rescale K from {fw}x{fh} to {tw}x{th}: aspect ratios differ "
            f"({a_from:.5f} vs {a_to:.5f}, tolerance {aspect_tol}) — the frames were "
            f"cropped, not only resized; declare the crop")
    g = grid_full_frame_resize(tw, th, fw, fh, name="rescale")
    return K_grid_to_native(K, g)


def camera_from_stray(K_stray: np.ndarray, stray_wh: Tuple[int, int], grid: GridMap,
                      aspect_tol: float, mask_grid: Optional[GridMap] = None,
                      camera_epoch: int = 0) -> CameraModel:
    """Session camera from Stray Scanner's ``camera_matrix.csv`` (given at the
    resolution of the RGB stream it was calibrated for), carried to the native
    frame resolution. Distortion 0 (Stray publishes none; F5 refines)."""
    K = rescale_K(np.asarray(K_stray, dtype=np.float64), stray_wh,
                  (grid.native_w, grid.native_h), aspect_tol)
    params = (K[0, 0], K[1, 1], K[0, 2], K[1, 2], 0.0, 0.0, 0.0, 0.0)
    report = {"stray_resolution": [int(stray_wh[0]), int(stray_wh[1])],
              "stray_K": np.asarray(K_stray, dtype=np.float64).tolist()}
    return CameraModel(width=grid.native_w, height=grid.native_h, params=params, source="stray",
                       camera_epoch=int(camera_epoch), omega_grid=grid, mask_grid=mask_grid,
                       report=report)


def read_stray_camera_matrix(path: os.PathLike) -> np.ndarray:
    K = np.loadtxt(str(path), delimiter=",", dtype=np.float64)
    if K.shape != (3, 3):
        raise CameraError(f"{path}: expected a 3x3 camera matrix, got shape {K.shape}")
    return K


def cross_check(cam: CameraModel, other: CameraModel) -> Dict[str, Any]:
    """How far two independent camera estimates disagree (reported, never gates)."""
    a, b = np.asarray(cam.params[:4]), np.asarray(other.params[:4])
    return {"reference": cam.source, "other": other.source,
            "params_reference": a.tolist(), "params_other": b.tolist(),
            "rel_diff_pct": (100.0 * (b - a) / np.where(a == 0, 1.0, a)).tolist()}


# ── native-resolution guard ──────────────────────────────────────────────

VIDEO_SUFFIXES = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm")


def find_source_video(session_dir: os.PathLike) -> Optional[Path]:
    root = Path(session_dir)
    cands = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES] \
        if root.exists() else []
    if not cands:
        return None
    pref = [p for p in cands if p.stem == "source_video"]
    return sorted(pref or cands)[0]


def video_size(video_path: os.PathLike) -> Tuple[int, int]:
    """(width, height) of the decoded frames of a video as cv2 delivers them
    (rotation metadata applied, the same decoder the extractor uses)."""
    cv2 = _cv2()
    cap = cv2.VideoCapture(str(video_path))
    try:
        ok, frame = cap.read()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            return int(w), int(h)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if w <= 0 or h <= 0:
        raise CameraError(f"cannot read the frame size of {video_path}")
    return w, h


def frame_sizes(frames_dir: os.PathLike, pattern: str = "*.jpg") -> Dict[Tuple[int, int], int]:
    """Histogram {(w, h): count} over the frames (header read only)."""
    from PIL import Image
    hist: Dict[Tuple[int, int], int] = {}
    for p in sorted(glob.glob(os.path.join(str(frames_dir), pattern))):
        with Image.open(p) as im:
            wh = (int(im.size[0]), int(im.size[1]))
        hist[wh] = hist.get(wh, 0) + 1
    return hist


def verify_native_frames(session_dir: os.PathLike, frames_subdir: str = "frames") -> Dict[str, Any]:
    """Assert frames/*.jpg are at the video's native resolution. Returns the
    report; raises NativeResolutionError naming every offending size."""
    root = Path(session_dir)
    fdir = root / frames_subdir
    hist = frame_sizes(fdir)
    if not hist:
        raise NativeResolutionError(f"{fdir} holds no frames")
    video = find_source_video(root)
    report: Dict[str, Any] = {"frames_dir": str(fdir),
                              "frame_sizes": {f"{w}x{h}": n for (w, h), n in hist.items()},
                              "n_frames": int(sum(hist.values())),
                              "video": str(video) if video else None}
    if len(hist) > 1:
        raise NativeResolutionError(
            f"{fdir} mixes frame sizes {sorted(hist)} — one video, one resolution")
    (fw, fh), = hist.keys()
    report.update({"native_w": fw, "native_h": fh})
    if video is not None:
        vw, vh = video_size(video)
        report.update({"video_w": vw, "video_h": vh})
        if (fw, fh) != (vw, vh):
            raise NativeResolutionError(
                f"frames are {fw}x{fh} but {video.name} decodes to {vw}x{vh}: something "
                f"rescaled the frames — re-extract at native resolution")
    return report


def verify_omega_grid(grid: GridMap, depth_hw: Tuple[int, int]) -> None:
    """The grid the mapping assumes must be the grid omega actually produced."""
    h, w = int(depth_hw[0]), int(depth_hw[1])
    if (w, h) != (grid.w, grid.h):
        raise CameraError(
            f"omega depth is {w}x{h} but the declared preprocessing yields {grid.w}x{grid.h} "
            f"({grid.name}) — the mode/resolution in camera.json does not match the run")


# ── session-level orchestration (F0 stage) ───────────────────────────────

OMEGA_CONFIG_NAME = "vggt_omega_config.yaml"
INTRINSIC_NAME = "intrinsic.txt"
CAMERA_JSON_NAME = "camera.json"
SEG_MASKS_NAME = "seg_masks.npz"
GEOMETRY_EPOCH_NAME = "geometry_epoch.json"


def read_geometry_epoch(output_dir: os.PathLike) -> int:
    """The session's current geometry epoch (0 = the original reconstruction)."""
    p = Path(output_dir) / GEOMETRY_EPOCH_NAME
    if not p.exists():
        return 0
    with open(p) as f:
        return int(json.load(f).get("epoch", 0))


def read_omega_preprocessing(output_dir: os.PathLike) -> Tuple[str, int]:
    """(mode, resolution) the omega run used, from the config the run wrote."""
    p = Path(output_dir) / OMEGA_CONFIG_NAME
    if not p.exists():
        raise CameraError(f"{p} not found — the omega run did not record its config; "
                          f"the grid mapping cannot be established without mode/resolution")
    import yaml
    with open(p) as f:
        cfg = yaml.safe_load(f) or {}
    model = cfg.get("Model") or {}
    if "omega_mode" not in model or "omega_resolution" not in model:
        raise CameraError(f"{p} lacks Model.omega_mode / Model.omega_resolution — cannot "
                          f"derive the omega grid")
    return str(model["omega_mode"]), int(model["omega_resolution"])


def omega_intrinsics_path(output_dir: os.PathLike) -> Path:
    """The intrinsic.txt :func:`read_omega_intrinsics` reads (output/ first, then maplong_run/)."""
    out = Path(output_dir)
    for cand in (out / INTRINSIC_NAME, out / "maplong_run" / INTRINSIC_NAME):
        if cand.exists():
            return cand
    raise CameraError(f"no {INTRINSIC_NAME} under {out} — the omega run did not write intrinsics")


def read_omega_intrinsics(output_dir: os.PathLike) -> np.ndarray:
    """intrinsic.txt rows [fx fy cx cy] on the omega grid (one per keyframe)."""
    cand = omega_intrinsics_path(output_dir)
    rows = np.loadtxt(str(cand), dtype=np.float64, ndmin=2)
    if rows.shape[1] < 4:
        raise CameraError(f"{cand}: expected 'fx fy cx cy' rows, got shape {rows.shape}")
    return rows[:, :4]


def k_source_record(source: str, path: os.PathLike, session_dir: os.PathLike) -> Dict[str, Any]:
    """Where the session camera's K came from (docs/plan_determinismo.md point 77): the source,
    the file (relative to the session) and its sha256 — camera.json carries it, so a camera
    that changed because its input changed says so."""
    import repro
    p, root = Path(path), Path(session_dir)
    try:
        rel = p.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        raise CameraError(f"the K source {p} is not inside the session {root}") from None
    return {"source": str(source), "file": rel, "sha256": repro.sha256_file(p)}


def omega_depth_shape(output_dir: os.PathLike) -> Optional[Tuple[int, int]]:
    """(H, W) of the omega depth actually written, from the first frame npz."""
    files = sorted(glob.glob(os.path.join(str(output_dir), "omega_run", "results_output",
                                          "frame_*.npz")))
    if not files:
        return None
    with np.load(files[0]) as z:
        if "depth" not in z.files:
            return None
        return tuple(int(v) for v in z["depth"].shape[:2])


def mask_grid_shape(output_dir: os.PathLike) -> Optional[Tuple[int, int]]:
    """(H, W) of the SAM3 mask grid (``scaled_res`` of seg_masks.npz) when the
    session already has masks."""
    p = Path(output_dir) / SEG_MASKS_NAME
    if not p.exists():
        return None
    with np.load(p, allow_pickle=False) as z:
        if "scaled_res" not in z.files:
            return None
        hw = np.asarray(z["scaled_res"]).reshape(-1)
        return int(hw[0]), int(hw[1])


def mask_grid_for(native_w: int, native_h: int, mask_hw: Tuple[int, int]) -> GridMap:
    """The SAM3 mask grid: masks come back at NATIVE resolution and
    ``_save_masks`` resizes the WHOLE frame to ``scaled_res`` — a full-frame
    resize, whatever crop omega applied (when omega did not crop, this is the
    omega grid itself)."""
    h, w = int(mask_hw[0]), int(mask_hw[1])
    return grid_full_frame_resize(native_w, native_h, w, h, name="mask")


def find_stray_camera_matrix(session_dir: os.PathLike) -> Optional[Path]:
    """Stray Scanner's camera_matrix.csv next to an odometry.csv of THIS scan only, in reading
    order: ``inputs/stray/`` (the capture data the replace wipe never touches —
    docs/plan_determinismo.md point 77, DECIDIDO 2026-10-07), then the scan directory, then its
    ``stray/`` subdirectory (ingestors.capture_inputs.stray_candidates — point 35: a sibling scan
    of the same day is another recording, its calibration is not this camera's, and which sibling
    an unsorted directory listing named first was the filesystem's choice). Two different
    matrices in the scan's own places are refused, naming both."""
    from ingestors.capture_inputs import stray_dirs
    root = Path(session_dir)
    found = [d / "camera_matrix.csv"
             for d in stray_dirs(root, required=("odometry.csv", "camera_matrix.csv"))]
    if len(found) > 1 and len({p.read_bytes() for p in found}) > 1:
        raise CameraError(f"two different Stray calibrations in this scan: "
                          f"{', '.join(str(p) for p in found)} — keep the one of this recording")
    return found[0] if found else None


def build_session_camera(session_dir: os.PathLike, cam_cfg, output_subdir: str = "output",
                         log=print) -> CameraModel:
    """F0 stage: verify native frames, derive the omega + mask grids, initialise
    the session camera (Stray when present and allowed, else omega's median K)
    and write output/camera.json stamped with the geometry epoch. Returns the
    camera. Fails hard, naming the reason, when the grid cannot be established."""
    root = Path(session_dir)
    out = root / output_subdir
    frames_rep = verify_native_frames(root)
    nw, nh = int(frames_rep["native_w"]), int(frames_rep["native_h"])
    mode, res = read_omega_preprocessing(out)
    grid = omega_grid_for(nw, nh, mode, res)
    hw = omega_depth_shape(out)
    if hw is not None:
        verify_omega_grid(grid, hw)
    # the camera's mask grid is the full-frame grid at Omega's size, ALWAYS: F0 runs before any
    # segmentation and its camera.json must not depend on whether a seg_masks.npz of an earlier
    # run happened to be on disk (point 36). Every reader of real masks derives their grid from
    # the masks' own scaled_res (mask_grid_for / mask_grid_shape).
    mask_grid = mask_grid_for(nw, nh, (grid.h, grid.w))
    rows = read_omega_intrinsics(out)
    epoch = read_geometry_epoch(out)
    omega_cam = camera_from_omega(rows, grid, cam_cfg.fx_spread_warn_pct, mask_grid=mask_grid)
    stray_csv = find_stray_camera_matrix(root) if cam_cfg.init_from in ("auto", "stray") else None
    if cam_cfg.init_from == "stray" and stray_csv is None:
        raise CameraError(f"precision.camera.init_from is 'stray' but no camera_matrix.csv "
                          f"(next to an odometry.csv) exists for {root}")
    if stray_csv is not None:
        K_stray = read_stray_camera_matrix(stray_csv)
        cam = camera_from_stray(K_stray, (nw, nh), grid, cam_cfg.aspect_tol, mask_grid=mask_grid)
        report = dict(cam.report)
        report["cross_check_omega"] = cross_check(cam, omega_cam)
        report["omega"] = omega_cam.report
        cam = replace(cam, report=report)
    else:
        cam = omega_cam
    report = dict(cam.report)
    report["frames"] = frames_rep
    report["omega_preprocessing"] = {"mode": mode, "resolution": res,
                                     "depth_shape_hw": list(hw) if hw else None}
    # the K source and the sha256 of the file K came from (point 77, DECIDIDO 2026-10-07)
    report["k_source"] = (k_source_record("stray", stray_csv, root) if stray_csv is not None
                          else k_source_record("omega", omega_intrinsics_path(out), root))
    cam = replace(cam, report=report)
    p = save_camera_json(out / CAMERA_JSON_NAME, cam, geometry_epoch=epoch)
    log(f"[precision.camera] {p}: {cam.source} K=({cam.fx:.2f},{cam.fy:.2f},{cam.cx:.2f},"
        f"{cam.cy:.2f}) native {nw}x{nh}, omega grid {grid.w}x{grid.h} ({mode} {res}), "
        f"fx spread {cam.report.get('fx_spread_pct', omega_cam.report['fx_spread_pct']):.2f}% "
        f"(warn > {cam_cfg.fx_spread_warn_pct}%), geometry_epoch {epoch}")
    return cam


def load_session_camera(session_dir: os.PathLike, output_subdir: str = "output") -> CameraModel:
    return load_camera_json(Path(session_dir) / output_subdir / CAMERA_JSON_NAME)


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser("precision.camera — F0 session camera + grid mappings")
    ap.add_argument("--session", required=True, help="session directory (holds frames/ and output/)")
    args = ap.parse_args(argv)
    from precision.config import load_precision_config
    cfg = load_precision_config()
    build_session_camera(args.session, cfg.camera)
    return 0


if __name__ == "__main__":     # pragma: no cover
    raise SystemExit(main())
