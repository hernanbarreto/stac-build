"""What a VARIANT delivers to the battery — a ``Product``: keyframe poses, a camera per keyframe
and a point cloud, with or without per-point provenance.

``StacProduct`` reads a finished session of the chain (the published cloud carries
``frame_global / pixel_row / pixel_col`` per point on the UNDISTORTED native grid, so the depth of
the product at an anchored pixel is the depth of its own surviving points there — exact, see
precision/depth_on_f5.py:853). ``DenseProduct`` is a reconstruction without provenance (COLMAP,
an external import): poses + camera + a dense cloud; its depth at a pixel comes from a z-buffer
of the cloud, and two observations triangulate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from evaluation import geometry as G
from evaluation.cloud_io import read_ply_memmap, xyz_of


class ProductError(RuntimeError):
    pass


@dataclass
class DepthSample:
    z: float            # depth along the camera z (product units)
    n: int              # samples pooled
    spread: float       # MAD of the pooled depths (0 for one sample)
    method: str         # provenance | zbuffer


@dataclass
class FrameProjection:
    """The whole cloud projected into one keyframe (cached per frame): undistorted pixels, depth
    and the row indices of the points in front of the camera."""
    uv: np.ndarray
    z: np.ndarray
    rows: np.ndarray


def _window_median(z: np.ndarray, method: str) -> DepthSample:
    z = np.asarray(z, np.float64)
    med = float(np.median(z))
    mad = float(np.median(np.abs(z - med))) if z.size > 1 else 0.0
    return DepthSample(med, int(z.size), mad, method)


class Product:
    """Base: poses + camera + cloud. Subclasses fill the depth lookups."""
    name: str = "product"
    kind: str = "dense"
    frames: List[int]
    c2w: np.ndarray
    xyz: np.ndarray

    def __init__(self) -> None:
        self._proj_cache: Dict[int, FrameProjection] = {}
        self._index: Dict[int, int] = {}

    # ── camera ──
    def K(self, frame: int) -> np.ndarray:
        raise NotImplementedError

    def undistort(self, frame: int, uv: np.ndarray) -> np.ndarray:
        """Clicked (distorted) native pixels → undistorted pixels of the product's camera."""
        return np.atleast_2d(np.asarray(uv, np.float64))

    def registered(self, frame: int) -> bool:
        return frame in self._index

    def pose(self, frame: int) -> np.ndarray:
        return self.c2w[self._index[frame]]

    def chainage(self) -> Dict[int, float]:
        ch = G.chainage(self.c2w)
        return {f: float(ch[i]) for f, i in self._index.items()}

    def _finish_init(self) -> None:
        self._index = {int(f): i for i, f in enumerate(self.frames)}
        if len(self.frames) != len(self.c2w):
            raise ProductError(f"{self.name}: {len(self.frames)} frames but {len(self.c2w)} poses")

    # ── depth ──
    def has_provenance(self) -> bool:
        return False

    def depth_by_provenance(self, frame: int, uv_und: np.ndarray, window_px: int
                            ) -> Optional[DepthSample]:
        return None

    def projection(self, frame: int) -> FrameProjection:
        """The cloud projected in ``frame`` (one frame cached at a time: 19 M points = 0.5 GB)."""
        if frame in self._proj_cache:
            return self._proj_cache[frame]
        uv, z = G.project(self.xyz, self.K(frame), self.pose(frame))
        front = np.isfinite(z) & (z > 0)
        rows = np.nonzero(front)[0]
        fp = FrameProjection(uv[front], z[front], rows)
        self._proj_cache = {frame: fp}
        return fp

    def depth_by_zbuffer(self, frame: int, uv_und: np.ndarray, window_px: int
                         ) -> Optional[DepthSample]:
        """The NEAREST splat within ± window_px of the pixel (a z-buffer read)."""
        fp = self.projection(frame)
        u, v = float(uv_und[0]), float(uv_und[1])
        m = (np.abs(fp.uv[:, 0] - u) <= window_px) & (np.abs(fp.uv[:, 1] - v) <= window_px)
        if not m.any():
            return None
        return DepthSample(float(fp.z[m].min()), int(m.sum()), 0.0, "zbuffer")

    # ── per-point fields (phase 2 / 5) ──
    def fields(self) -> Sequence[str]:
        return ()

    def points_of_frame(self, frame: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """(row indices of the points born in ``frame``, their undistorted (u, v) pixels) —
        empty without provenance."""
        return np.zeros(0, np.int64), None


class StacProduct(Product):
    """A finished session of the chain (output/: camera.json, camera_frames.txt,
    camera_poses.txt, cleaned_cloud.ply with provenance, instance_ids.npy when segmented)."""
    kind = "stac"

    def __init__(self, session_dir: Path, name: str = "stac", *, cloud_name: str = "cleaned_cloud.ply",
                 solver: Optional[dict] = None, log: Callable = print) -> None:
        super().__init__()
        from precision.camera import load_camera_json
        from precision.poses_epoch import load_poses
        self.name = name
        self.session_dir = Path(session_dir)
        out = self.session_dir / "output"
        if not (out / "camera.json").exists():
            raise ProductError(f"{out}/camera.json is missing — not a finished session of the chain")
        self.cam = load_camera_json(out / "camera.json")
        self.frames, self.c2w = load_poses(out)
        self._finish_init()
        cloud = out / cloud_name
        if not cloud.exists():
            raise ProductError(f"{cloud} is missing — the session published no cloud")
        self.cloud_path = cloud
        self.data = read_ply_memmap(cloud)
        names = self.data.dtype.names or ()
        for needed in ("x", "y", "z", "frame_global", "pixel_row", "pixel_col"):
            if needed not in names:
                raise ProductError(f"{cloud} has no '{needed}' column — provenance is required")
        self.xyz = xyz_of(self.data)
        fg = np.asarray(self.data["frame_global"], np.int64)
        self._order = np.argsort(fg, kind="stable")
        self._fg_sorted = fg[self._order]
        self.instance_ids = None
        p = out / "instance_ids.npy"
        if p.exists():
            ids = np.load(p, mmap_mode="r")
            if len(ids) == len(self.data):
                self.instance_ids = ids
            else:
                log(f"[evaluation] {p} has {len(ids)} rows for {len(self.data)} points — ignored")
        self._solver = solver if solver is not None else _undistort_solver_of(self.session_dir, log)
        self._K = self.cam.K()

    def K(self, frame: int) -> np.ndarray:
        return self._K

    def undistort(self, frame: int, uv: np.ndarray) -> np.ndarray:
        from precision.camera import undistort_points
        uv = np.atleast_2d(np.asarray(uv, np.float64))
        if not np.any(np.asarray(self.cam.params[4:], np.float64)):
            return uv
        return undistort_points(uv, self.cam, **{k: v for k, v in self._solver.items() if k != "source"})

    def has_provenance(self) -> bool:
        return True

    def fields(self) -> Sequence[str]:
        return tuple(n for n in (self.data.dtype.names or ())
                     if n not in ("x", "y", "z", "red", "green", "blue"))

    def rows_of_frame(self, frame: int) -> np.ndarray:
        lo = np.searchsorted(self._fg_sorted, frame, "left")
        hi = np.searchsorted(self._fg_sorted, frame, "right")
        return self._order[lo:hi]

    def points_of_frame(self, frame: int) -> Tuple[np.ndarray, np.ndarray]:
        rows = self.rows_of_frame(frame)
        uv = np.stack([np.asarray(self.data["pixel_col"][rows], np.float64),
                       np.asarray(self.data["pixel_row"][rows], np.float64)], axis=1)
        return rows, uv

    def depth_by_provenance(self, frame: int, uv_und: np.ndarray, window_px: int
                            ) -> Optional[DepthSample]:
        rows, uv = self.points_of_frame(frame)
        if rows.size == 0:
            return None
        u, v = float(uv_und[0]), float(uv_und[1])
        m = (np.abs(uv[:, 0] - u) <= window_px) & (np.abs(uv[:, 1] - v) <= window_px)
        if not m.any():
            return None
        z = G.camera_depth(self.xyz[rows[m]], self.pose(frame))
        z = z[np.isfinite(z) & (z > 0)]
        if z.size == 0:
            return None
        return _window_median(z, "provenance")

    def camera_centre(self, frame: int) -> np.ndarray:
        return self.pose(frame)[:3, 3]


def _undistort_solver_of(session_dir: Path, log: Callable) -> dict:
    """The chain's own undistortion solver bounds (reconstruction.precision.camera:
    undistort_max_iter / undistort_eps_px / undistort_roundtrip_ulps) from the session's frozen
    configuration when it carries them, else the server's — declared either way. Only the camera
    keys are read: a frozen configuration older than the current schema still serves."""
    from precision.config import raw_config_of
    try:
        raw, sha = raw_config_of(session_dir, log=lambda *a, **k: None)
    except Exception as e:  # noqa: BLE001 — declared, the server's config stands in
        log(f"[evaluation] {session_dir}: frozen configuration unreadable ({e}); the server's "
            f"camera solver bounds are used")
        raw, sha = None, None
    keys = ("undistort_max_iter", "undistort_eps_px", "undistort_roundtrip_ulps")
    sec = (((raw or {}).get("reconstruction") or {}).get("precision") or {}).get("camera") or {}
    source = f"frozen run_config ({sha})"
    if not all(k in sec for k in keys):
        from config import cfg
        sec = cfg["reconstruction"]["precision"]["camera"]
        source = "server config.yaml"
        if raw is not None:
            log(f"[evaluation] {session_dir}: the frozen configuration lacks the camera solver "
                f"bounds; the server's are used")
    return {"max_iter": int(sec["undistort_max_iter"]), "eps_px": float(sec["undistort_eps_px"]),
            "roundtrip_ulps": int(sec["undistort_roundtrip_ulps"]), "source": source}


@dataclass
class DenseProduct(Product):
    """Poses + a camera + a dense cloud without provenance (COLMAP, an import). ``K_of``,
    ``undistort_of`` and ``project_of`` are the camera model's own functions (per frame):
    ``undistort_of(frame, uv_distorted) -> uv_undistorted`` under the K that ``K_of`` returns."""
    frames: List[int]
    c2w: np.ndarray
    xyz: np.ndarray
    K_of: Callable[[int], np.ndarray]
    undistort_of: Optional[Callable[[int, np.ndarray], np.ndarray]] = None
    name: str = "dense"
    kind: str = "dense"
    registered_frames: Optional[set] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        Product.__init__(self)
        self.frames = [int(f) for f in self.frames]
        self.c2w = np.asarray(self.c2w, np.float64)
        self.xyz = np.asarray(self.xyz, np.float64)
        self._finish_init()

    def K(self, frame: int) -> np.ndarray:
        return self.K_of(frame)

    def undistort(self, frame: int, uv: np.ndarray) -> np.ndarray:
        uv = np.atleast_2d(np.asarray(uv, np.float64))
        return self.undistort_of(frame, uv) if self.undistort_of is not None else uv
