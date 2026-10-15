"""COLMAP sparse models (cameras / images / points3D, binary or text) and the camera models the
baseline uses, read without pycolmap (the formats are stable; the chain's envs hold three pycolmap
versions). A model becomes a :class:`DenseProduct` with its dense cloud.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from evaluation.cloud_io import read_ply_memmap, xyz_of
from evaluation.product import DenseProduct

# model id → (name, n_params)
CAMERA_MODELS = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5),
                 4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8), 6: ("FULL_OPENCV", 12), 7: ("FOV", 5),
                 8: ("SIMPLE_RADIAL_FISHEYE", 4), 9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12)}
MODEL_ID_OF = {name: mid for mid, (name, _) in CAMERA_MODELS.items()}
SUPPORTED = ("SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV")


class ColmapIOError(RuntimeError):
    pass


@dataclass
class ColmapCamera:
    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    def K(self) -> np.ndarray:
        p = self.params
        if self.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            fx = fy = p[0]; cx, cy = p[1], p[2]
        elif self.model in ("PINHOLE", "OPENCV"):
            fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        else:
            raise ColmapIOError(f"camera model {self.model} is not supported ({SUPPORTED})")
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)

    def distortion(self) -> np.ndarray:
        """(k1, k2, p1, p2) of the model."""
        p = self.params
        if self.model == "SIMPLE_RADIAL":
            return np.array([p[3], 0, 0, 0], np.float64)
        if self.model == "RADIAL":
            return np.array([p[3], p[4], 0, 0], np.float64)
        if self.model == "OPENCV":
            return np.array([p[4], p[5], p[6], p[7]], np.float64)
        if self.model in ("SIMPLE_PINHOLE", "PINHOLE"):
            return np.zeros(4)
        raise ColmapIOError(f"camera model {self.model} is not supported ({SUPPORTED})")

    def distort_normalized(self, xn: np.ndarray) -> np.ndarray:
        k1, k2, p1, p2 = self.distortion()
        x, y = xn[:, 0], xn[:, 1]
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2 * r2
        xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        return np.stack([xd, yd], axis=1)

    def undistort_pixels(self, uv: np.ndarray, max_iter: int = 50, eps: float = 1e-12) -> np.ndarray:
        """Distorted pixels → undistorted pixels under the same K (fixed-point inversion of the
        Brown model, as COLMAP's IterativeUndistortion)."""
        K = self.K()
        uv = np.atleast_2d(np.asarray(uv, np.float64))
        xd = np.stack([(uv[:, 0] - K[0, 2]) / K[0, 0], (uv[:, 1] - K[1, 2]) / K[1, 1]], axis=1)
        if not np.any(self.distortion()):
            return uv.copy()
        x = xd.copy()
        for _ in range(max_iter):
            d = self.distort_normalized(x) - x          # the distortion offset at the current estimate
            x_new = xd - d
            if np.max(np.abs(x_new - x)) < eps:
                x = x_new
                break
            x = x_new
        return np.stack([x[:, 0] * K[0, 0] + K[0, 2], x[:, 1] * K[1, 1] + K[1, 2]], axis=1)

    def project(self, X: np.ndarray, w2c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(distorted pixels, depth) of world points."""
        Xc = np.atleast_2d(X) @ w2c[:3, :3].T + w2c[:3, 3]
        z = Xc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            xn = np.stack([Xc[:, 0] / z, Xc[:, 1] / z], axis=1)
        xd = self.distort_normalized(xn)
        K = self.K()
        return np.stack([xd[:, 0] * K[0, 0] + K[0, 2], xd[:, 1] * K[1, 1] + K[1, 2]], axis=1), z


@dataclass
class ColmapImage:
    image_id: int
    name: str
    camera_id: int
    qvec: np.ndarray        # (qw, qx, qy, qz) of the w2c rotation
    tvec: np.ndarray        # w2c translation
    n_points2d: int = 0
    n_points3d: int = 0     # 2-D points with a 3-D track

    def w2c(self) -> np.ndarray:
        R = qvec_to_rotmat(self.qvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = self.tvec
        return T

    def c2w(self) -> np.ndarray:
        return np.linalg.inv(self.w2c())


@dataclass
class ColmapModel:
    path: Path
    cameras: Dict[int, ColmapCamera]
    images: Dict[int, ColmapImage]
    n_points3d: int

    @property
    def n_images(self) -> int:
        return len(self.images)


def qvec_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                     [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
                     [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]])


def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, np.float64)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    if i == 1:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
    s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])


# ── readers ───────────────────────────────────────────────────────────────

def _read(f, fmt: str):
    size = struct.calcsize(fmt)
    data = f.read(size)
    if len(data) != size:
        raise ColmapIOError("truncated COLMAP binary file")
    return struct.unpack("<" + fmt, data)


def read_cameras(path: Path) -> Dict[int, ColmapCamera]:
    path = Path(path)
    out: Dict[int, ColmapCamera] = {}
    if path.suffix == ".bin":
        with open(path, "rb") as f:
            (n,) = _read(f, "Q")
            for _ in range(n):
                cid, mid, w, h = _read(f, "iiQQ")
                name, npar = CAMERA_MODELS[mid]
                params = np.array(_read(f, "d" * npar), np.float64)
                out[cid] = ColmapCamera(cid, name, int(w), int(h), params)
    else:
        for line in path.read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            cid, name, w, h = int(parts[0]), parts[1], int(parts[2]), int(parts[3])
            out[cid] = ColmapCamera(cid, name, w, h, np.array([float(x) for x in parts[4:]]))
    return out


def read_images(path: Path) -> Dict[int, ColmapImage]:
    path = Path(path)
    out: Dict[int, ColmapImage] = {}
    if path.suffix == ".bin":
        with open(path, "rb") as f:
            (n,) = _read(f, "Q")
            for _ in range(n):
                (iid,) = _read(f, "i")
                q = np.array(_read(f, "dddd"))
                t = np.array(_read(f, "ddd"))
                (cid,) = _read(f, "i")
                name = b""
                while True:
                    c = f.read(1)
                    if c == b"\x00" or not c:
                        break
                    name += c
                (n2,) = _read(f, "Q")
                raw = f.read(n2 * 24)
                if len(raw) != n2 * 24:
                    raise ColmapIOError("truncated images.bin")
                ids = np.frombuffer(raw, dtype=np.dtype([("x", "<f8"), ("y", "<f8"), ("p", "<i8")]))["p"]
                out[iid] = ColmapImage(iid, name.decode("utf-8"), cid, q, t, int(n2), int((ids >= 0).sum()))
    else:
        # two lines per image: the header, then its 2-D points (possibly EMPTY — never skipped)
        lines = [ln for ln in path.read_text().splitlines() if not ln.startswith("#")]
        i = 0
        while i < len(lines):
            if not lines[i].strip():
                i += 1
                continue
            parts = lines[i].split()
            iid = int(parts[0]); q = np.array([float(x) for x in parts[1:5]]); t = np.array([float(x) for x in parts[5:8]])
            cid = int(parts[8]); name = " ".join(parts[9:])
            pts = lines[i + 1].split() if i + 1 < len(lines) else []
            n2 = len(pts) // 3
            n3 = sum(1 for j in range(n2) if int(float(pts[3 * j + 2])) >= 0)
            out[iid] = ColmapImage(iid, name, cid, q, t, n2, n3)
            i += 2
    return out


def count_points3d(path: Path) -> int:
    path = Path(path)
    if path.suffix == ".bin":
        with open(path, "rb") as f:
            (n,) = _read(f, "Q")
        return int(n)
    return sum(1 for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#"))


def read_points3d_xyz(path: Path) -> np.ndarray:
    """(N,3) xyz of the sparse points (binary or text)."""
    path = Path(path)
    if path.suffix == ".bin":
        xyz = []
        with open(path, "rb") as f:
            (n,) = _read(f, "Q")
            for _ in range(n):
                _pid, = _read(f, "Q")
                xyz.append(_read(f, "ddd"))
                f.read(3)                       # rgb
                _read(f, "d")                   # error
                (tl,) = _read(f, "Q")
                f.read(tl * 8)
        return np.asarray(xyz, np.float64).reshape(-1, 3)
    rows = [ln.split() for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    return np.array([[float(r[1]), float(r[2]), float(r[3])] for r in rows], np.float64).reshape(-1, 3)


def _find(model_dir: Path, stem: str) -> Path:
    for suffix in (".bin", ".txt"):
        p = model_dir / f"{stem}{suffix}"
        if p.exists():
            return p
    raise ColmapIOError(f"{model_dir}: no {stem}.bin / {stem}.txt")


def read_model(model_dir: Path) -> ColmapModel:
    model_dir = Path(model_dir)
    cams = read_cameras(_find(model_dir, "cameras"))
    imgs = read_images(_find(model_dir, "images"))
    n3 = count_points3d(_find(model_dir, "points3D"))
    return ColmapModel(model_dir, cams, imgs, n3)


def list_models(sparse_dir: Path) -> List[Path]:
    """COLMAP writes one sub-directory per model (0, 1, …); a directory holding the files
    itself is one model."""
    sparse_dir = Path(sparse_dir)
    if any((sparse_dir / f"cameras{s}").exists() for s in (".bin", ".txt")):
        return [sparse_dir]
    out = [d for d in sorted(sparse_dir.iterdir(), key=lambda d: d.name) if d.is_dir()
           and any((d / f"cameras{s}").exists() for s in (".bin", ".txt"))]
    return out


# ── writers (tests, imports) ──────────────────────────────────────────────

def write_model_text(model_dir: Path, cameras: Dict[int, ColmapCamera], images: Dict[int, ColmapImage],
                     points: np.ndarray) -> Path:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "cameras.txt", "w") as f:
        f.write("# Camera list\n")
        for c in cameras.values():
            f.write(f"{c.camera_id} {c.model} {c.width} {c.height} " + " ".join(repr(float(p)) for p in c.params) + "\n")
    with open(model_dir / "images.txt", "w") as f:
        f.write("# Image list\n")
        for im in images.values():
            f.write(f"{im.image_id} " + " ".join(repr(float(x)) for x in im.qvec) + " "
                    + " ".join(repr(float(x)) for x in im.tvec) + f" {im.camera_id} {im.name}\n")
            f.write("\n")
    with open(model_dir / "points3D.txt", "w") as f:
        f.write("# 3D point list\n")
        for i, p in enumerate(np.atleast_2d(points)):
            f.write(f"{i + 1} {p[0]!r} {p[1]!r} {p[2]!r} 128 128 128 0.5\n")
    return model_dir


def write_model_binary(model_dir: Path, cameras: Dict[int, ColmapCamera], images: Dict[int, ColmapImage],
                       points: np.ndarray) -> Path:
    """The binary form (cameras.bin / images.bin / points3D.bin), as COLMAP writes it."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "cameras.bin", "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for c in cameras.values():
            mid = MODEL_ID_OF[c.model]
            f.write(struct.pack("<iiQQ", c.camera_id, mid, c.width, c.height))
            f.write(struct.pack("<" + "d" * len(c.params), *[float(p) for p in c.params]))
    with open(model_dir / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for im in images.values():
            f.write(struct.pack("<i", im.image_id))
            f.write(struct.pack("<dddd", *[float(x) for x in im.qvec]))
            f.write(struct.pack("<ddd", *[float(x) for x in im.tvec]))
            f.write(struct.pack("<i", im.camera_id))
            f.write(im.name.encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", 0))
    pts = np.atleast_2d(np.asarray(points, np.float64))
    with open(model_dir / "points3D.bin", "wb") as f:
        f.write(struct.pack("<Q", len(pts)))
        for i, p in enumerate(pts):
            f.write(struct.pack("<Q", i + 1))
            f.write(struct.pack("<ddd", *[float(x) for x in p]))
            f.write(bytes([128, 128, 128]))
            f.write(struct.pack("<d", 0.5))
            f.write(struct.pack("<Q", 0))
    return model_dir


# ── product ───────────────────────────────────────────────────────────────

def frame_of_name(name: str) -> Optional[int]:
    """'000051.jpg' → 51 (the chain's frame file rule); None when the name is not a number."""
    stem = Path(name).stem
    try:
        return int(stem)
    except ValueError:
        return None


def model_product(model: ColmapModel, dense_xyz: np.ndarray, name: str, meta: Optional[dict] = None
                  ) -> DenseProduct:
    """A registered model + its dense cloud as a product (frames keyed by the image names)."""
    frames, c2w, cam_of = [], [], {}
    for im in sorted(model.images.values(), key=lambda i: i.name):
        f = frame_of_name(im.name)
        if f is None:
            continue
        frames.append(f); c2w.append(im.c2w()); cam_of[f] = model.cameras[im.camera_id]
    if not frames:
        raise ColmapIOError(f"{model.path}: no image name is a frame number")
    K_of = lambda f: cam_of[f].K()                                   # noqa: E731
    undistort_of = lambda f, uv: cam_of[f].undistort_pixels(uv)      # noqa: E731
    prod = DenseProduct(frames=frames, c2w=np.asarray(c2w), xyz=np.asarray(dense_xyz, np.float64),
                        K_of=K_of, undistort_of=undistort_of, name=name, kind="dense",
                        meta=dict(meta or {}))
    prod.cameras = cam_of
    return prod


def load_dense_xyz(ply_path: Path) -> np.ndarray:
    return xyz_of(read_ply_memmap(Path(ply_path)))
