"""Synthetic session for the evaluation battery's tests: a known camera, known poses, a
fronto-parallel plane sampled at every pixel with exact provenance, references anchored by
projecting known world points. Every number can be checked by hand."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from evaluation import references as R
from evaluation.cloud_io import write_ply

W, H = 320, 240
FX, FY, CX, CY = 300.0, 300.0, 160.0, 120.0
FRAMES = [10, 20, 30, 40]
Z0 = 4.0


@dataclass
class SynthScene:
    session_dir: Path
    K: np.ndarray
    c2w: np.ndarray             # (N,4,4) TRUE poses (unscaled)
    frames: List[int]
    scale: float                # product units per metre (the bias injected)
    lens: bool
    cam: object


def camera_model(lens: bool):
    from precision.camera import CameraModel, GridMap
    g = GridMap("omega", W, H, W, H, 0, 0, 0, 0, W, H, W, H)
    k = (0.05, -0.01, 0.0005, -0.0003) if lens else (0.0, 0.0, 0.0, 0.0)
    return CameraModel(W, H, (FX, FY, CX, CY) + k, "synthetic", 1, g, mask_grid=None)


def true_poses(n: int) -> np.ndarray:
    c2w = np.tile(np.eye(4), (n, 1, 1))
    for i in range(n):
        c2w[i, 0, 3] = 0.5 * i            # the camera walks along +x, looking down +z
        c2w[i, 1, 3] = 0.05 * (i % 2)
    return c2w


def plane_point(uv: np.ndarray, K: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """World point of an undistorted pixel on the plane z = Z0 (world)."""
    from evaluation import geometry as G
    o, d = G.world_ray(uv, K, c2w)
    t = (Z0 - o[2]) / d[:, 2]
    return o + d * t[:, None]


def make_session(tmp: Path, *, lens: bool = False, scale: float = 1.0, frames: Optional[List[int]] = None,
                 v_max: int = H, stride: int = 1, seed: int = 0, witness_fields: bool = False,
                 noise_m: float = 0.0) -> SynthScene:
    """A session dir with camera.json, camera_frames.txt, camera_poses.txt and a cloud whose
    every point is the plane z = Z0 seen through pixel (row, col) of its keyframe — rows up to
    ``v_max`` only, so anchors beyond it are not reconstructible. ``scale`` multiplies every
    product coordinate (a scale bias the gauge must undo)."""
    from precision.camera import save_camera_json
    from repro import write_poses_exact
    frames = list(frames or FRAMES)
    cam = camera_model(lens)
    K = cam.K()
    c2w = true_poses(len(frames))
    rng = np.random.default_rng(seed)
    cols: Dict[str, list] = {k: [] for k in ("x", "y", "z", "red", "green", "blue", "confidence",
                                            "frame_global", "pixel_row", "pixel_col")}
    extra = {k: [] for k in ("mv_votes", "mask_votes", "mask_conflicts", "status")} if witness_fields else {}
    for i, f in enumerate(frames):
        vv, uu = np.mgrid[0:v_max:stride, 0:W:stride]
        uv = np.stack([uu.ravel(), vv.ravel()], axis=1).astype(np.float64)
        P = plane_point(uv, K, c2w[i]) * scale
        n = len(uv)
        if noise_m > 0:
            P = P.copy()
            P[:, 2] += rng.normal(0.0, noise_m, n) * scale      # Gaussian noise along the plane normal
        cols["x"].append(P[:, 0]); cols["y"].append(P[:, 1]); cols["z"].append(P[:, 2])
        cols["red"].append(np.full(n, 100, np.uint8)); cols["green"].append(np.full(n, 100, np.uint8))
        cols["blue"].append(np.full(n, 100, np.uint8))
        cols["confidence"].append(rng.random(n).astype(np.float32))
        cols["frame_global"].append(np.full(n, f, np.int32))
        cols["pixel_row"].append(uv[:, 1].astype(np.int16)); cols["pixel_col"].append(uv[:, 0].astype(np.int16))
        if witness_fields:
            extra["mv_votes"].append(rng.integers(0, 6, n).astype(np.uint8))
            extra["mask_votes"].append(rng.integers(0, 4, n).astype(np.uint8))
            extra["mask_conflicts"].append(np.zeros(n, np.uint8))
            extra["status"].append(rng.integers(1, 3, n).astype(np.uint8))
    dtype = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1"),
             ("confidence", "<f4"), ("frame_global", "<i4"), ("pixel_row", "<i2"), ("pixel_col", "<i2")]
    dtype += [(k, "u1") for k in extra]
    n_all = sum(len(a) for a in cols["x"])
    data = np.zeros(n_all, dtype=dtype)
    for k, parts in list(cols.items()) + list(extra.items()):
        data[k] = np.concatenate(parts)
    out = tmp / "output"
    out.mkdir(parents=True, exist_ok=True)
    save_camera_json(out / "camera.json", cam, 0)
    (out / "camera_frames.txt").write_text("\n".join(str(f) for f in frames) + "\n")
    scaled = c2w.copy()
    scaled[:, :3, 3] *= scale
    write_poses_exact(out / "camera_poses.txt", scaled)
    write_ply(out / "cleaned_cloud.ply", data)
    (tmp / "frames").mkdir(exist_ok=True)
    return SynthScene(tmp, K, c2w, frames, scale, lens, cam)


def anchors_of(scene: SynthScene, X: np.ndarray, frames: Optional[List[int]] = None,
               round_px: bool = False) -> List[dict]:
    """Observations of the TRUE world point X in every keyframe that sees it (pixels as the user
    would click them: the DISTORTED native pixel when the camera has a lens). Float pixels by
    default: the plane is fronto-parallel, so any product point pooled within the lookup window
    has the exact depth and the anchor's own ray gives the exact point."""
    from evaluation import geometry as G
    from precision.camera import distort_points
    obs = []
    for i, f in enumerate(scene.frames):
        if frames is not None and f not in frames:
            continue
        uv, z = G.project(X[None], scene.K, scene.c2w[i])
        u, v = uv[0]
        if round_px:
            u, v = round(u), round(v)
        if not (0 <= u < W and 0 <= v < H) or z[0] <= 0:
            continue
        if scene.lens:
            u, v = distort_points(np.array([[float(u), float(v)]]), scene.cam)[0]
        obs.append(R.observation(f, float(u), float(v), f"{f:06d}.jpg"))
    return obs


def point_on_plane_at_pixel(scene: SynthScene, kf: int, u: int, v: int) -> np.ndarray:
    """The TRUE world point the cloud holds at (u, v) of keyframe index ``kf``."""
    return plane_point(np.array([[float(u), float(v)]]), scene.K, scene.c2w[kf])[0]


def references_doc(scene: SynthScene, pairs: List[Tuple[np.ndarray, np.ndarray, str, str]],
                   planes: Optional[List[dict]] = None, frames_of: Optional[Dict[str, List[int]]] = None
                   ) -> dict:
    """A reference document: ``pairs`` = (P_a, P_b, role, id) TRUE world points; the value is
    their true distance; sigma 0.005 m."""
    doc = R.empty_document()
    doc["image_size"] = [W, H]
    for Pa, Pb, role, iid in pairs:
        fr = (frames_of or {}).get(iid)
        doc["distances"].append(R.new_distance(iid, float(np.linalg.norm(Pa - Pb)), 0.005, role, "tape",
                                               anchors_of(scene, Pa, fr), anchors_of(scene, Pb, fr)))
    doc["planes"] = list(planes or [])
    return doc


def eval_config(**over):
    """The server's evaluation config with dotted overrides (evaluation.<a>.<b> = value)."""
    import copy
    import yaml
    from evaluation.config import load_evaluation_config
    raw = yaml.safe_load((Path(__file__).resolve().parent.parent / "config.yaml").read_text())
    raw = {"evaluation": copy.deepcopy(raw["evaluation"])}
    for k, v in over.items():
        node = raw["evaluation"]
        parts = k.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = v
    return load_evaluation_config(raw)
