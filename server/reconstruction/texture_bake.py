"""
Texture baking — bake a UV-atlas texture onto a reconstructed mesh.
================================================================================

A TSDF mesh carries at best voxel-resolution per-vertex colour (the "preview"
tier). This module bakes a crisp **UV-atlas texture** from the full-resolution
RGB keyframes with **texrecon** (MVS-Texturing, Waechter et al., ECCV 2014):
per-face view selection via an MRF, photo-consistency outlier removal, global
colour adjustment and Poisson seam levelling. The visual detail then lives in
the texture (1920×1440 RGB), not the mesh — which is exactly how Polycam keeps
a coarse LiDAR mesh looking sharp.

This is Phase 1B of the Polycam-quality plan
(``plans/polycam-quality-reconstruction.md``). CPU-only.

texrecon is built from ``vendor/mvs-texturing`` (OpenDroneMap fork). Override
the binary location with ``$STAC_TEXRECON_BIN``.

Authors: Hernán Barreto — Ingerop IN3
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger("TextureBake")
logger.propagate = False  # avoid double-logging via root


# ── texrecon binary ────────────────────────────────────────────────

def _texrecon_bin() -> Optional[Path]:
    """Locate the texrecon executable ($STAC_TEXRECON_BIN or the vendor build)."""
    env = os.environ.get("STAC_TEXRECON_BIN")
    if env and Path(env).is_file():
        return Path(env)
    repo_root = Path(__file__).resolve().parents[2]
    cand = (repo_root / "vendor" / "mvs-texturing" / "build"
            / "apps" / "texrecon" / "texrecon")
    if cand.is_file():
        return cand
    found = shutil.which("texrecon")
    return Path(found) if found else None


def _texrecon_env() -> dict:
    """Subprocess env with the vendored oneTBB on the library path.

    texrecon links the locally-built oneTBB; the build-tree rpath usually
    resolves it, but exporting LD_LIBRARY_PATH keeps the invocation robust to
    the binary being moved or copied.
    """
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[2]
    tbb_lib = repo_root / "vendor" / "oneTBB" / "lib"
    if tbb_lib.is_dir():
        prev = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{tbb_lib}:{prev}" if prev else str(tbb_lib)
    return env


# ── texrecon .cam writer ───────────────────────────────────────────

def _write_cam_file(cam_path: Path, c2w: np.ndarray, K: np.ndarray,
                    img_w: int, img_h: int) -> None:
    """Write an MVE/texrecon ``.cam`` file for one view.

    Format — two lines:
        tx ty tz  R00 R01 R02 R10 R11 R12 R20 R21 R22   (world→camera)
        flen  d0 d1  paspect  ppx ppy                    (normalised intrinsics)

    MVE normalises the focal length by the larger image dimension and the
    principal point by width/height. Deriving from a pixel K (fx,fy,cx,cy):
        flen    = fx / w   (landscape)  |  fy / h   (portrait)
        paspect = fy / fx
        ppx     = cx / w               ppy = cy / h
    """
    c2w_4 = np.eye(4, dtype=np.float64)
    c2w_4[:c2w.shape[0], :c2w.shape[1]] = c2w
    w2c = np.linalg.inv(c2w_4)              # texrecon wants world→camera
    R = w2c[:3, :3]
    t = w2c[:3, 3]

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    flen = (fx / img_w) if img_w >= img_h else (fy / img_h)
    paspect = fy / fx
    ppx, ppy = cx / img_w, cy / img_h

    with open(cam_path, "w") as f:
        f.write(f"{t[0]:.10f} {t[1]:.10f} {t[2]:.10f} "
                f"{R[0,0]:.10f} {R[0,1]:.10f} {R[0,2]:.10f} "
                f"{R[1,0]:.10f} {R[1,1]:.10f} {R[1,2]:.10f} "
                f"{R[2,0]:.10f} {R[2,1]:.10f} {R[2,2]:.10f}\n")
        f.write(f"{flen:.10f} 0 0 {paspect:.10f} {ppx:.10f} {ppy:.10f}\n")


# ── Public API ─────────────────────────────────────────────────────

def bake_texture(
    mesh_path: Path,
    frames_dir: Path,
    pose_map: Dict[int, np.ndarray],        # frame_idx → (4,4) c2w
    intrinsics_map: Dict[int, np.ndarray],  # frame_idx → (3,3) K at RGB res
    out_glb: Path,
    frame_indices: Optional[List[int]] = None,
    max_views: int = 400,
    data_term: str = "gmi",
    keep_work: bool = False,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Optional[Path]:
    """Bake a UV-atlas texture onto ``mesh_path`` and write it to ``out_glb``.

    Args:
        mesh_path: input mesh (GLB/PLY/OBJ) — geometry only; vertex colours,
            if any, are ignored.
        frames_dir: directory of RGB keyframes named ``{idx:06d}.jpg``.
        pose_map / intrinsics_map: per-frame camera-to-world poses and pixel
            intrinsics at RGB resolution (e.g. ``CameraSource.pose_map`` /
            ``.intrinsics_map``).
        out_glb: destination GLB (may equal ``mesh_path`` — it is fully read
            before being overwritten).
        frame_indices: which frames to texture from; default = all posed
            frames that have an intrinsic and a JPG on disk.
        max_views: cap on views fed to texrecon (evenly subsampled). More
            views ⇒ better coverage but slower MRF.
        data_term: texrecon data term — ``gmi`` (gradient magnitude, sharper)
            or ``area``.

    Returns:
        ``out_glb`` on success, or ``None`` on failure (the caller should keep
        the untextured mesh as a fallback).
    """
    t0 = time.time()
    mesh_path = Path(mesh_path)
    frames_dir = Path(frames_dir)
    out_glb = Path(out_glb)

    texrecon = _texrecon_bin()
    if texrecon is None:
        logger.error("[TextureBake] texrecon not found — build vendor/mvs-texturing "
                     "or set $STAC_TEXRECON_BIN")
        return None

    try:
        import trimesh
    except ImportError:
        logger.error("[TextureBake] trimesh not available — cannot convert mesh")
        return None

    # ── frame selection: posed + has intrinsic + JPG on disk ──
    if frame_indices is None:
        frame_indices = sorted(pose_map.keys())
    usable: List[int] = []
    for fi in frame_indices:
        fi = int(fi)
        if pose_map.get(fi) is None or intrinsics_map.get(fi) is None:
            continue
        if not (frames_dir / f"{fi:06d}.jpg").exists():
            continue
        usable.append(fi)
    if len(usable) < 2:
        logger.error(f"[TextureBake] too few usable views ({len(usable)})")
        return None
    if len(usable) > max_views:
        step = len(usable) / float(max_views)
        usable = [usable[int(i * step)] for i in range(max_views)]
    logger.info(f"[TextureBake] {len(usable)} views  mesh={mesh_path.name}  "
                f"texrecon={texrecon}")

    work = Path(tempfile.mkdtemp(prefix="texrecon_"))
    try:
        # ── input mesh for texrecon ──
        # texrecon reads PLY/OFF/OBJ natively — feed those straight through.
        # Anything else (e.g. GLB) is converted via trimesh. Passing a PLY
        # avoids a fragile GLB round-trip on large meshes.
        if progress_cb:
            progress_cb("preparing")
        if mesh_path.suffix.lower() in (".ply", ".off", ".obj"):
            ply_path = mesh_path
        else:
            tm = trimesh.load(str(mesh_path), force="mesh", process=False)
            ply_path = work / "mesh.ply"
            tm.export(str(ply_path))

        # ── views: image (symlink) + .cam per frame ──
        views = work / "views"
        views.mkdir()
        img_w = img_h = 0
        for fi in usable:
            jpg = (frames_dir / f"{fi:06d}.jpg").resolve()
            if img_w == 0:
                with Image.open(jpg) as im:
                    img_w, img_h = im.size
            link = views / f"{fi:06d}.jpg"
            try:
                os.symlink(jpg, link)
            except OSError:
                shutil.copyfile(jpg, link)
            _write_cam_file(
                views / f"{fi:06d}.cam",
                np.asarray(pose_map[fi], dtype=np.float64),
                np.asarray(intrinsics_map[fi], dtype=np.float64),
                img_w, img_h,
            )

        # ── run texrecon ──
        if progress_cb:
            progress_cb("texturing")
        out_prefix = work / "textured"
        cmd = [
            str(texrecon), str(views), str(ply_path), str(out_prefix),
            f"--data_term={data_term}",
            "--outlier_removal=gauss_damping",
            "--keep_unseen_faces",
            "--no_intermediate_results",
        ]
        logger.info(f"[TextureBake] running: {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True,
                               env=_texrecon_env())
        for line in (proc.stdout or "").splitlines():
            if line.strip():
                logger.info(f"[texrecon] {line}")
        if proc.returncode != 0:
            logger.error(f"[TextureBake] texrecon failed (code {proc.returncode})")
            for line in (proc.stderr or "").splitlines()[-25:]:
                logger.error(f"[texrecon] {line}")
            return None

        obj_path = out_prefix.with_suffix(".obj")
        if not obj_path.exists():
            logger.error(f"[TextureBake] texrecon produced no OBJ at {obj_path}")
            return None

        # ── textured OBJ (+ .mtl + texture PNG) → GLB ──
        if progress_cb:
            progress_cb("converting")
        textured = trimesh.load(str(obj_path), process=False)  # Scene or Trimesh
        # Sanitise before GLB export. trimesh tolerates NaN/Inf, but the glTF
        # JSON chunk it then writes carries `NaN` in accessor min/max — invalid
        # JSON, which three.js's JSON.parse rejects (mesh fails to load in the
        # browser). Drop degenerate faces (their normals are NaN) and zero any
        # remaining non-finite vertex/normal/uv values.
        _geoms = (list(textured.geometry.values())
                  if hasattr(textured, "geometry") else [textured])
        for _g in _geoms:
            try:
                _g.update_faces(_g.nondegenerate_faces())
                _g.remove_unreferenced_vertices()
                _g.vertices = np.nan_to_num(_g.vertices, nan=0.0,
                                            posinf=0.0, neginf=0.0)
                _g.vertex_normals = np.nan_to_num(_g.vertex_normals, nan=0.0,
                                                  posinf=0.0, neginf=0.0)
                _uv = getattr(_g.visual, "uv", None)
                if _uv is not None:
                    _g.visual.uv = np.nan_to_num(_uv, nan=0.0,
                                                 posinf=0.0, neginf=0.0)
            except Exception as e:
                logger.warning(f"[TextureBake] sanitize: {e}")
        out_glb.parent.mkdir(parents=True, exist_ok=True)
        textured.export(str(out_glb))

        logger.info(f"[TextureBake] ✅ {out_glb.name}  ({time.time() - t0:.1f}s)")
        if progress_cb:
            progress_cb("done")
        return out_glb

    except Exception as e:
        logger.error(f"[TextureBake] error: {e}", exc_info=True)
        return None
    finally:
        if keep_work:
            logger.info(f"[TextureBake] work dir kept: {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)
