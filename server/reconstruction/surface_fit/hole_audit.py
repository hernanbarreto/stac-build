"""
Stage-1 evidence-based hole audit (user concept 2026-08-29).

A hole in a fitted surface's point support is one of two very different
things: a REAL opening (door/window — the scan frames show it) or a
reconstruction gap (occlusion, low confidence, textureless surface — the
frames show continuous surface there). The cloud alone cannot tell them
apart; the SCAN FRAMES can. This module makes the surface see them:

  every unsupported cell of the surface's UV grid is projected into the
  SAM3 mask keyframes (poses + intrinsics from the session) and voted:
    - covered by the instance's own mask in most views → the surface IS
      there in the images → the cell is FILLED (provenance image_supported);
    - uncovered → the opening is real → the cell stays OPEN, and the
      opening border follows the mask evidence at cell resolution;
    - too few views / mixed evidence → AMBIGUOUS → stays open (we never
      invent; conservative default).

Nothing is generated from priors: every filled triangle is backed by
measured mask pixels from the frames that built the cloud, and the audit
report says exactly which cells were filled on image evidence.

Frame convention: RAW cloud frame (the fit frame) — poses and the cleaned
cloud share it; the viewer's floor transform is applied downstream.

Hernán Barreto - Ingerop IN3 Session IV - STAC
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger("SurfaceFit")

# ── the device (USER 2026-10-15: "toda esa etapa debe correr en gpu, no puede demorar tanto y
# correrlo en cpu") ─────────────────────────────────────────────────────────────────
# Every projection of the evidence — the per-frame z-buffers of the FULL cloud, the mask votes,
# the see-through tallies — runs on torch tensors on ONE device: the card the config declares
# (`surface_fit.evidence_device: cuda`, the meshing workers) or the CPU (the synthetic tests,
# tests/conftest.py → :func:`use_device`). The same code path on both, never a silent fallback:
# a declared cuda with no card FAILS. MEASURED on pccr 2408 (19.35 M points, 295 keyframes,
# F5's lens): one numpy z-buffer took 3.3 s (cv2.projectPoints on the whole cloud 2.7 s of it)
# and the floor's audit 22.5 min (07:01 → 07:23, run.log 2026-10-15) — every keyframe's z-buffer
# plus the calibration votes of 724 mask candidates. Deterministic by construction on the card:
# the z-buffer is a stable SORT by (depth, pixel) and the first entry of each pixel (no scatter,
# no atomics — precision.f6_torch.splat's recipe), votes are gathers and sums of 0/1, the 5-px
# minimum filter is a max-pool (precision.f6_torch.min_filter). The workers that mesh objects
# (run_surface_fit / run_poisson_objects / run_perfect_objects) run under torch's deterministic
# mode STRICT (repro.enable_deterministic_torch); the record of each evidence says whether its
# process was (``torch_record``).
_KEY_RE = re.compile(r"^f(\d+)_o(\d+)$")
_DEVICE: Optional[str] = None
_DEVICES = ("cuda", "cpu")


def use_device(dev: Optional[str]) -> Optional[str]:
    """The ONE device of every evidence built after this call ('cpu' in the synthetic tests);
    None = resolve `surface_fit.evidence_device` at the next evidence."""
    global _DEVICE
    _DEVICE = None if dev is None else str(dev)
    return _DEVICE


def evidence_device() -> str:
    """The device the evidence projects on: :func:`use_device` when set, else the declared
    `surface_fit.evidence_device` (cuda | cpu). A missing key, an unknown value or a cuda with
    no visible card FAILS naming the key — nothing runs somewhere else."""
    if _DEVICE is not None:
        dev = _DEVICE
    else:
        from config import cfg
        sf = (cfg.get("surface_fit") or {}) if hasattr(cfg, "get") else {}
        if "evidence_device" not in sf:
            raise RuntimeError("surface_fit.evidence_device is missing (cuda | cpu) — the hole-audit "
                               "evidence projects on one declared device, nothing is assumed")
        dev = str(sf["evidence_device"])
    if dev not in _DEVICES:
        raise RuntimeError(f"surface_fit.evidence_device = {dev!r} — one of {list(_DEVICES)}")
    if dev == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("surface_fit.evidence_device is cuda and no CUDA device is visible — the "
                           "evidence fails instead of running somewhere else")
    return dev


def torch_strict() -> bool:
    """torch's deterministic mode STRICT in this process (raise, never warn)."""
    return bool(torch.are_deterministic_algorithms_enabled()
                and not torch.is_deterministic_algorithms_warn_only_enabled())


def _to(a, dev: str, dtype=None) -> torch.Tensor:
    """numpy → a tensor on ``dev``; floats as float64 unless asked otherwise."""
    if isinstance(a, torch.Tensor):
        t = a.to(dev)
        return t if dtype is None else t.to(dtype)
    a = np.ascontiguousarray(a)
    if dtype is None:
        if a.dtype == np.bool_:
            dtype = torch.bool
        elif np.issubdtype(a.dtype, np.floating):
            dtype = torch.float64
        elif np.issubdtype(a.dtype, np.integer):
            dtype = torch.int64
    return torch.as_tensor(a, device=dev).to(dtype)


def _np(t: torch.Tensor, dtype=None) -> np.ndarray:
    a = t.detach().cpu().numpy()
    return a if dtype is None else a.astype(dtype)


def zbuffer_min(pix: torch.Tensor, z: torch.Tensor, mh: int, mw: int) -> torch.Tensor:
    """Per-pixel nearest depth of the (pixel, depth) pairs — +inf where nothing lands —
    as a stable SORT by depth, then by pixel, with one +inf sentinel per pixel, and the
    first entry of each pixel: ``np.minimum.at`` without a scatter (deterministic on every
    device; precision.f6_torch.splat's recipe)."""
    n_pix = int(mh) * int(mw)
    dev = z.device
    p_all = torch.cat([torch.arange(n_pix, dtype=torch.int64, device=dev), pix.to(torch.int64)])
    z_all = torch.cat([torch.full((n_pix,), float("inf"), dtype=torch.float64, device=dev),
                       z.to(torch.float64)])
    o1 = torch.sort(z_all, stable=True).indices
    p1 = p_all[o1]
    z1 = z_all[o1]
    o2 = torch.sort(p1, stable=True).indices
    p2 = p1[o2]
    z2 = z1[o2]
    first = torch.ones_like(p2, dtype=torch.bool)
    first[1:] = p2[1:] != p2[:-1]
    return z2[first.nonzero()[:, 0]].reshape(int(mh), int(mw))

# the files the evidence is built from — its identity (docs/plan_determinismo.md point 115:
# the cache below is keyed by their sha256, never by the directory alone, so an epoch that
# replaced the cloud or the poses is never audited with another epoch's z-buffers)
EVIDENCE_INPUTS = ("cleaned_cloud.ply", "corrected_cloud.json", "camera.json", "camera_poses.txt",
                   "camera_frames.txt", "seg_masks.npz")

# sha256 memo per file version: (path, inode, size, mtime_ns) → digest. The memo only spares
# re-hashing a file that is demonstrably the same version; the IDENTITY compared is the digest.
_SHA_MEMO: Dict[tuple, str] = {}


def _file_sha(p: Path) -> str:
    from repro import sha256_file
    st = p.stat()
    k = (str(p), st.st_ino, st.st_size, st.st_mtime_ns)
    h = _SHA_MEMO.get(k)
    if h is None:
        h = sha256_file(p)
        _SHA_MEMO[k] = h
    return h


def evidence_identity(output_dir: Path) -> str:
    """sha256 over (name, sha256) of every :data:`EVIDENCE_INPUTS` file present —
    what two evidence builds must share to be the same evidence."""
    from repro import sha256_json
    out = Path(output_dir)
    rows = [[name, _file_sha(out / name)] for name in EVIDENCE_INPUTS if (out / name).is_file()]
    return sha256_json(rows)


def _k_grid(output_dir: Path):
    """(W, H, cloud_xyz): the DECLARED grid the cloud's ``pixel_row`` /
    ``pixel_col`` live on (``correction.visit_drift.trace_grid`` — camera.json /
    corrected_cloud.json; point 108: never the maxima of the surviving pixels,
    which moved with whichever point happened to survive in the last row), plus
    the cloud points for the per-frame Z-buffers. RAISES when the grid is not
    declared or the cloud carries no provenance."""
    from correction.visit_drift import trace_grid
    from segmentation.pipeline import _load_ply_origins
    Ht, Wt = trace_grid(Path(output_dir))
    origins = _load_ply_origins(Path(output_dir) / "cleaned_cloud.ply")
    if origins is None:
        raise RuntimeError(f"{Path(output_dir) / 'cleaned_cloud.ply'} carries no per-point "
                           f"provenance (frame_global / pixel_row / pixel_col)")
    xyz, _fg, pr, pc = origins
    if len(pr) and (int(pr.max()) >= Ht or int(pc.max()) >= Wt):
        raise RuntimeError(f"the cloud's birth pixels reach ({int(pr.max())}, {int(pc.max())}) "
                           f"outside the declared {Wt}x{Ht} record grid — the cloud and the "
                           f"camera do not describe one grid")
    return (int(Wt), int(Ht), np.asarray(xyz))  # (W, H, pts)


class _Evidence:
    """Masks + cameras for one session, shared across instances.

    The camera is THE ONE THE CLOUD WAS BUILT WITH (``session_io.
    load_cloud_camera_source``: camera.json on the record grid + the live
    epoch's poses — point 113, never a Stray sibling's ARKit odometry). When
    the evidence cannot be built, ``ok`` is False and ``reason`` says why — the
    projection's mask audit FAILS on it (point 122); the on-demand hole-audit
    tools skip, declared."""

    def __init__(self, output_dir: Path, session_dir: Path):
        self.ok = False
        self.reason: Optional[str] = None
        self._others_cache: Dict[Tuple[int, int], Optional[np.ndarray]] = {}
        p = Path(output_dir) / "seg_masks.npz"
        if not p.exists():
            self.reason = f"{p} does not exist"
            return
        try:
            self.masks = np.load(p, allow_pickle=True)
            from segmentation.session_io import load_cloud_camera_source
            self.cam = load_cloud_camera_source(Path(output_dir))
            self.kw, self.kh, self._cloud = _k_grid(Path(output_dir))
            # THE TWO GRIDS (docs/plan_determinismo.md point 108, DECIDIDO): the camera
            # projects onto the RECORD grid (camera.json's omega grid — Omega's crop of the
            # native frame, resized — at the declared trace size) and the masks live on
            # their DECLARED grid (the store's `scaled_res`: a full-frame resize of the
            # native frame, precision.camera.mask_grid_for). `mask_pixels` takes one to the
            # other through the exact maps, as the matcher and the certification do; a
            # store that declares no grid FAILS here — nothing is deduced from the data.
            from precision.camera import (CAMERA_JSON_NAME, grid_like, load_camera_json,
                                          mask_grid_for)
            cm = load_camera_json(Path(output_dir) / CAMERA_JSON_NAME)
            g = cm.omega_grid
            self.record_grid = (g if (int(g.w), int(g.h)) == (self.kw, self.kh)
                                else grid_like(g, self.kw, self.kh, "record"))
            if "scaled_res" not in self.masks.files:
                raise RuntimeError(f"{p} declares no mask grid (no `scaled_res` member) — the "
                                   f"masks cannot be addressed without their declared grid")
            sr = np.asarray(self.masks["scaled_res"]).ravel()
            if sr.size < 2 or int(sr[0]) <= 0 or int(sr[1]) <= 0:
                raise RuntimeError(f"{p} declares an invalid mask grid scaled_res={sr.tolist()}")
            self.mask_hw: Tuple[int, int] = (int(sr[0]), int(sr[1]))
            self.mask_grid = mask_grid_for(cm.width, cm.height, self.mask_hw)
            # the z-buffers live on the device (``_zbuf_t``); :meth:`_zbuf` hands numpy copies
            self._zbuf_cache: Dict[int, Optional[torch.Tensor]] = {}
            self.device = evidence_device()
            self.torch_record = {"device": self.device, "deterministic_strict": torch_strict()}
            if self.device == "cuda" and not self.torch_record["deterministic_strict"]:
                logger.info("hole_audit: evidence on cuda in a process without torch's strict "
                            "deterministic mode (its own kernels are deterministic by construction)")
            self._cloud_t: Optional[torch.Tensor] = None   # uploaded at the first z-buffer — the
            #   backend's in-process readers (object_analysis) never need the cloud on the card
            self._frames_by_oid = self._index_frames()
            # The npz keys are NOT the frame the poses are keyed by. Measured
            # (or declared) once here; every lookup below goes through it.
            from segmentation import mask_space
            self.space = mask_space.resolve(Path(output_dir), masks=self.masks)
            if self.space.mixed:
                logger.warning("hole_audit: %s", self.space.describe())
            self.ok = True
        except Exception as e:  # noqa: BLE001 — declared in `reason`, never silent
            self.reason = f"{type(e).__name__}: {e}"
            logger.warning("hole_audit: evidence load failed: %s", self.reason)

    # ── frame spaces: a mask key is NOT a camera key ─────────────────
    #
    # ``frames_for`` / ``_others_mask`` / every ``f<N>_o<OID>`` key speaks the
    # MASK space (keyframe position on every store the pipeline writes); the
    # camera source, the JPEG filenames and the cloud's ``frame_global`` speak
    # the video frame number. These three are the only place the two meet.

    def cloud_frame(self, fidx: int) -> Optional[int]:
        """Mask key frame → the frame the poses and images are keyed by."""
        return self.space.to_cloud(int(fidx))

    def pose_at(self, fidx: int):
        """c2w for a MASK frame index (None when it has no keyframe/pose)."""
        cf = self.cloud_frame(fidx)
        return None if cf is None else self.cam.pose_map.get(cf)

    def K_at(self, fidx: int):
        """Intrinsics for a MASK frame index (K lives on the TRACE grid)."""
        cf = self.cloud_frame(fidx)
        return None if cf is None else self.cam.K_for(cf)

    def zbuf_of_mask_frame(self, fidx: int, mh: int, mw: int) -> Optional[np.ndarray]:
        """:meth:`_zbuf` addressed by a MASK frame index."""
        cf = self.cloud_frame(fidx)
        return None if cf is None else self._zbuf(cf, mh, mw)

    def mask_pixels(self, u: np.ndarray, v: np.ndarray, mh: int, mw: int
                    ) -> Tuple[np.ndarray, np.ndarray]:
        """Continuous pixel coordinates on the RECORD grid (what :meth:`CameraSource.pixels`
        returns, centres at integers) → the MASK-grid pixel (column, row) that shows them,
        through the exact grid maps (point 108): record → native (Omega's crop and scale) →
        mask (a full-frame resize) → the nearest pixel centre. A scale alone
        (``u * mw / kw``) ignored the crop and truncated. ``(mh, mw)`` must be the store's
        declared grid: one store, one grid. Out-of-frame inputs come back out of bounds —
        the caller's bounds test is unchanged."""
        if (int(mh), int(mw)) != self.mask_hw:
            raise RuntimeError(f"a mask of {int(mh)}x{int(mw)} px against a store that declares "
                               f"{self.mask_hw[0]}x{self.mask_hw[1]} — one store, one grid")
        from precision.camera import grid_to_native, native_to_grid
        uv = np.stack([np.asarray(u, np.float64), np.asarray(v, np.float64)], -1)
        g = native_to_grid(grid_to_native(uv, self.record_grid), self.mask_grid)
        with np.errstate(invalid="ignore"):
            mu = np.rint(g[..., 0])
            mv = np.rint(g[..., 1])
        # a non-finite input (a point behind the camera) is out of bounds, never a crash
        mu = np.where(np.isfinite(mu), mu, -1.0).astype(np.int64)
        mv = np.where(np.isfinite(mv), mv, -1.0).astype(np.int64)
        return mu, mv

    # ── the projection on tensors (CameraSource.pixels + mask_pixels, point 108) ──

    def _cloud_tensor(self) -> torch.Tensor:
        if self._cloud_t is None:
            self._cloud_t = _to(self._cloud, self.device, torch.float64)
        return self._cloud_t

    def _cam_t(self, c2w) -> Tuple[torch.Tensor, torch.Tensor]:
        """(R, t) of world → camera for a c2w pose, on the device."""
        c2w4 = np.eye(4)
        c2w4[:c2w.shape[0], :c2w.shape[1]] = c2w
        M = np.linalg.inv(c2w4)
        return _to(M[:3, :3], self.device), _to(M[:3, 3], self.device)

    def _lens(self) -> Optional[tuple]:
        cam = self.cam.lens
        if cam is None:
            return None
        k1, k2, p1, p2 = [float(x) for x in cam.dist()]
        sx, sy = self.cam.native_to_grid
        return (float(cam.fx), float(cam.fy), float(cam.cx), float(cam.cy), k1, k2, p1, p2,
                float(sx), float(sy))

    def _record_px_t(self, cloud_fidx: int, pcam: torch.Tensor, front: torch.Tensor
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``CameraSource.pixels`` on tensors: camera-frame points → continuous RECORD-grid
        pixels, the lens included (cv2.projectPoints' Brown–Conrady k1 k2 p1 p2 on the
        normalised point, rvec = tvec = 0). Rows outside ``front`` carry no meaning."""
        z = torch.where(front, pcam[:, 2], torch.ones_like(pcam[:, 2]))
        x = pcam[:, 0] / z
        y = pcam[:, 1] / z
        lens = self._lens()
        if lens is None:
            K = np.asarray(self.cam.K_for(cloud_fidx), np.float64)
            return float(K[0, 0]) * x + float(K[0, 2]), float(K[1, 1]) * y + float(K[1, 2])
        fx, fy, cx, cy, k1, k2, p1, p2, sx, sy = lens
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        return (fx * xd + cx) * sx, (fy * yd + cy) * sy

    def _mask_px_t(self, u: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """:meth:`mask_pixels` on tensors: record grid → native (Omega's crop and scale) → mask
        grid (a full-frame resize) → the nearest pixel centre (round half to even, np.rint's);
        a non-finite input lands at −1 (out of bounds)."""
        rg, mg = self.record_grid, self.mask_grid
        xn = (u - rg.pad_left + 0.5) * rg.scale_x - 0.5 + rg.crop_x
        yn = (v - rg.pad_top + 0.5) * rg.scale_y - 0.5 + rg.crop_y
        gx = (xn - mg.crop_x + 0.5) / mg.scale_x - 0.5 + mg.pad_left
        gy = (yn - mg.crop_y + 0.5) / mg.scale_y - 0.5 + mg.pad_top
        mu = torch.round(gx)
        mv = torch.round(gy)
        mu = torch.where(torch.isfinite(mu), mu, torch.full_like(mu, -1.0)).to(torch.int64)
        mv = torch.where(torch.isfinite(mv), mv, torch.full_like(mv, -1.0)).to(torch.int64)
        return mu, mv

    def _project_t(self, cloud_fidx: int, c2w, X: torch.Tensor):
        """World points on the device → (mask column, mask row, depth, in-bounds-and-in-front)
        in the camera of ``cloud_fidx`` (a CLOUD/POSE frame)."""
        mh, mw = self.mask_hw
        R, t = self._cam_t(c2w)
        pcam = X @ R.T + t
        z = pcam[:, 2]
        front = z > 0.05
        u, v = self._record_px_t(int(cloud_fidx), pcam, front)
        mu, mv = self._mask_px_t(u, v)
        inb = front & (mu >= 0) & (mu < mw) & (mv >= 0) & (mv < mh)
        return mu, mv, z, inb

    def _zbuf_t(self, cloud_fidx: int, mh: int, mw: int) -> Optional[torch.Tensor]:
        """Per-frame Z-buffer from the FULL cloud at mask resolution, on the device: what
        measured geometry is closest along each pixel ray. THE depth check the 2-D masks lack
        (without it, a phantom plane cell beyond the wall edge projected onto some other
        surface and voted 'occluded'). ``cloud_fidx`` is a CLOUD/POSE frame number — the
        camera's own space; :meth:`zbuf_of_mask_frame` takes a mask key's index."""
        fidx = int(cloud_fidx)
        if fidx in self._zbuf_cache:
            return self._zbuf_cache[fidx]
        if (int(mh), int(mw)) != self.mask_hw:
            raise RuntimeError(f"a z-buffer of {int(mh)}x{int(mw)} px against a store that declares "
                               f"{self.mask_hw[0]}x{self.mask_hw[1]} — one store, one grid")
        c2w = self.cam.pose_map.get(fidx)
        K = self.cam.K_for(fidx)
        if c2w is None or K is None:
            self._zbuf_cache[fidx] = None
            return None
        from precision.f6_torch import min_filter
        mu, mv, z, inb = self._project_t(fidx, c2w, self._cloud_tensor())
        zb = zbuffer_min((mv * mw + mu)[inb], z[inb], mh, mw)
        # window-min: the cloud is sparse at mask resolution, so single pixels
        # along a ray are often unmeasured (inf) while their NEIGHBOURS are —
        # wall3's arch kept filling because the background seen through it had
        # no point on the exact pixel (user 2026-08-29). A 5-px minimum filter
        # (scipy's minimum_filter(size=5, mode='nearest'), as a max-pool) gives
        # every pixel the closest measured depth nearby.
        zb = min_filter(zb, 5)
        self._zbuf_cache[fidx] = zb
        return zb

    def _zbuf(self, cloud_fidx: int, mh: int, mw: int) -> Optional[np.ndarray]:
        """:meth:`_zbuf_t` as a numpy map (the readers outside this module)."""
        zb = self._zbuf_t(cloud_fidx, mh, mw)
        return None if zb is None else _np(zb)

    def _index_frames(self) -> Dict[int, List[Tuple[int, str]]]:
        """``f<frame>_o<oid>`` keys of the store, by oid, in frame order — ONE pass over the
        members (the calibration tries hundreds of oids; a regex pass each was minutes)."""
        idx: Dict[int, List[Tuple[int, str]]] = {}
        for k in self.masks.files:
            m = _KEY_RE.match(k)
            if m:
                idx.setdefault(int(m.group(2)), []).append((int(m.group(1)), k))
        for v in idx.values():
            v.sort()
        return idx

    def frames_for(self, oid: int) -> List[Tuple[int, str]]:
        """(mask frame index, npz key) of every mask of ``oid``, in frame order."""
        return list(self._frames_by_oid.get(int(oid), []))

    def _others_mask(self, fidx: int, oid: int) -> Optional[np.ndarray]:
        """Union of every OTHER instance's mask in a frame — the occluders."""
        key = (fidx, oid)
        if key in self._others_cache:
            return self._others_cache[key]
        acc = None
        for k in self.masks.files:
            m = re.match(rf"^f{fidx}_o(\d+)$", k)
            if not m or int(m.group(1)) == oid:
                continue
            a = self.masks[k] > 0
            acc = a if acc is None else (acc | a)
        self._others_cache[key] = acc
        return acc

    def vote(self, world_pts: np.ndarray, oid: int, depth_tol: float = 0.15):
        """Per-point evidence tallies over the instance's mask keyframes,
        DEPTH-VERIFIED against the cloud Z-buffer:
        - occluded: measured geometry sits ≥depth_tol IN FRONT of the point
          along the ray — the frame cannot testify (real occlusion; also kills
          phantom votes where a plane cell beyond the wall edge merely
          projected onto some other surface);
        - covered: visible at its own depth AND the instance's OWN mask covers
          the pixel — a direct, depth-consistent witness;
        - neither (empty): visible, and the pixel is not this instance — the
          frame sees PAST the surface (or a coplanar in-fill object like a
          door leaf occupies it) → witness the surface is NOT there.
        Returns (covered, occluded, valid) as int32 arrays."""
        n = len(world_pts)
        dev = self.device
        covered = torch.zeros(n, dtype=torch.int32, device=dev)
        occluded = torch.zeros(n, dtype=torch.int32, device=dev)
        valid = torch.zeros(n, dtype=torch.int32, device=dev)
        if n == 0:
            return _np(covered), _np(occluded), _np(valid)
        X = _to(np.asarray(world_pts, dtype=np.float64), dev)
        mh, mw = self.mask_hw
        inf = torch.full((n,), float("inf"), dtype=torch.float64, device=dev)
        for fidx, key in self.frames_for(oid):
            c2w = self.pose_at(fidx)
            K = self.K_at(fidx)
            if c2w is None or K is None:
                continue
            m = np.asarray(self.masks[key])
            if m.shape != (mh, mw):
                raise RuntimeError(f"a mask of {m.shape[0]}x{m.shape[1]} px against a store that "
                                   f"declares {mh}x{mw} — one store, one grid")
            cf = self.cloud_frame(fidx)
            zb = self._zbuf_t(cf, mh, mw)
            mu, mv, z, inb = self._project_t(cf, c2w, X)
            lin = torch.where(inb, mv * mw + mu, torch.zeros_like(mu))
            valid += inb.to(torch.int32)
            if zb is not None:
                zpix = torch.where(inb, zb.reshape(-1)[lin], inf)
                occ = zpix < (z - depth_tol)
                # measured geometry BEHIND the surface through this cell: the
                # camera sees PAST it (wall3's arch: through the arch you see
                # the attached cone — same instance, so the 2-D mask covered
                # it and the arch got filled; depth says OPEN)
                behind = torch.isfinite(zpix) & (zpix > (z + depth_tol))
            else:
                occ = torch.zeros(n, dtype=torch.bool, device=dev)
                behind = torch.zeros(n, dtype=torch.bool, device=dev)
            occluded += occ.to(torch.int32)
            hit = inb & _to(m > 0, dev).reshape(-1)[lin]
            covered += (hit & ~occ & ~behind).to(torch.int32)
        return _np(covered), _np(occluded), _np(valid)

    def see_through(self, world_pts: np.ndarray, depth_tol: float = 0.15):
        """Per-point (valid, at_depth, behind) tallies over ALL posed keyframes
        — no mask involved, pure depth evidence. 'behind' counts frames whose
        cloud Z-buffer puts the nearest measured geometry ≥depth_tol BEHIND
        the point (the camera sees PAST it); 'at_depth' counts frames that
        measure geometry AT the point's depth (±depth_tol) — testimony the
        surface is real. Rays with no finite Z-buffer say nothing (not valid)."""
        mh, mw = self.mask_hw                    # the store's DECLARED grid, never a default
        n = len(world_pts)
        dev = self.device
        valid = torch.zeros(n, dtype=torch.int32, device=dev)
        at_depth = torch.zeros(n, dtype=torch.int32, device=dev)
        behind = torch.zeros(n, dtype=torch.int32, device=dev)
        if n == 0:
            return _np(valid), _np(at_depth), _np(behind)
        X = _to(np.asarray(world_pts, dtype=np.float64), dev)
        inf = torch.full((n,), float("inf"), dtype=torch.float64, device=dev)
        for fidx in sorted(self.cam.pose_map.keys()):
            c2w = self.cam.pose_map.get(fidx)
            K = self.cam.K_for(fidx)
            if c2w is None or K is None:
                continue
            zb = self._zbuf_t(fidx, mh, mw)
            if zb is None:
                continue
            mu, mv, z, inb = self._project_t(fidx, c2w, X)
            lin = torch.where(inb, mv * mw + mu, torch.zeros_like(mu))
            zpix = torch.where(inb, zb.reshape(-1)[lin], inf)
            measured = torch.isfinite(zpix)
            valid += measured.to(torch.int32)
            at_depth += (measured & (torch.abs(zpix - z) <= depth_tol)).to(torch.int32)
            behind += (measured & (zpix > z + depth_tol)).to(torch.int32)
        return _np(valid), _np(at_depth), _np(behind)

    def calibrate_oid(self, world_pts: np.ndarray, instance_id: int) -> Optional[int]:
        """Find the npz object id for this instance by projecting its OWN
        on-surface points: the right mask covers them (~100%). Convention is
        oid = instance_id - 1, but this self-check survives drift."""
        sample = world_pts[:: max(1, len(world_pts) // 2000)]
        candidates = [int(instance_id) - 1]
        try:
            candidates += [int(o) for o in self.masks["obj_ids"].tolist()
                           if int(o) != instance_id - 1]
        except Exception:  # noqa: BLE001
            pass
        for oid in candidates:
            if not self.frames_for(oid):
                continue
            cov, _occ, val = self.vote(sample, oid)
            seen = val > 0
            if seen.sum() >= 50 and (cov[seen] / val[seen]).mean() > 0.5:
                if oid != instance_id - 1:
                    logger.warning("hole_audit: instance %d matched mask o%d "
                                   "(non-standard mapping)", instance_id, oid)
                return oid
        return None


_EVIDENCE_CACHE: Dict[Tuple[str, str], _Evidence] = {}


def _evidence(output_dir: Path, session_dir: Path) -> _Evidence:
    """The session's evidence, cached by the IDENTITY of its inputs
    (:func:`evidence_identity`): the same directory with another cloud, other
    poses or another mask store is another evidence (point 115). One entry per
    directory — the previous version is dropped with its open npz handle."""
    out = Path(output_dir)
    key = (str(out), evidence_identity(out))
    ev = _EVIDENCE_CACHE.get(key)
    if ev is None:
        for old in [k for k in _EVIDENCE_CACHE if k[0] == key[0]]:
            stale = _EVIDENCE_CACHE.pop(old)
            try:
                stale.masks.close()
            except Exception:  # noqa: BLE001
                pass
        ev = _Evidence(out, session_dir)
        _EVIDENCE_CACHE[key] = ev
    return ev


def audit_and_fill(model, uv_support: np.ndarray,
                   output_dir: Path, session_dir: Path, instance_id: int,
                   resolution: float = 0.05, support_radius: float = 0.08,
                   fill_ratio: float = 0.6, open_ratio: float = 0.3,
                   occluded_ratio: float = 0.5,
                   min_votes: int = 2, snap_max_m: float = 0.2,
                   max_fill_dist_cells: Optional[int] = None,
                   interp_max_cells: int = 30):
    """Audit the holes of a fitted surface against the scan-frame masks and
    rebuild its mesh with image-supported fills.

    Returns (verts_world, faces, report) or None when the audit cannot run
    (no masks / no camera / no holes). ``report`` counts filled / open /
    ambiguous cells and the evidence used."""
    ev = _evidence(Path(output_dir), Path(session_dir))
    if not ev.ok or len(uv_support) < 100:
        return None

    uv = np.asarray(uv_support, dtype=np.float64)
    from .support import support_grid

    # ── supported quads (same grid convention as support.trimmed_quad_mesh)
    occ_cell = max(min(resolution, support_radius / 2.0), 1e-4)
    occ, ou0, ov0 = support_grid(uv, occ_cell, support_radius)
    u0, v0 = uv.min(0)
    u1, v1 = uv.max(0)
    nu = max(int(np.ceil((u1 - u0) / resolution)), 1)
    nv = max(int(np.ceil((v1 - v0) / resolution)), 1)
    uc = u0 + (np.arange(nu) + 0.5) * resolution
    vc = v0 + (np.arange(nv) + 0.5) * resolution
    ci = np.clip(np.floor((uc - ou0) / occ_cell).astype(np.int64), 0, occ.shape[1] - 1)
    cj = np.clip(np.floor((vc - ov0) / occ_cell).astype(np.int64), 0, occ.shape[0] - 1)
    keep = occ[np.ix_(cj, ci)]                        # (nv, nu) point-supported

    # ── audit EVERY unsupported cell of the UV domain. Real openings often
    # touch the domain border (a doorway reaches the floor edge), so a
    # border-connectivity filter would exempt exactly the interesting regions;
    # the mask vote is the arbiter everywhere. No extrapolation risk: the
    # domain itself is bounded by the measured points.
    hole_mask = ~keep
    n_holes = int(hole_mask.sum())
    if n_holes == 0:
        return None

    # ── mask calibration on the surface's own supported points
    sup_world = model.uv_to_world(uv[:: max(1, len(uv) // 4000)])
    oid = ev.calibrate_oid(np.asarray(sup_world, dtype=np.float64), instance_id)
    if oid is None:
        logger.info("hole_audit: instance %d — no matching mask, audit skipped",
                    instance_id)
        return None

    # ── vote every hole cell against the frames (three-way evidence)
    hj, hi = np.nonzero(hole_mask)
    centers_uv = np.column_stack([u0 + (hi + 0.5) * resolution,
                                  v0 + (hj + 0.5) * resolution])
    centers_world = np.asarray(model.uv_to_world(centers_uv), dtype=np.float64)
    cov, occ, val = ev.vote(centers_world, oid)
    vals = np.maximum(val, 1)
    r_cov = np.where(val > 0, cov / vals, 0.0)
    r_occ = np.where(val > 0, occ / vals, 0.0)
    r_empty = np.where(val > 0, 1.0 - r_cov - r_occ, 0.0)
    seen = val >= min_votes
    # direct witnesses: the object's own mask covers the cell
    fill_img = seen & (r_cov >= fill_ratio)
    # mixed: some views see the surface, the rest see an occluder — still a
    # direct witness exists, so image_supported
    fill_mix = seen & ~fill_img & (cov >= 1) & ((r_cov + r_occ) >= fill_ratio)
    # OCCLUSION REASONING (user 2026-08-29, "importantísimo"): no direct
    # witness, but measured geometry stands IN FRONT (Z-buffer verified) in
    # most views — the frames CANNOT testify either way, and on a fitted
    # surface the parsimonious reading is continuity behind the occluder
    # (e.g. the floor behind the ladder). RESTRICTED to INTERIOR holes:
    # a phantom plane extension past the wall edge is border-connected by
    # construction and gets no inferred continuity.
    from scipy import ndimage as _ndi
    # interior = enclosed by point support in all four grid directions — the
    # ladder's shadow on the floor qualifies (floor measured all around it);
    # a phantom extension past the wall edge has no support beyond it and
    # fails. (Strict border-connectivity lost the ladder case: unsupported
    # cells percolate to the domain border on real floors.)
    left = np.maximum.accumulate(keep, axis=1)
    right = np.maximum.accumulate(keep[:, ::-1], axis=1)[:, ::-1]
    top = np.maximum.accumulate(keep, axis=0)
    bot = np.maximum.accumulate(keep[::-1, :], axis=0)[::-1, :]
    interior_grid = left & right & top & bot & ~keep
    interior = interior_grid[hj, hi]
    fill_occl = (seen & ~fill_img & ~fill_mix & interior
                 & (r_occ >= occluded_ratio))
    # curved models (bspline/swept) extrapolate wildly far from their data —
    # cap every fill to cells NEAR the supported region (spikes on the
    # ceiling, user 2026-08-29)
    if max_fill_dist_cells is not None:
        near = _ndi.binary_dilation(keep, iterations=int(max_fill_dist_cells))
        near_cells = near[hj, hi]
        fill_img &= near_cells
        fill_mix &= near_cells
        fill_occl &= near_cells
    fill = fill_img | fill_mix | fill_occl
    # confirmed open: most views see PAST the surface (no mask at all there)
    open_ = seen & ~fill & (r_empty >= 1.0 - open_ratio)
    ambiguous = ~fill & ~open_
    # tiny ENCLOSED gaps with no image verdict: interpolate across the fitted
    # surface — the same bounded bridging Poisson does implicitly (user
    # 2026-08-29: Poisson had fewer holes than ransac). Provenance
    # 'interpolated'; confirmed openings are never touched.
    fill_interp = np.zeros_like(fill)
    if interp_max_cells > 0 and ambiguous.any():
        cand = np.zeros_like(keep)
        cand[hj[ambiguous], hi[ambiguous]] = True
        cand &= interior_grid
        lab2, n2 = _ndi.label(cand)
        if n2:
            sizes = np.bincount(lab2.ravel())
            small_ids = np.nonzero(sizes <= int(interp_max_cells))[0]
            small_ids = small_ids[small_ids > 0]
            if len(small_ids):
                ig = np.isin(lab2, small_ids) & cand
                fill_interp = ig[hj, hi] & ambiguous
                if max_fill_dist_cells is not None:
                    fill_interp &= near_cells
                fill |= fill_interp
                ambiguous = ~fill & ~open_

    keep2 = keep.copy()
    keep2[hj[fill], hi[fill]] = True

    # ── rebuild the quad mesh from keep2 (same corner assembly as support.py)
    jj, ii = np.nonzero(keep2)
    corners = np.stack([jj * (nu + 1) + ii,
                        jj * (nu + 1) + ii + 1,
                        (jj + 1) * (nu + 1) + ii,
                        (jj + 1) * (nu + 1) + ii + 1], axis=1)
    used = np.unique(corners)
    remap = np.full((nv + 1) * (nu + 1), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    c = remap[corners]
    faces = np.concatenate([c[:, [0, 1, 3]], c[:, [0, 3, 2]]], axis=0)
    gj, gi = np.divmod(used, nu + 1)
    verts_uv = np.column_stack([u0 + gi * resolution, v0 + gj * resolution])

    # border snap ONLY near measured points: the overhang ring snaps onto the
    # cloud border (as in support.py); image-filled interiors sit far from any
    # point and MUST stay on the grid — snapping them would collapse the fill.
    from scipy.spatial import cKDTree
    d, nearest = cKDTree(uv).query(verts_uv, k=1)
    band = (d > max(0.6 * resolution, 1e-6)) & (d < snap_max_m)
    if band.any():
        verts_uv = verts_uv.copy()
        verts_uv[band] = uv[nearest[band]]
        tri = verts_uv[faces]
        area2 = np.abs((tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1])
                       - (tri[:, 2, 0] - tri[:, 0, 0]) * (tri[:, 1, 1] - tri[:, 0, 1]))
        faces = faces[area2 > resolution * resolution * 1e-4]

    verts_world = np.asarray(model.uv_to_world(verts_uv), dtype=np.float64)
    report = {
        "hole_cells": n_holes,
        "filled_cells": int(fill.sum()),
        "filled_image_supported": int((fill_img | fill_mix).sum()),
        "filled_occlusion_inferred": int(fill_occl.sum()),
        "filled_interpolated": int(fill_interp.sum()),
        "open_cells": int(open_.sum()),
        "ambiguous_cells": int(ambiguous.sum()),
        "filled_area_m2": round(float(fill.sum()) * resolution * resolution, 3),
        "filled_occluded_area_m2": round(float(fill_occl.sum()) * resolution * resolution, 3),
        "open_area_m2": round(float(open_.sum()) * resolution * resolution, 3),
        "mask_oid": int(oid),
        "frames_used": [ev.cloud_frame(f) for f, _ in ev.frames_for(oid)],
        "mask_frames_used": [f for f, _ in ev.frames_for(oid)],
        "fill_ratio_gate": fill_ratio,
        "open_ratio_gate": open_ratio,
        "occluded_ratio_gate": occluded_ratio,
        "_grid": (keep2,
                  _cells_grid(keep.shape, hj[open_], hi[open_]),
                  float(u0), float(v0), float(resolution)),
        "provenance": "image_supported fills = the instance's own mask covers "
                      "the cell (direct witness); occlusion_inferred fills = "
                      "another object blocks the view in most frames, surface "
                      "continuity inferred behind it; open cells = the frames "
                      "see past the surface (image-confirmed opening); "
                      "ambiguous cells stay open — nothing is invented",
    }
    return verts_world, faces, report


def silhouette_report(verts_world: np.ndarray,
                      output_dir: Path, session_dir: Path,
                      instance_id: int) -> Optional[dict]:
    """Stage-2 metric: how well does the FINAL surface match the images?
    The mesh's footprint is projected into each mask keyframe and compared
    with the instance's own mask: precision (mesh pixels the mask confirms),
    recall (mask pixels the mesh reaches) and IoU, per frame + mean. A low
    score flags 'the curve/shape does not match what the frames show' —
    tool_measured, drives future refinement, never edits geometry itself."""
    from scipy import ndimage as _ndi

    ev = _evidence(Path(output_dir), Path(session_dir))
    if not ev.ok or len(verts_world) < 50:
        return None
    sample = np.asarray(verts_world, dtype=np.float64)
    sample = sample[:: max(1, len(sample) // 20000)]
    oid = ev.calibrate_oid(sample, instance_id)
    if oid is None:
        return None
    frames = []
    for fidx, key in ev.frames_for(oid):
        c2w = ev.pose_at(fidx)
        K = ev.K_at(fidx)
        if c2w is None or K is None:
            continue
        m = ev.masks[key] > 0
        mh, mw = m.shape
        c2w4 = np.eye(4)
        c2w4[:c2w.shape[0], :c2w.shape[1]] = c2w
        M = np.linalg.inv(c2w4)
        pcam = (M[:3, :3] @ sample.T).T + M[:3, 3]
        z = pcam[:, 2]
        front = z > 0.05
        if front.sum() < 50:
            continue
        u, v = ev.cam.pixels(ev.cloud_frame(fidx), pcam[front])                    # lens included
        mu, mv = ev.mask_pixels(u, v, mh, mw)                                       # point 108
        inb = (mu >= 0) & (mu < mw) & (mv >= 0) & (mv < mh)
        if inb.sum() < 50:
            continue
        foot = np.zeros((mh, mw), dtype=bool)
        foot[mv[inb], mu[inb]] = True
        foot = _ndi.binary_dilation(foot, iterations=2)   # close vertex-grid gaps
        inter = float((foot & m).sum())
        union = float((foot | m).sum())
        frames.append({
            "frame": ev.cloud_frame(fidx),
            "mask_frame": fidx,
            "precision": round(inter / max(foot.sum(), 1), 3),
            "recall": round(inter / max(m.sum(), 1), 3),
            "iou": round(inter / max(union, 1.0), 3),
        })
    if not frames:
        return None
    return {
        "mean_iou": round(float(np.mean([f["iou"] for f in frames])), 3),
        "mean_precision": round(float(np.mean([f["precision"] for f in frames])), 3),
        "mean_recall": round(float(np.mean([f["recall"] for f in frames])), 3),
        "frames": frames,
        "source": "tool_measured",
    }


def _cells_grid(shape, jj, ii) -> np.ndarray:
    g = np.zeros(shape, dtype=bool)
    g[jj, ii] = True
    return g


def save_report(out_dir: Path, report: dict) -> None:
    try:
        from correction.epoch import stamp_nearest as _stamp_epoch
        _stamp_epoch(report, out_dir)
        (Path(out_dir) / "hole_audit.json").write_text(json.dumps(report, indent=2))
    except Exception as e:  # noqa: BLE001
        logger.warning("hole_audit: report write failed: %s", e)


def see_through_filter(mesh_verts: np.ndarray, mesh_faces: np.ndarray,
                       output_dir: Path, session_dir: Path,
                       behind_ratio: float = 0.5, min_votes: int = 2,
                       depth_tol: float = 0.15):
    """PHANTOM-SURFACE cut (user 2026-08-29: a decomposition plane covered
    wall3's access arch). A RANSAC primitive collects real points but its MESH
    can bridge across open space between them; the witness is depth: a face
    whose center the cameras consistently see PAST — nearest measured geometry
    ≥depth_tol BEHIND it in ≥behind_ratio of the depth-valid views, and NO
    view measuring geometry AT its depth — is not a real surface.

    Returns (keep_face_mask, report) or None when evidence is unavailable.
    Conservative by design: one at-depth witness keeps the face."""
    ev = _evidence(Path(output_dir), Path(session_dir))
    if not ev.ok:
        return None
    verts = np.asarray(mesh_verts, dtype=np.float64)
    faces = np.asarray(mesh_faces, dtype=np.int64)
    if not len(faces):
        return None
    centers = verts[faces].mean(axis=1)
    valid, at_depth, behind = ev.see_through(centers, depth_tol=depth_tol)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = behind / np.maximum(valid, 1)
    refuted = (valid >= int(min_votes)) & (at_depth == 0) & \
        (frac >= float(behind_ratio))
    report = {
        "faces": int(len(faces)),
        "refuted": int(refuted.sum()),
        "min_votes": int(min_votes),
        "behind_ratio": float(behind_ratio),
        "depth_tol_m": float(depth_tol),
        "provenance": "tool_measured",
    }
    return ~refuted, report
