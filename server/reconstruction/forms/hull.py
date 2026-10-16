"""The measured envelope of an object: the VISUAL HULL of its masks carved by the TSDF of its own
points (USER 2026-10-16: "no respeta la nube, sobresale por todas partes … debe respetar la forma").

Per object, on the card:

1. A voxel grid over the object (the points' box, widened while the hull touches its walls).
2. THE HULL: every view that masks the object votes on every voxel it can see (in frame, not
   occluded by the cloud's z-buffer): inside the mask or outside. A voxel belongs to the object
   when it is inside in at least ``min_inside_frac`` of at least ``min_votes`` views — the same
   majority the point filter uses. Nothing outside the silhouettes survives: a mesh cannot stick
   out of what the images show.
3. THE TSDF of the object's own points: per view, the z-buffer of the points (filled over
   ``fill_px``), the signed distance measured − voxel depth, truncated at ``sdf_trunc``,
   averaged over the views. Where the points measured a surface, the surface is theirs; where
   they did not (a dark cabinet front with points on 28 % of its mask), the hull is the surface.
4. The field: the TSDF where known, −trunc inside the hull and +trunc outside it elsewhere;
   marching cubes at zero → the mesh, in the cloud's frame, at the voxel's resolution.

Every number is the session's or the declared TSDF configuration's (``tsdf.object``: voxel,
truncation, fill), the majority rule the point filter's; nothing here chooses a form.
"""
from __future__ import annotations

import math
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from .fit import subsample

EXTENT_Q = 0.005


_SCENE_DEPTH_CACHE: Dict[tuple, torch.Tensor] = {}


def scene_depth(vb, i: int, scene: torch.Tensor, fill_px: int, trunc: float, key: Optional[tuple] = None
                ) -> torch.Tensor:
    """(H, W) float32: the depth of the SCENE's nearest surface in view ``i`` of the batch — the
    full cloud's z-buffer (the nearest point per pixel: with about one point per pixel per view
    its noise bias is under a σ; a mean over the truncation band mixed the far side in at
    grazing pixels and carved every rim inward), holes filled from the nearest measured depth
    within ``fill_px`` and only those; inf where nothing. The object's own sparse points could
    not do this: a pixel whose only point lay on the far side emptied the interior. Cached per
    (session, keyframe): every object of the session reuses it."""
    from precision.f6_torch import min_filter
    from reconstruction.surface_fit.hole_audit import zbuffer_min
    if key is not None and key in _SCENE_DEPTH_CACHE:
        return _SCENE_DEPTH_CACHE[key]
    px, py, z, inb = vb.project_one(i, scene)
    zb = torch.full((vb.H, vb.W), float("inf"), dtype=torch.float64, device=vb.dev)
    if int(inb.sum()):
        ci = torch.clamp(torch.round(px[inb]), 0, vb.W - 1).to(torch.int64)
        ri = torch.clamp(torch.round(py[inb]), 0, vb.H - 1).to(torch.int64)
        zb = zbuffer_min(ri * vb.W + ci, z[inb].to(torch.float64), vb.H, vb.W)
        if int(fill_px) > 0:
            filled = min_filter(zb, 2 * int(fill_px) + 1)
            zb = torch.where(torch.isfinite(zb), zb, filled)
    out = zb.to(torch.float32)
    if key is not None:
        _SCENE_DEPTH_CACHE[key] = out
    return out


class HullCarver:
    """The votes and the TSDF of one object over its voxel grid, view batch by view batch."""

    def __init__(self, batches: list, P: np.ndarray, *, voxel: float, trunc: float, fill_px: int,
                 min_votes: int, min_inside_frac: float, dev, sigma: float = 0.0,
                 scene: Optional[torch.Tensor] = None, session_key: str = "", max_voxels: int = 30_000_000):
        self.batches = batches
        self.dev = dev
        self.voxel = float(voxel)
        self.trunc = float(trunc)
        self.min_votes = int(min_votes)
        self.frac = float(min_inside_frac)
        self.max_voxels = int(max_voxels)
        self.P = torch.as_tensor(np.asarray(P, np.float64), dtype=torch.float32, device=dev)
        self.sigma_t = float(sigma)
        # the scene's nearest surface per view (the whole cloud; the object's own points when no
        # scene was given — the synthetic tests)
        src = self.P if scene is None else scene.to(torch.float32)
        self.depth = []
        for vb in batches:
            maps = []
            for i, v in enumerate(vb.views):
                key = (session_key, int(v.cloud_frame)) if (session_key and scene is not None) else None
                maps.append(scene_depth(vb, i, src, int(fill_px), self.trunc, key))
            self.depth.append(torch.stack(maps))

    def grid(self, lo: np.ndarray, hi: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int], float]:
        """(origin, shape, voxel) of a grid covering [lo, hi]; the voxel grows when the grid would
        exceed ``max_voxels`` (declared: a floor of 20 m at 1.5 cm is 27 M voxels)."""
        v = self.voxel
        n = np.ceil((hi - lo) / v).astype(np.int64) + 1
        while int(np.prod(n)) > self.max_voxels:
            v *= 1.25
            n = np.ceil((hi - lo) / v).astype(np.int64) + 1
        return lo.astype(np.float64), (int(n[0]), int(n[1]), int(n[2])), float(v)

    @torch.no_grad()
    def field(self, origin: np.ndarray, shape: Tuple[int, int, int], voxel: float, chunk: int = 2_000_000
              ) -> Dict[str, np.ndarray]:
        """The signed field over the grid and its evidence counts (numpy, shape order x, y, z)."""
        nx, ny, nz = shape
        N = nx * ny * nz
        f = np.full(N, self.trunc, np.float32)
        n_seen = np.zeros(N, np.int16)
        n_in = np.zeros(N, np.int16)
        tsdf_sum = np.zeros(N, np.float32)
        tsdf_n = np.zeros(N, np.int16)
        ix = torch.arange(nx, device=self.dev)
        for a in range(0, N, int(chunk)):
            idx = torch.arange(a, min(N, a + int(chunk)), device=self.dev)
            kx = idx // (ny * nz)
            ky = (idx // nz) % ny
            kz = idx % nz
            X = torch.stack([kx, ky, kz], 1).to(torch.float32) * float(voxel) \
                + torch.as_tensor(origin, dtype=torch.float32, device=self.dev)
            seen = torch.zeros(len(idx), dtype=torch.int32, device=self.dev)
            inside = torch.zeros(len(idx), dtype=torch.int32, device=self.dev)
            s_sum = torch.zeros(len(idx), dtype=torch.float32, device=self.dev)
            s_n = torch.zeros(len(idx), dtype=torch.int32, device=self.dev)
            for vb, dep in zip(self.batches, self.depth):
                px, py, z, front, _visible, inside_m = vb.project(X)
                ci = torch.clamp(torch.round(px), 0, vb.W - 1).to(torch.int64)
                ri = torch.clamp(torch.round(py), 0, vb.H - 1).to(torch.int64)
                bi = torch.arange(vb.n, device=self.dev)[:, None].expand_as(ci)
                d = dep[bi, ri, ci]                                  # the scene's nearest surface
                has = front & torch.isfinite(d)
                s = d - z                                             # + in front of the surface
                # THE HULL VOTE: a voxel inside the object's mask is inside its silhouette cone
                # whatever its depth (behind the object's own front it is still the object —
                # the interior must be solid, not a shell); outside the mask it is outside the
                # cone unless something closer hides that pixel (another object in front):
                # then the view abstains
                tol = self.sigma_t + z / vb.f_px[:, None]
                occluded = has & (z > d + tol)
                in_vote = front & inside_m
                out_vote = front & ~inside_m & ~occluded
                seen += (in_vote | out_vote).to(torch.int32).sum(0)
                inside += in_vote.to(torch.int32).sum(0)
                # FREE where the voxel is in front of whatever the scene measured (any object);
                # the object's own surface where the pixel is the object's (its mask) and the
                # voxel sits within trunc behind the nearest surface; beyond that: unknown
                free = has & (s > self.trunc)
                own = has & inside_m & (s >= -self.trunc) & (s <= self.trunc)
                integ = free | own
                s = torch.clamp(s, -self.trunc, self.trunc)
                s_sum += torch.where(integ, s, torch.zeros_like(s)).sum(0)
                s_n += integ.to(torch.int32).sum(0)
            sl = slice(a, a + len(idx))
            n_seen[sl] = seen.clamp(max=32000).cpu().numpy()
            n_in[sl] = inside.clamp(max=32000).cpu().numpy()
            tsdf_sum[sl] = s_sum.cpu().numpy()
            tsdf_n[sl] = s_n.clamp(max=32000).cpu().numpy()
        judged = n_seen >= self.min_votes
        hull = judged & (n_in >= self.frac * n_seen)
        known = tsdf_n > 0
        tsdf = np.where(known, tsdf_sum / np.maximum(tsdf_n, 1), 0.0).astype(np.float32)
        f = np.where(known, tsdf, np.where(hull, -self.trunc, self.trunc)).astype(np.float32)
        # the hull is the envelope: outside the silhouettes nothing is a surface, whatever the
        # points said (a flyer, a leak of another object's depth)
        f = np.where(judged & ~hull, self.trunc, f).astype(np.float32)
        return {"f": f.reshape(shape), "hull": hull.reshape(shape), "judged": judged.reshape(shape),
                "known": known.reshape(shape), "n_seen": n_seen.reshape(shape)}


def _touches_wall(hull: np.ndarray) -> np.ndarray:
    """Which of the six walls of the grid the hull reaches (x−, x+, y−, y+, z−, z+)."""
    return np.array([hull[0].any(), hull[-1].any(), hull[:, 0].any(), hull[:, -1].any(),
                     hull[:, :, 0].any(), hull[:, :, -1].any()])


@torch.no_grad()
def mask_agreement(batches: list, P: np.ndarray, dev, n_max: int = 20000, seed: int = 0) -> dict:
    """How often the object's OWN points fall inside its masks in the views that see them — the
    instrument's agreement between the cloud and the silhouettes. The hull asks a voxel for at
    least that share of its views (never under the declared majority): with six exact synthetic
    views the hull is their intersection; with SAM3 leaking on a tenth of the views, a tenth is
    forgiven."""
    Ps = torch.as_tensor(subsample(P, n_max, seed), dtype=torch.float32, device=dev)
    seen = 0
    inside = 0
    for vb in batches:
        _px, _py, _z, front, visible, ins = vb.project(Ps)
        in_vote = front & ins
        out_vote = visible & ~ins            # outside the mask and nothing in front of it
        seen += int((in_vote | out_vote).sum())
        inside += int(in_vote.sum())
    share = inside / seen if seen else float("nan")
    return {"pairs": int(seen), "inside_share": share}


def hull_mesh(P: np.ndarray, batches: list, *, voxel: float, trunc: float, fill_px: int, min_votes: int,
              min_inside_frac: float, sigma: float, dev, scene: Optional[torch.Tensor] = None,
              session_key: str = "", log: Callable[[str], None] = lambda m: None,
              max_rounds: int = 3) -> Tuple[np.ndarray, np.ndarray, dict]:
    """The object's measured envelope as a mesh: (vertices, faces, record)."""
    from skimage import measure
    t0 = time.time()
    P = np.asarray(P, np.float64)
    agree = mask_agreement(batches, P, dev) if batches else {"pairs": 0, "inside_share": float("nan")}
    share = agree["inside_share"]
    frac = max(float(min_inside_frac), float(share)) if np.isfinite(share) else float(min_inside_frac)
    carver = HullCarver(batches, P, voxel=voxel, trunc=trunc, fill_px=fill_px, min_votes=min_votes,
                        min_inside_frac=frac, dev=dev, sigma=sigma, scene=scene, session_key=session_key)
    lo = np.percentile(P, 100 * EXTENT_Q, axis=0)
    hi = np.percentile(P, 100 * (1 - EXTENT_Q), axis=0)
    margin = max(3.0 * float(sigma), 4.0 * float(voxel))
    lo, hi = lo - margin, hi + margin
    rec: Dict[str, object] = {"rounds": [], "mask_agreement": agree, "inside_frac": frac,
                              "inside_frac_is": "the share of the object's own points inside its masks over the "
                                                "views that see them, never under the declared majority"}
    res = None
    for rnd in range(int(max_rounds)):
        origin, shape, v = carver.grid(lo, hi)
        res = carver.field(origin, shape, v)
        walls = _touches_wall(res["hull"])
        rec["rounds"].append({"shape": list(shape), "voxel_m": v, "hull_voxels": int(res["hull"].sum()),
                              "known_voxels": int(res["known"].sum()), "touches": walls.tolist()})
        if not walls.any() or rnd == int(max_rounds) - 1:
            break
        # the silhouettes reach past the box the points gave: widen those walls and look again
        grow = max(0.25 * float(np.max(hi - lo)), 10.0 * v)
        lo = lo - walls[[0, 2, 4]] * grow
        hi = hi + walls[[1, 3, 5]] * grow
    f = res["f"]
    if not (f.min() < 0 < f.max()):
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64), {**rec, "empty": True, "seconds": round(time.time() - t0, 1)}
    verts, faces, _normals, _vals = measure.marching_cubes(f, level=0.0, spacing=(v, v, v))
    V = verts + origin[None, :]
    rec.update({"voxel_m": v, "n_vertices": int(len(V)), "n_faces": int(len(faces)),
                "hull_voxels": int(res["hull"].sum()), "known_voxels": int(res["known"].sum()),
                "seconds": round(time.time() - t0, 1),
                "what": "the visual hull of the masks (majority of the views that see each voxel), carved by the "
                        "TSDF of the object's own points; marching cubes at the voxel"})
    log(f"[forms] hull: grid {shape} at {v * 100:.1f} cm, inside in ≥ {frac:.0%} of the views (the cloud's own "
        f"agreement {share:.0%}), hull {rec['hull_voxels']:,} voxels, measured {rec['known_voxels']:,}, mesh "
        f"{len(V):,} vertices / {len(faces):,} faces in {rec['seconds']} s")
    return V.astype(np.float64), faces.astype(np.int64), rec
