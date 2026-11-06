"""THE CUBES (USER 2026-11-06): the scene is cut into cubes of SPACE, as GenRecon cuts it, and a view belongs to a cube by
WHAT IT SEES there, never by where its camera stands ("el tema es lo que ven, ¿ven lo mismo?"; "en bufferstop los rayos
podrían ser concéntricos pero ven cosas diferentes porque el objeto está en el medio"; "van juntas porque ven lo mismo,
total o parcialmente … no importa si son del principio o del final del recorrido"; "un kf puede servir para otros
objetos porque no ve una sola cosa, esa covisibilidad es la que importa para armar los cubos y el solapamiento"; "con
esa información sacás un mapa de cobertura de lo escaneado, y qué frames fueron cubriendo qué cosa, vas a armar cubos de
x m, con no más de y kf, y ya sabés quiénes vas a incluir en cada cubo porque el propio mapa de distribución te lo dice").

1. WHAT A VIEW SEES. The structure probe is AMB3R-SLAM over the whole video (reconstruction/amb3r_probe.py): every
   frame it reconstructed with what it saw — its depth over its whole image, its camera — so every point with the frame
   it came from; a view of ours sees what its own frame saw, or the nearest reconstructed frame when AMB3R did not
   reconstruct it. (The older probe, one Omega pass over our views, gives each view its own depth.) Each point is the
   first surface of its ray: what lies behind is occluded by construction. No ray is cast.
2. UNITS: a parallax keyframe with the rotation keyframes it stands for and the turn frames after them — what a unit
   sees is what its views see; a turn is never cut.
3. The GRAVITY FRAME: up = −(the sum of the keyframes' image-down axes), the floor plan's axes the principal axes of the
   cameras on the plane ⊥ up. Metric through the session's own walk (the I3 walk over the probe's keyframe path).
4. CUBES of space: a grid of cubes of ``cube_m`` metres over everything the views see (one layer when the scene is no
   taller than a cube, else stacked: several floors are cut the same way). A unit BELONGS to a cube when at least τ of
   what it sees falls in it (τ the planner's co-visibility bar) — a unit that sees two regions belongs to both: that is
   the overlap, and the seam the chunk graph joins the cubes on.
5. THE CAP: a cube seen by more than ``max_views`` views is halved on its longest side, and again, while halving still
   lowers the views of a half; what no halving can separate keeps the views that see the most of it (declared).
6. OWNER: a unit's keyframes take their geometry from the cube where it sees the most. A unit no cube holds joins the
   cube where it sees the most among those with room (the most, over the cap, when none has room); seeing only space no
   cube holds, the cell of the grid where it sees the most becomes a cube; seeing nothing, it joins its walk
   neighbour's. Two units consecutive along the walk always share a cube (the turn frames between them are Omega views
   of a cube that holds both, and the cubes form one graph). A cube needs two keyframes.
7. HOLES (reported): the share of what the views see that falls in a cube they belong to, per unit and overall. The
   keyframes themselves were completed before (intake/coverage_fill.py: where no keyframe sees what the video saw, or
   one sees it alone).

A cube's Omega views are its keyframes and the turn frames between two of its keyframes that are consecutive in the
walk; its F6 witnesses (the interval frames) follow the same rule. Deterministic: fixed grids, integer counts, sorted
reductions, every order with a tie-break by index; what the views see is counted on the card."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

MAX_CUBE_VIEWS = 64          # USER 2026-11-04 ("ese es el límite") / 2026-11-06 ("siempre con tope de 64")
TAU = 0.3                    # the co-visibility bar (walk_structure.TAU, chunk_covis.TAU)
CUBE_M = 5.0                 # USER 2026-11-06 ("el tamaño del cubo, no más de por ejemplo como genrecon 5 m o 6 m")
CUBES_NAME = "cubes.json"
CUBES_VERSION = 3            # 3: what a view sees = its frame's own points (AMB3R over the whole video; 2: rays
                             #    cast against AMB3R's cloud; 1: k-d cells of camera positions)


class CubePlanError(RuntimeError):
    pass


def _frame_no(name) -> int:
    return int(Path(str(name)).stem)


# ── 2. units ──────────────────────────────────────────────────────────────────────────────────

def units_of(kf_frames: Sequence[int], plan_index: Sequence[int], turn_frames: Sequence[int]) -> List[Dict]:
    """The planning units in walk order: each parallax keyframe (``plan_index`` into ``kf_frames``) with the rotation
    keyframes up to the next parallax keyframe and every turn frame after them up to the next unit's first keyframe.
    The keyframes before the first parallax keyframe (none on a normal selection) join the first unit."""
    n = len(kf_frames)
    fr = [int(f) for f in kf_frames]
    if fr != sorted(fr) or len(set(fr)) != n:
        raise CubePlanError("the keyframes are not in strictly increasing frame order")
    pi = sorted({int(i) for i in plan_index})
    if not pi or pi[0] < 0 or pi[-1] >= n:
        raise CubePlanError(f"plan_index {list(plan_index)[:5]}… does not index {n} keyframes")
    starts = [0] + pi[1:] if pi[0] != 0 else pi
    bounds = starts + [n]
    turns = sorted(int(t) for t in turn_frames)
    kf_set = set(fr)
    if any(t in kf_set for t in turns):
        raise CubePlanError("a turn frame is a keyframe")
    out = []
    ti = 0
    for u in range(len(starts)):
        a, b = bounds[u], bounds[u + 1]
        hi = fr[b] if b < n else None
        mine = []
        while ti < len(turns) and turns[ti] < fr[a]:
            ti += 1          # a turn frame before the first keyframe: never a view (no keyframe before it)
        while ti < len(turns) and (hi is None or turns[ti] < hi):
            mine.append(turns[ti]); ti += 1
        out.append({"keyframes": list(range(a, b)), "turn": mine, "views": (b - a) + len(mine)})
    return out


# ── 3. the gravity frame ──────────────────────────────────────────────────────────────────────

def _device(device: str):
    import torch
    return torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")


def gravity_frame(c2w_kf: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Rows e1, e2, up of the floor-plan frame (orthonormal, right-handed): up = −Σ image-down axes (OpenCV +y); e1 =
    the principal horizontal axis of the camera centres, pointing from the walk's start to its end (else its largest
    component positive); e2 = up × e1. On the card, float64."""
    import torch
    T = torch.as_tensor(np.asarray(c2w_kf, np.float64), device=_device(device))
    down = T[:, :3, 1].sum(0)
    if not bool(torch.isfinite(down).all()) or float(down.norm()) < 1e-9:
        raise CubePlanError("the keyframes' image-down axes cancel out — no gravity direction")
    up = -down / down.norm()
    C = T[:, :3, 3]
    H = C - (C @ up)[:, None] * up[None]
    H = H - H.mean(0)
    _, V = torch.linalg.eigh(H.T @ H)
    e1 = V[:, -1] - (V[:, -1] @ up) * up
    if float(e1.norm()) < 1e-9:                        # every camera on one vertical line
        a = torch.eye(3, dtype=torch.float64, device=T.device)[int(torch.argmin(up.abs()))]
        e1 = a - (a @ up) * up
    e1 = e1 / e1.norm()
    s = float((C[-1] - C[0]) @ e1)
    if abs(s) < 1e-12:
        s = float(e1[int(torch.argmax(e1.abs()))])
    if s < 0:
        e1 = -e1
    e2 = torch.linalg.cross(up, e1)
    return torch.stack([e1, e2 / e2.norm(), up]).cpu().numpy()


# ── 1.–7. what the views see, the cubes, the owners ──────────────────────────────────────────

def depth_points(c2w: np.ndarray, K: np.ndarray, depth: np.ndarray, valid: np.ndarray, views: Sequence[int], *,
                 device: str = "cuda"):
    """The points of ``views``: every valid pixel of each one's depth, back-projected with its K and camera (on the
    card, float64), stacked — what those views saw."""
    import torch
    dev = _device(device)
    out = []
    for i in views:
        m = torch.as_tensor(np.asarray(valid[i], bool), device=dev)
        if not bool(m.any()):
            continue
        vv, uu = torch.nonzero(m, as_tuple=True)
        d = torch.as_tensor(np.asarray(depth[i], np.float64), device=dev)[vv, uu]
        k = torch.as_tensor(np.asarray(K[i], np.float64), device=dev)
        T = torch.as_tensor(np.asarray(c2w[i], np.float64), device=dev)
        cam = torch.stack([(uu.double() - k[0, 2]) * d / k[0, 0], (vv.double() - k[1, 2]) * d / k[1, 1], d], 1)
        out.append(cam @ T[:3, :3].T + T[:3, 3])
    return torch.cat(out) if out else torch.zeros((0, 3), dtype=torch.float64, device=dev)


def _axis_cells(lo: float, hi: float, size: float) -> List[Tuple[float, float]]:
    """The fewest cells of ``size`` that cover [lo, hi], centred on it."""
    n = max(1, int(np.ceil((hi - lo) / size - 1e-9)))
    a0 = 0.5 * (lo + hi) - 0.5 * n * size
    return [(a0 + i * size, a0 + (i + 1) * size) for i in range(n)]


class _Seen:
    """What every unit sees (its points in the gravity frame) on the card: the share of each unit's points inside a
    box — integer counts (a cumulative sum over the units' consecutive runs)."""

    def __init__(self, pts, R: np.ndarray, device: str):
        import torch
        self.torch = torch
        self.dev = _device(device)
        Rt = torch.as_tensor(np.asarray(R, np.float64), device=self.dev)
        self.U = len(pts)
        n = [int(len(p)) for p in pts]
        self.n = torch.as_tensor([max(1, x) for x in n], dtype=torch.float64, device=self.dev)
        self.off = torch.as_tensor(np.concatenate([[0], np.cumsum(n)]), dtype=torch.int64, device=self.dev)
        self.P = (torch.cat([p.to(self.dev, torch.float64) for p in pts]) @ Rt.T if sum(n)
                  else torch.zeros((0, 3), dtype=torch.float64, device=self.dev))
        self.empty = [u for u in range(self.U) if n[u] == 0]

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.P.min(0).values.cpu().numpy(), self.P.max(0).values.cpu().numpy()

    def share(self, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        t = self.torch
        lo_t = t.as_tensor(np.asarray(lo, np.float64), device=self.dev)
        hi_t = t.as_tensor(np.asarray(hi, np.float64), device=self.dev)
        inside = ((self.P >= lo_t) & (self.P < hi_t)).all(1).to(t.int64)
        c = t.cat([t.zeros(1, dtype=t.int64, device=self.dev), t.cumsum(inside, 0)])
        cnt = c[self.off[1:]] - c[self.off[:-1]]
        return (cnt.double() / self.n).cpu().numpy()


def plan_cubes(c2w_kf: np.ndarray, units: List[Dict], unit_pts, *, metres_per_unit: float,
               cube_m: float = CUBE_M, max_views: int = MAX_CUBE_VIEWS, tau: float = TAU,
               device: str = "cuda") -> Dict:
    """The cube plan of ``units`` (units_of) from what each unit sees (``unit_pts``: its points in the probe frame, on
    the card) and the probe's keyframe cameras ``c2w_kf`` (the gravity frame); ``metres_per_unit``: the probe's scale."""
    U = len(units)
    if U == 0:
        raise CubePlanError("no unit to plan")
    R = gravity_frame(c2w_kf, device)
    seen = _Seen(unit_pts, R, device)
    if len(seen.P) == 0:
        raise CubePlanError("no view sees anything — the probe holds no valid depth")
    size = float(cube_m) / float(metres_per_unit)              # the cube's side in the probe's units
    w = np.asarray([int(u["views"]) for u in units], np.int64)
    lo_all, hi_all = seen.bounds()
    hi_all = hi_all + 1e-9
    zcells = ([(lo_all[2], lo_all[2] + size)] if hi_all[2] - lo_all[2] <= size
              else _axis_cells(lo_all[2], hi_all[2], size))
    grid = [(np.array([x0, y0, z0]), np.array([x1, y1, z1]))
            for (z0, z1) in zcells for (y0, y1) in _axis_cells(lo_all[1], hi_all[1], size)
            for (x0, x1) in _axis_cells(lo_all[0], hi_all[0], size)]
    cubes: List[Dict] = []
    empty_leaves: List[Tuple[np.ndarray, np.ndarray]] = []
    splits, trimmed = 0, 0

    def members_of(lo, hi):
        sh = seen.share(lo, hi)
        return sh, [u for u in range(U) if sh[u] >= tau]

    stack = [(lo, hi) for lo, hi in reversed(grid)]
    while stack:
        lo, hi = stack.pop()
        sh, mem = members_of(lo, hi)
        if not mem:
            if float(sh.max()) > 0:
                empty_leaves.append((lo, hi))               # seen, by no unit enough: a cell for the forced
            continue
        nv = int(w[mem].sum())
        if nv > int(max_views):
            ax = int(np.argmax(hi - lo))
            mid = 0.5 * (lo[ax] + hi[ax])
            hiA, loB = hi.copy(), lo.copy()
            hiA[ax], loB[ax] = mid, mid
            _, mA = members_of(lo, hiA)
            _, mB = members_of(loB, hi)
            if max(int(w[mA].sum()) if mA else 0, int(w[mB].sum()) if mB else 0) < nv:
                splits += 1
                stack.append((loB, hi))
                stack.append((lo, hiA))                     # popped first: the cubes come out in a fixed order
                continue
            # no halving separates what these views see: the views that see the most of it stay (declared)
            order = sorted(mem, key=lambda u: (-sh[u], u))
            keep, tot = [], 0
            for u in order:
                if tot + int(w[u]) > int(max_views) and keep:
                    continue
                keep.append(u)
                tot += int(w[u])
            trimmed += len(mem) - len(keep)
            mem = sorted(keep)
        cubes.append({"lo": lo.tolist(), "hi": hi.tolist(), "units": sorted(mem),
                      "share": {str(u): float(sh[u]) for u in mem}})
    if not cubes:
        raise CubePlanError("no cube: no unit sees τ of what it sees in any cell")
    shares = np.zeros((U, len(cubes)))
    for ci, cb in enumerate(cubes):
        shares[:, ci] = seen.share(np.asarray(cb["lo"]), np.asarray(cb["hi"]))
    forced = 0

    def views_in(ci):
        return int(w[cubes[ci]["units"]].sum())

    def place(u, ci):
        cubes[ci]["units"] = sorted(cubes[ci]["units"] + [u])
        cubes[ci]["share"][str(u)] = float(shares[u, ci])

    # a unit no cube holds: the cube where it sees the most among those with room (the most, over the cap, when none
    # has room); seeing only cells no cube holds, the one where it sees the most becomes a cube
    leaf_sh = None
    for u in range(U):
        if any(u in cb["units"] for cb in cubes) or u in seen.empty:
            continue
        cand = [ci for ci in range(len(cubes)) if shares[u, ci] > 0]
        if not cand:
            if leaf_sh is None:
                leaf_sh = np.stack([seen.share(lo, hi) for lo, hi in empty_leaves], 1) if empty_leaves \
                    else np.zeros((U, 0))
            if leaf_sh.shape[1] == 0 or float(leaf_sh[u].max()) <= 0:
                continue                                    # nothing it sees is in any cell: its walk neighbour's
            li = int(np.argmax(leaf_sh[u]))
            lo, hi = empty_leaves[li]
            cubes.append({"lo": lo.tolist(), "hi": hi.tolist(), "units": [], "share": {}})
            shares = np.concatenate([shares, leaf_sh[:, li][:, None]], axis=1)
            leaf_sh[:, li] = 0.0                            # one cube per cell
            cand = [len(cubes) - 1]
        room = [ci for ci in cand if views_in(ci) + int(w[u]) <= int(max_views)]
        place(u, max(room or cand, key=lambda c: (shares[u, c], -c)))
        forced += 1
    # a unit that sees nothing a cube holds joins its walk neighbour's cube (the previous unit's, else the next's)
    blind = [u for u in range(U) if not any(u in cb["units"] for cb in cubes)]
    for u in blind:
        nb = next((x for x in list(range(u - 1, -1, -1)) + list(range(u + 1, U))
                   if any(x in cb["units"] for cb in cubes)), None)
        if nb is None:
            raise CubePlanError("no unit sees anything a cube holds")
        mine = [ci for ci, cb in enumerate(cubes) if nb in cb["units"]]
        place(u, max(mine, key=lambda c: (shares[nb, c], -c)))
        forced += 1

    # a cube needs two keyframes: a smaller one's units join where else they see the most (with room first)
    def n_kf(cb):
        return sum(len(units[u]["keyframes"]) for u in cb["units"])
    dropped = [ci for ci, cb in enumerate(cubes) if n_kf(cb) < 2]
    others = [cj for cj in range(len(cubes)) if cj not in dropped]
    if not others:
        raise CubePlanError("no cube holds two keyframes")
    for ci in dropped:
        for u in cubes[ci]["units"]:
            if any(u in cubes[cj]["units"] for cj in others):
                continue
            cand = [cj for cj in others if shares[u, cj] > 0]
            if not cand:                                    # its walk neighbour's kept cube
                nb = next((x for x in list(range(u - 1, -1, -1)) + list(range(u + 1, U))
                           if any(x in cubes[cj]["units"] for cj in others)), None)
                cand = [cj for cj in others if nb is not None and nb in cubes[cj]["units"]]
            room = [cj for cj in cand if views_in(cj) + int(w[u]) <= int(max_views)]
            place(u, max(room or cand, key=lambda c: (shares[u, c], -c)))
    keep_idx = others
    cubes = [cubes[ci] for ci in keep_idx]
    shares = shares[:, keep_idx]

    def owner_of(u):
        mine = [ci for ci, cb in enumerate(cubes) if u in cb["units"]]
        return max(mine, key=lambda c: (shares[u, c], -c))
    # two units consecutive along the walk share a cube
    joined = 0
    for u in range(U - 1):
        if any(u in cb["units"] and (u + 1) in cb["units"] for cb in cubes):
            continue
        a, b = owner_of(u), owner_of(u + 1)
        room_a = int(w[cubes[a]["units"]].sum()) + int(w[u + 1]) <= int(max_views)
        ci, add = (a, u + 1) if room_a or not (int(w[cubes[b]["units"]].sum()) + int(w[u]) <= int(max_views)) \
            else (b, u)
        cubes[ci]["units"] = sorted(cubes[ci]["units"] + [add])
        cubes[ci]["share"][str(add)] = float(shares[add, ci])
        joined += 1
    for cb in cubes:
        cb["keyframes"] = sorted(k for u in cb["units"] for k in units[u]["keyframes"])
        cb["views"] = int(w[cb["units"]].sum())
        cb["over_cap"] = bool(cb["views"] > int(max_views))
        cb["size_m"] = ((np.asarray(cb["hi"]) - np.asarray(cb["lo"])) * float(metres_per_unit)).tolist()
    order = sorted(range(len(cubes)), key=lambda ci: (cubes[ci]["keyframes"][0], cubes[ci]["lo"]))
    cubes = [cubes[ci] for ci in order]
    shares = shares[:, order]
    owner_unit = [owner_of(u) for u in range(U)]
    n_kf_all = int(sum(len(u["keyframes"]) for u in units))
    owner = np.full(n_kf_all, -1, np.int64)
    for u, ci in enumerate(owner_unit):
        owner[units[u]["keyframes"]] = ci
    if (owner < 0).any():
        raise CubePlanError("a keyframe has no owner cube")
    # one graph: cubes joined by the units they share
    par = list(range(len(cubes)))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for u in range(U):
        mine = [ci for ci, cb in enumerate(cubes) if u in cb["units"]]
        for ci in mine[1:]:
            par[find(ci)] = find(mine[0])
    if len({find(ci) for ci in range(len(cubes))}) != 1:
        raise CubePlanError("the cubes do not form one graph through the views they share")
    # holes (reported): the share of what each unit sees that falls in a cube it belongs to
    covered = np.array([sum(shares[u, ci] for ci, cb in enumerate(cubes) if u in cb["units"]) for u in range(U)])
    n_pts = (seen.off[1:] - seen.off[:-1]).cpu().numpy().astype(np.float64)
    return {"gravity_frame": R.tolist(), "max_views": int(max_views), "tau": float(tau), "cube_m": float(cube_m),
            "metres_per_unit": float(metres_per_unit), "cubes": cubes, "owner_unit": owner_unit,
            "owner": owner.tolist(),
            "stats": {"splits": splits, "trimmed": trimmed, "forced": forced, "dropped_small": len(dropped),
                      "walk_joins": joined, "units_seeing_nothing": len(seen.empty),
                      "covered_overall": float((covered * n_pts).sum() / max(n_pts.sum(), 1.0)),
                      "units_mostly_uncovered": int((covered < 0.5).sum()),
                      "memberships_per_unit": float(np.mean([sum(u in cb["units"] for cb in cubes)
                                                             for u in range(U)]))}}


def cube_extras(cubes: List[Dict], kf_frames: Sequence[int], extra_frames: Sequence[int]) -> List[List[int]]:
    """Per cube, the frames of ``extra_frames`` (turn frames or interval witnesses) that lie strictly between two
    keyframes CONSECUTIVE in the walk that are both in the cube — the views Omega can place between them."""
    fr = [int(f) for f in kf_frames]
    ex = sorted(int(x) for x in extra_frames)
    out = []
    for cb in cubes:
        mine = set(int(k) for k in cb["keyframes"])
        spans = [(fr[k], fr[k + 1]) for k in sorted(mine) if k + 1 < len(fr) and (k + 1) in mine]
        out.append([x for x in ex if any(lo < x < hi for lo, hi in spans)])
    return out


# ── the analysis on the card (a process of its own: the map worker never touches the card before Omega) ──────────

def _omega_valid(conf: np.ndarray, depth: np.ndarray, device: str) -> np.ndarray:
    """The valid pixels of an Omega probe's views: the chunk confidence floor (walk_structure._floor_mask), on the
    card."""
    import torch
    from reconstruction.walk_structure import CONF_FLOOR
    dev = _device(device)
    out = np.zeros(depth.shape, bool)
    for i in range(len(depth)):
        c = torch.as_tensor(np.asarray(conf[i], np.float64), device=dev)
        d = torch.as_tensor(np.asarray(depth[i], np.float64), device=dev)
        lo, hi = c.min(), c.max()
        out[i] = ((c >= lo + CONF_FLOOR * (hi - lo)) & (c > 1e-5) & (d > 1e-3) & torch.isfinite(d)).cpu().numpy()
    return out


def analyse(probe_npz: Path, inputs: Dict, *, device: str = "cuda", log: Callable[[str], None] = print) -> Dict:
    """What every unit sees and the cube plan. ``probe_npz``: AMB3R's whole-video pass (intake/amb3r_video.npz,
    reconstruction/amb3r_probe.py) or one Omega pass over our views (intake/structure_probe.npz). ``inputs``: keyframe
    and turn-frame numbers, the plan index, the witness frames, the cap, the cube's side and the session's walk in
    metres."""
    import torch
    kf = [int(f) for f in inputs["keyframes"]]
    turns = [int(f) for f in inputs["turn"]]
    nums = sorted(set(kf) | set(turns))
    if len(nums) != len(kf) + len(turns):
        raise CubePlanError("a turn frame is a keyframe, or a frame is listed twice")
    units = units_of(kf, inputs["plan_index"], turns)
    slot = {f: i for i, f in enumerate(nums)}
    with np.load(probe_npz) as z:
        source = str(z["source"]) if "source" in z.files else "omega"
    extra: Dict = {}
    if source == "amb3r-video":
        from intake.coverage_fill import CoverageMap
        from reconstruction import amb3r_probe as A3
        video = A3.load_video(probe_npz)
        c2w = np.asarray(video["pred"], np.float64)[A3.video_index(video, nums)]
        src = A3.source_records(video, nums)
        cmap = CoverageMap(video, device, state=False)
        unit_pts, unit_src = [], []
        for u in units:
            recs = sorted({int(src[slot[kf[k]]]) for k in u["keyframes"]} | {int(src[slot[t]]) for t in u["turn"]})
            unit_src.append(recs)
            unit_pts.append(torch.cat([cmap.X[cmap.own(r)] for r in recs]).double())
        used = sorted({int(r) for r in src})
        rec_frame = np.asarray(video["frame_no"], np.int64)[np.asarray(video["rec_index"], np.int64)]
        own = sum(1 for f in nums if int(rec_frame[src[slot[f]]]) == f)
        log(f"[cubes] what {len(nums)} views see: the points of {len(used)} frames AMB3R reconstructed "
            f"({sum(int(cmap.n_per[r]) for r in used):,} points) — {own} view(s) their own frame's, the others the "
            f"nearest reconstructed frame's")
        probe_resolution = int(max(int(v) for v in video["grid_wh"]))
        extra = {"unit_records": unit_src, "views": nums,
                 "view_source_frames": [int(rec_frame[src[slot[f]]]) for f in nums]}
        del cmap
    else:
        z = np.load(probe_npz)
        pnums = [_frame_no(p) for p in z["paths"]]
        if pnums != nums:
            raise CubePlanError("the probe's views are not this session's keyframes and turn frames in frame order")
        c2w, K = np.asarray(z["c2w"], np.float64), np.asarray(z["K"], np.float64)
        depth = np.asarray(z["depth"], np.float64)
        valid = _omega_valid(np.asarray(z["conf"]), depth, device)
        unit_pts = [depth_points(c2w, K, depth, valid, [slot[kf[k]] for k in u["keyframes"]]
                                 + [slot[t] for t in u["turn"]], device=device) for u in units]
        probe_resolution = int(z["resolution"])
    kf_c2w = c2w[[slot[f] for f in kf]]
    Ct = torch.as_tensor(kf_c2w[:, :3, 3], dtype=torch.float64, device=_device(device))
    path = float(torch.linalg.norm(Ct[1:] - Ct[:-1], dim=1).sum()) if len(kf) > 1 else 0.0
    walk_m = float(inputs["walk_m"])
    if not (path > 0 and walk_m > 0):
        raise CubePlanError(f"the probe's keyframe path {path} or the session's walk {walk_m} m is not positive — no "
                            f"metre to size the cubes with")
    plan = plan_cubes(kf_c2w, units, unit_pts, metres_per_unit=walk_m / path,
                      cube_m=float(inputs.get("cube_m", CUBE_M)), max_views=int(inputs["max_views"]), device=device)
    ctx = cube_extras(plan["cubes"], kf, turns)
    wit = cube_extras(plan["cubes"], kf, inputs.get("witness") or [])
    for cb, cx, wx in zip(plan["cubes"], ctx, wit):
        cb["context"] = cx
        cb["witnesses"] = wx
        cb["omega_views"] = len(cb["keyframes"]) + len(cx)
    plan.update({"units": units, "n_keyframes": len(kf), "n_turn_frames": len(turns),
                 "probe_resolution": probe_resolution, "kf_slot": [slot[f] for f in kf],
                 "kf_centres": kf_c2w[:, :3, 3].tolist(), "probe_source": source, **extra})
    sizes = [cb["omega_views"] for cb in plan["cubes"]]
    st = plan["stats"]
    log(f"[cubes] {len(units)} units ({len(kf)} keyframes, {len(turns)} turn frames) by what they see → "
        f"{len(plan['cubes'])} cube(s) of {plan['cube_m']:g} m or less, {min(sizes)}–{max(sizes)} Omega views (cap "
        f"{plan['max_views']}); a unit in {st['memberships_per_unit']:.2f} cube(s) on average; {st['splits']} "
        f"halving(s), {st['walk_joins']} walk join(s), {st['forced']} unit(s) placed where they see the most; what "
        f"the views see inside their cubes: {st['covered_overall'] * 100:.1f} % ({st['units_mostly_uncovered']} "
        f"unit(s) under half)"
        + (f"; over the cap: {[i for i, cb in enumerate(plan['cubes']) if cb['over_cap']]}"
           if any(cb["over_cap"] for cb in plan["cubes"]) else "")
        + (f"; {st['trimmed']} view(s) left out of a cube no halving could separate" if st["trimmed"] else ""))
    return plan


def figure(plan: Dict, centres: np.ndarray, out_png: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    R = np.asarray(plan["gravity_frame"])
    C = np.asarray(centres, np.float64) @ R.T
    fig, ax = plt.subplots(1, 2, figsize=(17, 8), gridspec_kw={"width_ratios": [3, 1.2]})
    ax[0].plot(C[:, 0], C[:, 1], "-", color="0.8", lw=0.8, zorder=1)
    n = max(1, len(plan["cubes"]))
    cols = plt.cm.hsv(np.arange(n) / n)
    for ci, cb in enumerate(plan["cubes"]):
        lo, hi = cb["lo"], cb["hi"]
        ax[0].add_patch(Rectangle((lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1], fill=False, ec=cols[ci], lw=1.0,
                                  ls="--" if cb["over_cap"] else "-", zorder=2))
        ax[0].text(0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), str(ci), fontsize=7, ha="center", va="center",
                   color=cols[ci], zorder=5, bbox=dict(boxstyle="round,pad=0.1", fc="w", ec="none", alpha=0.7))
    own = np.asarray(plan["owner"])
    for ci in range(n):
        m = own == ci
        if m.any():
            ax[0].scatter(C[m, 0], C[m, 1], s=10, color=cols[ci], zorder=3)
    ax[0].scatter(C[0, 0], C[0, 1], marker="*", s=220, c="g", zorder=6, label="start")
    ax[0].set_aspect("equal")
    ax[0].legend(fontsize=8)
    ax[0].set_title(f"{title}: {len(plan['cubes'])} cubes of {plan['cube_m']:g} m or less by what the views see "
                    f"(≤ {plan['max_views']} views); cameras coloured by the cube that owns them — from above")
    ax[1].scatter(C[:, 0], C[:, 2], s=6, c="0.4")
    ax[1].set_title("side view (floors)")
    ax[1].set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def save(plan: Dict, path: Path) -> None:
    doc = {k: v for k, v in plan.items() if k != "covisibility"}
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    os.replace(tmp, path)


def code_sha() -> str:
    h = hashlib.sha256()
    for p in (Path(__file__), Path(__file__).with_name("walk_structure.py"),
              Path(__file__).resolve().parents[1] / "intake" / "coverage_fill.py",
              Path(__file__).with_name("amb3r_probe.py")):
        h.update(p.read_bytes())
    return h.hexdigest()


def session_cubes(scan_dir: Path, frames_dir: Path, sel_files: Sequence[str], turn_files: Sequence[str],
                  witness_files: Sequence[str], cfg_v: dict, *, resolution: int, max_views: int,
                  sky_cache: Optional[Path], log: Callable[[str], None], probe: str = "omega",
                  cube_m: float = CUBE_M, walk_m: float = 0.0) -> Dict:
    """The session's cube plan on the STRUCTURE PROBE — ``probe`` "amb3r": AMB3R-SLAM's whole-video pass
    (intake/amb3r_video.npz, run right after the intake's keyframes — workers/map_worker._run_intake_selection —
    and the keyframes completed on it, intake/coverage_fill.py); "omega": one Omega pass over every keyframe and turn
    frame at ``resolution`` (vendor/VGGT-Long/stac_frames_pass.py, intake/structure_probe.npz, reused when its stamp is
    the same) — and the PLAN (a process of its own on the card, ``python -m reconstruction.cubes``; reused when the
    probe, the inputs and this code are the same). intake/cubes.json and intake/cubes.png."""
    import subprocess
    import sys
    import repro
    from reconstruction.chunk_covis import planning_indices
    from reconstruction.zones import PASS_SCRIPT, _sha_file, run_frames_pass
    intake = Path(scan_dir) / "intake"
    intake.mkdir(parents=True, exist_ok=True)
    views = sorted(set(sel_files) | set(turn_files), key=_frame_no)
    views_sha = hashlib.sha256("\n".join(views).encode()).hexdigest()
    if probe == "amb3r":
        from reconstruction import amb3r_probe as A3
        npz, st_p = intake / A3.VIDEO_NAME, intake / "amb3r_video.stamp.json"
        if not (npz.exists() and st_p.exists()):
            raise CubePlanError(f"{npz} is missing — AMB3R's whole-video pass runs right after the intake's keyframes "
                                f"(workers/map_worker._run_intake_selection)")
        stamp = {"video_pass": json.loads(st_p.read_text()), "views": views_sha}
        log(f"[cubes] the structure probe: AMB3R-SLAM's whole-video pass ({npz.name}) — what our "
            f"{len(sel_files)} keyframes + {len(turn_files)} turn frames see decides the cut")
    elif probe == "omega":
        npz, stamp_p = intake / "structure_probe.npz", intake / "structure_probe.stamp.json"
        stamp = {"views": views_sha, "resolution": int(resolution),
                 "omega_mode": str((cfg_v.get("Model") or {}).get("omega_mode")),
                 "pass_code": _sha_file(PASS_SCRIPT), "weights": str((cfg_v.get("Weights") or {}).get("VGGTOmega")),
                 "mask_sky": bool((cfg_v.get("Model") or {}).get("mask_sky", True))}
        reuse = False
        if npz.exists() and stamp_p.exists():
            try:
                reuse = json.loads(stamp_p.read_text()) == stamp
            except ValueError:
                reuse = False
        if reuse:
            log(f"[cubes] the structure probe reused ({npz.name}: the same {len(views)} views, resolution, code and "
                f"weights)")
        else:
            stamp_p.unlink(missing_ok=True)
            log(f"[cubes] the structure probe: ONE Omega pass over {len(sel_files)} keyframes + {len(turn_files)} "
                f"turn frames at {resolution} — not one of precision, the cut is decided on it")
            run_frames_pass([str(Path(frames_dir) / f) for f in views], int(resolution), npz, cfg_v,
                            sky_cache=sky_cache, salad=False, log=log)
            tmp = stamp_p.with_name(stamp_p.name + ".tmp")
            tmp.write_text(json.dumps(stamp, indent=1))
            os.replace(tmp, stamp_p)
    else:
        raise CubePlanError(f"structure probe {probe!r}: 'amb3r' or 'omega'")
    kf = [_frame_no(f) for f in sel_files]
    inputs = {"keyframes": kf, "turn": sorted(_frame_no(f) for f in turn_files),
              "witness": sorted(_frame_no(f) for f in witness_files),
              "plan_index": [int(i) for i in planning_indices(Path(scan_dir), kf)], "max_views": int(max_views),
              "cube_m": float(cube_m), "walk_m": float(walk_m)}
    h = hashlib.sha256(json.dumps(stamp, sort_keys=True).encode())
    h.update(code_sha().encode())
    h.update(json.dumps(inputs, sort_keys=True).encode())
    inputs["stamp"] = h.hexdigest()
    out_p, png = intake / CUBES_NAME, intake / "cubes.png"
    if out_p.exists():
        try:
            doc = json.loads(out_p.read_text())
        except ValueError:
            doc = {}
        if doc.get("version") == CUBES_VERSION and doc.get("stamp") == inputs["stamp"]:
            log(f"[cubes] the cube plan reused ({out_p.name}: the same probe, inputs and code) — "
                f"{len(doc['cubes'])} cube(s)")
            return doc
    inp_p = intake / "cubes_inputs.json"
    inp_p.write_text(json.dumps(inputs))
    cmd = [sys.executable, "-m", "reconstruction.cubes", "--probe", str(npz), "--inputs", str(inp_p),
           "--out", str(out_p), "--png", str(png), "--title", Path(scan_dir).parent.parent.name, "--device", "cuda"]
    r = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[1]), env=repro.deterministic_env(),
                       capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.strip():
            log(line)
    if r.returncode != 0 or not out_p.exists():
        raise RuntimeError(f"the cube plan failed (exit {r.returncode}):\n{r.stderr[-2000:]}")
    inp_p.unlink(missing_ok=True)
    doc = json.loads(out_p.read_text())
    if doc.get("stamp") != inputs["stamp"]:
        raise RuntimeError("the cube plan on disk is not the one just made (stamp)")
    return doc


def main(argv=None) -> int:
    """The cube plan in a process of its own, on the card (USER 2026-11-05: "ningún cálculo va en CPU"; the map worker
    never touches the card before Omega's exclusive check)."""
    import argparse
    ap = argparse.ArgumentParser(prog="python -m reconstruction.cubes")
    ap.add_argument("--probe", required=True)
    ap.add_argument("--inputs", required=True, help="a JSON file: keyframes, turn, witness, plan_index, max_views")
    ap.add_argument("--out", required=True)
    ap.add_argument("--png", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    inputs = json.loads(Path(a.inputs).read_text())
    plan = analyse(Path(a.probe), inputs, device=a.device, log=lambda m: print(m, flush=True))
    figure(plan, np.asarray(plan["kf_centres"]), Path(a.png), a.title)
    plan["stamp"] = inputs.get("stamp")
    plan["version"] = CUBES_VERSION
    save(plan, Path(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
