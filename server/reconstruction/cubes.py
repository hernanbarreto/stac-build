"""THE CUBES (USER 2026-11-06/07): the views are cut into groups Omega reconstructs together — "cubos de 64 imágenes
solapadas al 50 % con los vecinos"; "el tema es lo que ven, ¿ven lo mismo?"; "van juntas porque ven lo mismo, total o
parcialmente … no importa si son del principio o del final del recorrido"; "en bufferstop los rayos podrían ser
concéntricos pero ven cosas diferentes"; "los 5 m es una estimación no de profundidad de lo que ven, sino de
desplazamiento"; "los cubos deben contener las vistas que lo conforman"; and, asked what comes first, the number of
views or the 5 m: "si hay muy pocas vistas eso sí es un problema" — the views first ("me parece bien").

1. WHAT A VIEW SEES. The structure probe is AMB3R-SLAM over the whole video (reconstruction/amb3r_probe.py): every frame
   it reconstructed with what it saw — its depth over its whole image, its camera; a view of ours sees what its own frame
   saw, or the nearest reconstructed frame when AMB3R did not reconstruct it. (The older probe, one Omega pass over our
   views, gives each view its own depth.)
2. UNITS: a parallax keyframe with the rotation keyframes it stands for and the turn frames after them — a turn is never
   cut.
3. WHAT TWO UNITS SEE OF EACH OTHER: c(u→v) = the share of what u saw that some record of v sees at a consistent depth
   (|z − d| ≤ tol·d, tol the session's own — reconstruction/coverage_fill.CoverageMap's test); c_uv = min(c(u→v),
   c(v→u)), for every two units whose cameras lie within two cubes of each other and every two consecutive along the walk.
4. THE GROUPS. A. From the first unit of the walk no group holds, a group grows by the unit that sees the most of what
   it sees (c ≥ τ with one of its units — total or partially, any moment of the walk) until its own views reach half
   the cap, its own cameras always within half a cube and its pass within the cube of ``cube_m`` around them (C): the
   DISPLACEMENT of a pass, never exceeded, half for its own views and half for the shared ones — beyond about 5 m
   Omega bends what it infers, and a cube longer than the others bends differently (USER 2026-11-07: "a más de 5 m omega
   deforma y puede generar dobleces para un cubo más largo que otros … deben ser coherentes"). B. A group left with fewer
   own views than half the cap joins another small group first, else the smallest of the others — among the groups it
   sees the same as (c ≥ τ) before those it sees anything of, else its walk neighbour's — while the two stay within the
   cap and their cameras within half a cube; one that cannot stays as it is, its pass filled with the shared views
   around it.
5. THE WALK: two units consecutive along the walk share a cube (the turn frames between them are Omega views of a cube
   that holds both, and the cubes form one graph).
6. THE OVERLAP: each cube takes the units of other cubes that see what its own see (c ≥ τ with one of them) and whose
   cameras fall within its cube (its own cameras' box widened to ``cube_m``), the most co-visible first, while it stays
   within the cap: neighbouring cubes share the strip between them, and a pass with few own views gets its views there. The OWNER of a
   unit's keyframes is its own group's cube.

A cube's box (for the viewer) holds the cameras of all its views, its own and the shared ones: cubes that share views
overlap. A cube's Omega views are its keyframes and the turn
frames between two of its keyframes consecutive in the walk; its F6 witnesses (the interval frames) follow the same rule.
7. WHAT EACH KEYFRAME SEES OF EVERY OTHER (USER 2026-11-07, on the duplicated wooden wall and signs of observatorio: "me
   gusta tu enfoque de corrección … aplícalo a conciencia y que se corrija en serio"): c(a→b), the test of 3 with one
   keyframe on each side and the band of one surface (coverage_fill.ONE_SURFACE: two moments that disagree are still on
   the surface each other saw), for every two keyframes; the pairs that reach τ either way are kept in the plan
   (``kf_covisibility``, frame numbers). Two moments of the walk that see the same surface were tied by nothing after
   the cut — F4 tracked only along the walk, F6 voted only among walk neighbours — so two cubes that disagree left the
   surface twice. F4 tracks across these pairs (precision/tracks.revisit_pairs: F5 aligns the two moments' poses on
   landmarks both see) and F6's vote hears them (precision/depth_on_f5.covisible_witnesses). The band, not the session's
   tolerance: AMB3R closed no loop on pccr, and at that tolerance its map never saw the walk's end meet its start — on
   one surface it does.
Deterministic: fixed strides, every order with a tie-break by index; what the views see is tested on the card."""
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
SAMPLES = 2000               # the points of a unit tested in another (a fixed stride over what its records saw)
CUBES_NAME = "cubes.json"
CUBES_VERSION = 5            # 5: + what each keyframe sees of every other (kf_covisibility: F4's revisit tracks, F6's vote)
                             #    (4: groups by what the units see of each other, the views first, the displacement a
                             #    ceiling; 3: boxes of seen space; 2: rays cast against AMB3R's cloud; 1: k-d camera cells)


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


# ── 3.–6. what the units see of each other, and the cubes ────────────────────────────────────────

def _sees_many(cmap, recs, X, tol: float):
    """(len(recs), len(X)) — which records see each point of X at a consistent depth (CoverageMap's test, batched)."""
    t = cmap.t
    r = t.as_tensor([int(x) for x in recs], dtype=t.int64, device=cmap.dev)
    Xc = t.einsum("kab,nb->kna", cmap.R_w2c[r], X) + cmap.t_w2c[r][:, None, :]
    z = Xc[..., 2]
    zs = t.where(z > 1e-6, z, t.ones_like(z))
    K = cmap.K[r]
    u = t.round(K[:, 0, 0, None] * Xc[..., 0] / zs + K[:, 0, 2, None]).long()
    v = t.round(K[:, 1, 1, None] * Xc[..., 1] / zs + K[:, 1, 2, None]).long()
    ok = (z > 1e-3) & (u >= 0) & (u < cmap.Ws) & (v >= 0) & (v < cmap.Hs)
    d = cmap.D[r[:, None].expand_as(u), v.clamp(0, cmap.Hs - 1), u.clamp(0, cmap.Ws - 1)]
    return ok & (d > 0) & ((z - d).abs() <= tol * d)


def directed_covisibility(cmap, unit_recs: Sequence[Sequence[int]], centres: np.ndarray, reach: float, tol: float, *,
                          samples: int = SAMPLES) -> Dict[Tuple[int, int], float]:
    """c(u→v) for every two units whose cameras lie within ``reach`` of each other, and every two consecutive along the
    walk, both ways: the share of what u saw (a fixed stride of its records' points) that some record of v sees at a
    consistent depth. On the card."""
    t = cmap.t
    U = len(unit_recs)
    Ct = t.as_tensor(np.asarray(centres, np.float64), device=cmap.dev)
    close = (t.cdist(Ct, Ct) <= float(reach)).cpu().numpy()
    near = []
    for u in range(U):
        s = set(int(v) for v in np.nonzero(close[u])[0]) | {x for x in (u - 1, u + 1) if 0 <= x < U}
        s.discard(u)
        near.append(sorted(s))
    directed: Dict[Tuple[int, int], float] = {}
    for u in range(U):
        if not near[u]:
            continue
        P = t.cat([cmap.X[cmap.own(r)] for r in unit_recs[u]])
        if len(P) == 0:
            continue
        idx = t.linspace(0, len(P) - 1, min(len(P), int(samples)), dtype=t.float64, device=cmap.dev).floor().long()
        X = P[idx]
        recs = sorted({int(r) for v in near[u] for r in unit_recs[v]})
        row = {r: i for i, r in enumerate(recs)}
        seen = t.cat([_sees_many(cmap, recs[a:a + 512], X, tol) for a in range(0, len(recs), 512)])
        for v in near[u]:
            rows = t.as_tensor([row[int(r)] for r in unit_recs[v]], dtype=t.int64, device=cmap.dev)
            directed[(u, v)] = float(seen[rows].any(0).double().mean())
    return directed


def unit_covisibility(cmap, unit_recs: Sequence[Sequence[int]], centres: np.ndarray, reach: float, tol: float, *,
                      samples: int = SAMPLES) -> Dict[Tuple[int, int], float]:
    """c_uv = min(c(u→v), c(v→u)) (directed_covisibility) for u < v."""
    directed = directed_covisibility(cmap, unit_recs, centres, reach, tol, samples=samples)
    return {(u, v): min(c, directed.get((v, u), 0.0)) for (u, v), c in directed.items() if u < v}


def keyframe_covisibility(cmap, kf_frames: Sequence[int], kf_recs: Sequence[int], *, band: Optional[float] = None,
                          tau: float = TAU, samples: int = SAMPLES) -> Dict:
    """What each keyframe sees of every other (7), for EVERY two keyframes: c(a→b) = the share of what keyframe a saw (a
    fixed stride of its record's points) that keyframe b's record sees on one surface, |z − d| ≤ ``band``·d — the band
    the session's tolerance is measured in (coverage_fill.ONE_SURFACE): two moments of the walk that disagree by less
    than that still see the surface each other saw, and those are the ones to tie. Each keyframe sees with its record
    (AMB3R: its own frame's, else the nearest reconstructed one's). Kept: the pairs that reach ``tau`` either way, as
    [frame a, frame b, c(a→b), c(b→a)], a before b along the walk. On the card."""
    from reconstruction.coverage_fill import ONE_SURFACE
    band = ONE_SURFACE if band is None else float(band)
    t = cmap.t
    recs = sorted({int(r) for r in kf_recs})
    row = {r: i for i, r in enumerate(recs)}
    kr = t.as_tensor([row[int(r)] for r in kf_recs], dtype=t.int64, device=cmap.dev)
    n = len(kf_recs)
    M = t.zeros((n, n), dtype=t.float64, device=cmap.dev)
    for a in range(n):
        P = cmap.X[cmap.own(int(kf_recs[a]))]
        if len(P) == 0:
            continue
        idx = t.linspace(0, len(P) - 1, min(len(P), int(samples)), dtype=t.float64, device=cmap.dev).floor().long()
        seen = t.cat([_sees_many(cmap, recs[x:x + 512], P[idx], band) for x in range(0, len(recs), 512)])
        M[a] = seen.double().mean(1)[kr]
    M.fill_diagonal_(0.0)
    Mc = M.cpu().numpy()
    fr = [int(f) for f in kf_frames]
    ii, jj = np.nonzero(np.triu(np.maximum(Mc, Mc.T) >= float(tau), 1))
    pairs = [[fr[i], fr[j], float(Mc[i, j]), float(Mc[j, i])] for i, j in zip(ii.tolist(), jj.tolist())]
    return {"tau": float(tau), "band_rel": band, "samples": int(samples), "n_keyframes": n, "pairs": pairs}


def cube_box(cams: np.ndarray, size: float):
    """(lo of the cameras, hi of the cameras, the box's centre, its half extents): the box drawn for a cube — around its
    views' cameras, at least a quarter of a cube on each horizontal half side and half a cube up and down."""
    lo, hi = cams.min(0), cams.max(0)
    mid = 0.5 * (lo + hi)
    hx = np.maximum(0.5 * (hi - lo), 0.25 * size)
    hx[2] = 0.5 * size
    return lo, hi, mid, hx


def plan_cubes(c2w_kf: np.ndarray, units: List[Dict], cov: Dict[Tuple[int, int], float], *, metres_per_unit: float,
               cube_m: float = CUBE_M, max_views: int = MAX_CUBE_VIEWS, tau: float = TAU,
               device: str = "cuda") -> Dict:
    """The cubes of ``units`` (units_of) from what they see of each other (``cov``, unit_covisibility) and the probe's
    keyframe cameras ``c2w_kf`` (the gravity frame, the displacement); ``metres_per_unit``: the probe's scale."""
    U = len(units)
    if U == 0:
        raise CubePlanError("no unit to plan")
    R = gravity_frame(c2w_kf, device)
    Cg = np.asarray(c2w_kf, np.float64)[:, :3, 3] @ np.asarray(R, np.float64).T
    ucam = [Cg[units[u]["keyframes"]] for u in range(U)]
    size = float(cube_m) / float(metres_per_unit)
    w = np.asarray([int(u["views"]) for u in units], np.int64)
    half = int(max_views) // 2
    nbr: Dict[int, Dict[int, float]] = {u: {} for u in range(U)}
    for (u, v), c in cov.items():
        nbr[u][v] = c
        nbr[v][u] = c
    # A. GROUPS THAT GROW BY WHAT THEY SEE: from the first unit of the walk no group holds, a group takes the unit that
    #    sees the most of what it sees (c ≥ τ with one of its units — total or partially, any moment of the walk) until its
    #    own views reach half the cap, its cameras always within a cube (USER 2026-11-07: "a más de 5 m omega deforma y
    #    puede generar dobleces para un cubo más largo que otros … deben ser coherentes")
    assigned = [-1] * U
    members: Dict[int, List[int]] = {}
    own: Dict[int, int] = {}

    # a group's OWN cameras take half a cube; its pass, the cube widened around them, the other half (the shared views)
    # — "64 imágenes solapadas al 50 % con los vecinos" in space as in views
    own_size = 0.5 * size

    def fits_box(lo, hi, pts):
        return bool(np.all(np.maximum(hi, pts.max(0)) - np.minimum(lo, pts.min(0)) <= own_size * (1.0 + 1e-9)))
    for seed in range(U):
        if assigned[seed] >= 0:
            continue
        g = seed
        assigned[seed] = g
        mem, n_own = [seed], int(w[seed])
        lo, hi = ucam[seed].min(0), ucam[seed].max(0)
        best = {y: c for y, c in nbr[seed].items() if assigned[y] < 0}
        while n_own < half:
            cands = [y for y, c in best.items() if c >= tau and assigned[y] < 0]
            if not cands:
                break
            within = [y for y in cands if fits_box(lo, hi, ucam[y])]
            if not within:
                break
            y = max(within, key=lambda y: (best[y], -y))
            assigned[y] = g
            mem.append(y)
            n_own += int(w[y])
            lo, hi = np.minimum(lo, ucam[y].min(0)), np.maximum(hi, ucam[y].max(0))
            best.pop(y)
            for z, c in nbr[y].items():
                if assigned[z] < 0 and c > best.get(z, -1.0):
                    best[z] = c
        members[g] = sorted(mem)
        own[g] = n_own
    par = list(range(U))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for u in range(U):
        par[u] = assigned[u]

    def merge(a, b):
        a, b = sorted((a, b))
        for x in members[b]:
            par[x] = a
        par[b] = a
        members[a] = sorted(members[a] + members.pop(b))
        own[a] += own.pop(b)
        return a
    # B. FEW OWN VIEWS (USER 2026-11-07: "si hay muy pocas vistas eso sí es un problema"): a group left with fewer own
    #    views than half the cap joins another small group first, else the smallest of the others — among the groups it
    #    sees the same as (≥ τ) before those it sees anything of, else its walk neighbour's — while the two stay within
    #    the cap and their cameras within a cube (the coherence of every pass comes first: a group that cannot join stays
    #    as it is, and its cube is filled with the shared views around it, C)
    merges_b, stuck = 0, set()
    gbox = {g: (np.min([ucam[x].min(0) for x in members[g]], axis=0), np.max([ucam[x].max(0) for x in members[g]], axis=0))
            for g in members}

    def fits_groups(a, b):
        return bool(np.all(np.maximum(gbox[a][1], gbox[b][1]) - np.minimum(gbox[a][0], gbox[b][0])
                           <= own_size * (1.0 + 1e-9)))

    def affinity(a):
        out: Dict[int, float] = {}
        for x in members[a]:
            for y, c in nbr[x].items():
                gy = find(y)
                if gy != a and c > out.get(gy, -1.0):
                    out[gy] = c
        return out
    while True:
        small = [g for g in members if own[g] < half and g not in stuck]
        if not small or len(members) == 1:
            break
        a = min(small, key=lambda g: (own[g], g))
        aff = affinity(a)
        cands = [g for g, c in aff.items() if c > 0 and own[a] + own[g] <= int(max_views) and fits_groups(a, g)]
        if cands:
            pools = ([g for g in cands if own[g] < half and aff[g] >= tau], [g for g in cands if own[g] < half],
                     [g for g in cands if aff[g] >= tau], cands)
            pool = next(pl for pl in pools if pl)
            b = min(pool, key=lambda g: (own[g], -aff[g], g))
        else:
            wn = sorted({find(x) for y in members[a] for x in (y - 1, y + 1) if 0 <= x < U} - {a})
            wn = [g for g in wn if own[a] + own[g] <= int(max_views) and fits_groups(a, g)]
            if not wn:
                stuck.add(a)
                continue
            b = min(wn, key=lambda g: (own[g], g))
        lo_ab = np.minimum(gbox[a][0], gbox[b][0])
        hi_ab = np.maximum(gbox[a][1], gbox[b][1])
        m = merge(a, b)
        gbox.pop(a, None)
        gbox.pop(b, None)
        gbox[m] = (lo_ab, hi_ab)
        stuck.discard(m)
        merges_b += 1
    groups = sorted(members, key=lambda g: (members[g][0], g))
    gi = {g: i for i, g in enumerate(groups)}
    core = [list(members[g]) for g in groups]
    owner_unit = [gi[find(u)] for u in range(U)]
    cube_units = [set(c) for c in core]

    def views_of(i):
        return int(w[sorted(cube_units[i])].sum())
    # D. two units consecutive along the walk share a cube (the turn frames between them, one graph)
    joined, over = 0, 0
    for u in range(U - 1):
        if any(u in s and (u + 1) in s for s in cube_units):
            continue
        a, b = owner_unit[u], owner_unit[u + 1]
        if views_of(a) + int(w[u + 1]) <= int(max_views):
            cube_units[a].add(u + 1)
        elif views_of(b) + int(w[u]) <= int(max_views):
            cube_units[b].add(u)
        else:
            cube_units[a].add(u + 1)
            over += 1
        joined += 1
    # C. THE OVERLAP: each cube takes the units of other cubes that see what its own see (≥ τ with one of them) and whose
    #    cameras fall within its cube — its own cameras' box widened to ``cube_m`` (USER 2026-11-07: a shared view enters
    #    "si su cámara cae dentro de sus 5 m") —, the most co-visible first, while it stays within the cap
    shared = 0
    for i in range(len(groups)):
        oc = np.concatenate([ucam[u] for u in core[i]])
        olo, ohi = oc.min(0), oc.max(0)
        pad = np.maximum(size - (ohi - olo), 0.0) / 2.0
        blo, bhi = olo - pad, ohi + pad
        best: Dict[int, float] = {}
        for x in core[i]:
            for y, c in nbr[x].items():
                if (c >= tau and y not in cube_units[i] and c > best.get(y, -1.0)
                        and bool(np.all(ucam[y] >= blo - 1e-9 * size)) and bool(np.all(ucam[y] <= bhi + 1e-9 * size))):
                    best[y] = c
        for y in sorted(best, key=lambda y: (-best[y], y)):
            if views_of(i) + int(w[y]) <= int(max_views):
                cube_units[i].add(y)
                shared += 1
    # one graph: cubes joined by the units they share
    gpar = list(range(len(groups)))

    def gfind(x):
        while gpar[x] != x:
            gpar[x] = gpar[gpar[x]]
            x = gpar[x]
        return x
    for u in range(U):
        mine = [i for i, s in enumerate(cube_units) if u in s]
        for i in mine[1:]:
            gpar[gfind(i)] = gfind(mine[0])
    if len({gfind(i) for i in range(len(groups))}) != 1:
        raise CubePlanError("the cubes do not form one graph through the views they share")
    cubes: List[Dict] = []
    for i in range(len(groups)):
        # the box drawn holds the cameras of ALL its views, its own and the shared ones (USER 2026-11-07: "no veo
        # solapamiento … los cubos deberían estar uno dentro del otro"): two cubes that share views overlap
        lo, hi, mid, hx = cube_box(np.concatenate([ucam[u] for u in sorted(cube_units[i])]), size)
        o_cams = np.concatenate([ucam[u] for u in core[i]])
        olo, ohi = o_cams.min(0), o_cams.max(0)
        aff = {}
        for y in sorted(cube_units[i]):
            aff[str(y)] = 1.0 if y in core[i] else max(nbr[y].get(x, 0.0) for x in core[i])
        units_i = sorted(cube_units[i])
        cubes.append({"lo": (mid - hx).tolist(), "hi": (mid + hx).tolist(), "units": units_i, "core": sorted(core[i]),
                      "share": aff, "keyframes": sorted(k for u in units_i for k in units[u]["keyframes"]),
                      "own_views": int(w[core[i]].sum()), "views": int(w[units_i].sum()),
                      "over_cap": bool(int(w[units_i].sum()) > int(max_views)),
                      "camera_span_m": ((hi - lo) * float(metres_per_unit)).tolist(),
                      "own_camera_span_m": ((ohi - olo) * float(metres_per_unit)).tolist(),
                      "size_m": (2.0 * hx * float(metres_per_unit)).tolist()})
    n_kf_all = int(sum(len(u["keyframes"]) for u in units))
    owner = np.full(n_kf_all, -1, np.int64)
    for u, i in enumerate(owner_unit):
        owner[units[u]["keyframes"]] = i
    if (owner < 0).any():
        raise CubePlanError("a keyframe has no owner cube")
    own_v = [cb["own_views"] for cb in cubes]
    return {"gravity_frame": R.tolist(), "max_views": int(max_views), "tau": float(tau), "cube_m": float(cube_m),
            "metres_per_unit": float(metres_per_unit), "cubes": cubes, "owner_unit": owner_unit,
            "owner": owner.tolist(),
            "stats": {"covisible_pairs": len(cov), "groups_grown": len(set(assigned)),
                      "merges_for_views": merges_b,
                      "small_left": int(sum(1 for v in own_v if v < half)), "walk_joins": joined,
                      "over_cap_joins": over, "shared_memberships": shared,
                      "memberships_per_unit": float(np.mean([sum(u in s for s in cube_units) for u in range(U)])),
                      "own_views_min": int(min(own_v)), "own_views_max": int(max(own_v))}}


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
    """What the units see of each other and the cube plan. ``probe_npz``: AMB3R's whole-video pass
    (intake/amb3r_video.npz, reconstruction/amb3r_probe.py) or one Omega pass over our views (intake/structure_probe.npz).
    ``inputs``: keyframe and turn-frame numbers, the plan index, the witness frames, the cap, the cube's side and the
    session's walk in metres."""
    import torch
    from reconstruction.coverage_fill import CoverageMap
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
        from reconstruction import amb3r_probe as A3
        video = A3.load_video(probe_npz)
        c2w = np.asarray(video["pred"], np.float64)[A3.video_index(video, nums)]
        rec_of = [int(r) for r in A3.source_records(video, nums)]
        rec_frame = np.asarray(video["frame_no"], np.int64)[np.asarray(video["rec_index"], np.int64)]
        own_frames = sum(1 for f, r in zip(nums, rec_of) if int(rec_frame[r]) == f)
        log(f"[cubes] what {len(nums)} views see: the records of {len(set(rec_of))} frames AMB3R reconstructed — "
            f"{own_frames} view(s) their own frame's, the others the nearest reconstructed frame's")
        cmap = CoverageMap(video, device, state=False)
        probe_resolution = int(max(int(v) for v in video["grid_wh"]))
        extra = {"views": nums, "view_source_frames": [int(rec_frame[r]) for r in rec_of]}
        del video
    else:
        z = np.load(probe_npz)
        if [_frame_no(p) for p in z["paths"]] != nums:
            raise CubePlanError("the probe's views are not this session's keyframes and turn frames in frame order")
        c2w = np.asarray(z["c2w"], np.float64)
        depth = np.asarray(z["depth"], np.float32)
        valid = _omega_valid(np.asarray(z["conf"]), depth, device)
        cmap = CoverageMap({"rec_depth": np.where(valid, depth, 0.0).astype(np.float32),
                            "rec_K": np.asarray(z["K"], np.float64), "rec_c2w": c2w, "stride": 1},
                           device, state=False)
        rec_of = list(range(len(nums)))
        probe_resolution = int(z["resolution"])
    unit_recs = [sorted({rec_of[slot[kf[k]]] for k in u["keyframes"]} | {rec_of[slot[t]] for t in u["turn"]})
                 for u in units]
    kf_recs = [rec_of[slot[f]] for f in kf]
    tol, _noise = cmap.tolerance([(a, b) for a, b in zip(kf_recs[:-1], kf_recs[1:]) if a != b], log)
    kf_c2w = c2w[[slot[f] for f in kf]]
    Ct = torch.as_tensor(kf_c2w[:, :3, 3], dtype=torch.float64, device=_device(device))
    path = float(torch.linalg.norm(Ct[1:] - Ct[:-1], dim=1).sum()) if len(kf) > 1 else 0.0
    walk_m = float(inputs["walk_m"])
    if not (path > 0 and walk_m > 0):
        raise CubePlanError(f"the probe's keyframe path {path} or the session's walk {walk_m} m is not positive — no "
                            f"metre to size the cubes with")
    mpu = walk_m / path
    cube_m = float(inputs.get("cube_m", CUBE_M))
    centres = np.stack([kf_c2w[u["keyframes"], :3, 3].mean(0) for u in units])
    cov = unit_covisibility(cmap, unit_recs, centres, 2.0 * cube_m / mpu, tol)
    kf_cov = keyframe_covisibility(cmap, kf, kf_recs)
    del cmap
    plan = plan_cubes(kf_c2w, units, cov, metres_per_unit=mpu, cube_m=cube_m, max_views=int(inputs["max_views"]),
                      device=device)
    ctx = cube_extras(plan["cubes"], kf, turns)
    wit = cube_extras(plan["cubes"], kf, inputs.get("witness") or [])
    for cb, cx, wx in zip(plan["cubes"], ctx, wit):
        cb["context"] = cx
        cb["witnesses"] = wx
        cb["omega_views"] = len(cb["keyframes"]) + len(cx)
    plan.update({"units": units, "unit_records": unit_recs, "n_keyframes": len(kf), "n_turn_frames": len(turns),
                 "probe_resolution": probe_resolution, "kf_slot": [slot[f] for f in kf],
                 "kf_centres": kf_c2w[:, :3, 3].tolist(), "probe_source": source, "tol_rel": tol,
                 "kf_covisibility": kf_cov, **extra})
    st = plan["stats"]
    sizes = [cb["omega_views"] for cb in plan["cubes"]]
    half = int(inputs["max_views"]) // 2
    log(f"[cubes] {len(units)} units ({len(kf)} keyframes, {len(turns)} turn frames) → {len(plan['cubes'])} cube(s): "
        f"own views {st['own_views_min']}–{st['own_views_max']} (the views first: at least {half} when the walk has "
        f"them), {min(sizes)}–{max(sizes)} Omega views with the shared ones (cap {plan['max_views']}); "
        f"{st['groups_grown']} group(s) grown by what the units see within {cube_m:g} m of displacement "
        f", {st['merges_for_views']} merge(s) of small groups within it; a unit in {st['memberships_per_unit']:.2f} "
        f"cube(s); "
        f"{st['walk_joins']} walk join(s)"
        + (f"; {st['small_left']} cube(s) under {half} own views (nothing left to join)" if st["small_left"] else "")
        + (f"; over the cap by a walk join: {st['over_cap_joins']}" if st["over_cap_joins"] else ""))
    at = {f: i for i, f in enumerate(kf)}
    gap = np.asarray([at[b] - at[a] for a, b, _c, _d in kf_cov["pairs"]], np.int64)
    log(f"[cubes] what each keyframe sees of every other: {len(kf_cov['pairs'])} pair(s) of keyframes at c ≥ {TAU:g} "
        f"either way on one surface (within {kf_cov['band_rel'] * 100:g} % of the depth)"
        + (f", the farthest {int(gap.max())} keyframes apart along the walk" if len(gap) else "")
        + " — what F4 tracks across and F6's vote hears")
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
              Path(__file__).with_name("coverage_fill.py"),
              Path(__file__).with_name("amb3r_probe.py")):
        h.update(p.read_bytes())
    return h.hexdigest()


def session_cubes(scan_dir: Path, frames_dir: Path, sel_files: Sequence[str], turn_files: Sequence[str],
                  witness_files: Sequence[str], cfg_v: dict, *, resolution: int, max_views: int,
                  sky_cache: Optional[Path], log: Callable[[str], None], probe: str = "omega",
                  cube_m: float = CUBE_M, walk_m: float = 0.0) -> Dict:
    """The session's cube plan on the STRUCTURE PROBE — ``probe`` "amb3r": AMB3R-SLAM's whole-video pass
    (intake/amb3r_video.npz, run right after the intake's keyframes — workers/map_worker._run_intake_selection —
    and the keyframes completed on it, reconstruction/coverage_fill.py); "omega": one Omega pass over every keyframe
    and turn
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
