"""THE CUBES (USER 2026-11-07, night): "los cubos son de 5 m, no los decide nadie … y el solape con los cubos vecinos
es el 50 % de los kf que hay en los cubos"; and ONE Omega pass holds 64 views at most (USER 2026-11-04: "ese es el
límite"; 2026-11-08: "corremos todo así como vos proponés" — pccr's cubes of 5 m with no view cap, 94–358 keyframes and
up to 385 Omega views each, did not hold together: seams 5.9 cm median / 15.8 max after the graph where cubes of ≤ 64
views closed to 0.0–0.6 cm, the floor duplicated 6 cm between two neighbouring cubes, a table's edge bent 4–5 cm by
copies of its keyframes drifting along the walk).

1. UNITS: a parallax keyframe with the rotation keyframes it stands for and the turn frames after them — a turn is never
   cut (units_of).
2. A CUBE: along the walk, the units in order while their cameras (each unit's mean camera) fit a box of ``cube_m`` on
   every axis of the gravity frame — the displacement of ONE Omega pass (USER 2026-11-07: "a más de 5 m omega deforma")
   — AND its views (keyframes + turn frames) stay within ``max_views`` — the views of ONE Omega pass. A cube takes at
   least two units (a unit alone over the cap is never cut — Omega sees all of it, declared ``over_cap``), so that two
   neighbours always share a unit.
3. THE OVERLAP: the next cube starts at the unit where half of this cube's keyframes lie behind it — two neighbouring
   cubes share half their keyframes, the seams that join them in ONE Sim(3) graph.
4. THE OWNER of a unit's keyframes is the cube where the unit lies deepest (farthest from both of its ends; ties → the
   earlier cube).
Earlier cuts (versions 1–6: k-d camera cells, rays against AMB3R's cloud, boxes of seen space, groups by what the units
see with a cap of 64 views by halving; 7: 5 m along the walk with no view cap) are gone. Revisits across cubes are tied
downstream — F4's revisit windows and F6's vote (7).

A cube's box (for the viewer) holds the cameras of all its views. A cube's Omega views are its keyframes and the turn
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
   one surface it does. The test of what a view sees: a point of its record inside the other's image at a consistent
   depth (reconstruction/coverage_fill.CoverageMap's test).
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
CUBES_VERSION = 8            # 8: along the walk, ≤ cube_m AND ≤ max_views views, neighbours sharing half their
                             #    keyframes (USER 2026-11-08); 7: 5 m along the walk, no view cap (2026-11-07, night);
                             #    6: no view cap; 5: + kf_covisibility; 4: groups by what the units see; 3: boxes of
                             #    seen space; 2: rays cast against AMB3R's cloud; 1: k-d camera cells


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


def plan_cubes(c2w_kf: np.ndarray, units: List[Dict], *, metres_per_unit: float, cube_m: float = CUBE_M,
               max_views: int = MAX_CUBE_VIEWS, device: str = "cuda") -> Dict:
    """The cubes of ``units`` (units_of) along the walk (module doc 2–4): a cube takes the units in order while their
    mean cameras fit a box of ``cube_m`` (the probe's metre: ``metres_per_unit``) on every axis of the gravity frame
    and its views stay within ``max_views`` — at least two units, so that neighbours share one; the next cube starts at
    the unit where half of this cube's keyframes lie behind it; a unit's owner is the cube where it lies deepest."""
    U = len(units)
    if U == 0:
        raise CubePlanError("no unit to plan")
    if int(max_views) < 2:
        raise CubePlanError(f"max_views {max_views}: a cube holds at least two views")
    R = gravity_frame(c2w_kf, device)
    Cg = np.asarray(c2w_kf, np.float64)[:, :3, 3] @ np.asarray(R, np.float64).T
    ucam = [Cg[units[u]["keyframes"]] for u in range(U)]
    centre = np.stack([c.mean(0) for c in ucam])
    size = float(cube_m) / float(metres_per_unit)
    nk = np.asarray([len(u["keyframes"]) for u in units], np.int64)
    w = np.asarray([int(u["views"]) for u in units], np.int64)
    spans: List[Tuple[int, int]] = []
    s0 = 0
    while True:
        lo = hi = centre[s0]
        e, nv = s0, int(w[s0])
        while e + 1 < U:
            nlo, nhi = np.minimum(lo, centre[e + 1]), np.maximum(hi, centre[e + 1])
            fits = not bool(np.any(nhi - nlo > size * (1.0 + 1e-9))) and nv + int(w[e + 1]) <= int(max_views)
            if not fits and e > s0:                       # at least two units: neighbours always share one
                break
            lo, hi, e, nv = nlo, nhi, e + 1, nv + int(w[e + 1])
            if not fits:
                break
        spans.append((s0, e))
        if e == U - 1:
            break
        # half of this cube's keyframes behind the next start (at least one unit shared, always a step forward)
        behind, nxt = 0, e
        for u in range(s0, e + 1):
            if behind * 2 >= int(nk[s0:e + 1].sum()):
                nxt = u
                break
            behind += int(nk[u])
        s0 = min(max(nxt, s0 + 1), e)
    owner_unit = []
    for u in range(U):
        cand = [(min(u - a, b - u), -i) for i, (a, b) in enumerate(spans) if a <= u <= b]
        owner_unit.append(-max(cand)[1])
    cubes: List[Dict] = []
    for i, (a, b) in enumerate(spans):
        units_i = list(range(a, b + 1))
        core = [u for u in units_i if owner_unit[u] == i]
        lo_b, hi_b, mid, hx = cube_box(np.concatenate([ucam[u] for u in units_i]), size)
        o_cams = np.concatenate([ucam[u] for u in core]) if core else np.concatenate([ucam[u] for u in units_i])
        shared = [u for u in units_i if u not in core]
        cubes.append({"lo": (mid - hx).tolist(), "hi": (mid + hx).tolist(), "units": units_i, "core": core,
                      "share": {str(u): (1.0 if u in core else 0.5) for u in units_i},
                      "keyframes": sorted(k for u in units_i for k in units[u]["keyframes"]),
                      "own_views": int(w[core].sum()) if core else 0, "views": int(w[units_i].sum()),
                      "shared_keyframes": int(nk[shared].sum()) if shared else 0,
                      "over_cap": bool(int(w[units_i].sum()) > int(max_views)),
                      "camera_span_m": ((hi_b - lo_b) * float(metres_per_unit)).tolist(),
                      "own_camera_span_m": ((o_cams.max(0) - o_cams.min(0)) * float(metres_per_unit)).tolist(),
                      "size_m": (2.0 * hx * float(metres_per_unit)).tolist()})
    n_kf_all = int(sum(len(u["keyframes"]) for u in units))
    owner = np.full(n_kf_all, -1, np.int64)
    for u, i in enumerate(owner_unit):
        owner[units[u]["keyframes"]] = i
    if (owner < 0).any():
        raise CubePlanError("a keyframe has no owner cube")
    nkc = [len(cb["keyframes"]) for cb in cubes]
    ov = [cubes[i]["shared_keyframes"] / max(len(cubes[i]["keyframes"]), 1) for i in range(len(cubes))]
    return {"gravity_frame": R.tolist(), "max_views": int(max_views), "cube_m": float(cube_m),
            "metres_per_unit": float(metres_per_unit), "cubes": cubes, "owner_unit": owner_unit,
            "owner": owner.tolist(),
            "stats": {"cubes": len(cubes), "keyframes_min": int(min(nkc)), "keyframes_max": int(max(nkc)),
                      "views_min": int(min(cb["views"] for cb in cubes)),
                      "views_max": int(max(cb["views"] for cb in cubes)),
                      "shared_keyframes_frac_min": float(min(ov)), "shared_keyframes_frac_max": float(max(ov)),
                      "memberships_per_unit": float(np.mean([sum(a <= u <= b for a, b in spans) for u in range(U)])),
                      "max_camera_span_m": float(max(max(cb["camera_span_m"][:2]) for cb in cubes))}}


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
    kf_cov = keyframe_covisibility(cmap, kf, kf_recs)
    del cmap
    plan = plan_cubes(kf_c2w, units, metres_per_unit=mpu, cube_m=cube_m, max_views=int(inputs["max_views"]),
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
    log(f"[cubes] {len(units)} units ({len(kf)} keyframes, {len(turns)} turn frames) → {len(plan['cubes'])} cube(s) of "
        f"≤ {cube_m:g} m and ≤ {plan['max_views']} views along the walk, neighbours sharing half their keyframes ({st['shared_keyframes_frac_min'] * 100:.0f}"
        f"–{st['shared_keyframes_frac_max'] * 100:.0f} %): {st['keyframes_min']}–{st['keyframes_max']} keyframes, "
        f"{min(sizes)}–{max(sizes)} Omega views per cube (cap {plan['max_views']}"
        + (f", {sum(cb['over_cap'] for cb in plan['cubes'])} over it: a unit the cut never splits" if any(
            cb["over_cap"] for cb in plan["cubes"]) else "")
        + f"); cameras spanning ≤ {st['max_camera_span_m']:.1f} m; "
        f"a unit in {st['memberships_per_unit']:.2f} cube(s)")
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
    ax[0].set_title(f"{title}: {len(plan['cubes'])} cubes of ≤ {plan['cube_m']:g} m and ≤ {plan['max_views']} views "
                    f"along the walk, neighbours sharing half their keyframes; cameras coloured by the cube that owns "
                    f"them — from above")
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
