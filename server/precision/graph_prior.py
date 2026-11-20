"""F5 CARRIES THE CUBE GRAPH (USER 2026-11-20, docs/pipeline_checklist.md line 4; the audit: "F5 does not carry the
cube graph: where the tracks say nothing it moves the cameras freely").

What the cube stage knows about the cameras, held inside F5's cost as RELATIVE-POSE edges between keyframes:

  - EVERY CUBE IS ONE INFERENCE: its keyframes stand in known relative poses. Each keyframe of a cube is tied to the
    cube's middle keyframe (a star per cube). The cubes overlap, so a keyframe two cubes share is tied to both: the
    seams are carried by the stars themselves. How well: as two cubes agree on what they share — the residual of the
    cube's own seams (their median), the number the cube graph itself weighs them with.
  - THE CLOSURES tie keyframes of different passes: the two keyframes of the revisit, as well as the closure's own
    fit says and as far as the graph's solution honours it — sigma = hypot(its fit's residual, what it disagrees by
    with the solved graph at its surfaces). A bridge the graph could not honour ties nothing; one it honours ties
    with its own weight. No threshold.

For an edge (k, l) the REFERENCE is the relative pose of the state F5 starts from (the solved graph, at the session's
scale): F5 may move any camera where the images ask for it, against what the graph knows of that camera's place among
its neighbours. Translation is weighed in metres (sigma), rotation in radians (sigma / lever): the graph's own
weights (loop_utils.chunk_graph.edge_residual).

No camera has a prior of its own any more (until this date: one independent position prior per keyframe, sigma grown
along the walk — an open chain that knew nothing of the closures). Where the images say nothing a camera stays with
its cube and its closures, and the walk can neither stretch nor shrink. The gauge is one held keyframe.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

KIND_CUBE, KIND_CLOSURE = 0, 1
KIND_NAMES = {KIND_CUBE: "within a cube", KIND_CLOSURE: "closure"}
GRAPH_NAME, LAYOUT_NAME = "chunk_graph.json", "chunk_sim3.json"


class GraphPriorError(RuntimeError):
    pass


def _vendor():
    import sys
    p = str(Path(__file__).resolve().parents[2] / "vendor" / "VGGT-Long")
    if p not in sys.path:
        sys.path.insert(0, p)
    from loop_utils import chunk_graph as CG
    return CG


def cube_sigmas(n_cubes: int, seams: Sequence[dict]) -> Dict[int, tuple]:
    """Per cube (sigma_m, lever_m): the medians over the seams the cube takes part in — how well it and its
    neighbours agree on what they share, and the arm that agreement was measured at."""
    by: Dict[int, list] = {c: [] for c in range(int(n_cubes))}
    for e in seams:
        for c in (int(e["a"]), int(e["b"])):
            by[c].append((float(e["sigma_m"]), float(e["lever_m"])))
    out = {}
    for c, v in by.items():
        if v:
            a = np.asarray(v, np.float64)
            out[c] = (float(np.median(a[:, 0])), float(np.median(a[:, 1])))
    return out


def graph_edges(maplong_dir, n_keyframes: int) -> Dict[str, np.ndarray]:
    """The edges F5 holds, from what the cube run left (``chunk_sim3.json``: which keyframes each cube holds;
    ``chunk_graph.json``: the seams, the closures and the solved poses). Returns arrays over the edges:
    ``k``, ``l`` (keyframe positions), ``sigma_m``, ``lever_m``, ``kind``; and ``held`` — the keyframe that fixes
    the gauge (the middle keyframe of the first cube)."""
    d = Path(maplong_dir)
    for name in (GRAPH_NAME, LAYOUT_NAME):
        if not (d / name).exists():
            raise GraphPriorError(f"{d / name} is missing — F5 holds the cube graph and the cube run did not leave it")
    layout = json.loads((d / LAYOUT_NAME).read_text())
    graph = json.loads((d / GRAPH_NAME).read_text())
    sets = [sorted(int(x) for x in s) for s in layout["chunk_sets"]]
    N = int(n_keyframes)
    if len(layout["owner"]) != N:
        raise GraphPriorError(f"{d / LAYOUT_NAME} places {len(layout['owner'])} keyframes, the session has {N}")
    seen = sorted(set(x for s in sets for x in s))
    if seen != list(range(N)):
        raise GraphPriorError(f"the cubes of {d / LAYOUT_NAME} do not hold every keyframe 0 … {N - 1} exactly")
    sig = cube_sigmas(len(sets), graph.get("seams") or [])
    k, l, s_m, lev, kind = [], [], [], [], []
    if len(sets) == 1 and not sig:
        raise GraphPriorError("a session of ONE cube has no seam: nothing measured how well its keyframes stand "
                              "together, so F5's graph cannot be weighed")
    for c, members in enumerate(sets):
        if c not in sig:
            raise GraphPriorError(f"cube {c} takes part in no seam — its own agreement was not measured")
        hub = members[len(members) // 2]
        for m in members:
            if m != hub:
                k.append(hub); l.append(m); s_m.append(sig[c][0]); lev.append(sig[c][1]); kind.append(KIND_CUBE)
    closures = graph.get("closures") or []
    if closures:
        CG = _vendor()
        poses = [(float(p["s"]), np.asarray(p["R"], np.float64), np.asarray(p["t"], np.float64)) for p in graph["poses"]]
        for e in closures:
            i, j = (int(x) for x in e["revisit"])
            if not (0 <= i < N and 0 <= j < N) or i == j:
                raise GraphPriorError(f"closure {e['a']}<->{e['b']} names keyframes {i}, {j} outside 0 … {N - 1}")
            miss = float(CG.edge_surface_error_m(poses[int(e["a"])], poses[int(e["b"])], e))
            k.append(i); l.append(j); s_m.append(float(np.hypot(float(e["sigma_m"]), miss)))
            lev.append(float(e["lever_m"])); kind.append(KIND_CLOSURE)
    out = {"k": np.asarray(k, np.int64), "l": np.asarray(l, np.int64), "sigma_m": np.asarray(s_m, np.float64),
           "lever_m": np.asarray(lev, np.float64), "kind": np.asarray(kind, np.int64),
           "held": np.asarray([sets[0][len(sets[0]) // 2]], np.int64)}
    if not (np.all(np.isfinite(out["sigma_m"])) and np.all(out["sigma_m"] > 0) and np.all(out["lever_m"] > 0)):
        raise GraphPriorError("an edge of the cube graph came with a sigma or a lever that is not a positive number")
    return out


def summary(edges: Dict[str, np.ndarray]) -> dict:
    out = {"held_keyframe": int(edges["held"][0]), "edges": int(len(edges["k"]))}
    for kind, name in KIND_NAMES.items():
        sel = edges["kind"] == kind
        out[name] = ({"edges": int(sel.sum()), "sigma_m": {"median": float(np.median(edges["sigma_m"][sel])),
                                                           "min": float(edges["sigma_m"][sel].min()),
                                                           "max": float(edges["sigma_m"][sel].max())}}
                     if sel.any() else {"edges": 0})
    return out


# ── the edges inside the solver (torch; R, t: world -> camera, the solver's own state) ───────────────────────────────
def so3_log(R):
    """Rotation matrices (…, 3, 3) -> rotation vectors (…, 3); the series below the solver's small angle."""
    import torch
    from precision.ba_torch import SMALL_ANGLE
    v = torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1) / 2
    c = ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]) - 1) / 2
    th = torch.atan2(torch.linalg.norm(v, dim=-1), c)
    small = th < SMALL_ANGLE
    th_safe = torch.where(small, torch.ones_like(th), th)
    f = torch.where(small, 1 + th ** 2 / 6, th_safe / torch.sin(th_safe))
    return v * f[..., None]


def left_jacobian_inverse(phi):
    """J_l^-1(phi) of SO(3): Log(exp(a) exp(phi)) = phi + J_l^-1(phi) a to first order in a."""
    import torch
    from precision.ba_torch import SMALL_ANGLE, skew
    th = torch.linalg.norm(phi, dim=-1)
    small = th < SMALL_ANGLE
    th_safe = torch.where(small, torch.ones_like(th), th)
    half = th_safe / 2
    c = torch.where(small, 1 / 12 + th ** 2 / 720, (1 - half * torch.cos(half) / torch.sin(half)) / th_safe ** 2)
    K = skew(phi)
    I = torch.eye(3, dtype=phi.dtype, device=phi.device).expand(K.shape)
    return I - K / 2 + c[..., None, None] * (K @ K)


def relative(R, t, k, l):
    """(R_kl, t_kl) of the edges: camera k's frame seen from camera l — p_l = R_kl p_k + t_kl."""
    Rkl = R[l] @ R[k].transpose(-1, -2)
    return Rkl, t[l] - (Rkl @ t[k][..., None])[..., 0]


def edge_terms(R, t, k, l, R0, t0, sig_t, sig_r, with_jac: bool = True):
    """The whitened residuals of the edges at (R, t) and their Jacobians in the solver's unknowns (per camera: a
    rotation step dw applied on the LEFT of R, then a translation step dt).

    Returns e (E, 6) = [translation (m) / sig_t, rotation (rad) / sig_r] against the reference (R0, t0), and — with
    ``with_jac`` — Jk, Jl (E, 6, 6): d e / d [dw, dt] of camera k and of camera l."""
    import torch
    from precision.ba_torch import skew
    Rkl, tkl = relative(R, t, k, l)
    phi = so3_log(Rkl @ R0.transpose(-1, -2))
    e = torch.cat([(tkl - t0) / sig_t[:, None], phi / sig_r[:, None]], -1)
    if not with_jac:
        return e, None, None
    E = int(k.shape[0])
    Jk = torch.zeros(E, 6, 6, dtype=R.dtype, device=R.device)
    Jl = torch.zeros(E, 6, 6, dtype=R.dtype, device=R.device)
    st, sr = sig_t[:, None, None], sig_r[:, None, None]
    Jk[:, :3, :3] = -(Rkl @ skew(t[k])) / st
    Jk[:, :3, 3:] = -Rkl / st
    Jl[:, :3, :3] = skew((Rkl @ t[k][..., None])[..., 0]) / st
    Jl[:, :3, 3:] = torch.eye(3, dtype=R.dtype, device=R.device).expand(E, 3, 3) / st
    Jinv = left_jacobian_inverse(phi)
    Jk[:, 3:, :3] = -(Jinv @ Rkl) / sr
    Jl[:, 3:, :3] = Jinv / sr
    return e, Jk, Jl
