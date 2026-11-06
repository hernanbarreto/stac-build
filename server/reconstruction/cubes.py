"""THE CUBES (USER 2026-11-06): the scene is cut the way GenRecon cuts it — cubes in space — from a first Omega pass that
is not one of precision ("la primer pasada de omega que no es de precisión es para saber cómo cortar"): the keyframes
and, in the turns, the turn frames; cubes of at most 64 images, each sharing 50 % with its neighbours; a place the walk
came back to is ONE place, and several floors are cut the same way ("si tuviera varios pisos lo mismo debería armarse
igual").

1. The STRUCTURE PROBE (reconstruction/zones.structure_probe): ONE VGGT-Ω pass over every keyframe and every turn frame
   at the resolution where they all fit the card; its co-visibility (walk_structure.covisibility) says who sees whom.
2. UNITS: a parallax keyframe with the rotation keyframes it stands for and the turn frames after them — the view cap's
   unit since 2026-11-04 (chunk_covis.planning_views): a turn is never cut.
3. The GRAVITY FRAME: up = −(the sum of the keyframes' image-down axes), the floor plan's axes the principal axes of the
   units' positions on the plane ⊥ up (signs fixed by the walk's start → end).
4. CELLS: the space of the units' positions cut recursively at the weighted median (views) of its longest side until a
   cell holds at most half the cap (32 views) — a k-d tree, so the cells are boxes that tile the scene: small where the
   walk lingered, large where it passed once; a revisit falls in the cell of the first visit, a floor above in its own
   cells. A unit is never split; a unit alone over half the cap is a cell over it (a turn dense with views), declared.
5. BLOCKS: the units of a cell that see each other (co-visibility ≥ τ, or consecutive along the walk) — a cell across a
   wall or across two floors is two blocks.
6. CUBES: two blocks that are NEIGHBOURS — their cells touch face to face and they see each other, or they are
   consecutive along the walk — are one cube (≤ 64 views); two cubes that share a block share 50 % of their views, the
   exact seam the chunk graph joins them on. One block alone (a session of at most 32 views) is one cube.
7. OWNER: a block's keyframes take their geometry from ONE of its cubes — the cube that pairs it with the neighbour it
   sees most; the others measure the seams.

A cube's Omega views are its keyframes and the turn frames between two of its keyframes that are consecutive in the
walk; its F6 witnesses (the interval frames) follow the same rule. Deterministic: no random draw, every order with a
tie-break by index; the co-visibility runs on the card (walk_structure.covisibility), the rest is bookkeeping over a
few hundred units."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

MAX_CUBE_VIEWS = 64          # USER 2026-11-04 ("ese es el límite") / 2026-11-06 ("siempre con tope de 64")
TAU = 0.3                    # the co-visibility bar (walk_structure.TAU, chunk_covis.TAU)
CUBES_NAME = "cubes.json"
CUBES_VERSION = 1
_BOX_EPS = 1e-9


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

def gravity_frame(c2w_kf: np.ndarray) -> np.ndarray:
    """Rows e1, e2, up of the floor-plan frame (orthonormal, right-handed): up = −Σ image-down axes (OpenCV +y); e1 =
    the principal horizontal axis of the camera centres, pointing from the walk's start to its end (else its largest
    component positive); e2 = up × e1."""
    T = np.asarray(c2w_kf, np.float64)
    down = T[:, :3, 1].sum(0)
    if not np.isfinite(down).all() or np.linalg.norm(down) < 1e-9:
        raise CubePlanError("the keyframes' image-down axes cancel out — no gravity direction")
    up = -down / np.linalg.norm(down)
    C = T[:, :3, 3]
    H = C - np.outer(C @ up, up)
    H = H - H.mean(0)
    _, V = np.linalg.eigh(H.T @ H)
    e1 = V[:, -1] - (V[:, -1] @ up) * up
    if np.linalg.norm(e1) < 1e-9:                       # every camera on one vertical line
        a = np.eye(3)[int(np.argmin(np.abs(up)))]
        e1 = a - (a @ up) * up
    e1 = e1 / np.linalg.norm(e1)
    s = float((C[-1] - C[0]) @ e1)
    if abs(s) < 1e-12:
        s = float(e1[int(np.argmax(np.abs(e1)))])
    if s < 0:
        e1 = -e1
    e2 = np.cross(up, e1)
    return np.stack([e1, e2 / np.linalg.norm(e2), up])


# ── 4. cells ──────────────────────────────────────────────────────────────────────────────────

def kd_cells(P: np.ndarray, w: np.ndarray, half: int) -> List[Dict]:
    """The k-d cells of the units at positions ``P`` (U, 3) with views ``w``: a node over ``half`` views is cut on its
    longest side (ties → the lower axis) at the weighted median between two distinct coordinates (ties of the cut →
    the earlier); a node that cannot be cut (one unit, or every unit at one point) is a leaf over the cap. Each cell:
    its units (sorted) and its box [lo, hi] — the boxes tile the root box."""
    P = np.asarray(P, np.float64)
    w = np.asarray(w, np.int64)
    lo0, hi0 = P.min(0), P.max(0)
    pad = np.maximum(1e-6, 1e-3 * np.maximum(hi0 - lo0, 1e-9))
    out: List[Dict] = []
    stack = [(np.arange(len(P)), lo0 - pad, hi0 + pad)]
    while stack:
        ids, lo, hi = stack.pop()
        tot = int(w[ids].sum())
        cut = None
        if tot > half and len(ids) > 1:
            ext = P[ids].max(0) - P[ids].min(0)
            for ax in sorted(range(3), key=lambda a: (-ext[a], a)):
                if ext[ax] <= 0:
                    continue
                xs = P[ids, ax]
                o = np.lexsort((ids, xs))
                sid, sx = ids[o], xs[o]
                cw = np.cumsum(w[sid])
                cand = np.nonzero(sx[:-1] < sx[1:])[0]
                if len(cand) == 0:
                    continue
                i = int(cand[int(np.argmin(np.abs(cw[cand] - tot / 2.0)))])
                cut = (ax, 0.5 * (sx[i] + sx[i + 1]), np.sort(sid[:i + 1]), np.sort(sid[i + 1:]))
                break
        if cut is None:
            out.append({"units": [int(u) for u in np.sort(ids)], "lo": lo.tolist(), "hi": hi.tolist(),
                        "views": tot, "over_cap": bool(tot > half)})
            continue
        ax, x, left, right = cut
        hiL, loR = hi.copy(), lo.copy()
        hiL[ax], loR[ax] = x, x
        stack.append((right, loR, hi))
        stack.append((left, lo, hiL))           # popped first: the cells come out in a fixed order
    return out


def _touch(a: Dict, b: Dict) -> bool:
    """Two boxes touch face to face: equal on one side along one axis, overlapping with positive length on the others."""
    alo, ahi, blo, bhi = (np.asarray(x, np.float64) for x in (a["lo"], a["hi"], b["lo"], b["hi"]))
    n_face, n_over = 0, 0
    for ax in range(3):
        if abs(ahi[ax] - blo[ax]) <= _BOX_EPS or abs(bhi[ax] - alo[ax]) <= _BOX_EPS:
            n_face += 1
        elif min(ahi[ax], bhi[ax]) - max(alo[ax], blo[ax]) > _BOX_EPS:
            n_over += 1
    return n_face == 1 and n_over == 2


# ── 5.–7. blocks, cubes, owners ───────────────────────────────────────────────────────────────

def _components(units: Sequence[int], cu: np.ndarray, tau: float) -> List[List[int]]:
    left = sorted(int(u) for u in units)
    out = []
    while left:
        comp, todo = {left[0]}, [left[0]]
        while todo:
            u = todo.pop()
            for v in left:
                if v not in comp and (abs(u - v) == 1 or cu[u, v] >= tau):
                    comp.add(v); todo.append(v)
        out.append(sorted(comp))
        left = [u for u in left if u not in comp]
    return out


def plan_cubes(c2w_kf: np.ndarray, units: List[Dict], cu: np.ndarray, *, max_views: int = MAX_CUBE_VIEWS,
               tau: float = TAU) -> Dict:
    """The cube plan of ``units`` (units_of) from the probe's keyframe cameras ``c2w_kf`` (n, 4, 4) and the units'
    co-visibility ``cu`` (U, U): the strongest co-visibility between any view of one and any view of the other."""
    U = len(units)
    if U == 0:
        raise CubePlanError("no unit to plan")
    half = int(max_views) // 2
    R = gravity_frame(c2w_kf)
    C = np.asarray(c2w_kf, np.float64)[:, :3, 3] @ R.T
    P = np.stack([C[u["keyframes"]].mean(0) for u in units])
    w = np.asarray([int(u["views"]) for u in units], np.int64)
    cells = kd_cells(P, w, half)
    blocks: List[Dict] = []
    for ci, cell in enumerate(cells):
        for comp in _components(cell["units"], cu, tau):
            blocks.append({"units": comp, "cell": ci, "views": int(w[comp].sum()),
                           "keyframes": [k for u in comp for k in units[u]["keyframes"]]})
    blocks.sort(key=lambda b: (min(b["keyframes"]), b["cell"]))
    unit_block = np.full(U, -1, np.int64)
    for bi, b in enumerate(blocks):
        unit_block[b["units"]] = bi
    B = len(blocks)
    link = np.zeros((B, B), np.float64)
    for a in range(B):
        ua = blocks[a]["units"]
        for b in range(a + 1, B):
            link[a, b] = link[b, a] = float(cu[np.ix_(ua, blocks[b]["units"])].max())
    edges: Dict[Tuple[int, int], List[str]] = {}
    for a in range(B):
        for b in range(a + 1, B):
            if link[a, b] >= tau and _touch(cells[blocks[a]["cell"]], cells[blocks[b]["cell"]]):
                edges.setdefault((a, b), []).append("neighbour cells that see each other")
    for u in range(U - 1):
        a, b = sorted((int(unit_block[u]), int(unit_block[u + 1])))
        if a != b:
            edges.setdefault((a, b), []).append("consecutive along the walk")
    if B == 1:
        cubes = [{"blocks": [0]}]
    else:
        cubes = [{"blocks": [a, b], "why": sorted(set(why)), "link": float(link[a, b])}
                 for (a, b), why in edges.items()]
    for cb in cubes:
        cb["keyframes"] = sorted(k for bi in cb["blocks"] for k in blocks[bi]["keyframes"])
        cb["views"] = int(sum(blocks[bi]["views"] for bi in cb["blocks"]))
        cb["over_cap"] = bool(cb["views"] > int(max_views))
    cubes.sort(key=lambda cb: (cb["keyframes"][0], cb["blocks"]))
    # connected (the walk joins every block to the next); a block in no cube cannot happen — said if it does
    seen = {bi for cb in cubes for bi in cb["blocks"]}
    if seen != set(range(B)):
        raise CubePlanError(f"block(s) {sorted(set(range(B)) - seen)} in no cube")
    owner_block = []
    for bi in range(B):
        mine = [(ci, cb) for ci, cb in enumerate(cubes) if bi in cb["blocks"]]
        best = max(mine, key=lambda t: (max([link[bi, o] for o in t[1]["blocks"] if o != bi] or [1.0]), -t[0]))
        owner_block.append(int(best[0]))
    n_kf = int(sum(len(u["keyframes"]) for u in units))
    owner = np.full(n_kf, -1, np.int64)
    for bi, b in enumerate(blocks):
        owner[b["keyframes"]] = owner_block[bi]
    if (owner < 0).any():
        raise CubePlanError("a keyframe has no owner cube")
    return {"gravity_frame": R.tolist(), "max_views": int(max_views), "half": half, "tau": float(tau),
            "unit_position": P.tolist(), "cells": cells, "blocks": blocks, "unit_block": unit_block.tolist(),
            "block_link": link.tolist(), "cubes": cubes, "owner_block": owner_block, "owner": owner.tolist()}


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

def analyse(probe_npz: Path, inputs: Dict, *, device: str = "cuda", log: Callable[[str], None] = print) -> Dict:
    """The probe's co-visibility on the card and the cube plan. ``inputs``: keyframe and turn-frame numbers, the plan
    index, the witness frames, the cap."""
    from reconstruction.walk_structure import covisibility
    z = np.load(probe_npz)
    paths = [str(p) for p in z["paths"]]
    nums = [_frame_no(p) for p in paths]
    kf = [int(f) for f in inputs["keyframes"]]
    turns = [int(f) for f in inputs["turn"]]
    if sorted(nums) != nums or set(nums) != set(kf) | set(turns) or len(nums) != len(kf) + len(turns):
        raise CubePlanError("the probe's views are not this session's keyframes and turn frames in frame order")
    slot = {f: i for i, f in enumerate(nums)}
    kf_slot = [slot[f] for f in kf]
    # the tolerance on consecutive KEYFRAMES (the session's own disagreement between two keyframes on one surface)
    cv = covisibility(z["c2w"], z["K"], z["depth"], z["conf"], device=device, log=log,
                      tol_pairs=list(zip(kf_slot[:-1], kf_slot[1:])))
    c = cv["c"]
    units = units_of(kf, inputs["plan_index"], turns)
    uviews = [[kf_slot[k] for k in u["keyframes"]] + [slot[t] for t in u["turn"]] for u in units]
    Uc = len(units)
    cu = np.zeros((Uc, Uc), np.float64)
    for a in range(Uc):
        for b in range(a + 1, Uc):
            cu[a, b] = cu[b, a] = float(c[np.ix_(uviews[a], uviews[b])].max())
    plan = plan_cubes(z["c2w"][kf_slot], units, cu, max_views=int(inputs["max_views"]))
    ctx = cube_extras(plan["cubes"], kf, turns)
    wit = cube_extras(plan["cubes"], kf, inputs.get("witness") or [])
    for cb, cx, wx in zip(plan["cubes"], ctx, wit):
        cb["context"] = cx
        cb["witnesses"] = wx
        cb["omega_views"] = len(cb["keyframes"]) + len(cx)
    plan.update({"units": units, "tol_rel": cv["tol_rel"], "n_keyframes": len(kf), "n_turn_frames": len(turns),
                 "probe_resolution": int(z["resolution"]), "covisibility": c, "kf_slot": kf_slot})
    sizes = [cb["omega_views"] for cb in plan["cubes"]]
    log(f"[cubes] {len(units)} units ({len(kf)} keyframes, {len(turns)} turn frames) → {len(plan['cells'])} cells, "
        f"{len(plan['blocks'])} blocks, {len(plan['cubes'])} cube(s) of {min(sizes)}–{max(sizes)} Omega views "
        f"(cap {plan['max_views']})"
        + (f"; over the cap (a turn dense with views kept whole): "
           f"{[i for i, cb in enumerate(plan['cubes']) if cb['over_cap']]}"
           if any(cb["over_cap"] for cb in plan["cubes"]) else ""))
    return plan


def figure(plan: Dict, c2w_kf: np.ndarray, out_png: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    R = np.asarray(plan["gravity_frame"])
    C = np.asarray(c2w_kf, np.float64)[:, :3, 3] @ R.T
    fig, ax = plt.subplots(1, 2, figsize=(17, 8), gridspec_kw={"width_ratios": [3, 1.2]})
    ax[0].plot(C[:, 0], C[:, 1], "-", color="0.8", lw=0.8, zorder=1)
    cols = plt.cm.tab20(np.arange(max(1, len(plan["blocks"]))) % 20)
    for cell in plan["cells"]:
        lo, hi = cell["lo"], cell["hi"]
        ax[0].add_patch(Rectangle((lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1], fill=False, ec="0.45", lw=0.8,
                                  ls="--" if cell["over_cap"] else "-", zorder=2))
    cen = []
    for bi, b in enumerate(plan["blocks"]):
        P = C[b["keyframes"]]
        ax[0].scatter(P[:, 0], P[:, 1], s=10, color=cols[bi], zorder=3)
        cen.append(P.mean(0))
    for ci, cb in enumerate(plan["cubes"]):
        if len(cb["blocks"]) == 2:
            a, b = cb["blocks"]
            ax[0].plot([cen[a][0], cen[b][0]], [cen[a][1], cen[b][1]], "-", color="k", lw=1.2, alpha=0.6, zorder=4)
            m = 0.5 * (cen[a] + cen[b])
            ax[0].text(m[0], m[1], str(ci), fontsize=7, ha="center", va="center", zorder=5,
                       bbox=dict(boxstyle="round,pad=0.15", fc="w", ec="none", alpha=0.8))
    ax[0].scatter(C[0, 0], C[0, 1], marker="*", s=220, c="g", zorder=6, label="start")
    ax[0].set_aspect("equal")
    ax[0].legend(fontsize=8)
    ax[0].set_title(f"{title}: {len(plan['cells'])} cells, {len(plan['blocks'])} blocks, {len(plan['cubes'])} cubes "
                    f"(≤ {plan['max_views']} views, each pair of neighbour blocks one cube) — from above")
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
    for p in (Path(__file__), Path(__file__).with_name("walk_structure.py")):
        h.update(p.read_bytes())
    return h.hexdigest()


def session_cubes(scan_dir: Path, frames_dir: Path, sel_files: Sequence[str], turn_files: Sequence[str],
                  witness_files: Sequence[str], cfg_v: dict, *, resolution: int, max_views: int,
                  sky_cache: Optional[Path], log: Callable[[str], None]) -> Dict:
    """The session's cube plan: the STRUCTURE PROBE (one Omega pass over every keyframe and every turn frame at
    ``resolution``, vendor/VGGT-Long/stac_frames_pass.py; reused when the views, the resolution, the pass's code and the
    weights are the same — its stamp beside it) and the PLAN (a process of its own on the card, ``python -m
    reconstruction.cubes``; reused when the probe, the inputs and this code are the same). intake/structure_probe.npz,
    intake/cubes.json and intake/cubes.png."""
    import subprocess
    import sys
    import repro
    from reconstruction.chunk_covis import planning_indices
    from reconstruction.zones import PASS_SCRIPT, _sha_file, run_frames_pass
    intake = Path(scan_dir) / "intake"
    intake.mkdir(parents=True, exist_ok=True)
    views = sorted(set(sel_files) | set(turn_files), key=_frame_no)
    npz, stamp_p = intake / "structure_probe.npz", intake / "structure_probe.stamp.json"
    stamp = {"views": hashlib.sha256("\n".join(views).encode()).hexdigest(), "resolution": int(resolution),
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
        log(f"[cubes] the structure probe: ONE Omega pass over {len(sel_files)} keyframes + {len(turn_files)} turn "
            f"frames at {resolution} — not one of precision, the cut is decided on it")
        run_frames_pass([str(Path(frames_dir) / f) for f in views], int(resolution), npz, cfg_v, sky_cache=sky_cache,
                        salad=False, log=log)
        tmp = stamp_p.with_name(stamp_p.name + ".tmp")
        tmp.write_text(json.dumps(stamp, indent=1))
        os.replace(tmp, stamp_p)
    kf = [_frame_no(f) for f in sel_files]
    inputs = {"keyframes": kf, "turn": sorted(_frame_no(f) for f in turn_files),
              "witness": sorted(_frame_no(f) for f in witness_files),
              "plan_index": [int(i) for i in planning_indices(Path(scan_dir), kf)], "max_views": int(max_views)}
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
    plan = analyse(Path(a.probe), inputs, device=a.device)
    z = np.load(a.probe)
    figure(plan, z["c2w"][plan["kf_slot"]], Path(a.png), a.title)
    plan["stamp"] = inputs.get("stamp")
    plan["version"] = CUBES_VERSION
    save(plan, Path(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
