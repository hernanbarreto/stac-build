"""Co-visibility-planned chunks (docs/design_adaptive_chunking.md §1) — THE chunk planner of the
chunked Omega path (USER 2026-10-06: always this plan, no switch, no alternative, deterministic:
the same inputs give a bit-identical plan).

VGG-T3 idea: two views are co-visible when their depth maps agree (a depth-consistency check).
VGGT-Motion idea: the walk is priced by how fast the view changes, so static stretches cost
nothing, turns and near-field sideways motion cost a lot.

From the I3 DA3 windows (output/da3_windows, chained into one metric trajectory) this measures,
for every keyframe i, its forward CO-VISIBILITY LENGTH ℓ(i) — how many following keyframes still
share ≥ τ of i's content (majority rule), scanned over the WHOLE walk — and the motion budget
δ(i) = 1/max(ℓ(i), 1). D(a,b) = Σ δ over [a,b) counts "co-visibility lengths" walked.

THE PLAN (design §1.5–1.7, ``plan``): ONE pass when D(0,n) ≤ H; otherwise half-chunk BLOCKS
B_0..B_{m−1} with chunk k = B_k ∪ B_{k+1}, so consecutive chunks share exactly one block (the 50 %
overlap) and chunk k ends where chunk k+2 starts. Every block holds ≥ MIN_CHUNK_FRAMES // 2 frames
and at most H/2 lengths. The boundaries come from an exact DP, choosing lexicographically: fewest
blocks → shallowest worst seam → gentlest worst cut → tightest block budget → leftmost.

THE CARD NEVER SHAPES THE PLAN (USER 2026-10-06: "el plan de chunk no cambia, el tamaño de la
tarjeta no lo podemos cambiar, para que encaje adaptamos la resolución"): no capacity splits or
shrinks a chunk here and the scan horizon is the whole walk; the map worker then chooses Omega's
processing resolution so that the LARGEST planned chunk fits the card
(reconstruction.chunk_plan.omega_resolution_for).

    python -m reconstruction.chunk_covis --session <scan> --plan [--cache f.json] [--out f.json]
    python -m reconstruction.chunk_covis --session <scan> --layout walk:5 --layout walk:15
    python -m reconstruction.chunk_covis --session <scan> --total

``--plan`` prints the plan and persists the plan-independent measurement (ℓ, δ, z̄, θ, tol_rel)
in ``<session>/intake/covis.json`` (or ``--cache``), stamped by its inputs, so a resume replans
from it bit-identically after the window files are deleted. ``--layout`` / ``--total`` are the
calibration readouts (D per chunk of a uniform layout of X metres of chainage, or the walk's D).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TAU = 0.3                  # VGG-T3's published co-visibility threshold (H is calibrated AT this τ)
TOL_QUANTILE = 0.95        # the depth-agreement tolerance is MEASURED: this quantile of the
                           # disagreement between two windows' depth of the same keyframe
SAMPLES = 1500             # points per keyframe for the co-visibility test

# H — co-visibility lengths per chunk. CALIBRATED on the user's visual verdicts; a constant of the
# method, not a configuration and not a user option (USER 2026-10-06). Every D below was measured
# by this module (τ 0.3, tol_rel measured per session, scan horizon 883 keyframes — the WHOLE walk
# on all three scenes, n ≤ 289, so the same numbers as the whole-walk horizon the plan uses):
#   pccr 2026-08-31, uniform chunks of 5 m of chainage, 50 % overlap — VALIDATED GOOD:
#       D per chunk 13.67 / 7.45 / 11.11 / 12.07 / 7.40 / 6.00                  (max 13.674)
#   pccr 2026-08-31, uniform chunks of 15 m, 50 % overlap — FAILED (a 90.5 cm seam, which no
#       single chunk can be blamed for, so both count as failed):
#       D per chunk 32.19 / 18.06                                               (min 18.06)
#   zaragoza 2026-06-03, the whole walk as ONE pass — validated good:            D 5.01
#   observatorio 2026-08-11, the whole walk as ONE pass — validated good:        D 3.39
# lo = the largest D validated good = 13.67, hi = the smallest D that failed = 18.06: lo < hi, so
# co-visibility separates the verdicts, and H = lo — no extrapolation beyond what was seen to work.
H_LENGTHS_PER_CHUNK = 13.67

# THE I3 WINDOW LAYOUT H WAS CALIBRATED AT (review 2026-10-06). ℓ, δ, z̄, θ and tol_rel are read off
# the I3 DA3 windows, and their SIZE is the card's (intake/vram.window_size: the largest window the
# card's total memory holds at native resolution, at most gauge.window_frames; an OOM halves it).
# Every D above was measured on windows of DA3NESTED-GIANT-LARGE-1.1 at native process_res (pccr
# 840, zaragoza 1932, observatorio 1036), window_frames 32 requested, overlap 0.5, sized on an
# NVIDIA A100 80GB PCIe (81 920 MiB, vram_margin_frac 0.15): pccr 22 windows over 289 keyframes,
# zaragoza 60 over 183, observatorio 7 over 71 (the calibration log of 2026-10-06) — 25–27 / 6–7 /
# 17–20 keyframes per window, INFERRED from those counts with intake.walk.plan_windows (the
# calibration's windows.json / walk.json are no longer on disk). Windows of another size (another
# card, an OOM halving) give other depths, poses and tol_rel, hence other D: H is not known to hold
# on them. Every measurement records its own layout (``window_layout``, in covis.json and in the
# plan report) next to this one. DECLARED, not decided: what to do when the two differ (fix the I3
# window size for this measurement, or refuse to plan) is the user's call.
H_CALIBRATION_LAYOUT: Dict[str, Any] = {
    "model_id": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
    "process_res": "native",
    "window_frames_requested": 32,
    "window_overlap_frac": 0.5,
    "card": "NVIDIA A100 80GB PCIe, 81920 MiB",
    # the same card as repro.card_key names it: the card MODEL (name | board MiB — nvidia-smi
    # memory.total — | compute capability); never torch's usable-memory bytes (they moved by 3 MiB
    # across a pod restart of 2026-10-07 on this same card)
    "card_key": "NVIDIA A100 80GB PCIe | 81920 MiB | sm_8.0",
    "vram_margin_frac": 0.15,
    "scenes": {
        "pccr 2026-08-31": {"keyframes": 289, "windows": 22, "process_res": 840,
                            "window_frames_inferred": [25, 27]},
        "zaragoza 2026-06-03": {"keyframes": 183, "windows": 60, "process_res": 1932,
                                "window_frames_inferred": [6, 7]},
        "observatorio 2026-08-11": {"keyframes": 71, "windows": 7, "process_res": 1036,
                                    "window_frames_inferred": [17, 20]},
    },
}

# The shortest chunk, in frames: the existing clamp, declared. It is ≥ the loop-bridge window
# (loop_chunk_size 20) and ≥ the intra-chunk floor (8), and its half — one block, i.e. one seam —
# is 12 frames, above the fork's 8-frame seam-alignment floor (sim3utils.py:958).
MIN_CHUNK_FRAMES = 24

COVIS_NAME = "covis.json"
COVIS_VERSION = 4          # 4: measured on the PARALLAX keyframes only (plan_index; USER
                           #    2026-10-07) — 3: the stamp holds the code; τ margins recorded (2: whole-walk
                           # horizon; 1 scanned up to a card capacity)
LOG_TAG = "[covis]"
_INF = 1 << 62             # an unreachable DP cost (int64-safe with every block cost added)


class CovisError(RuntimeError):
    """The co-visibility inputs cannot be produced (a file missing, a stale cache whose window
    files are gone) — always with the exact reason."""


class PlanError(ValueError):
    """No admissible block layout exists — always with the exact reason."""


# ── the co-visibility measurement ────────────────────────────────────────────────────────────

def _load_windows(wdir: Path):
    from intake.walk import load_window
    spec = json.loads((wdir / "windows.json").read_text())
    return [load_window(wdir / f"window_{i:04d}.npz") for i in range(len(spec["windows"]))]


def _placements(windows) -> List[np.ndarray]:
    """G_k for every window (intake.walk.chain_windows' construction, kept per window)."""
    from intake.walk import _chordal_mean
    G = [np.eye(4)]
    poses = {f: T for f, T in zip(windows[0]["frames"], windows[0]["c2w"])}
    for k in range(1, len(windows)):
        w = windows[k]; local = dict(zip(w["frames"], w["c2w"]))
        shared = [f for f in w["frames"] if f in poses]
        Ms = [poses[f] @ np.linalg.inv(local[f]) for f in shared]
        R = _chordal_mean([M[:3, :3] for M in Ms])
        t = np.mean([poses[f][:3, 3] - R @ local[f][:3, 3] for f in shared], axis=0)
        Gk = np.eye(4); Gk[:3, :3] = R; Gk[:3, 3] = t
        G.append(Gk)
        for f in w["frames"]:
            if f not in poses:
                poses[f] = Gk @ local[f]
    return G


def keyframe_views(wdir: Path) -> Tuple[List[int], Dict[int, dict], List[dict]]:
    """Every keyframe's view from the window where it is most central: c2w (chained), depth, conf,
    K — all from THAT window, so pose and depth agree. Each frame's maps are COPIED out of their
    window, so only the n kept frames stay in memory, never every window whole."""
    windows = _load_windows(wdir)
    G = _placements(windows)
    best: Dict[int, Tuple[float, int, int]] = {}
    for k, w in enumerate(windows):
        n = len(w["frames"])
        for i, f in enumerate(w["frames"]):
            d = abs(i - (n - 1) / 2.0)
            if f not in best or d < best[f][0]:
                best[f] = (d, k, i)
    views: Dict[int, dict] = {}
    cache: Dict[int, dict] = {}
    for f, (_d, k, i) in sorted(best.items()):
        if k not in cache:
            with np.load(windows[k]["path"]) as z:
                cache = {k: {"depth": z["depth"].astype(np.float32), "conf": z["conf"].astype(np.float32),
                             "K": z["intrinsics"].astype(np.float64)}}
        z = cache[k]
        views[f] = {"c2w": G[k] @ windows[k]["c2w"][i], "depth": np.array(z["depth"][i]),
                    "conf": np.array(z["conf"][i]), "K": np.array(z["K"][i]), "window": k}
    return sorted(views), views, windows


def measured_tol(windows) -> float:
    """TOL_QUANTILE of |z_a − z_b| / mean over keyframes seen by two consecutive windows."""
    pool = []
    for k in range(1, len(windows)):
        a, b = windows[k - 1], windows[k]
        shared = [f for f in b["frames"] if f in a["frames"]]
        if not shared:
            continue
        with np.load(a["path"]) as za, np.load(b["path"]) as zb:
            for f in shared[:: max(1, len(shared) // 4)]:
                da = za["depth"][a["frames"].index(f)]; db = zb["depth"][b["frames"].index(f)]
                ok = (da > 0) & (db > 0)
                r = np.abs(da[ok] - db[ok]) / (0.5 * (da[ok] + db[ok]))
                pool.append(r[:: max(1, r.size // 4000)])
    return float(np.quantile(np.concatenate(pool), TOL_QUANTILE)) if pool else 0.05


def _samples(v: dict) -> np.ndarray:
    Z = v["depth"]; K = v["K"]; h, w = Z.shape
    ok = (Z > 0) & np.isfinite(Z)
    s = max(1, int(np.sqrt(ok.sum() / SAMPLES)))
    vv, uu = np.mgrid[0:h:s, 0:w:s]
    m = ok[::s, ::s]
    z = Z[::s, ::s][m]
    X = np.stack([(uu[m] + 0.5 - K[0, 2]) / K[0, 0] * z, (vv[m] + 0.5 - K[1, 2]) / K[1, 1] * z, z], 1)
    T = v["c2w"]
    return X @ T[:3, :3].T + T[:3, 3]


def covis_dir(Pi: np.ndarray, vj: dict, tol: float) -> float:
    """c(i→j): share of i's samples that land inside j, in front, and agree with j's depth."""
    if len(Pi) == 0:
        return 0.0
    w2c = np.linalg.inv(vj["c2w"]); X = Pi @ w2c[:3, :3].T + w2c[:3, 3]
    K = vj["K"]; Zj = vj["depth"]; h, w = Zj.shape
    z = X[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * X[:, 0] / z + K[0, 2]; v = K[1, 1] * X[:, 1] / z + K[1, 2]
    ok = (z > 1e-6) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if not ok.any():
        return 0.0
    zj = Zj[v[ok].astype(int), u[ok].astype(int)]
    agree = (zj > 0) & (np.abs(z[ok] - zj) <= tol * zj)
    return float(agree.sum()) / len(Pi)


def covis_lengths_margins(frames: List[int], views: Dict[int, dict], tol: float,
                          horizon: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """(ℓ, τ-margin): ℓ(i) by the majority rule (design §1.3, -1 = a censored scan) and, per
    keyframe, the smallest |c − τ| among the co-visibility tests that decided its ℓ — how close
    one flipped test is (docs/plan_determinismo.md point 15: recorded, never acted on)."""
    n = len(frames)
    reach = n if horizon is None else int(horizon)
    P = {f: _samples(views[f]) for f in frames}
    ell = np.zeros(n, int)
    margin = np.full(n, np.inf)
    for a in range(n):
        passes = fails = 0; last = a; censored = True
        for b in range(a + 1, min(n, a + reach)):
            c = min(covis_dir(P[frames[a]], views[frames[b]], tol), covis_dir(P[frames[b]], views[frames[a]], tol))
            margin[a] = min(margin[a], abs(c - TAU))
            if c >= TAU:
                passes += 1; last = b
            else:
                fails += 1
            if fails > passes:
                censored = False
                break
        ell[a] = (last - a) if not censored else -1
    return ell, margin


def covis_lengths(frames: List[int], views: Dict[int, dict], tol: float,
                  horizon: Optional[int] = None) -> np.ndarray:
    """ℓ(i) by the majority rule (design §1.3); -1 marks a censored scan (never failed). The scan
    from i reaches i + horizon − 1 at most; None = the whole walk (the plan's horizon)."""
    return covis_lengths_margins(frames, views, tol, horizon)[0]


def budget(ell: np.ndarray) -> np.ndarray:
    """δ(i) = 1/max(ℓ, 1); the censored tail (every later scan reached the end) costs ONE length."""
    n = len(ell)
    d = 1.0 / np.maximum(np.where(ell < 0, 1, ell), 1).astype(float)
    cens = ell < 0
    if cens.any():
        ic = n
        while ic > 0 and cens[ic - 1]:
            ic -= 1
        d[ic:] = 1.0 / max(n - ic, 1)
        d[:ic][cens[:ic]] = 0.0          # censored by a finite horizon: static/far beyond it
    return d


def frame_depth_median(v: dict) -> float:
    """z̄ of one keyframe: the median of its valid DA3 depth (conf > 0, finite, Z > 0); +inf when
    nothing is valid (such a frame is never a good seam)."""
    Z = v["depth"]; ok = (Z > 0) & np.isfinite(Z) & (v["conf"] > 0)
    return float(np.median(Z[ok])) if ok.any() else float("inf")


def rotation_steps_deg(c2w: np.ndarray) -> np.ndarray:
    """θ[i] = the rotation angle (degrees) from keyframe i to i+1 of the chained c2w (n−1 values),
    from ‖R_{i+1} − R_i‖_F = 2√2·sin(θ/2) — well conditioned at small angles."""
    R = np.asarray(c2w, dtype=np.float64)[:, :3, :3]
    if len(R) < 2:
        return np.zeros(0)
    chord = np.linalg.norm(R[1:] - R[:-1], axis=(1, 2))
    return np.degrees(2.0 * np.arcsin(np.clip(chord / (2.0 * np.sqrt(2.0)), 0.0, 1.0)))


def walk_layout(chainage: np.ndarray, metres: float) -> List[Tuple[int, int]]:
    """Uniform 50 % layout of `metres` of CHAINAGE (keyframe ranges [a, b))."""
    n = len(chainage); out = []
    start = 0.0
    while True:
        a = int(np.searchsorted(chainage, start, "left"))
        b = int(np.searchsorted(chainage, start + metres, "right"))
        b = min(max(b, a + 2), n)
        out.append((a, b))
        if b >= n:
            return out
        start += metres / 2.0


# ── the planner (design §1.5–1.7) ────────────────────────────────────────────────────────────

def _lower_median(x: np.ndarray) -> float:
    """The ⌈L/2⌉-th smallest value: one frame's own measurement (no interpolation), and
    ``lower_median ≤ Z  ⇔  2·#{x ≤ Z} ≥ L`` — the count the DP tests."""
    s = np.sort(np.asarray(x, dtype=np.float64), kind="stable")
    return float(s[(len(s) - 1) // 2])


def _first_true(k: int, pred: Callable[[int], bool]) -> int:
    """Smallest i in [0, k) with pred(i), for a monotone pred that holds at k − 1."""
    lo, hi = 0, k - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if pred(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


class _Blocks:
    """The block DP over one walk (design §1.6–1.7).

    Block [a, b) is admissible when L_min ≤ b − a and D(a, b) ≤ H/2 — no upper length: the card
    never splits a chunk (USER 2026-10-06), Omega's resolution adapts instead — or, over budget,
    when b − a ≤ L_ex = 2·L_min − 1 (a block that cannot be split into two minimum blocks; only a
    stretch so fast that even the minimum exceeds the budget, or the integer remainder of such a
    stretch, needs it). A block costs 1 + W1·[over budget] + W2·[over budget and longer than L_min],
    so the minimum total cost is, in this order: the fewest blocks pushed over budget ABOVE the
    minimum length, the fewest blocks over budget, the fewest blocks. With no over-budget stretch
    that is just the fewest blocks. The first block never ends at n (a plan has ≥ 2 blocks).

    Optional constraints, each one MONOTONE in its threshold (so the bottlenecks are found by a
    binary search over the candidate values): ``zc`` (prefix counts of z̄ ≤ Z — an interior block's
    lower-median z̄ ≤ Z), ``tok`` (tok[b]: a cut at b, between b−1 and b, turns ≤ T), ``dthr`` (a
    block within budget holds ≤ dthr lengths). R[a] = the minimum cost from a to the end."""

    def __init__(self, P: np.ndarray, min_chunk_frames: int, H: float):
        self.P = P
        self.n = len(P) - 1
        self.half = H / 2.0
        self.L_min = min_chunk_frames // 2
        self.L_max = self.n
        self.L_ex = 2 * self.L_min - 1
        self.W1 = self.n + 1
        self.W2 = (self.n + 1) ** 2
        self._slack = 1e-9 * (1.0 + float(P[-1]))     # the search bound only; D decides exactly

    def ends(self, a: int, zc=None, tok=None, dthr=None):
        n = self.n
        lo = a + self.L_min
        hi = min(a + self.L_max, n - 1 if a == 0 else n)
        bb = int(np.searchsorted(self.P, self.P[a] + self.half + self._slack, side="right")) - 1
        hi = min(hi, max(a + self.L_ex, bb))
        if lo > hi:
            return None
        bs = np.arange(lo, hi + 1)
        D = self.P[bs] - self.P[a]
        L = bs - a
        over = D > self.half
        ok = ~over | (L <= self.L_ex)
        if dthr is not None:
            ok &= over | (D <= dthr)
        if zc is not None and a > 0:
            ok &= (bs == n) | (2 * (zc[bs] - zc[a]) >= L)
        if tok is not None:
            ok &= (bs == n) | tok[bs]
        cost = 1 + over.astype(np.int64) * self.W1 + (over & (L > self.L_min)).astype(np.int64) * self.W2
        return bs, ok, cost, D, over

    def solve(self, **kw) -> np.ndarray:
        R = np.full(self.n + 1, _INF, dtype=np.int64)
        R[self.n] = 0
        for a in range(self.n - 1, -1, -1):
            e = self.ends(a, **kw)
            if e is None:
                continue
            bs, ok, cost = e[0], e[1], e[2]
            if not ok.any():
                continue
            v = int((R[bs][ok] + cost[ok]).min())
            if v < _INF:
                R[a] = v
        return R

    def leftmost(self, R: np.ndarray, **kw) -> List[int]:
        """The lexicographically smallest boundary vector among the optimal ones."""
        bnd = [0]
        a = 0
        while a < self.n:
            bs, ok, cost = self.ends(a, **kw)[:3]
            hit = np.flatnonzero(ok & (R[bs] + cost == R[a]))
            a = int(bs[hit[0]])
            bnd.append(a)
        return bnd

    def budget_candidates(self, **kw) -> np.ndarray:
        vals = []
        for a in range(self.n):
            e = self.ends(a, **kw)
            if e is None:
                continue
            sel = e[1] & ~e[4]
            if sel.any():
                vals.append(e[3][sel])
        return np.unique(np.concatenate(vals)) if vals else np.zeros(0)


def _single(n: int, D_total: float, why: str, flags: List[str], H: float) -> Dict[str, Any]:
    return {"single_pass": True, "why": why, "ranges": [(0, n)], "blocks": [], "cuts": [],
            "chunks": [{"range": [0, n], "frames": n, "D": D_total, "seam_after": None,
                        "flags": list(flags)}],
            "flags": list(flags), "objective": None,
            "margins": {"D_total_over_H": D_total - H, "blocks_half_H_minus_D": [],
                        "min_block_margin": None, "block_budget_margin": None}}


def plan_detail(delta, zbar, theta, min_chunk_frames: int = MIN_CHUNK_FRAMES,
                H: float = H_LENGTHS_PER_CHUNK) -> Dict[str, Any]:
    """The plan with its evidence: ``ranges`` ([start, end) keyframe indices), the blocks, chunks,
    cuts, flags and the optimum reached on each criterion. See ``plan``."""
    d = np.asarray(delta, dtype=np.float64).ravel()
    z = np.asarray(zbar, dtype=np.float64).ravel()
    t = np.asarray(theta, dtype=np.float64).ravel()
    n, mcf, H = len(d), int(min_chunk_frames), float(H)
    if n < 1:
        raise PlanError("no keyframe to plan")
    if len(z) != n or len(t) != n - 1:
        raise PlanError(f"{n} budgets need {n} seam depths and {n - 1} cut angles "
                        f"(got {len(z)} and {len(t)})")
    if not np.all(np.isfinite(d)) or (d < 0).any():
        raise PlanError("the motion budget must be finite and ≥ 0")
    if np.isnan(z).any():
        raise PlanError("a seam depth is NaN")
    if not np.all(np.isfinite(t)) or (t < 0).any():
        raise PlanError("a cut angle is not a finite angle ≥ 0")
    if mcf < 2 or not (np.isfinite(H) and H > 0):
        raise PlanError(f"min_chunk_frames {mcf} / H {H} cannot plan")
    P = np.concatenate(([0.0], np.cumsum(d)))
    D_total = float(P[n])
    L_min = mcf // 2
    head = {"n": n, "H": H, "min_chunk_frames": mcf, "block_min": L_min, "D_total": D_total}
    if D_total <= H:
        return {**head, **_single(n, D_total, f"D_total {D_total:.2f} ≤ H {H:g}", [], H)}
    if n < 2 * L_min:
        return {**head, **_single(n, D_total, f"{n} keyframes are fewer than the shortest "
                                              f"chunk ({2 * L_min}): one pass, over budget "
                                              f"(D_total {D_total:.2f} > H {H:g})",
                                  ["too_short_to_chunk"], H)}
    bl = _Blocks(P, mcf, H)
    # (a) fewest blocks (after the over-budget terms, which are 0 on any walk within budget)
    C = int(bl.solve()[0])
    if C >= _INF:
        raise PlanError(f"internal: {n} keyframes ≥ two blocks of {L_min} admit no block layout")
    m = C % bl.W1
    kw: Dict[str, Any] = {}
    obj: Dict[str, Any] = {"blocks": m, "chunks": m - 1,
                           "over_budget_blocks": (C % bl.W2) // bl.W1,
                           "over_budget_above_minimum": C // bl.W2,
                           "worst_seam_depth_m": None, "worst_cut_deg": None, "block_budget": None}

    def feasible(**extra) -> bool:
        return int(bl.solve(**kw, **extra)[0]) == C

    # (b) the worst interior seam block's (lower-)median depth, as low as the fewest blocks allow
    if m >= 3:
        cz = np.unique(z)

        def zc_of(Z: float) -> np.ndarray:
            return np.concatenate(([0], np.cumsum(z <= Z)))
        i = _first_true(len(cz), lambda i: feasible(zc=zc_of(cz[i])))
        kw["zc"] = zc_of(cz[i])
        obj["worst_seam_depth_m"] = float(cz[i])
    # (c) the worst rotation across a cut
    ct = np.unique(t)

    def tok_of(T: float) -> np.ndarray:
        return np.concatenate(([True], t <= T, [True]))
    i = _first_true(len(ct), lambda i: feasible(tok=tok_of(ct[i])))
    kw["tok"] = tok_of(ct[i])
    obj["worst_cut_deg"] = float(ct[i])
    # (c') the tightest budget per block (within budget) that keeps (a)–(c): the walk's budget is
    # spread evenly over the blocks instead of packed to the left (equal budgets, design §core)
    cd = bl.budget_candidates(**kw)
    if len(cd):
        i = _first_true(len(cd), lambda i: feasible(dthr=float(cd[i])))
        kw["dthr"] = float(cd[i])
        obj["block_budget"] = float(cd[i])
    # (d) leftmost
    R = bl.solve(**kw)
    if int(R[0]) != C:
        raise PlanError("internal: the constrained optimum lost the block count")
    bnd = bl.leftmost(R, **kw)

    blocks = []
    for k in range(m):
        a, b = bnd[k], bnd[k + 1]
        Db = float(P[b] - P[a])
        flag = None
        if Db > bl.half:
            flag = "over_budget_at_minimum" if b - a == L_min else "over_budget_partition"
        blocks.append({"range": [a, b], "frames": b - a, "D": Db, "z_med_m": _lower_median(z[a:b]),
                       "flag": flag})
    ranges = [(bnd[k], bnd[k + 2]) for k in range(m - 1)]
    chunks = []
    for k, (s, e) in enumerate(ranges):
        seam = blocks[k + 1] if k + 1 < m - 1 else None
        chunks.append({"range": [s, e], "frames": e - s, "D": float(P[e] - P[s]),
                       "seam_after": ({"range": seam["range"], "frames": seam["frames"],
                                       "z_med_m": seam["z_med_m"]} if seam else None),
                       "flags": [x["flag"] for x in (blocks[k], blocks[k + 1]) if x["flag"]]})
    cuts = [{"index": b, "theta_deg": float(t[b - 1])} for b in bnd[1:-1]]
    flags = sorted({x["flag"] for x in blocks if x["flag"]})
    # every bar's margin (point 15): how far each decision sits from flipping — D_total vs H (one
    # pass or chunks), each block's D vs H/2, the chosen block budget vs H/2
    within = [bl.half - x["D"] for x in blocks if x["D"] <= bl.half]
    margins = {"D_total_over_H": D_total - H,
               "blocks_half_H_minus_D": [bl.half - x["D"] for x in blocks],
               "min_block_margin": min(within) if within else None,
               "block_budget_margin": (bl.half - obj["block_budget"]
                                       if obj["block_budget"] is not None else None)}
    return {**head, "single_pass": len(ranges) == 1,
            "why": f"D_total {D_total:.2f} > H {H:g}",
            "ranges": ranges, "blocks": blocks, "chunks": chunks, "cuts": cuts, "flags": flags,
            "objective": obj, "margins": margins}


def plan(delta, zbar, theta, min_chunk_frames: int = MIN_CHUNK_FRAMES,
         H: float = H_LENGTHS_PER_CHUNK) -> List[Tuple[int, int]]:
    """The co-visibility chunk plan: [start, end) keyframe-index ranges (design §1.5–1.7).

    delta: δ per keyframe in walk order; zbar: each keyframe's median valid depth (m); theta: the
    rotation (deg) between consecutive keyframes (n − 1 values). ONE pass when D(0,n) ≤ H;
    otherwise blocks B_0..B_{m−1}, chunk k = B_k ∪ B_{k+1}, every block ≥ L_min =
    min_chunk_frames // 2 frames and D ≤ H/2 (a block over budget only at the minimum length or
    as the integer remainder of such a stretch, flagged). No upper length: the card never splits
    a chunk (USER 2026-10-06 — Omega's resolution adapts to the largest one). Boundaries:
    (a) fewest blocks, (b) the worst interior seam block's median depth lowest, (c) the worst
    rotation across a cut lowest, (c') the tightest per-block budget, (d) leftmost. Exact and
    deterministic."""
    return [(int(a), int(b)) for a, b in plan_detail(delta, zbar, theta, min_chunk_frames, H)["ranges"]]


def inputs_sha256(delta, zbar, theta, min_chunk_frames: int, H: float) -> str:
    """sha256 of exactly what the planner reads — two equal hashes plan bit-identically."""
    h = hashlib.sha256()
    for x in (delta, zbar, theta):
        h.update(np.ascontiguousarray(np.asarray(x, dtype=np.float64), dtype="<f8").tobytes())
    h.update(json.dumps([len(delta), int(min_chunk_frames), float(H)]).encode())
    return h.hexdigest()


# ── the session: measure once, persist, replan bit-identically ───────────────────────────────

def _sha_file(p: Path) -> str:
    import repro
    return repro.sha256_file(p)


def _code_stamp() -> Dict[str, str]:
    """sha256 of the code the measurement depends on: this planner and the walk (the windows'
    chaining, the anchors) — point 21: a code change re-measures instead of reusing an old plan
    as if a fresh run would give it."""
    import repro
    return repro.stamp(code=[__file__, "intake.walk"])["code"]


SCAN_HORIZON = "whole_walk"   # every keyframe's co-visibility scan may reach the end of the walk


def input_stamp(scan_dir: Path) -> Tuple[str, Dict[str, Any]]:
    """The stamp of the measurement's inputs, computable WITHOUT the window files: the I3 window
    plan (windows.json — since 2026-10-07 it carries the DA3 identity: pinned weights, extractor
    and DA3 code, torch / cuDNN, card) and the walk measured on those windows (walk.json —
    rewritten whenever I3 re-measures, so new DA3 windows change it), the method constants, the
    scan horizon (the whole walk — no card capacity enters the plan) and the code of this planner
    and of the walk (point 21)."""
    from intake.walk import WALK_NAME, WINDOWS_DIRNAME
    scan_dir = Path(scan_dir)
    spec = scan_dir / "output" / WINDOWS_DIRNAME / "windows.json"
    walk = scan_dir / "intake" / WALK_NAME
    for p in (spec, walk):
        if not p.exists():
            raise CovisError(f"{p} is missing — the co-visibility plan needs I3 (the DA3 windows "
                             f"and the walk measured on them)")
    # the keyframes' KINDS (parallax / rotation) decide which ones the plan is measured on
    sel = scan_dir / "frames" / "selected_frames.json"
    parts = {"covis_version": COVIS_VERSION, "tau": TAU, "tol_quantile": TOL_QUANTILE,
             "samples_per_frame": SAMPLES, "scan_horizon": SCAN_HORIZON,
             "windows_json_sha256": _sha_file(spec), "walk_json_sha256": _sha_file(walk),
             "selected_frames_sha256": (_sha_file(sel) if sel.exists() else None),
             "code_sha256": _code_stamp()}
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest(), parts


def window_layout(scan_dir: Path) -> Dict[str, Any]:
    """The I3 window layout a measurement is read from — what the co-visibility depends on beyond
    the frames (review 2026-10-06): the model and process_res, the windows' size and seams as they
    RAN (windows.json), the requested size and overlap (walk.json), and the card that sized them
    (windows.json's window_sizing — the committed card table's key; None for windows planned
    before the record existed). Readable without the window files.
    RECORDED and compared with H_CALIBRATION_LAYOUT's card_key — never acted on here."""
    from intake.walk import WINDOWS_DIRNAME, load_walk
    scan_dir = Path(scan_dir)
    spec = json.loads((scan_dir / "output" / WINDOWS_DIRNAME / "windows.json").read_text())
    wins = spec.get("windows") or []
    lens = [len(w) for w in wins]
    shared = [len(set(a) & set(b)) for a, b in zip(wins, wins[1:])]
    # the card that sized them: windows.json's own sizing record (the committed card table's key,
    # repro.card_key) — None for windows planned before the record existed
    card = ((spec.get("window_sizing") or {}).get("card")
            or (spec.get("da3_environment") or {}).get("card"))
    params = (load_walk(scan_dir) or {}).get("params") or {}
    return {"model_id": spec.get("model_id"), "process_res": spec.get("process_res"),
            "n_windows": len(wins), "window_frames": max(lens) if lens else 0,
            "window_frames_min": min(lens) if lens else 0,
            "window_frames_requested": params.get("window_frames"),
            "window_overlap_frac": params.get("window_overlap_frac"),
            "seam_frames": [min(shared), max(shared)] if shared else [],
            "card": card,
            "card_matches_calibration": (None if card is None
                                         else card == H_CALIBRATION_LAYOUT["card_key"])}


def measure_inputs(scan_dir: Path, log: Callable = print) -> Dict[str, Any]:
    """ℓ, δ, z̄, θ, tol_rel and the chainage from the I3 windows (they must be on disk)."""
    from intake.walk import WINDOWS_DIRNAME, load_walk
    scan_dir = Path(scan_dir)
    stamp, parts = input_stamp(scan_dir)
    wdir = scan_dir / "output" / WINDOWS_DIRNAME
    spec = json.loads((wdir / "windows.json").read_text())
    missing = [i for i in range(len(spec["windows"])) if not (wdir / f"window_{i:04d}.npz").exists()]
    if missing:
        raise CovisError(f"{len(missing)} of the {len(spec['windows'])} I3 window files are gone "
                         f"(first window_{missing[0]:04d}.npz) and no co-visibility measurement with "
                         f"stamp {stamp[:12]} is persisted — regenerate the windows "
                         f"(intake.walk.run_da3_windows) first")
    t0 = time.time()
    frames, views, windows = keyframe_views(wdir)
    walk_n = int((load_walk(scan_dir) or {}).get("n_keyframes", -1))
    if walk_n != len(frames):
        raise CovisError(f"walk.json counts {walk_n} keyframes, the windows hold {len(frames)} — "
                         f"the walk was not measured on these windows")
    tol = measured_tol(windows)
    # THE PLAN IS MEASURED ON THE PARALLAX KEYFRAMES ONLY (USER 2026-10-07). H was calibrated on
    # keyframes chosen by parallax alone; the keyframes the selector adds in a TURN
    # (intake/parallax.py, closed_by "rotation") give every view more co-visible neighbours
    # without a metre of walk, so counted here they made turns CHEAP (pccr: D_total 34.7 → 31.9,
    # 5 chunks → 4 longer ones, the chunks spanned the turn and Omega drifted −54 cm inside them).
    # They fill the turns INSIDE the chunks; they never move a cut or stretch a budget.
    plan_idx = planning_indices(scan_dir, frames)
    pframes = [frames[i] for i in plan_idx]
    ell, tau_margin = covis_lengths_margins(pframes, views, tol)
    delta = budget(ell)
    zbar = [frame_depth_median(views[f]) for f in pframes]
    c2w = np.stack([views[f]["c2w"] for f in pframes])
    theta = rotation_steps_deg(c2w)
    c = c2w[:, :3, 3]
    chain = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])
    doc = {"version": COVIS_VERSION, "provenance": "tool_measured", "stamp": stamp,
           "stamp_parts": parts, "n": len(frames), "frames": [int(f) for f in frames],
           # the planning subsequence: full keyframe index and frame number of every parallax
           # keyframe; ell / delta / zbar / theta / chainage below are over THESE
           "plan_index": [int(i) for i in plan_idx], "plan_frames": [int(f) for f in pframes],
           "n_plan": len(pframes), "n_rotation_keyframes": len(frames) - len(pframes),
           "tau": TAU, "tol_quantile": TOL_QUANTILE, "samples_per_frame": SAMPLES,
           "scan_horizon": SCAN_HORIZON, "tol_rel": float(tol),
           "ell": [int(x) for x in ell], "delta": [float(x) for x in delta],
           # point 15: how close each ℓ is to flipping (smallest |c − τ| among its tests)
           "tau_margin": [float(x) if np.isfinite(x) else None for x in tau_margin],
           "tau_margin_min": (float(np.min(tau_margin[np.isfinite(tau_margin)]))
                              if np.isfinite(tau_margin).any() else None),
           "zbar_m": [float(x) for x in zbar], "theta_deg": [float(x) for x in theta],
           "chainage_m": [float(x) for x in chain],
           "D_total": float(np.concatenate(([0.0], np.cumsum(delta)))[-1]),
           "window_layout": window_layout(scan_dir)}
    log(f"{LOG_TAG} measured {len(frames)} keyframes in {time.time() - t0:.1f} s: tol_rel "
        f"{tol * 100:.1f} %, D_total {doc['D_total']:.2f}")
    return doc


def covis_inputs(scan_dir: Path, *, cache_path: Optional[Path] = None,
                 log: Callable = print) -> Tuple[Dict[str, Any], str]:
    """The measurement for the plan: reused from ``intake/covis.json`` (or ``cache_path``) when its
    stamp matches — no window file needed — otherwise measured from the windows and persisted.
    Returns (doc, "reused" | "measured"). The fresh doc goes through the same JSON round trip
    as a persisted one, so both plan bit-identically."""
    scan_dir = Path(scan_dir)
    path = Path(cache_path) if cache_path else scan_dir / "intake" / COVIS_NAME
    stamp, _parts = input_stamp(scan_dir)
    if path.exists():
        try:
            old = json.loads(path.read_text())
        except (OSError, ValueError):
            old = None
        if isinstance(old, dict) and old.get("version") == COVIS_VERSION and old.get("stamp") == stamp:
            log(f"{LOG_TAG} reused {path} (stamp {stamp[:12]})")
            return old, "reused"
        log(f"{LOG_TAG} {path} is stale or unreadable — measuring again")
    doc = json.loads(json.dumps(measure_inputs(scan_dir, log)))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    os.replace(tmp, path)
    return doc, "measured"


FROZEN_NAME = "chunk_plan_frozen.json"
FROZEN_VERSION = 1


def _planner_constants() -> Dict[str, Any]:
    """What a plan is a function of beyond the measurement: the method's constants."""
    return {"H": H_LENGTHS_PER_CHUNK, "min_chunk_frames": MIN_CHUNK_FRAMES, "tau": TAU,
            "tol_quantile": TOL_QUANTILE, "samples_per_frame": SAMPLES,
            "scan_horizon": SCAN_HORIZON}


def _walk_keyframes(scan_dir: Path) -> Optional[List[int]]:
    """The keyframes walk.json was measured over (its chainage frames), None when not recorded."""
    from intake.walk import load_walk
    ch = (load_walk(scan_dir) or {}).get("chainage")
    if not ch:
        return None
    return [int(c["frame"]) for c in ch]


def _load_frozen(scan_dir: Path) -> Optional[Dict[str, Any]]:
    p = Path(scan_dir) / "intake" / FROZEN_NAME
    if not p.exists():
        return None
    try:
        doc = json.loads(p.read_text())
    except (OSError, ValueError) as e:
        raise CovisError(f"{p} is unreadable ({e}) — the session's frozen chunk plan cannot be read "
                         f"back; delete it to plan again") from e
    return doc if doc.get("version") == FROZEN_VERSION else None


def _reusable_measurement(scan_dir: Path, cache_path: Optional[Path]) -> Optional[Dict[str, Any]]:
    """The persisted measurement when its stamp is current — never measured here."""
    path = Path(cache_path) if cache_path else Path(scan_dir) / "intake" / COVIS_NAME
    try:
        stamp, _parts = input_stamp(scan_dir)
        old = json.loads(path.read_text()) if path.exists() else None
    except (CovisError, OSError, ValueError):
        return None
    if isinstance(old, dict) and old.get("version") == COVIS_VERSION and old.get("stamp") == stamp:
        return old
    return None


def planning_indices(scan_dir: Path, frames: List[int]) -> List[int]:
    """The indices (into ``frames``, the keyframes in walk order) of the PARALLAX keyframes — the
    ones the plan is measured on. A keyframe the selector closed by rotation (selected_frames.json
    keyframes[].closed_by == "rotation") is not one; a selection written before that field
    existed has only parallax keyframes. A frame the selection does not list FAILS: the windows
    were built on another keyframe set."""
    p = Path(scan_dir) / "frames" / "selected_frames.json"
    try:
        sel = json.loads(p.read_text())
    except (OSError, ValueError) as e:
        raise CovisError(f"{p} is unreadable ({e}) — the plan needs the keyframes' kinds") from e
    kinds = {int(k["frame"]): str(k.get("closed_by", "quantum")) for k in sel.get("keyframes", [])}
    if kinds:
        missing = [int(f) for f in frames if int(f) not in kinds]
        if missing:
            raise CovisError(f"{len(missing)} keyframe(s) of the windows are not in {p} (first "
                             f"{missing[0]}) — the windows were built on another keyframe set")
    return [i for i, f in enumerate(frames) if kinds.get(int(f), "quantum") != "rotation"]


def _report_of(scan_dir: Path, doc: Dict[str, Any], source: str) -> Dict[str, Any]:
    delta = np.asarray(doc["delta"], dtype=np.float64)
    zbar = np.asarray(doc["zbar_m"], dtype=np.float64)
    theta = np.asarray(doc["theta_deg"], dtype=np.float64)
    det = plan_detail(delta, zbar, theta)
    chain = doc["chainage_m"]
    frames = doc["frames"]
    n_full = len(frames)
    # the plan is measured on the planning subsequence (the parallax keyframes); every range it
    # reports is mapped to FULL keyframe indices — a boundary at planning keyframe p becomes the
    # full index of p, the end of the walk stays n — so the rotation keyframes between two
    # planning keyframes fall inside the chunk that holds both
    plan_index = [int(i) for i in doc.get("plan_index", list(range(n_full)))]
    full_of = plan_index + [n_full]
    pframes = [frames[i] for i in plan_index]
    n_plan = len(plan_index)

    def _full(r):
        return [int(full_of[int(r[0])]), int(full_of[int(r[1])])]
    for ch in det["chunks"]:
        s, e = ch["range"]
        ch["metres"] = float(chain[e - 1] - chain[s])
        ch["keyframes"] = [pframes[s], pframes[e - 1]]
        ch["plan_range"] = [int(s), int(e)]
        ch["plan_frames"] = int(e - s)
        ch["range"] = _full(ch["range"]); ch["frames"] = ch["range"][1] - ch["range"][0]
        if ch.get("seam_after"):
            ch["seam_after"]["plan_range"] = list(ch["seam_after"]["range"])
            ch["seam_after"]["range"] = _full(ch["seam_after"]["range"])
            ch["seam_after"]["frames"] = ch["seam_after"]["range"][1] - ch["seam_after"]["range"][0]
    for bl in det["blocks"]:
        bl["plan_range"] = list(bl["range"]); bl["range"] = _full(bl["range"])
        bl["frames"] = bl["range"][1] - bl["range"][0]
    for cu in det["cuts"]:
        cu["plan_index"] = int(cu["index"]); cu["index"] = int(full_of[int(cu["index"])])
        cu["frame"] = pframes[cu["plan_index"]]
        cu["chainage_m"] = float(chain[cu["plan_index"]])
    det["plan_ranges"] = [(int(a), int(b)) for a, b in det["ranges"]]
    ranges = [tuple(_full(r)) for r in det["ranges"]]
    det["n_plan"] = n_plan
    det["n"] = n_full
    det["n_rotation_keyframes"] = n_full - n_plan
    return {"version": 2, "method": "covis", "session": str(scan_dir), "tau": TAU,
            "tol_rel": doc["tol_rel"], "walk_m": float(chain[-1]), **det, "ranges": ranges,
            "input_stamp": doc["stamp"], "measurement": source,
            "inputs_sha256": inputs_sha256(delta, zbar, theta, MIN_CHUNK_FRAMES,
                                           H_LENGTHS_PER_CHUNK),
            # how close every ℓ sat to flipping (point 15; None on a measurement older than it)
            "tau_margin_min": doc.get("tau_margin_min"),
            # the windows this measurement was read from (a measurement persisted before the
            # layout was recorded: windows.json is the stamped one, so read it there) and the
            # layout H was calibrated at — recorded side by side, never acted on (review
            # 2026-10-06; the decision is the user's)
            "window_layout": doc.get("window_layout") or window_layout(scan_dir),
            "H_calibration_layout": H_CALIBRATION_LAYOUT}


def plan_session(scan_dir: Path, *, cache_path: Optional[Path] = None,
                 log: Callable = print) -> Dict[str, Any]:
    """The session's chunk plan: {"ranges": [(start, end), ...], "report": {...}}. The report
    carries n, D_total, H, tol_rel, every chunk's D / frames / metres / seam depth, the cuts, the
    flags, every bar's margin, the input stamp and the sha256 of exactly what the planner read.
    It depends on the co-visibility measurement ONLY — never on the card (USER 2026-10-06).

    THE PLAN IS FROZEN PER SESSION (docs/plan_determinismo.md point 15, 2026-10-07): the first
    plan of a keyframe set is written to ``<session>/intake/chunk_plan_frozen.json`` and every
    later run over the SAME keyframes and planner constants returns it — a re-measurement whose
    integer ℓ or DP thresholds flip on noise never re-plans the session (it is DECLARED, with the
    plan it would have given). Another keyframe set plans again and freezes that plan. With the
    keyframes known from walk.json, the frozen plan needs no measurement at all."""
    scan_dir = Path(scan_dir)
    planner = _planner_constants()
    frozen = _load_frozen(scan_dir)
    fpath = scan_dir / "intake" / FROZEN_NAME

    def _frozen_for(frames: Optional[List[int]]) -> bool:
        return (frozen is not None and frames is not None and frozen.get("planner") == planner
                and [int(f) for f in frozen.get("keyframes", [])] == [int(f) for f in frames])

    def _from_frozen(doc_now: Optional[Dict[str, Any]], source: str) -> Dict[str, Any]:
        ranges = [(int(a), int(b)) for a, b in frozen["ranges"]]
        rep = dict(frozen["report"], ranges=ranges, measurement=source, plan_frozen=str(fpath))
        if doc_now is not None:
            now = _report_of(scan_dir, doc_now, "reused")
            # the informational fields follow the CURRENT measurement (the window layout it was
            # read from, the τ margins) — recorded, never acted on; the ranges are the frozen ones
            for k in ("window_layout", "tau_margin_min"):
                if now.get(k) is not None:
                    rep[k] = now[k]
            if [tuple(r) for r in now["ranges"]] != ranges:
                rep["replan_declined"] = {"ranges_now": now["ranges"],
                                          "input_stamp_now": now["input_stamp"],
                                          "margins_now": now.get("margins")}
                log(f"{LOG_TAG} ⚠ the current measurement (stamp {now['input_stamp'][:12]}) would "
                    f"plan {now['ranges']}; the session's FROZEN plan {ranges} stays (point 15 — "
                    f"declared; delete {fpath.name} to plan again)")
        log(f"{LOG_TAG} the session's frozen chunk plan reused ({fpath.name}: "
            f"{len(ranges)} chunk(s) over {len(frozen['keyframes'])} keyframes)")
        return {"ranges": ranges, "report": rep}

    if _frozen_for(_walk_keyframes(scan_dir)):
        return _from_frozen(_reusable_measurement(scan_dir, cache_path), "frozen")
    doc, source = covis_inputs(scan_dir, cache_path=cache_path, log=log)
    if _frozen_for(doc["frames"]):
        return _from_frozen(doc, "frozen")
    report = _report_of(scan_dir, doc, source)
    ranges = [(int(a), int(b)) for a, b in report["ranges"]]
    if frozen is not None:
        log(f"{LOG_TAG} {fpath.name} froze another keyframe set or planner — this one is planned "
            f"and frozen instead")
    fz = {"version": FROZEN_VERSION, "keyframes": [int(f) for f in doc["frames"]],
          "planner": planner, "ranges": [[a, b] for a, b in ranges],
          "report": json.loads(json.dumps(report))}
    fpath.parent.mkdir(parents=True, exist_ok=True)
    tmp = fpath.with_name(fpath.name + ".tmp")
    tmp.write_text(json.dumps(fz, indent=1))
    os.replace(tmp, fpath)
    report["plan_frozen"] = str(fpath)
    return {"ranges": ranges, "report": report}


def format_plan(rep: Dict[str, Any]) -> List[str]:
    out = [f"{LOG_TAG} plan {rep.get('session', '')}: {rep['n']} keyframes"
           + (f" ({rep['n_plan']} parallax planned on, {rep['n_rotation_keyframes']} rotation inside the chunks)"
              if rep.get("n_plan") is not None else "") + ", "
           f"walk {rep.get('walk_m', float('nan')):.1f} m, tol_rel {rep.get('tol_rel', float('nan')) * 100:.1f} %, "
           f"D_total {rep['D_total']:.2f}, H {rep['H']:g} → "
           + ("ONE pass" if rep["single_pass"] else f"{len(rep['ranges'])} chunks ({len(rep['blocks'])} blocks)")
           + f" — {rep['why']}"]
    for k, ch in enumerate(rep["chunks"]):
        s, e = ch["range"]
        seam = ch["seam_after"]
        out.append(f"  chunk {k:2d}  [{s:4d}, {e:4d})  {ch['frames']:4d} kf  "
                   f"{ch.get('metres', float('nan')):6.2f} m  D {ch['D']:6.2f}"
                   + (f"  seam [{seam['range'][0]}, {seam['range'][1]}) {seam['frames']} kf "
                      f"z̄ {seam['z_med_m']:.2f} m" if seam else "")
                   + (f"  {','.join(ch['flags'])}" if ch["flags"] else ""))
    if rep["cuts"]:
        out.append("  cuts: " + ", ".join(f"{c['index']} (θ {c['theta_deg']:.2f}°)" for c in rep["cuts"]))
    if rep.get("objective"):
        o = rep["objective"]
        out.append(f"  optimum: {o['blocks']} blocks, worst seam z̄ {o['worst_seam_depth_m']}, "
                   f"worst cut {o['worst_cut_deg']}°, block budget {o['block_budget']}, "
                   f"over budget {o['over_budget_blocks']}")
    mg = rep.get("margins")
    if mg:
        out.append(f"  margins: D_total − H {mg['D_total_over_H']:+.4f}"
                   + (f", tightest block H/2 − D {mg['min_block_margin']:.4f}"
                      if mg.get("min_block_margin") is not None else "")
                   + (f", block budget H/2 − {mg['block_budget_margin']:.4f}"
                      if mg.get("block_budget_margin") is not None else "")
                   + (f", tightest co-visibility test |c − τ| {rep['tau_margin_min']:.4f}"
                      if rep.get("tau_margin_min") is not None else ""))
    wl = rep.get("window_layout")
    if wl:
        cal = rep.get("H_calibration_layout") or H_CALIBRATION_LAYOUT
        out.append(f"  measured on {wl['n_windows']} I3 window(s) of {wl['window_frames']} kf "
                   f"({wl.get('model_id')} at process_res {wl.get('process_res')}, card "
                   f"{wl.get('card') or 'unknown'}); H {rep['H']:g} was calibrated on windows sized "
                   f"for {cal['card']}"
                   + ("" if wl.get("card_matches_calibration") else
                      " — ⚠ " + ("another card" if wl.get("card_matches_calibration") is False
                                 else "the card that sized these windows is not recorded")
                      + ": H is not known to hold on this window layout (declared, not acted on)"))
    out.append(f"  input stamp {rep.get('input_stamp', '')[:16]}  inputs sha256 "
               f"{rep.get('inputs_sha256', '')[:16]}  measurement {rep.get('measurement', '')}"
               + (f"  frozen in {Path(rep['plan_frozen']).name}" if rep.get("plan_frozen") else ""))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m reconstruction.chunk_covis")
    ap.add_argument("--session", required=True)
    ap.add_argument("--layout", action="append", default=[], help="walk:<metres>")
    ap.add_argument("--total", action="store_true")
    ap.add_argument("--plan", action="store_true",
                    help="print the co-visibility chunk plan (persists the measurement in --cache)")
    ap.add_argument("--cache", default=None,
                    help="--plan: the measurement file (default <session>/intake/covis.json)")
    ap.add_argument("--out", default=None, help="--plan: write {ranges, report} as JSON here")
    a = ap.parse_args(argv)
    sess = Path(a.session)
    if a.plan:
        res = plan_session(sess, cache_path=Path(a.cache) if a.cache else None)
        for line in format_plan(res["report"]):
            print(line)
        if a.out:
            Path(a.out).write_text(json.dumps(res, indent=1))
        return 0
    frames, views, windows = keyframe_views(sess / "output" / "da3_windows")
    tol = measured_tol(windows)
    ell = covis_lengths(frames, views, tol)
    d = budget(ell)
    c = np.array([views[f]["c2w"][:3, 3] for f in frames])
    chain = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])
    depth_med = np.array([float(np.median(views[f]["depth"][views[f]["depth"] > 0])) for f in frames])
    rep = {"session": str(sess), "n": len(frames), "walk_m": float(chain[-1]), "tol_rel": tol, "tau": TAU,
           "ell": ell.tolist(), "delta": d.tolist(), "D_total": float(d.sum()),
           "depth_median_m": float(np.median(depth_med)), "layouts": {}}
    print(f"[covis] {sess}: {len(frames)} keyframes, walk {chain[-1]:.1f} m, scene depth {np.median(depth_med):.2f} m, "
          f"tol_rel {tol * 100:.1f} % (measured), median ℓ {int(np.median(np.where(ell < 0, len(frames), ell)))}, "
          f"D_total {d.sum():.2f}")
    for lay in a.layout:
        m = float(lay.split(":")[1])
        chunks = walk_layout(chain, m)
        Ds = [float(d[x:y].sum()) for x, y in chunks]
        rep["layouts"][lay] = {"chunks": chunks, "D": Ds}
        print(f"[covis] layout {lay}: {len(chunks)} chunks, D per chunk "
              f"min {min(Ds):.2f} / median {np.median(Ds):.2f} / max {max(Ds):.2f}")
    out = sess / "output" / "chunk_covis.json"
    out.write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
