"""§5 on the finished session: the scale GRAPH re-solved with what the
reconstruction could not measure yet — the copies of every revisited
instance (loop rows, §5.1) and the regulated dimensions (absolute rows,
§5.2) — against the chunks' DA3 agreement RELATIVE to the metric lock
(anchor rows: how much the geometry moved against the anchors since the
lock, 1 = nothing new) and the glued seams (the current chain is
consistent: a correction that differs between neighbouring chunks pays a
seam residual).

Unknowns: one correction factor r_k per reconstruction chunk. Applied per
keyframe through the correction package: depth × r_k about the keyframe's
own camera plus the translation that keeps the walk continuous (each chunk
scales about the first keyframe it owns, the shift accumulates along the chain).

After the precision gauge (claude_stac.txt §4-F2) only the closures measured on
the current epoch enter: the anchor, DA3 trend and absolute rows are not
relative to that geometry and stand down (`stand_down_for_gauge`).

DETERMINISM (docs/plan_determinismo.md, 2026-10-08):
  * point 127 — the solution is APPLIED only by THE USER'S RULE
    (``loop_utils.metric_lock.decide_change``): the judges are the closures, each
    left out once (the graph solved without it predicts it); ≥ 5 judges, the 95 %
    interval of the paired improvement on the improving side, median improvement
    ≥ the error factor × the closures' own error; between identity, ramp and free
    the SIMPLEST model whose held-out error lies within the factor × that error of
    the best wins. Gone: the split by list position, the 9-row shortcut, the 0.75
    factor, the 6 σ seam cap and the 0.01 significance switch;
    ``max_correction_log`` stays as a declared BOUND.
  * point 147 — an anchor row stands down when its chunk has NOT moved: |log K| of
    the depth factor the epoch chain applied to it below the factor × the anchors'
    own error (mad_rel / √n), never a float equality on a rounded history.
  * point 132 — a closure is attributed to chunk pairs by the chunk SHARES of its
    two copies' birth keyframes (the row moves continuously as frames come and go;
    the share inside one chunk weighs nothing).
  * point 134 — the closure rows enter only on a matching evidence stamp.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


def _vendor_on_path():
    vendor = Path(__file__).resolve().parents[3] / "vendor" / "VGGT-Long"
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))


def chunk_of_keyframes(output_dir, n_kf: int) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """(chunk_ranges, owner[kf]) — the reconstruction's real chunks
    (chunk_plan.json); a single-pass session is one chunk."""
    from correction.units import load_chunk_plan
    _vendor_on_path()
    from loop_utils.metric_lock import frame_owner
    plan = load_chunk_plan(output_dir)
    if plan is None:
        ranges = [(0, int(n_kf))]
    else:
        ranges = [(int(a), int(b)) for a, b in plan["chunk_ranges"]]
    owner = frame_owner(ranges, int(n_kf))
    if (owner < 0).any():
        raise RuntimeError(f"chunk_plan.json leaves {int((owner < 0).sum())} keyframe(s) without "
                           f"a chunk — the plan and camera_frames.txt disagree")
    return ranges, owner


def _baseline_agreements(diag: dict) -> Optional[Dict[int, float]]:
    """The per-anchor agreement the METRIC LOCK consumed (epoch 0: the
    original estimate, preserved untouched across epochs — s_f / s_applied)."""
    s_applied = diag.get("s_applied")
    frames = ((diag.get("anchors") or {}).get("frames")) or []
    if not s_applied or not frames:
        return None
    return {int(fr["num"]): float(fr["s_f"]) / float(s_applied) for fr in frames if fr.get("s_f")}


def applied_depth_factor(output_dir, frames: List[int]) -> np.ndarray:
    """The cumulative depth factor already applied to each keyframe over the reconstruction's
    PRODUCT epoch — the epoch chain composed (``correction.chain.applied_depth_factor``; the
    per-keyframe ``k_kf`` of every transform epoch's npz is exactly what was applied).

    `_current_agreements` cannot answer this: it reads an `epochs` history in
    `scale_diagnostics.json` that restated whatever the previous run wrote (found
    2026-09-19, after epoch 1 had already applied 0.9586-1.0432 and every anchor
    still read "nothing moved")."""
    from correction.chain import applied_depth_factor as _chain_k
    return np.asarray(_chain_k(output_dir, list(frames)), np.float64)


def anchor_rows(output_dir, owner: np.ndarray, frames: List[int],
                applied_k: Optional[np.ndarray] = None, error_factor: Optional[float] = None
                ) -> Tuple[Dict[int, float], Dict[int, int], Dict[int, bool], Dict[int, dict]]:
    """Per-chunk DA3 evidence RELATIVE to the metric lock: median over the
    chunk's anchors of agreement_now / agreement_at_lock → (s_da3, n_anchors,
    moved, margins).

    The lock already weighed the raw anchors against the seams and chose;
    re-solving the raw ratios post-hoc only pulled the chunks back toward
    the DA3 medians the lock had overruled (pccr 2026-09-13 21:03: 0 loop
    rows, r down to 0.87, the start↔end revisit went from 20 cm to 1.15 m).
    What the anchors can still say is whether the geometry MOVED against
    them since the lock — an injected or accumulated scale error changes
    the agreement by 1/k (known-answer §10.10), an untouched session leaves
    every ratio at 1 → identity. A ratio of 1 means: nothing new.

    `moved[k]` (docs/plan_determinismo.md point 147): the chunk MOVED when the
    depth factor the epoch chain applied to it (``applied_k`` per keyframe,
    ``correction.chain.applied_depth_factor``) reads |log K| ≥ ``error_factor``
    × the anchors' own measured error, ``mad_rel / √n_k``. It used to be
    ``any(x != 1.0)`` over a history rounded to 6 decimals — a rounding residue,
    not a motion. "The geometry has not moved" restates the gauge, it does not
    measure whether the lock was RIGHT; `solve_scale_stage` needs to tell those
    two apart — see there. ``margins[k]`` = |log K| − factor × error."""
    p = Path(output_dir) / "scale_diagnostics.json"
    if not p.exists():
        return {}, {}, {}, {}
    from correction.diagnose import _current_agreements
    diag = json.loads(p.read_text())
    now = _current_agreements(diag)
    base = _baseline_agreements(diag)
    if not now or not base:
        return {}, {}, {}, {}
    if error_factor is None:
        from correction.config import judge_of
        error_factor = judge_of(None)[0]
    mad = float((diag.get("anchors") or {}).get("mad_rel") or 0.0)
    kf_of = {int(f): k for k, f in enumerate(frames)}
    per_chunk: Dict[int, List[float]] = {}
    k_chunk: Dict[int, List[float]] = {}
    for frame, ratio in now.items():
        k = kf_of.get(int(frame))
        b = base.get(int(frame))
        if k is None or b is None or not np.isfinite(ratio) or ratio <= 0 or not np.isfinite(b) or b <= 0:
            continue
        per_chunk.setdefault(int(owner[k]), []).append(float(ratio) / float(b))
        k_chunk.setdefault(int(owner[k]), []).append(
            float(applied_k[k]) if applied_k is not None else 1.0)
    s = {k: float(np.median(v)) for k, v in per_chunk.items()}
    n = {k: len(v) for k, v in per_chunk.items()}
    moved: Dict[int, bool] = {}
    margins: Dict[int, dict] = {}
    for k, v in per_chunk.items():
        log_k = float(np.median(np.abs(np.log(np.asarray(k_chunk[k], np.float64)))))
        err = (mad / np.sqrt(len(v))) if (np.isfinite(mad) and mad > 0) else 0.0
        margin = log_k - float(error_factor) * err
        moved[k] = bool(margin >= 0.0 and log_k > 0.0)
        margins[k] = {"abs_log_k": log_k, "anchor_error": float(err),
                      "error_factor": float(error_factor), "margin": float(margin)}
    return s, n, moved, margins


def da3_trend_rows(output_dir, owner: np.ndarray, frames: List[int],
                   applied: Optional[np.ndarray] = None
                   ) -> Tuple[Dict[Tuple[int, int], Tuple[float, float]], List[dict]]:
    """The SHAPE of the DA3 drift along the walk, as RELATIVE rows between
    consecutive chunks — never as absolute pins.

    USER 2026-09-19. This is not the move rejected on 2026-09-13: that one fed
    each chunk's ABSOLUTE DA3 median (±8-15 % monocular noise) and dragged the
    session toward medians the lock had overruled. What goes in here is only
    how the per-frame scale CHANGES from one chunk to the next — the trend
    that pccr measured at Spearman rho +0.459, p 0.0038 over 38 anchors, and
    that the silhouette closures independently agree with (+14.0 % vs
    +12.3-15.7 %). A relative row carries no opinion about the metre; it
    cannot move the session's size, only how the size is DISTRIBUTED.

    Already-applied corrections are subtracted, or the row would ask for the
    same drift every epoch and compound it. The lock-relative agreements are
    what measure that: expanding a chunk by r improves its agreement by 1/r,
    so with B_k the chunk's agreement AT the lock and K_k the depth factor
    ALREADY APPLIED to it (``applied``, from the epoch chain — see
    `applied_depth_factor` — NOT from `_current_agreements`, which is inert),

        log r_{k+1} - log r_k  =  log(B_{k+1}/B_k) - log(K_{k+1}/K_k)

    and the row goes to zero exactly when the drift has been corrected.

    σ is MEASURED, never chosen: the session's own anchor dispersion
    (`anchors.mad_rel`) over √n of each chunk's anchors, the two chunks of a
    row combined in quadrature. A chunk with three noisy anchors prices itself
    out on its own evidence.
    """
    p = Path(output_dir) / "scale_diagnostics.json"
    if not p.exists():
        return {}, []
    from correction.diagnose import _current_agreements
    diag = json.loads(p.read_text())
    now = _current_agreements(diag)
    base = _baseline_agreements(diag)
    if not now or not base:
        return {}, []
    mad = float((diag.get("anchors") or {}).get("mad_rel") or 0.0)
    if not np.isfinite(mad) or mad <= 0:
        return {}, []
    kf_of = {int(f): k for k, f in enumerate(frames)}
    if applied is None:
        applied = applied_depth_factor(output_dir, frames)
    B: Dict[int, List[float]] = {}
    A: Dict[int, List[float]] = {}
    for frame, b in base.items():
        k = kf_of.get(int(frame))
        if k is None or not np.isfinite(b) or b <= 0:
            continue
        c = int(owner[k])
        B.setdefault(c, []).append(float(b))
        A.setdefault(c, []).append(float(applied[k]))
    rows: Dict[Tuple[int, int], Tuple[float, float]] = {}
    rep: List[dict] = []
    for c in sorted(B):
        if (c + 1) not in B:
            continue
        b0, b1 = float(np.median(B[c])), float(np.median(B[c + 1]))
        a0, a1 = float(np.median(A[c])), float(np.median(A[c + 1]))
        # what the chunks still want, MINUS what has already been applied
        log_r = float(np.log(b1 / b0) - np.log(a1 / a0))
        sig = float(np.hypot(mad / max(np.sqrt(len(B[c])), 1.0),
                             mad / max(np.sqrt(len(B[c + 1])), 1.0)))
        if not np.isfinite(log_r) or not (sig > 0):
            continue
        rows[(c, c + 1)] = (log_r, sig)
        rep.append({"chunks": [c, c + 1], "log_r": log_r, "sigma": sig,
                    "s_at_lock": [b0, b1], "already_applied": [a0, a1],
                    "n_anchors": [len(B[c]), len(B[c + 1])]})
    return rows, rep


def loop_rows_artifact(output_dir, ccfg=None) -> Tuple[List[dict], Optional[int], Optional[dict]]:
    """The §5.1 loop rows `correction.visit_drift` measured, the epoch it
    measured them on, and — when the file is NOT this geometry's — what is
    refused and why (``{n, measured_on_epoch, why}``; the rows are then []).

    On pccr the stage solved with ZERO loop rows while the correction module
    was measuring five good ones every pass and spending them all on a
    translation solver (USER 2026-09-19). The closures are 97-99 % RADIAL —
    a DEPTH ratio, which is exactly what this graph solves for.

    FRESHNESS IS THE EVIDENCE STAMP (point 134): the rows carry a ``repro.stamp``
    over the cloud, the poses, the segmentation, the masks, the camera, the
    configuration and the code they were measured with, verified here; never
    the equality of an epoch number.
    """
    from correction.visit_drift_run import read_evidence
    doc, fresh, why = read_evidence(output_dir, "scale_loop_rows.json", ccfg)
    if doc is None:
        return [], None, None
    ep = doc.get("measured_on_epoch")
    ep = int(ep) if ep is not None else None
    rows = list(doc.get("rows") or [])
    if not fresh:
        return [], ep, {"n": len(rows), "measured_on_epoch": ep, "why": why}
    return rows, ep, None


def absolute_rows(output_dir) -> List[Tuple[int, float, float, str]]:
    p = Path(output_dir) / "scale_absolute_rows.json"
    if not p.exists():
        return []
    rep = json.loads(p.read_text())
    return [(int(r["chunk"]), float(r["log_s"]), float(r["sigma"]), str(r.get("source", "regulated")))
            for r in rep.get("rows", []) if r.get("chunk") is not None]


def _chainage_of_chunks(output_dir, session, ranges) -> np.ndarray:
    """Walked distance at the MIDDLE keyframe of every chunk (metres).

    The drift model is a function of the distance WALKED, so the free
    per-chunk solution has to be compared against it on the same axis.
    """
    c = np.asarray(session.poses, np.float64)[:, :3, 3]
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])
    return np.array([float(d[min(int((a + b) // 2), len(d) - 1)]) for a, b in ranges])


def _weighted_median(v, w) -> float:
    """Median of `v` under weights `w`; with equal weights it is `np.median`
    (the two middle values averaged on an even count)."""
    v = np.asarray(v, np.float64)
    w = np.asarray(w, np.float64)
    o = np.argsort(v, kind="stable")
    v, w = v[o], w[o]
    c = np.cumsum(w) / float(w.sum())
    k = int(np.searchsorted(c, 0.5 - 1e-12))
    if abs(float(c[k]) - 0.5) <= 1e-12 and k + 1 < len(v):
        return float(0.5 * (v[k] + v[k + 1]))
    return float(v[k])


# ── the closures as rows of the graph ──────────────────────────────────────

def _closure_rows(loop_measurements: List[dict], owner: np.ndarray, sigma_floor: float
                  ) -> Tuple[List[dict], int]:
    """Every trusted closure as ONE record: its log r, its σ (log units, floored by the row
    σ floor) and its chunk PAIRS with weights — the product of the two copies' chunk shares
    (``chunks_a`` / ``chunks_b``, point 132); a closure without shares (an older row, the
    post-hoc copy rows) is the owner of its keyframe midpoints at weight 1. The share inside
    one chunk weighs nothing; a closure entirely inside one chunk is counted, not a row.
    Returns (closures, n_intra)."""
    out: List[dict] = []
    n_intra = 0
    for m in loop_measurements:
        if not m.get("scale_trusted") or "s_ab" not in m:
            continue
        extent = float(m.get("extent_m", 0.0))
        sig = max(float(sigma_floor),
                  (float(m["residual_m"]) / extent) if extent > 0 else float(sigma_floor))
        _vendor_on_path()
        from loop_utils.loop_bridges import loop_scale_row
        log_r = loop_scale_row(float(m["s_ab"]))
        if isinstance(m.get("chunks_a"), dict) and isinstance(m.get("chunks_b"), dict) \
                and m["chunks_a"] and m["chunks_b"]:
            wa = {int(k): float(v) for k, v in m["chunks_a"].items()}
            wb = {int(k): float(v) for k, v in m["chunks_b"].items()}
        else:
            wa = {int(owner[int(m["i"])]): 1.0}
            wb = {int(owner[int(m["j"])]): 1.0}
        pairs = [(ci, cj, wi * wj) for ci, wi in sorted(wa.items()) for cj, wj in sorted(wb.items())
                 if ci != cj and wi * wj > 0.0]
        intra = float(sum(wi * wj for ci, wi in wa.items() for cj, wj in wb.items() if ci == cj))
        if not pairs:
            # both visits in one chunk: a relative row between a chunk and
            # itself constrains nothing a per-chunk factor can change
            n_intra += 1
            continue
        out.append({"instance_id": m.get("instance_id"), "label": m.get("label"),
                    "s_ab": float(m["s_ab"]), "log_r": float(log_r), "sigma": float(sig),
                    "chunks_a": wa, "chunks_b": wb, "pairs": pairs, "intra_share": intra,
                    # the chunk pair the row mostly stands on, for the report
                    "chunks": [int(max(wa, key=wa.get)), int(max(wb, key=wb.get))]})
    return out, n_intra


def _loop_rel_of(closures: List[dict], extra: Dict[Tuple[int, int], Tuple[float, float]]
                 ) -> Dict[Tuple[int, int], Tuple[float, float]]:
    """{(ci, cj): (log_r, σ)} fused per chunk pair at 1/σ² — every closure's pair rows
    (σ / √weight: a row that stands on a pair with share w weighs w/σ²) and the extra relative
    rows (the DA3 trend)."""
    acc: Dict[Tuple[int, int], Tuple[float, float]] = {}

    def _add(key, log_r, sig):
        if key in acc:
            lr, sg = acc[key]
            w1, w2 = 1.0 / sg ** 2, 1.0 / sig ** 2
            acc[key] = ((lr * w1 + log_r * w2) / (w1 + w2), (1.0 / (w1 + w2)) ** 0.5)
        else:
            acc[key] = (float(log_r), float(sig))

    for c in closures:
        for ci, cj, w in c["pairs"]:
            _add((int(ci), int(cj)), c["log_r"], c["sigma"] / np.sqrt(float(w)))
    for key, (log_r, sig) in extra.items():
        _add((int(key[0]), int(key[1])), log_r, sig)
    return acc


def _predict(closure: dict, x: np.ndarray) -> float:
    """What the per-chunk log factors ``x`` say this closure should read: the share-weighted
    log ratio of copy B's chunks over copy A's (the intra share cancels by construction)."""
    return float(sum(w * x[int(cj)] for cj, w in closure["chunks_b"].items())
                 - sum(w * x[int(ci)] for ci, w in closure["chunks_a"].items()))


def _ramp_eps(closures: List[dict], chain: np.ndarray) -> float:
    """eps of k = 1 + eps·d — ONE parameter along the walk — by weighted least squares (1/σ²)
    on the closures; each closure spans the share-weighted chainage difference of its pairs."""
    num = den = 0.0
    for c in closures:
        d = float(sum(w * (chain[int(cj)] - chain[int(ci)]) for ci, cj, w in c["pairs"]))
        w = 1.0 / c["sigma"] ** 2
        num += w * d * c["log_r"]
        den += w * d * d
    return float(num / den) if den > 0 else 0.0


def judge_models(closures: List[dict], solve_free: Callable[[List[dict]], Optional[np.ndarray]],
                 chain: np.ndarray, error_factor: float, confidence: float,
                 hold_size: bool = False, log: Callable = print) -> dict:
    """Which model earns the right to move the geometry — identity, the ramp or the free
    per-chunk solution — by THE USER'S RULE on the closures themselves (point 127).

    USER 2026-09-22, after test2's epoch 1 came out torn: *"debiamos tener una
    solucion que se adaptara a todos los casos y no se esta logrando"*. The
    question is whether a model explains closures it was NOT fitted on. The
    judges are the closures, each left out once: the model solved without it
    predicts it; ``before`` = what identity leaves of it (|log r|), ``after`` =
    what the model leaves (|log r − prediction|) — BOTH IN UNITS OF THAT
    CLOSURE'S OWN σ, so the error of what is judged is 1 σ and a closure the
    graph priced out (a false identity with σ 1.3 against 0.02 for a genuine
    one) cannot swing the judge: the standing doctrine (USER 2026-09-09 /
    2026-09-22) is that a closure's σ prices it, never that it votes at full
    weight. Then:

      1. the held-out error of each model = the median of its ``after`` (σ units);
      2. the SIMPLEST model whose held-out error lies within ``error_factor`` ×
         the error (1 σ) of the best wins (identity < ramp < free);
      3. a model other than identity is applied only when ``decide_change``
         passes — ≥ 5 judges, the 95 % interval of the paired improvement
         entirely on the improving side, median improvement ≥ the factor × the
         error; otherwise identity, declared with the rule's own reason.

    Measured on the two sessions the first version was written for:
      pccr  13 closures agreeing on x1.10-1.15, r monotone     → applies
      test2 18 closures contradicting each other (0.90-1.18),
            r zigzagging with a -16.7 % and a +19.2 % jump
            between ADJACENT chunks (~2 m of tearing at 10 m) → refuses

    ``hold_size`` (after the precision gauge): the ramp keeps the session's size
    the way the minimum-norm free solution does — mean log r = 0 — instead of
    pinning chunk 0. The gauge is the absolute scale along the WHOLE walk, so
    the start is no more right than the end; the closures are relative and only
    say how the residual is distributed.
    """
    _vendor_on_path()
    from loop_utils.loop_judge import min_judge_closures
    from loop_utils.metric_lock import decide_change
    n = len(closures)
    n_chunks = len(chain)
    min_judges = int(min_judge_closures(float(confidence)))
    rep: Dict[str, Any] = {"rows": n, "n_chunks": n_chunks, "min_judges": min_judges,
                           "hold_size": bool(hold_size), "error_factor": float(error_factor),
                           "confidence": float(confidence)}
    if n == 0:
        rep.update({"verdict": "identity", "reason": "no closure to judge — identity"})
        return rep
    chain = np.asarray(chain, np.float64)

    def _ramp_x(sub):
        x = _ramp_eps(sub, chain) * (chain - float(chain[0]))
        return x - float(np.mean(x)) if hold_size else x

    sig = np.array([float(c["sigma"]) for c in closures])
    before = np.array([abs(c["log_r"]) for c in closures]) / sig
    after_free = np.full(n, np.nan)
    after_ramp = np.full(n, np.nan)
    for i, c in enumerate(closures):
        others = closures[:i] + closures[i + 1:]
        x_f = solve_free(others)
        if x_f is not None:
            after_free[i] = abs(c["log_r"] - _predict(c, np.asarray(x_f, np.float64))) / sig[i]
        after_ramp[i] = abs(c["log_r"] - _predict(c, _ramp_x(others))) / sig[i]
    err = 1.0                                   # σ units: each closure's own measured error
    ok_free = np.all(np.isfinite(after_free))
    e_id = float(np.median(before))
    e_ramp = float(np.median(after_ramp))
    e_free = float(np.median(after_free)) if ok_free else float("inf")
    rep.update({"held_out": n, "error": err, "error_unit": "closure sigma (log units)",
                "sigma_median_log": float(np.median(sig)),
                "err_identity": e_id, "err_ramp": e_ramp,
                "err_free": (e_free if ok_free else None),
                "ramp_eps": float(_ramp_eps(closures, chain))})
    best = min(e_id, e_ramp, e_free)
    bar = best + float(error_factor) * err
    verdict = next(name for name, e in (("identity", e_id), ("ramp", e_ramp), ("free", e_free))
                   if e <= bar)
    rep.update({"best_error": best, "within_error_bar": bar, "model": verdict})
    if verdict == "identity":
        rep.update({"verdict": "identity",
                    "reason": f"identity is within the row σ ({error_factor:g} x each closure's own "
                              f"error) of the best model (held-out errors in σ: identity {e_id:.3f}, "
                              f"ramp {e_ramp:.3f}, free {e_free:.3f}) — the geometry is left alone"})
        log(f"[earn] IDENTITY: {rep['reason']}")
        return rep
    after = after_ramp if verdict == "ramp" else after_free
    dc = decide_change(before, after, error=err, error_factor=float(error_factor),
                       confidence=float(confidence), min_judges=min_judges)
    rep["judge"] = dc
    if not dc["improves"]:
        why = ("within the row σ" if not dc["beyond_error"] else "not significant" if not dc["significant"]
               else "too few judges")
        rep.update({"verdict": "identity", "model_refused": verdict,
                    "reason": f"the {verdict} model does not pass the user's rule ({why}): {dc['reason']}"})
        log(f"[earn] IDENTITY ({verdict} refused): {rep['reason']}")
        return rep
    rep.update({"verdict": verdict,
                "reason": f"the {verdict} model explains the held-out closures by the user's rule "
                          f"(in units of each closure's σ): {dc['reason']} (held-out errors: identity "
                          f"{e_id:.3f}, ramp {e_ramp:.3f}, free {e_free:.3f})"})
    if verdict == "ramp":
        rep["r_ramp"] = [float(v) for v in np.exp(_ramp_x(closures))]
    log(f"[earn] {verdict.upper()}: {rep['reason']}")
    return rep


def stand_down_for_gauge(output_dir, s_da3: Dict[int, float], abs_rows: list,
                         trend_rep: List[dict], vd_rows: List[dict], vd_epoch: Optional[int],
                         log: Callable = print, stale_rows: Optional[dict] = None) -> Optional[dict]:
    """After the precision gauge, only what is MEASURED ON THE CURRENT GEOMETRY
    may drive this graph: the visit-drift closures whose evidence stamp matches
    the session (USER 2026-09-29 — the depth correction from the closures applies
    ALSO after the gauge, on the RESIDUAL, never applying the gauge's drift twice).
    Returns what stands down (None when the gauge did not apply).

    Read from the code, none of the other rows is relative to that geometry:
      · DA3 TREND rows are the lock-time anchor agreements minus the depth
        factor the epoch chain says was applied (`applied_depth_factor`). The
        chain starts at the PRODUCT epoch (`precision/corrected_cloud.py`'s
        ``new_cloud`` epoch) — so on a precision session the row is the RAW
        lock-time drift between chunks, the very drift the gauge already
        removed (pccr epoch 6: −19.7 % between chunks 0-1, +12.6 % between
        4-5, at σ 0.012-0.014 — tighter than any closure).
      · ANCHOR rows restate the lock, not the current geometry.
      · ABSOLUTE rows (`scale_absolute_rows.json`) carry no evidence stamp:
        nothing says which geometry they were measured on. The gauge is the
        session's absolute scale (its own `known_dims` instrument included).
      · visit-drift rows whose stamp does not match ask for a correction the
        geometry may already hold (``stale_rows``, point 134).

    What holds the size then: the gauge. The remaining rows are all RELATIVE
    (closures and seams), so `solve_scale_graph` returns the minimum-norm
    solution — the geometric mean of the factors is 1 and the session's size is
    the one the gauge set (CLAUDE.md, 2026-09-19 block). When the judge prefers
    its RAMP it is re-centred to the same size (`hold_size`), never pinned at
    chunk 0."""
    from precision.gauge import gauge_applied
    from correction.epoch import current_epoch
    if not gauge_applied(output_dir):
        return None
    now = int(current_epoch(output_dir))
    rep = {"reason": "the precision gauge applied (gauge.json): only the closures measured "
                     f"on the current geometry (epoch {now}) drive the depth — the rows below are "
                     "not relative to the current geometry and would apply the gauge's drift twice",
           "current_epoch": now,
           "anchor_rows": sorted(int(k) for k in s_da3),
           "da3_trend_rows": list(trend_rep),
           "absolute_rows": len(abs_rows),
           "visit_drift_rows_stale": ({"n": int(stale_rows["n"]),
                                       "measured_on_epoch": stale_rows.get("measured_on_epoch")}
                                      if stale_rows else None)}
    log(f"[scale-posthoc] precision gauge applied — stood down: {len(rep['anchor_rows'])} "
        f"anchor row(s), {len(trend_rep)} DA3 trend row(s), {len(abs_rows)} absolute row(s)"
        + (f", {stale_rows['n']} closure row(s) whose evidence stamp is not this geometry's "
           f"(measured on epoch {stale_rows.get('measured_on_epoch')}: "
           f"{'; '.join((stale_rows.get('why') or [])[:2])})" if stale_rows else "")
        + " — the closures measured on the current geometry are what drive the depth")
    return rep


def solve_scale_stage(output_dir, session, loop_measurements: List[dict], scfg,
                      log: Callable = print, *, graph=None, ccfg=None) -> dict:
    """Per-chunk correction factors from the post-hoc rows. Returns the
    report: r per chunk, the rows, the judge's verdict, the bound, "applied".

    ``graph``: the loops config's ``graph`` block (the user's error factor and
    confidence); ``ccfg``: the correction configuration (its ``raw`` names the
    configuration the evidence stamp and the judge read). Either missing is read
    from the server's configuration (a CLI / test; the pipeline passes both).

    After the precision gauge only the rows measured on the current geometry
    enter (`stand_down_for_gauge`)."""
    _vendor_on_path()
    from loop_utils.metric_lock import solve_scale_graph
    from correction.config import judge_of
    output_dir = Path(output_dir)
    if graph is not None:
        fac, conf = float(graph.improvement_error_factor), float(graph.heldout_confidence)
    else:
        fac, conf = judge_of(ccfg)
    ranges, owner = chunk_of_keyframes(output_dir, session.n_kf)
    n_chunks = len(ranges)
    applied_k = applied_depth_factor(output_dir, session.frames)
    s_da3, n_anch, moved, anchor_margins = anchor_rows(output_dir, owner, session.frames,
                                                       applied_k=applied_k, error_factor=fac)
    abs_rows = absolute_rows(output_dir)
    vd_rows, vd_epoch, stale_rows = loop_rows_artifact(output_dir, ccfg)
    trend_rel, trend_rep = da3_trend_rows(output_dir, owner, session.frames, applied=applied_k)
    gauge_rep = stand_down_for_gauge(output_dir, s_da3, abs_rows, trend_rep, vd_rows,
                                     vd_epoch, log=log, stale_rows=stale_rows)
    if gauge_rep is not None:
        s_da3, n_anch, moved, abs_rows, trend_rel, trend_rep = {}, {}, {}, [], {}, []
    closures, n_intra = _closure_rows(list(loop_measurements) + list(vd_rows), owner,
                                      float(scfg.sigma_loop_min_log))
    rows_used = [{"instance_id": c["instance_id"], "label": c["label"], "chunks": c["chunks"],
                  "chunks_a": {str(k): v for k, v in c["chunks_a"].items()},
                  "chunks_b": {str(k): v for k, v in c["chunks_b"].items()},
                  "intra_share": c["intra_share"], "s_ab": c["s_ab"], "log_r": c["log_r"],
                  "sigma": c["sigma"]} for c in closures]

    seam_rel = {k: 1.0 for k in range(n_chunks - 1)}
    # An anchor row of a chunk that has NOT moved since the lock is 1.0 by
    # construction: "the geometry has not moved" restates the gauge, it does
    # not measure whether the lock was RIGHT. With no loop row that is exactly
    # the anchor to keep — it is what stopped the chunks drifting back to the
    # raw DA3 medians the lock overruled (pccr 2026-09-13). With loop rows it
    # stops being harmless: on pccr seven such rows at σ 0.03, plus six seam
    # rows at 0.02, outvoted the one loop row 13 to 1 and turned a measured
    # +17.2 % into +2.3 % (USER 2026-09-19). So they stand down exactly when
    # something else can speak, and the report says which and why.
    #
    # WHAT HOLDS THE GAUGE once they do: anchor and absolute rows are the only
    # ABSOLUTE rows (x_k = log s_k); seams and loops are both RELATIVE. With
    # every anchor stood down and no absolute row the system is rank-deficient
    # by exactly one, and `solve_scale_graph`'s lstsq returns the MINIMUM-NORM
    # solution — mean(log r) = 0, the geometric mean of the factors is 1. That
    # is the right gauge here and it is not an accident to be "fixed": the
    # metric lock already set the session's overall size from all its anchors
    # at once, and what stands down is only each chunk's individual pin. The
    # drift is redistributed along the walk; the total size does not move.
    stood_down = sorted(k for k in list(s_da3) if not moved.get(k, True)) if (closures or trend_rel) else []
    for k in stood_down:
        s_da3.pop(k, None)
        n_anch.pop(k, None)
    rep = {"n_chunks": n_chunks, "chunk_ranges": [list(r) for r in ranges],
           "visit_drift_rows": len(vd_rows), "visit_drift_epoch": vd_epoch,
           "loop_rows_refused": stale_rows,
           "da3_trend_rows": trend_rep, "anchor_rows_stood_down": stood_down,
           "anchor_margins": {str(k): v for k, v in sorted(anchor_margins.items())},
           "stood_down_for_gauge": gauge_rep, "loop_rows_intra_chunk": n_intra,
           "loop_rows": rows_used, "anchor_rows": {str(k): {"s": s_da3[k], "n": n_anch[k]} for k in s_da3},
           "absolute_rows": [{"chunk": k, "log_s": ls, "sigma": sg, "source": src} for k, ls, sg, src in abs_rows],
           "sigma_seam_log": float(scfg.sigma_seam_log), "sigma_anchor_log": float(scfg.sigma_anchor_log),
           "error_factor": fac, "confidence": conf}
    # Every row is RELATIVE to the metric lock's state: anchors = how much the
    # geometry moved against DA3 since the lock (1 = nothing new), seams = the
    # chain is glued (1), loop rows = the copies' s_ab, absolute rows = the
    # regulated dimensions. An untouched session with no loop row solves to
    # r ≡ 1 and stays identity; nothing pulls the chunks back toward the raw
    # DA3 medians the lock overruled (pccr 2026-09-13).
    if not s_da3 and not closures and not trend_rel and not abs_rows:
        reason = "no anchor, loop or absolute row — nothing to solve (identity)"
        if gauge_rep is not None:
            # DECLARED: a single-pass session is ONE chunk, so its only degree of
            # freedom is the session's size — the gauge's. Every closure is then
            # relative inside that one chunk and cannot move it.
            reason = (f"after the precision gauge no closure measured on epoch "
                      f"{gauge_rep['current_epoch']} spans two chunks ({n_intra} inside one"
                      + (f"; the session is ONE chunk — its only degree of freedom is its "
                         f"size, which the gauge holds" if n_chunks == 1 else "")
                      + ") — nothing to solve (identity)")
        rep.update({"r": [1.0] * n_chunks, "applied": False, "reason": reason})
        log(f"[scale-posthoc] IDENTITY — {rep['reason']}")
        return rep

    def _solve_free(subset: List[dict]) -> Optional[np.ndarray]:
        """log r per chunk from the anchors, seams, absolute rows, the DA3 trend and the
        closures of ``subset``; None when the graph has nothing to solve or no finite
        positive answer."""
        try:
            r_ = np.asarray(solve_scale_graph(
                s_da3, n_anch, seam_rel, n_chunks, sigma_seam=float(scfg.sigma_seam_log),
                sigma_anchor=float(scfg.sigma_anchor_log),
                loop_rel=_loop_rel_of(subset, trend_rel) or None,
                absolute=[(k, ls, sg) for k, ls, sg, _ in abs_rows] or None), np.float64)
        except ValueError:
            return None
        if not np.all(np.isfinite(r_)) or np.any(r_ <= 0):
            return None
        return np.log(r_)

    # THE JUDGE (point 127): the closures, each left out once, by the user's rule
    earn = judge_models(closures, _solve_free, _chainage_of_chunks(output_dir, session, ranges),
                        fac, conf, hold_size=gauge_rep is not None, log=log)
    rep["earned"] = earn
    if earn["verdict"] == "identity":
        rep.update({"r": [1.0] * n_chunks, "applied": False,
                    "reason": f"did not earn the right: {earn['reason']}"})
        log(f"[scale-posthoc] IDENTITY — {rep['reason']}")
        return rep
    if earn["verdict"] == "ramp":
        r = np.asarray(earn["r_ramp"], np.float64)
    else:
        x = _solve_free(closures)
        if x is None:
            rep.update({"r": [1.0] * n_chunks, "applied": False,
                        "reason": "scale graph solution not finite/positive — declared, identity"})
            log(f"[scale-posthoc] IDENTITY — {rep['reason']}")
            return rep
        r = np.exp(x)
    max_log = float(np.max(np.abs(np.log(r))))
    rep["r"] = [float(v) for v in r]
    rep["max_abs_log_r"] = max_log
    # the BOUND (a 'bounded' gate, point 19's twin): declared in the report, never a verdict
    # on the evidence — a solution beyond it is refused as implausible, by name
    rep["gate"] = {"name": "max_correction_log", "value": max_log,
                   "threshold": float(scfg.max_correction_log), "passed": max_log <= float(scfg.max_correction_log)}
    if not rep["gate"]["passed"]:
        rep.update({"applied": False, "reason": f"|log r| {max_log:.3f} beyond max_correction_log "
                                                 f"{scfg.max_correction_log} — declared, not applied"})
        rep["r"] = [1.0] * n_chunks
        log(f"[scale-posthoc] IDENTITY — {rep['reason']}")
        return rep
    rep["applied"] = True
    rep["judge"] = earn.get("judge")
    rep["reason"] = f"scale graph solved ({earn['verdict']} model by the user's rule)"
    log(f"[scale-posthoc] {len(closures)} closure(s) ({len(vd_rows)} from visit-drift on epoch "
        f"{vd_epoch}, {len(trend_rep)} DA3 trend row(s)), {len(abs_rows)} absolute, "
        f"{len(s_da3)} anchor chunk(s)"
        + (f", {len(stood_down)} anchor row(s) stood down as not moved {stood_down}" if stood_down else "")
        + f" → {earn['verdict']} r = {np.round(r, 4).tolist()} (max |log r| {max_log:.4f})")
    return rep


def scale_transforms(session, ranges: List[Tuple[int, int]], owner: np.ndarray, r: np.ndarray
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """(k_kf, t_kf): depth factor per keyframe (its chunk's r) and the
    translation that scales every chunk about the first keyframe it OWNS while
    the chain stays continuous across the seams: every step of the walk is
    scaled by the factor of the chunk that owns its first keyframe.

    The pivot is the first OWNED keyframe (`owner`, the Omega records' `chunk`
    field), not the first keyframe of the chunk's range: with the 50 % overlap
    of a chunked plan the range starts inside the previous chunk's frames, and
    pivoting there opened a camera step of (r_k − r_{k−1})·(owned start − range
    start) at every ownership boundary — pccr's 83/41 plan puts ~21 keyframes
    (~1.2 m of walk) between the two. Without overlap the two are the same
    keyframe."""
    n = session.n_kf
    r = np.asarray(r, np.float64)
    k_kf = np.array([float(r[int(owner[g])]) for g in range(n)])
    t_kf = np.zeros((n, 3))
    centres = session.poses[:, :3, 3]
    shift = np.zeros(3)
    prev_ref, prev_k = None, None
    for k in range(len(ranges)):
        sel = np.flatnonzero(owner == k)
        if not len(sel):
            continue
        c_ref = centres[int(sel[0])]
        if prev_ref is not None:
            shift = shift + (float(r[prev_k]) - 1.0) * (c_ref - prev_ref)
        for g in sel:
            t_kf[g] = (float(r[k]) - 1.0) * (centres[g] - c_ref) + shift
        prev_ref, prev_k = c_ref, k
    return k_kf, t_kf
