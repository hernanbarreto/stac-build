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


def anchor_rows(output_dir, owner: np.ndarray, frames: List[int]
                ) -> Tuple[Dict[int, float], Dict[int, int], Dict[int, bool]]:
    """Per-chunk DA3 evidence RELATIVE to the metric lock: median over the
    chunk's anchors of agreement_now / agreement_at_lock → (s_da3, n_anchors,
    moved).

    The lock already weighed the raw anchors against the seams and chose;
    re-solving the raw ratios post-hoc only pulled the chunks back toward
    the DA3 medians the lock had overruled (pccr 2026-09-13 21:03: 0 loop
    rows, r down to 0.87, the start↔end revisit went from 20 cm to 1.15 m).
    What the anchors can still say is whether the geometry MOVED against
    them since the lock — an injected or accumulated scale error changes
    the agreement by 1/k (known-answer §10.10), an untouched session leaves
    every ratio at 1 → identity. A ratio of 1 means: nothing new.

    `moved[k]` says whether that chunk's agreement changed AT ALL since the
    lock. It is the honest reading of the row: when nothing moved, every ratio
    is exactly 1 by construction, and "the geometry has not moved" is a
    restatement of the gauge, not a measurement of whether the lock was RIGHT.
    `solve_scale_stage` needs to tell those two apart — see there."""
    p = Path(output_dir) / "scale_diagnostics.json"
    if not p.exists():
        return {}, {}, {}
    from correction.diagnose import _current_agreements
    diag = json.loads(p.read_text())
    now = _current_agreements(diag)
    base = _baseline_agreements(diag)
    if not now or not base:
        return {}, {}, {}
    kf_of = {int(f): k for k, f in enumerate(frames)}
    per_chunk: Dict[int, List[float]] = {}
    for frame, ratio in now.items():
        k = kf_of.get(int(frame))
        b = base.get(int(frame))
        if k is None or b is None or not np.isfinite(ratio) or ratio <= 0 or not np.isfinite(b) or b <= 0:
            continue
        per_chunk.setdefault(int(owner[k]), []).append(float(ratio) / float(b))
    s = {k: float(np.median(v)) for k, v in per_chunk.items()}
    n = {k: len(v) for k, v in per_chunk.items()}
    moved = {k: any(x != 1.0 for x in v) for k, v in per_chunk.items()}
    return s, n, moved


def applied_depth_factor(output_dir, n_kf: int) -> np.ndarray:
    """The cumulative depth factor already applied to each keyframe, walking
    the epoch chain from the live epoch back to 0.

    `_current_agreements` cannot answer this: it reads an `epochs` history in
    `scale_diagnostics.json` that NOTHING writes, so it always falls through to
    `s_f / s_applied` — identical to the baseline, ratio 1.0000 forever (found
    2026-09-19, after epoch 1 had already applied 0.9586-1.0432 and every
    anchor still read "nothing moved"). The epoch npz is the record that DOES
    travel: `k_kf` is exactly what was applied, per keyframe, and
    `parent_epoch` gives the chain — so showing an older epoch reports that
    epoch's history, not the newest one's.
    """
    from correction.epoch import current_epoch, EPOCH_FILE
    from correction import ledger

    output_dir = Path(output_dir)
    k = np.ones(int(n_kf), np.float64)
    seen, ep = set(), int(current_epoch(output_dir))
    while ep > 0 and ep not in seen:
        seen.add(ep)
        try:
            d = ledger.load_epoch_npz(output_dir, ep)
        except Exception:                      # noqa: BLE001 — a missing epoch
            break                              # ends the chain, it never lies
        kk = np.asarray(d.get("k_kf"), np.float64)
        if kk.shape == k.shape:
            k *= kk
        rec = output_dir / f"_epoch_{ep}" / EPOCH_FILE
        if not rec.exists():
            rec = output_dir / EPOCH_FILE
        try:
            ep = int(json.loads(rec.read_text()).get("parent_epoch", ep - 1))
        except Exception:                      # noqa: BLE001
            ep = ep - 1
    return k


def da3_trend_rows(output_dir, owner: np.ndarray, frames: List[int],
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
    ALREADY APPLIED to it (from the epoch chain, see `applied_depth_factor` —
    NOT from `_current_agreements`, which is inert),

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
    applied = applied_depth_factor(output_dir, len(frames))
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


def loop_rows_artifact(output_dir) -> Tuple[List[dict], Optional[int]]:
    """The §5.1 loop rows `correction.visit_drift` measured, and the epoch it
    measured them on.

    On pccr the stage solved with ZERO loop rows while the correction module
    was measuring five good ones every pass and spending them all on a
    translation solver (USER 2026-09-19). The closures are 97-99 % RADIAL —
    a DEPTH ratio, which is exactly what this graph solves for.
    """
    p = Path(output_dir) / "scale_loop_rows.json"
    if not p.exists():
        return [], None
    doc = json.loads(p.read_text())
    ep = doc.get("measured_on_epoch")
    return list(doc.get("rows") or []), (int(ep) if ep is not None else None)


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


def earn_the_right(r_free, closures, chain, sigma_seam: float,
                   min_holdout: int = 3, improve: float = 0.75,
                   hold_size: bool = False, log: Callable = print) -> dict:
    """Does the per-chunk solution EARN the right to touch the geometry?

    USER 2026-09-22, after test2's epoch 1 came out torn: *"debiamos tener una
    solucion que se adaptara a todos los casos y no se esta logrando"*.

    Every other stage that moves this geometry already answers this question
    against data it never saw — `metric_lock.scale_drift_gate` on held-out seam
    ratios, `depth_graph_verdict` on held-out frame pairs, and the intra-chunk
    consensus on held-out strata (on pccr 2026-09-22 it earned NOTHING and
    applied nothing, which is the behaviour being copied here). The post-hoc
    depth correction was the only one that applied unconditionally.

    This is NOT the kind of gate the USER abolished on 2026-09-09 ("no debes
    rechazar correcciones por umbrales arbitrarios"): there is no threshold on
    HOW MUCH may be corrected. The question is whether the model explains
    closures it was not fitted on — the same discipline, and the same 25 %
    margin, the vendor gate already uses.

    Three outcomes, decided by the held-out error:
      · the FREE solution (one factor per chunk) explains them best → apply it
      · the RAMP (k = 1 + eps*d, ONE parameter along the walk) explains them
        best → apply the ramp: a drift ACCUMULATES, so a smooth ramp cannot
        tear a seam, and CLAUDE.md already records it fitting pccr's closures
        better than a 3-DOF translation (rms 16.7 vs 22.8 cm)
      · neither beats leaving the geometry alone → apply NOTHING

    Measured on the two sessions this was written for:
      pccr  13 closures agreeing on x1.10-1.15, r monotone     → applies
      test2 18 closures contradicting each other (0.90-1.18),
            r zigzagging with a -16.7 % and a +19.2 % jump
            between ADJACENT chunks (~2 m of tearing at 10 m) → refuses

    Every closure speaks with its OWN σ (`sigma` of the row, the tangential
    residual over the object's extent — `visit_drift.scale_rows`): the ramp is
    fitted by weighted least squares and the held-out error is a weighted
    median, both at 1/σ², the weights the scale graph itself gives the rows. A
    false identity the graph prices out (pccr epoch 6: a monitor matched 2.8 m
    to the side, σ 1.30 against 0.01 for a genuine closure) used to decide the
    ramp's slope and the verdict anyway, at full weight (2026-09-29).

    `hold_size` (after the precision gauge): the ramp keeps the session's size
    the way the minimum-norm free solution does — mean log r = 0 — instead of
    pinning chunk 0. The gauge is the absolute scale along the WHOLE walk, so
    the start is no more right than the end; the closures are relative and only
    say how the residual is distributed.
    """
    # The sample is the CLOSURES, not the rows the solver fuses them into: one
    # row per chunk PAIR would have turned pccr's 13 independent measurements
    # into 7 and left nothing to hold out (found while testing this, 2026-09-22).
    n = len(r_free)
    rows = []
    for c in (closures or []):
        ch = c.get("chunks") or ()
        if len(ch) == 2 and c.get("log_r") is not None:
            sig = float(c.get("sigma") or 1.0)
            rows.append((int(ch[0]), int(ch[1]), float(c["log_r"]), 1.0 / sig ** 2))
    rep: Dict[str, Any] = {"rows": len(rows), "n_chunks": n}
    if len(rows) < int(min_holdout) * 3:
        rep.update({"verdict": "free", "reason":
                    f"{len(rows)} relative row(s) — too few to hold any out, "
                    f"the solution stands on its own evidence"})
        return rep
    held = [rows[i] for i in range(len(rows)) if i % 3 == 2]
    fit = [rows[i] for i in range(len(rows)) if i % 3 != 2]
    if len(held) < int(min_holdout):
        rep.update({"verdict": "free", "reason": f"held-out too thin ({len(held)})"})
        return rep

    x_free = np.log(np.asarray(r_free, np.float64))

    def _ramp_from(sample):
        """eps of k = 1 + eps*d, weighted least squares (1/σ²) on the
        sample's own rows."""
        A, b, sw = [], [], []
        for i, j, lr, w in sample:
            A.append(chain[j] - chain[i])
            b.append(lr)
            sw.append(np.sqrt(w))
        sw = np.asarray(sw, np.float64)
        A = np.asarray(A, np.float64)[:, None] * sw[:, None]
        b = np.asarray(b, np.float64) * sw
        if not len(A) or float(np.squeeze(A.T @ A)) <= 0:
            return 0.0
        return float(np.linalg.lstsq(A, b, rcond=None)[0][0])

    eps = _ramp_from(fit)
    x_ramp = eps * (np.asarray(chain, np.float64) - float(chain[0]))
    if hold_size:
        x_ramp = x_ramp - float(np.mean(x_ramp))   # differences — the judge — unchanged

    def _err(x):
        if not held:
            return np.inf
        return _weighted_median([abs((x[j] - x[i]) - lr) for i, j, lr, _w in held],
                                [w for _i, _j, _lr, w in held])

    e_free, e_ramp, e_none = _err(x_free), _err(x_ramp), _err(np.zeros(n))
    rep.update({"held_out": len(held), "fit_rows": len(fit), "hold_size": bool(hold_size),
                "err_free": round(e_free, 5), "err_ramp": round(e_ramp, 5),
                "err_identity": round(e_none, 5), "ramp_eps": round(eps, 6)})

    # the seam sensor: neighbouring chunks share ~30 frames that measure their
    # relative scale to 0.3-1 %, so a solution demanding far more than the seam
    # prior allows is contradicted by frames it never consulted
    jumps = np.abs(np.diff(x_free))
    worst = float(jumps.max()) if len(jumps) else 0.0
    seam_cap = 6.0 * float(sigma_seam)          # 6 sigma of the seam sensor
    rep.update({"worst_seam_jump": round(float(np.exp(worst) - 1.0), 4),
                "seam_cap": round(float(np.exp(seam_cap) - 1.0), 4)})

    best = min(e_free, e_ramp, e_none)
    if worst > seam_cap and e_ramp <= e_none:
        rep.update({"verdict": "ramp", "reason":
                    f"adjacent chunks demand {np.exp(worst) - 1:.1%} while the "
                    f"shared frames measure their relative scale to "
                    f"{np.exp(seam_cap) - 1:.1%} — the free solution is refuted "
                    f"by evidence it never used; the ramp explains the held-out "
                    f"closures ({e_ramp:.4f} vs identity {e_none:.4f})"})
    elif best == e_free and e_free <= improve * e_none:
        rep.update({"verdict": "free", "reason":
                    f"the per-chunk solution explains the held-out closures "
                    f"({e_free:.4f} vs identity {e_none:.4f})"})
    elif best == e_ramp and e_ramp <= improve * e_none:
        rep.update({"verdict": "ramp", "reason":
                    f"the ramp explains the held-out closures better "
                    f"({e_ramp:.4f} vs free {e_free:.4f}, identity {e_none:.4f})"})
    else:
        rep.update({"verdict": "none", "reason":
                    f"no model explains the closures it did not see "
                    f"(free {e_free:.4f}, ramp {e_ramp:.4f}, identity "
                    f"{e_none:.4f}) — the geometry is left alone"})
    if rep["verdict"] == "ramp":
        rep["r_ramp"] = [float(v) for v in np.exp(x_ramp)]
    log(f"[earn] {rep['verdict'].upper()}: {rep['reason']}")
    return rep


def stand_down_for_gauge(output_dir, s_da3: Dict[int, float], abs_rows: list,
                         trend_rep: List[dict], vd_rows: List[dict], vd_epoch: Optional[int],
                         log: Callable = print) -> Optional[dict]:
    """After the precision gauge, only what is MEASURED ON THE CURRENT GEOMETRY
    may drive this graph: the visit-drift closures stamped with the live epoch
    (USER 2026-09-29 — the depth correction from the closures applies ALSO
    after the gauge, on the RESIDUAL, never applying the gauge's drift twice).
    Returns what stands down (None when the gauge did not apply).

    Read from the code, none of the other rows is relative to that geometry:
      · DA3 TREND rows are the lock-time anchor agreements minus the depth
        factor the epoch chain says was applied (`applied_depth_factor`). The
        chain stops at the first epoch without a persisted transform, and the
        core's corrected cloud (`precision/corrected_cloud.py`) is a
        ``new_cloud`` epoch with none — so on a precision session the row is
        the RAW lock-time drift between chunks, the very drift the gauge
        already removed (pccr epoch 6: −19.7 % between chunks 0-1, +12.6 %
        between 4-5, at σ 0.012-0.014 — tighter than any closure).
      · ANCHOR rows read the `epochs` history of scale_diagnostics.json, which
        the gauge's epoch appends with NO depth factor (`apply_pose_epoch`
        passes an empty map) and the corrected cloud does not append at all:
        they restate the lock, not the current geometry. (Their `moved` test is
        also defeated by the history's 6-decimal rounding — on pccr every
        chunk read "moved", so none ever stood down.)
      · ABSOLUTE rows (`scale_absolute_rows.json`) carry no epoch stamp: nothing
        says which geometry they were measured on. The gauge is the session's
        absolute scale (its own `known_dims` instrument included).
      · visit-drift rows stamped with an OLDER epoch ask for a correction the
        geometry may already hold.

    What holds the size then: the gauge. The remaining rows are all RELATIVE
    (closures and seams), so `solve_scale_graph` returns the minimum-norm
    solution — the geometric mean of the factors is 1 and the session's size is
    the one the gauge set (CLAUDE.md, 2026-09-19 block). When `earn_the_right`
    prefers its RAMP it is re-centred to the same size (`hold_size`), never
    pinned at chunk 0."""
    from precision.gauge import gauge_applied
    from correction.epoch import current_epoch
    if not gauge_applied(output_dir):
        return None
    now = int(current_epoch(output_dir))
    stale = bool(vd_rows) and vd_epoch != now
    rep = {"reason": "the precision gauge applied (gauge.json): only the closures measured "
                     f"on the current epoch ({now}) drive the depth — the rows below are not "
                     "relative to the current geometry and would apply the gauge's drift twice",
           "current_epoch": now,
           "anchor_rows": sorted(int(k) for k in s_da3),
           "da3_trend_rows": list(trend_rep),
           "absolute_rows": len(abs_rows),
           "visit_drift_rows_stale": ({"n": len(vd_rows), "measured_on_epoch": vd_epoch}
                                      if stale else None)}
    log(f"[scale-posthoc] precision gauge applied — stood down: {len(rep['anchor_rows'])} "
        f"anchor row(s), {len(trend_rep)} DA3 trend row(s), {len(abs_rows)} absolute row(s)"
        + (f", {len(vd_rows)} closure row(s) measured on epoch {vd_epoch} (stale, now {now})"
           if stale else "")
        + " — the closures measured on the current epoch are what drive the depth")
    return rep


def solve_scale_stage(output_dir, session, loop_measurements: List[dict], scfg, log: Callable = print
                      ) -> dict:
    """Per-chunk correction factors from the post-hoc rows. Returns the
    report: r per chunk, the rows, the gate, "applied".

    After the precision gauge only the rows measured on the current geometry
    enter (`stand_down_for_gauge`)."""
    _vendor_on_path()
    from loop_utils.metric_lock import solve_scale_graph
    from loop_utils.loop_bridges import loop_scale_row
    output_dir = Path(output_dir)
    ranges, owner = chunk_of_keyframes(output_dir, session.n_kf)
    n_chunks = len(ranges)
    s_da3, n_anch, moved = anchor_rows(output_dir, owner, session.frames)
    abs_rows = absolute_rows(output_dir)
    vd_rows, vd_epoch = loop_rows_artifact(output_dir)
    trend_rel, trend_rep = da3_trend_rows(output_dir, owner, session.frames)
    gauge_rep = stand_down_for_gauge(output_dir, s_da3, abs_rows, trend_rep, vd_rows,
                                     vd_epoch, log=log)
    if gauge_rep is not None:
        s_da3, n_anch, moved, abs_rows, trend_rel, trend_rep = {}, {}, {}, [], {}, []
        if gauge_rep["visit_drift_rows_stale"]:
            vd_rows = []
    loop_rel: Dict[Tuple[int, int], Tuple[float, float]] = {}
    rows_used = []
    n_intra = 0
    for m in list(loop_measurements) + list(vd_rows):
        if not m.get("scale_trusted") or "s_ab" not in m:
            continue
        ci, cj = int(owner[int(m["i"])]), int(owner[int(m["j"])])
        if ci == cj:
            # both visits in one chunk: a relative row between a chunk and
            # itself constrains nothing a per-chunk factor can change
            n_intra += 1
            continue
        extent = float(m.get("extent_m", 0.0))
        sig = max(float(scfg.sigma_loop_min_log),
                  (float(m["residual_m"]) / extent) if extent > 0 else float(scfg.sigma_loop_min_log))
        key = (ci, cj)
        log_r = loop_scale_row(float(m["s_ab"]))
        if key in loop_rel:                   # several instances on the same chunk pair: fuse
            lr, sg = loop_rel[key]
            w1, w2 = 1.0 / sg ** 2, 1.0 / sig ** 2
            loop_rel[key] = ((lr * w1 + log_r * w2) / (w1 + w2), (1.0 / (w1 + w2)) ** 0.5)
        else:
            loop_rel[key] = (log_r, sig)
        rows_used.append({"instance_id": m.get("instance_id"), "chunks": [ci, cj], "s_ab": m["s_ab"],
                          "log_r": log_r, "sigma": sig})
    # the SHAPE of the DA3 drift, as relative rows between consecutive chunks.
    # Same fusion as the loop rows: several sources on one chunk pair are one
    # weighted measurement, not two votes.
    for key, (log_r, sig) in trend_rel.items():
        if key in loop_rel:
            lr, sg = loop_rel[key]
            w1, w2 = 1.0 / sg ** 2, 1.0 / sig ** 2
            loop_rel[key] = ((lr * w1 + log_r * w2) / (w1 + w2), (1.0 / (w1 + w2)) ** 0.5)
        else:
            loop_rel[key] = (log_r, sig)

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
    stood_down = sorted(k for k in list(s_da3) if not moved.get(k, True)) if loop_rel else []
    for k in stood_down:
        s_da3.pop(k, None)
        n_anch.pop(k, None)
    rep = {"n_chunks": n_chunks, "chunk_ranges": [list(r) for r in ranges],
           "visit_drift_rows": len(vd_rows), "visit_drift_epoch": vd_epoch,
           "da3_trend_rows": trend_rep, "anchor_rows_stood_down": stood_down,
           "stood_down_for_gauge": gauge_rep, "loop_rows_intra_chunk": n_intra,
           "loop_rows": rows_used, "anchor_rows": {str(k): {"s": s_da3[k], "n": n_anch[k]} for k in s_da3},
           "absolute_rows": [{"chunk": k, "log_s": ls, "sigma": sg, "source": src} for k, ls, sg, src in abs_rows],
           "sigma_seam_log": float(scfg.sigma_seam_log), "sigma_anchor_log": float(scfg.sigma_anchor_log)}
    # Every row is RELATIVE to the metric lock's state: anchors = how much the
    # geometry moved against DA3 since the lock (1 = nothing new), seams = the
    # chain is glued (1), loop rows = the copies' s_ab, absolute rows = the
    # regulated dimensions. An untouched session with no loop row solves to
    # r ≡ 1 and stays identity below the row σ; nothing pulls the chunks back
    # toward the raw DA3 medians the lock overruled (pccr 2026-09-13).
    if not s_da3 and not loop_rel and not abs_rows:
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
    r = np.asarray(solve_scale_graph(
        s_da3, n_anch, seam_rel, n_chunks, sigma_seam=float(scfg.sigma_seam_log),
        sigma_anchor=float(scfg.sigma_anchor_log), loop_rel=loop_rel or None,
        absolute=[(k, ls, sg) for k, ls, sg, _ in abs_rows] or None), np.float64)
    if not np.all(np.isfinite(r)) or np.any(r <= 0):
        rep.update({"r": [1.0] * n_chunks, "applied": False,
                    "reason": "scale graph solution not finite/positive — declared, identity"})
        log(f"[scale-posthoc] IDENTITY — {rep['reason']}")
        return rep
    # EARN THE RIGHT (USER 2026-09-22): the solution must explain closures it
    # was not fitted on, or step aside — see `earn_the_right`.
    try:
        # after the gauge the ramp keeps the gauge's size, as the free solution does
        _earn = earn_the_right(r, rows_used, _chainage_of_chunks(output_dir, session, ranges),
                               float(scfg.sigma_seam_log), hold_size=gauge_rep is not None,
                               log=log)
        rep["earned"] = _earn
        if _earn.get("verdict") == "ramp" and _earn.get("r_ramp"):
            r = np.asarray(_earn["r_ramp"], np.float64)
        elif _earn.get("verdict") == "none":
            rep.update({"r": [1.0] * n_chunks, "applied": False,
                        "reason": f"did not earn the right: {_earn['reason']}"})
            log(f"[scale-posthoc] IDENTITY — {rep['reason']}")
            return rep
    except Exception as _e:  # noqa: BLE001 — the judge never breaks the run
        log(f"[scale-posthoc] ⚠ self-validation failed ({_e}) — solution applied "
            f"as solved, undeclared")
        rep["earned"] = {"verdict": "error", "reason": str(_e)}

    max_log = float(np.max(np.abs(np.log(r))))
    rep["r"] = [float(x) for x in r]
    rep["max_abs_log_r"] = max_log
    rep["gate"] = {"name": "max_correction_log", "value": max_log,
                   "threshold": float(scfg.max_correction_log), "passed": max_log <= float(scfg.max_correction_log)}
    if not rep["gate"]["passed"]:
        rep.update({"applied": False, "reason": f"|log r| {max_log:.3f} beyond max_correction_log "
                                                 f"{scfg.max_correction_log} — declared, not applied"})
        rep["r"] = [1.0] * n_chunks
        log(f"[scale-posthoc] IDENTITY — {rep['reason']}")
        return rep
    # significance: a correction smaller than the tightest row's σ is within
    # the measurement noise — declared, not applied
    rep["applied"] = bool(max_log > float(scfg.sigma_loop_min_log))
    if not rep["applied"]:
        rep["r"] = [1.0] * n_chunks
    rep["reason"] = ("scale graph solved" if rep["applied"]
                     else f"solution within the row σ ({max_log:.5f} ≤ {scfg.sigma_loop_min_log}) — identity")
    log(f"[scale-posthoc] {len(loop_rel)} relative row(s) ({len(vd_rows)} closure(s) from "
        f"visit-drift on epoch {vd_epoch}, {len(trend_rep)} DA3 trend), "
        f"{len(abs_rows)} absolute, {len(s_da3)} anchor chunk(s)"
        + (f", {len(stood_down)} anchor row(s) stood down as identity-by-construction "
           f"{stood_down}" if stood_down else "")
        + f" → r = {np.round(r, 4).tolist()} (max |log r| {max_log:.4f})")
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
