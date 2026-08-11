"""
Scale-model estimators for scale_align v2 (precision task, Phase A).
================================================================================
The DA3/Omega metric scale today is ONE scalar: s = trimmed median of per-frame
near-band ratios (see scale_align.estimate_scale — that code path is untouched
and stays the production baseline, mode "global_median").

DA3's metric depth has per-scene systematic bias and potentially depth-dependent
structure; a single scalar averages it all in and we never see it. This module
adds two structured estimators of the DA3↔Omega depth relation:

  - "affine_robust":   d_da3 ≈ s·z_omega + b        (Huber IRLS, gain+offset)
  - "depth_dependent": low-order gain in depth —
        depth_linear:  d_da3 ≈ a0·z + a1·z²          (gain r(z) = a0 + a1·z)
        two_segment:   d_da3 ≈ s_near·z | s_far·z    (split at the pooled median z)

Model selection is EVIDENCE-GATED, never cosmetic: a structured model is kept
only if it beats the simpler one under leave-frames-out cross-validation (Huber
predictive loss) AND BIC. Otherwise the mode DEGRADES automatically to the
simpler model and reports why. No overfitting by construction.

GEOMETRY INVARIANT (why the applied correction is always one scalar): Omega's
reconstruction is internally consistent up to exactly ONE gauge degree of
freedom — a global similarity. An offset b or a depth-dependent gain is NOT a
similarity of space; applied per-pixel it would move each frame's surface
differently along its rays and break the multi-view consistency Omega already
has. So the structured models serve two purposes only:
  1. absorb DA3's bias structure so the GAIN estimate is less biased, and
  2. report that structure as a per-session DA3 bias DIAGNOSTIC.
The applied correction is always the single similarity s derived from the
selected model over the near band (the trusted regime — same rationale as the
production near-band ratio).

Everything here is numpy-only and read-only over the output artifacts.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("ScaleModel")

# Per-frame pixel budget for the pooled fits. Deterministic strided subsample —
# no RNG, identical across runs (resume-aware discipline).
MAX_PX_PER_FRAME = 20000

# A structured model must beat the simpler one by this RELATIVE margin on the
# cross-validated predictive loss AND have a lower BIC, or it is degraded.
CV_IMPROVE_MARGIN = 0.02


# ──────────────────────────────────────────────────────────────────────────────
# Sample collection
# ──────────────────────────────────────────────────────────────────────────────

def collect_frame_samples(output_dir: Path, near_frac: float = 0.25,
                          conf_top_frac: float = 0.0) -> List[dict]:
    """Per anchor frame: paired (z_omega, d_da3) samples + the production per-frame
    ratio s_f. Gates (validity, near band, confidence) replicate scale_align._ratio
    semantics so "global_median" built on these s_f is bit-identical to production.

    Returns a list of dicts:
      {num, s_f, z (1D), d (1D), z_near_mask (1D bool), n_px, z_median, d_median}
    ordered by frame number (= capture time). Frames whose ratio fails stay out,
    exactly like production.
    """
    from reconstruction.scale_align import _da3_npz_dir, _omega_depth, _ratio

    frames: List[dict] = []
    da3_dir = _da3_npz_dir(output_dir)
    if da3_dir is None:
        return frames
    omega_depth = _omega_depth(output_dir)
    for n in sorted(omega_depth.keys()):
        p = da3_dir / f"frame_{n}.npz"
        if not p.exists():
            continue
        od = omega_depth[n]
        try:
            npz = np.load(p)
            dd = npz["depth"].astype(np.float32)
            cc = npz["conf"].astype(np.float32) if "conf" in npz.files else None
        except Exception:
            continue
        s_f = _ratio(dd, od, conf=cc, near_frac=near_frac, conf_top_frac=conf_top_frac)
        if s_f is None or not np.isfinite(s_f) or s_f <= 0:
            continue

        H, W = od.shape

        def _fit(a):
            if a is None or a.shape[:2] == (H, W):
                return a
            yi = (np.arange(H) * a.shape[0] / H).astype(int).clip(0, a.shape[0] - 1)
            xi = (np.arange(W) * a.shape[1] / W).astype(int).clip(0, a.shape[1] - 1)
            return a[yi][:, xi]

        dd = _fit(dd)
        cc = _fit(cc)
        m = np.isfinite(dd) & np.isfinite(od) & (dd > 1e-3) & (od > 1e-3)
        if cc is not None and conf_top_frac and conf_top_frac > 0:
            thr = np.percentile(cc[m], 100.0 * (1.0 - float(conf_top_frac))) if m.any() else np.inf
            mc = m & (cc >= thr)
            if mc.sum() >= 50:
                m = mc
        if m.sum() < 100:
            continue
        z = od[m].astype(np.float64)
        d = dd[m].astype(np.float64)
        if z.size > MAX_PX_PER_FRAME:                    # deterministic stride
            step = int(np.ceil(z.size / MAX_PX_PER_FRAME))
            z, d = z[::step], d[::step]
        near = z <= np.percentile(z, float(near_frac) * 100.0)
        frames.append({
            "num": int(n), "s_f": float(s_f),
            "z": z, "d": d, "z_near_mask": near, "n_px": int(z.size),
            "z_median": float(np.median(z)), "d_median": float(np.median(d)),
        })
    return frames


def _pooled(frames: List[dict], near_only: bool = True
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool samples across frames with EQUAL per-frame total weight (a big frame
    must not dominate the fit — the scale is a property of the session, not of the
    frame with the most valid pixels). near_only keeps the trusted near band."""
    zs, ds, ws = [], [], []
    for f in frames:
        m = f["z_near_mask"] if near_only else np.ones(f["z"].size, bool)
        if m.sum() < 20:
            m = np.ones(f["z"].size, bool)               # starved near band → all valid
        z, d = f["z"][m], f["d"][m]
        zs.append(z); ds.append(d)
        ws.append(np.full(z.size, 1.0 / z.size))
    if not zs:
        return np.array([]), np.array([]), np.array([])
    return np.concatenate(zs), np.concatenate(ds), np.concatenate(ws)


# ──────────────────────────────────────────────────────────────────────────────
# Robust fitting (Huber IRLS)
# ──────────────────────────────────────────────────────────────────────────────

def _huber_irls(X: np.ndarray, y: np.ndarray, w0: Optional[np.ndarray] = None,
                iters: int = 30, tol: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    """Weighted Huber IRLS for y ≈ X·θ. Returns (θ, final robust weights·w0).
    Huber delta = 1.345·σ̂ (σ̂ = 1.4826·MAD of residuals), re-estimated per iteration."""
    w0 = np.ones(y.size) if w0 is None else np.asarray(w0, np.float64)
    w = w0.copy()
    theta = np.zeros(X.shape[1])
    for _ in range(iters):
        WX = X * w[:, None]
        A = WX.T @ X
        b = WX.T @ y
        try:
            new = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            break
        r = y - X @ new
        sigma = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-12
        delta = 1.345 * sigma
        a = np.abs(r)
        w = w0 * np.where(a <= delta, 1.0, delta / a)
        if np.max(np.abs(new - theta)) < tol * (1.0 + np.max(np.abs(theta))):
            theta = new
            break
        theta = new
    return theta, w


def _huber_loss(r: np.ndarray, w: Optional[np.ndarray] = None,
                sigma: Optional[float] = None) -> float:
    """Mean (weighted) Huber loss of residuals r — the CV comparison metric."""
    w = np.ones(r.size) if w is None else np.asarray(w, np.float64)
    if sigma is None:
        sigma = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-12
    delta = 1.345 * sigma
    a = np.abs(r)
    loss = np.where(a <= delta, 0.5 * a * a, delta * (a - 0.5 * delta))
    return float(np.sum(loss * w) / (np.sum(w) + 1e-12))


def _fit_model(kind: str, z: np.ndarray, d: np.ndarray, w: np.ndarray,
               z_split: Optional[float] = None) -> dict:
    """Fit one model kind on pooled samples. z is normalized internally by its
    median for conditioning; params are returned DE-normalized.
      scale_only:   d = s·z
      affine:       d = s·z + b
      depth_linear: d = a0·z + a1·z²          (gain r(z) = a0 + a1·z)
      two_segment:  d = s_near·z | s_far·z    (split at z_split)
    """
    zm = float(np.median(z)) + 1e-12
    u = z / zm
    if kind == "scale_only":
        th, _ = _huber_irls(u[:, None], d, w)
        params = {"s": float(th[0] / zm)}
    elif kind == "affine":
        th, _ = _huber_irls(np.stack([u, np.ones_like(u)], 1), d, w)
        params = {"s": float(th[0] / zm), "b": float(th[1])}
    elif kind == "depth_linear":
        th, _ = _huber_irls(np.stack([u, u * u], 1), d, w)
        params = {"a0": float(th[0] / zm), "a1": float(th[1] / (zm * zm))}
    elif kind == "two_segment":
        zs = float(np.median(z)) if z_split is None else float(z_split)
        lo, hi = z <= zs, z > zs
        out = {"z_split": zs}
        for name, m in (("s_near", lo), ("s_far", hi)):
            if m.sum() >= 50:
                th, _ = _huber_irls(u[m][:, None], d[m], w[m])
                out[name] = float(th[0] / zm)
        if "s_near" not in out or "s_far" not in out:   # a starved side → not fittable
            return {"kind": kind, "params": None, "n_params": 2}
        params = out
    else:
        raise ValueError(f"unknown model kind: {kind}")
    return {"kind": kind, "params": params, "n_params": 1 if kind == "scale_only" else 2}


def _predict(model: dict, z: np.ndarray) -> np.ndarray:
    p = model["params"]
    k = model["kind"]
    if k == "scale_only":
        return p["s"] * z
    if k == "affine":
        return p["s"] * z + p["b"]
    if k == "depth_linear":
        return p["a0"] * z + p["a1"] * z * z
    if k == "two_segment":
        return np.where(z <= p["z_split"], p["s_near"] * z, p["s_far"] * z)
    raise ValueError(k)


def _cv_frame_loss(frames: List[dict], kind: str, k: int = 6,
                   z_split: Optional[float] = None) -> Optional[float]:
    """Leave-frames-out cross-validation: fit on train FRAMES, predictive Huber
    loss on held-out FRAMES (relative residuals (d̂−d)/d so near/far frames weigh
    comparably). Folding by frame — not by pixel — is what makes this an honest
    generalization test: pixels within a frame share the frame's bias."""
    n = len(frames)
    if n < 3:
        return None
    k = max(2, min(k, n))
    losses = []
    for fold in range(k):
        test_idx = set(range(fold, n, k))
        train = [f for i, f in enumerate(frames) if i not in test_idx]
        test = [f for i, f in enumerate(frames) if i in test_idx]
        if not train or not test:
            continue
        z, d, w = _pooled(train)
        if z.size < 100:
            continue
        model = _fit_model(kind, z, d, w, z_split=z_split)
        if model["params"] is None:
            return None                                   # not fittable on some fold
        zt, dt, wt = _pooled(test)
        if zt.size < 50:
            continue
        r = (_predict(model, zt) - dt) / np.maximum(dt, 1e-6)
        losses.append(_huber_loss(r, wt))
    return float(np.mean(losses)) if losses else None


def _bic(frames: List[dict], model: dict) -> Optional[float]:
    """BIC over the pooled near-band samples with robust variance (Huber loss in
    place of the Gaussian RSS — standard robust-BIC surrogate)."""
    z, d, w = _pooled(frames)
    if z.size < 100 or model["params"] is None:
        return None
    r = (_predict(model, z) - d) / np.maximum(d, 1e-6)
    loss = _huber_loss(r, w)
    n = z.size
    return float(n * np.log(max(loss, 1e-18)) + model["n_params"] * np.log(n))


# ──────────────────────────────────────────────────────────────────────────────
# Model selection + applied gain
# ──────────────────────────────────────────────────────────────────────────────

def select_model(frames: List[dict], mode: str) -> dict:
    """Evidence-gated selection for the requested mode. Returns:
      {mode_requested, mode_used, degraded, degrade_reason, model, cv, bic}
    where model is the WINNING fitted model on all frames. "global_median" never
    enters here (it is the untouched production path)."""
    z, d, w = _pooled(frames)
    if z.size < 200:
        return {"mode_requested": mode, "mode_used": "global_median", "degraded": True,
                "degrade_reason": f"only {z.size} pooled samples (<200) — structured fit "
                                  f"unjustifiable, degraded to global_median",
                "model": None, "cv": {}, "bic": {}}

    cv: Dict[str, Optional[float]] = {"scale_only": _cv_frame_loss(frames, "scale_only")}
    fitted: Dict[str, dict] = {"scale_only": _fit_model("scale_only", z, d, w)}
    bic: Dict[str, Optional[float]] = {"scale_only": _bic(frames, fitted["scale_only"])}

    def _consider(kind: str):
        fitted[kind] = _fit_model(kind, z, d, w)
        cv[kind] = None if fitted[kind]["params"] is None else _cv_frame_loss(frames, kind)
        bic[kind] = _bic(frames, fitted[kind])

    def _wins(challenger: str, incumbent: str) -> bool:
        c, i = cv.get(challenger), cv.get(incumbent)
        cb, ib = bic.get(challenger), bic.get(incumbent)
        if c is None or i is None or cb is None or ib is None:
            return False
        return (i - c) / max(i, 1e-18) >= CV_IMPROVE_MARGIN and cb < ib

    if mode == "affine_robust":
        _consider("affine")
        if _wins("affine", "scale_only"):
            return {"mode_requested": mode, "mode_used": "affine_robust", "degraded": False,
                    "degrade_reason": None, "model": fitted["affine"], "cv": cv, "bic": bic}
        return {"mode_requested": mode, "mode_used": "scale_only_robust", "degraded": True,
                "degrade_reason": "offset b did not improve leave-frames-out CV/BIC over "
                                  "scale-only — degraded (no overfitting)",
                "model": fitted["scale_only"], "cv": cv, "bic": bic}

    if mode == "depth_dependent":
        _consider("affine")
        _consider("depth_linear")
        _consider("two_segment")
        # incumbent ladder: scale_only → affine (if it earns its keep)
        incumbent = "affine" if _wins("affine", "scale_only") else "scale_only"
        # best structured candidate by CV
        cand = [k for k in ("depth_linear", "two_segment") if cv.get(k) is not None]
        cand.sort(key=lambda k: cv[k])
        if cand and _wins(cand[0], incumbent):
            return {"mode_requested": mode, "mode_used": f"depth_dependent:{cand[0]}",
                    "degraded": False, "degrade_reason": None,
                    "model": fitted[cand[0]], "cv": cv, "bic": bic}
        used = "affine_robust" if incumbent == "affine" else "scale_only_robust"
        return {"mode_requested": mode, "mode_used": used, "degraded": True,
                "degrade_reason": "residual structure does not justify a depth-dependent "
                                  "gain under leave-frames-out CV/BIC — degraded",
                "model": fitted[incumbent], "cv": cv, "bic": bic}

    raise ValueError(f"unknown scale mode: {mode}")


def applied_gain(model: Optional[dict], frames: List[dict]) -> Optional[float]:
    """The single similarity s the pipeline applies, derived from the selected
    model over the NEAR-BAND depth distribution (the trusted regime): the robust
    median of the model's local gain d̂(z)/z over pooled near-band samples. For
    scale_only/affine this is s (the offset absorbed DA3's bias and does NOT
    enter the applied similarity — see the module docstring invariant)."""
    if model is None or model["params"] is None:
        return None
    if model["kind"] in ("scale_only", "affine"):
        return float(model["params"]["s"])
    z, _, _ = _pooled(frames)
    if z.size < 50:
        return None
    gain = _predict(model, z) / np.maximum(z, 1e-9)
    return float(np.median(gain))


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostics: dispersion, jackknife, residual profile, confidence
# ──────────────────────────────────────────────────────────────────────────────

def trimmed_median_s(frames: List[dict]) -> Optional[float]:
    """Production baseline aggregate: 10–90% trimmed median of per-frame s_f —
    bit-identical to scale_align.estimate_scale on the same frames."""
    ratios = np.array([f["s_f"] for f in frames], np.float64)
    if ratios.size < 3:
        return None
    lo, hi = np.percentile(ratios, [10, 90])
    return float(np.median(ratios[(ratios >= lo) & (ratios <= hi)]))


def jackknife_s(frames: List[dict], estimator=None) -> Optional[dict]:
    """Leave-one-anchor-out stability of the aggregate s. estimator defaults to
    the production trimmed median. Reports the spread as a health metric."""
    est = estimator or trimmed_median_s
    if len(frames) < 4:
        return None
    vals = []
    for i in range(len(frames)):
        s = est([f for j, f in enumerate(frames) if j != i])
        if s is not None and np.isfinite(s):
            vals.append(s)
    if len(vals) < 3:
        return None
    v = np.array(vals)
    s_all = est(frames)
    mad = float(np.median(np.abs(v - np.median(v))))
    return {"n": int(v.size), "s_min": float(v.min()), "s_max": float(v.max()),
            "max_dev_rel": float(np.max(np.abs(v - s_all)) / abs(s_all)) if s_all else None,
            "mad_rel": float(mad / abs(np.median(v))) if np.median(v) else None}


def residual_depth_profile(frames: List[dict], s_applied: float,
                           n_bins: int = 8) -> List[dict]:
    """Median relative residual (s_applied·z − d)/d binned by METRIC depth d —
    the 'residual vs depth' diagnostic: systematic structure here is DA3 bias
    (or a depth-dependent gain the selected model chose not to keep)."""
    zs, ds = [], []
    for f in frames:
        zs.append(f["z"]); ds.append(f["d"])
    if not zs:
        return []
    z = np.concatenate(zs); d = np.concatenate(ds)
    r = (s_applied * z - d) / np.maximum(d, 1e-6)
    edges = np.percentile(d, np.linspace(0, 100, n_bins + 1))
    out = []
    for i in range(n_bins):
        m = (d >= edges[i]) & (d <= edges[i + 1] if i == n_bins - 1 else d < edges[i + 1])
        if m.sum() < 50:
            continue
        out.append({"d_bin_center_m": float(0.5 * (edges[i] + edges[i + 1])),
                    "median_residual_pct": float(100.0 * np.median(r[m])),
                    "n_px": int(m.sum())})
    return out


def scale_confidence(mad_rel: Optional[float], n_anchors: int,
                     heldout_loss: Optional[float],
                     vio_agreement_pct: Optional[float]) -> Tuple[float, dict]:
    """Aggregate health metric in [0,1]. Monotone, documented terms:
      dispersion: 1 at MAD/s = 0, 0 at ≥15% (anchor ratios all over the place)
      anchors:    n/12 capped at 1 (12 = validated production default)
      residual:   1 at CV predictive Huber loss (relative residuals) 0, 0 at ≥0.02
                  (0.02 ≈ 14% typical residual under Huber — depth model useless)
      vio:        1 at 0% VIO↔DA3 disagreement, 0 at ≥15% (only when VIO present)
    Weighted mean; weights renormalize over the terms that exist."""
    terms: Dict[str, float] = {}
    weights: Dict[str, float] = {}
    if mad_rel is not None:
        terms["dispersion"] = float(np.clip(1.0 - mad_rel / 0.15, 0.0, 1.0))
        weights["dispersion"] = 0.40
    terms["anchors"] = float(np.clip(n_anchors / 12.0, 0.0, 1.0))
    weights["anchors"] = 0.20
    if heldout_loss is not None:
        terms["residual"] = float(np.clip(1.0 - heldout_loss / 0.02, 0.0, 1.0))
        weights["residual"] = 0.25
    if vio_agreement_pct is not None:
        terms["vio_agreement"] = float(np.clip(1.0 - abs(vio_agreement_pct) / 15.0, 0.0, 1.0))
        weights["vio_agreement"] = 0.15
    wsum = sum(weights.values())
    conf = sum(terms[k] * weights[k] for k in terms) / wsum if wsum else 0.0
    return float(conf), terms


# ──────────────────────────────────────────────────────────────────────────────
# Anchor top-up: depth-range coverage (Phase A.2)
# ──────────────────────────────────────────────────────────────────────────────

def plan_depth_coverage_topup(output_dir: Path, sel_files: List[str],
                              max_topup: int = 8, n_bins: int = 6) -> List[str]:
    """Anchors are picked uniformly in TIME before any depth exists. After the
    omega pass the per-keyframe omega depth is on disk — check that the anchors
    also cover the scene's DEPTH RANGE and plan extra DA3 extractions for the
    uncovered bins (greedy: the keyframe whose median depth is closest to each
    empty bin's center). Returns frame FILE names to extract (may be empty).
    Deterministic; read-only."""
    from reconstruction.scale_align import _da3_npz_dir, _omega_depth
    import os as _os

    omega = _omega_depth(output_dir)
    if not omega or max_topup <= 0:
        return []
    da3_dir = _da3_npz_dir(output_dir)
    have = set()
    if da3_dir is not None:
        for p in da3_dir.glob("frame_*.npz"):
            try:
                have.add(int(p.stem.split("_")[1]))
            except Exception:
                continue
    num_by_file = {}
    for f in sel_files:
        try:
            num_by_file[int(_os.path.splitext(f)[0])] = f
        except Exception:
            continue

    med = {}
    for n, dep in omega.items():
        v = dep[np.isfinite(dep) & (dep > 1e-3)]
        if v.size >= 100:
            med[n] = float(np.median(v[:: max(1, v.size // 5000)]))
    if len(med) < n_bins:
        return []
    vals = np.array(list(med.values()))
    lo, hi = np.percentile(vals, [5, 95])
    if hi <= lo:
        return []
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    def _bin(x):
        return int(np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n_bins - 1))

    covered = {_bin(med[n]) for n in have if n in med}
    plan: List[str] = []
    for bi in range(n_bins):
        if bi in covered or len(plan) >= max_topup:
            continue
        cands = [(abs(med[n] - centers[bi]), n) for n in med
                 if n not in have and n in num_by_file and _bin(med[n]) == bi]
        if not cands:
            continue
        _, best = min(cands)
        plan.append(num_by_file[best])
        have.add(best)
    return plan
