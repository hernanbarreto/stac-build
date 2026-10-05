"""P2 — the continuous metric gauge along the walk (claude_stac.txt §4-F2).

Every instrument that can say "how many metres is this part of the cloud" becomes
ROWS in log space, each with a MEASURED σ:

  ABSOLUTE  da3_windows  per I3 window: the Huber gain of the window's multi-view
                         DA3 depth over Omega's (every valid pixel, weighted by DA3
                         confidence), the window's value = weighted median over its
                         frames, σ = bootstrap over those frames; the near-band
                         variant (scale_align's instrument) is reported beside it
            da3_mono     per keyframe: the same gain on the NESTED model's own
                         monocular metric branch, captured in the same inference
            vio          per VIO segment (reconstruction/vio_scale.py), when present
            stray        per keyframe, Stray LiDAR over Omega, when present
            known_dims   output/known_dimensions.json entries with role "gauge" (F8)
  RELATIVE  visit_drift  scale_loop_rows.json measured on epoch 0: the depth factor
                         between the two visits of a duplicated object

Rows are what the CURRENT cloud still needs: Omega's per-frame depth is written
before the global scale is applied, so an absolute row is log(inst/Omega) − log
s_applied. Chunk seams (metric_lock.json) and DA3 trend rows are NOT rows: the
fork's metric lock already spent the seams, and the trend rows are made of the
same DA3 anchors the absolute rows read — adding them would count one piece of
evidence twice. The seams are reported.

The model: x(c) = log s(c), piecewise linear in the chainage c with knots every
``knot_walk_m``, weighted least squares (1/σ²) plus a second-difference
smoothness whose weight λ is chosen by leave-one-window-out (the grid is
dimensionless: λ × the rows' total weight per knot). σ of a row = max(its own,
its instrument's measured scatter — 1.4826 × MAD of the residuals of the
instrument fitted alone). Each absolute instrument is fitted alone and judged
held-out (leave-one-window-out RMS in log units); the applied one is the DEFAULT
(the first of ``gauge.instruments`` that was judged) unless the one with the lowest
held-out error beats it beyond the sample's noise (``heldout_change`` improves) —
a lower RMS within the noise switches nothing. It is fitted with the relative rows;
every instrument is reported (``sigma_by_instrument``, ``heldout_by_instrument``)
and the choice carries its verdict and decision (``choice_verdict``).

Each per-frame gain is a Huber IRLS iterated to ``huber_tol``; one that reaches
``huber_max_iter`` still moving is not converged — its row is excluded and reported.

The application: keyframe k gets s_k = exp x(c_k) — depth × s_k about its own
camera, the camera moved so the walk stays continuous (c'_k = c'_{k−1} +
½(s_k + s_{k−1})(c_k − c_{k−1})) — published as an epoch through
``precision.poses_epoch.apply_pose_epoch`` (poses only — no cloud before F7), the path every other
correction takes. ``scale_diagnostics.json`` keeps its v1 keys and gains ``v2``.

CLI: ``python -m precision.gauge --session <dir> [--no-apply]``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

GAUGE_NAME = "gauge.json"
GAUGE_VERSION = 1
PROVENANCE = "tool_measured"
LOG_TAG = "[gauge]"
MAD_TO_SIGMA = 1.0 / 0.6744897501960817     # the normal distribution's MAD → σ
BOOT_SEED = 0
BOOT_N = 2000


class GaugeError(RuntimeError):
    """A structural impossibility of the gauge — always with the exact reason."""


@dataclass
class Row:
    instrument: str
    y: float                            # log scale factor (abs) or log ratio b/a (rel)
    sigma: float                        # log-space σ as measured
    group: int                          # the I3 window the row belongs to (held-out unit)
    c: Optional[float] = None           # chainage (absolute rows)
    c_ab: Optional[Tuple[float, float]] = None   # (c_a, c_b) — relative rows: x(c_b) − x(c_a)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def absolute(self) -> bool:
        return self.c_ab is None


# ── the model (pure numpy) ───────────────────────────────────────────────

def knots_for(c_max: float, knot_walk_m: float) -> np.ndarray:
    n = max(2, int(math.ceil(max(c_max, 0.0) / float(knot_walk_m))) + 1)
    return np.linspace(0.0, max(c_max, float(knot_walk_m)), n)


def hat(c: float, knots: np.ndarray) -> np.ndarray:
    """Piecewise-linear interpolation weights of chainage c over the knots
    (clamped to the knot span)."""
    w = np.zeros(len(knots))
    c = float(np.clip(c, knots[0], knots[-1]))
    j = int(np.clip(np.searchsorted(knots, c, side="right") - 1, 0, len(knots) - 2))
    t = (c - knots[j]) / (knots[j + 1] - knots[j])
    w[j], w[j + 1] = 1.0 - t, t
    return w


def design(rows: Sequence[Row], knots: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    A = np.array([hat(r.c, knots) if r.absolute else hat(r.c_ab[1], knots) - hat(r.c_ab[0], knots)
                  for r in rows])
    y = np.array([r.y for r in rows])
    s = np.array([r.sigma for r in rows])
    return A, y, s


def second_diff(K: int) -> np.ndarray:
    D = np.zeros((max(K - 2, 0), K))
    for i in range(K - 2):
        D[i, i:i + 3] = (1.0, -2.0, 1.0)
    return D


def fit(rows: Sequence[Row], knots: np.ndarray, lam: float) -> np.ndarray:
    """x at the knots: min Σ((A x − y)/σ)² + λ_eff ‖D2 x‖², λ_eff = λ × Σ(1/σ²)/K
    (dimensionless λ). Minimum-norm when the rows leave the level free (only
    relative rows)."""
    A, y, s = design(rows, knots)
    w = 1.0 / s
    D = second_diff(len(knots))
    lam_eff = float(lam) * float(np.sum(w ** 2)) / len(knots)
    M = np.vstack([A * w[:, None], math.sqrt(lam_eff) * D]) if len(D) else A * w[:, None]
    b = np.concatenate([y * w, np.zeros(len(D))])
    x, *_ = np.linalg.lstsq(M, b, rcond=None)
    return x


def predict(x: np.ndarray, knots: np.ndarray, r: Row) -> float:
    return float(hat(r.c, knots) @ x) if r.absolute else \
        float((hat(r.c_ab[1], knots) - hat(r.c_ab[0], knots)) @ x)


def loo_residuals(rows: Sequence[Row], knots: np.ndarray, lam: float) -> Optional[np.ndarray]:
    """Every row's residual predicted with its whole window held out; None with fewer
    than two groups."""
    groups = sorted({r.group for r in rows})
    if len(groups) < 2:
        return None
    res = np.full(len(rows), np.nan)
    for g in groups:
        train = [r for r in rows if r.group != g]
        if not any(r.absolute for r in train):
            continue
        x = fit(train, knots, lam)
        for i, r in enumerate(rows):
            if r.group == g:
                res[i] = predict(x, knots, r) - r.y
    res = res[np.isfinite(res)]
    return res if len(res) else None


def loo(rows: Sequence[Row], knots: np.ndarray, lam: float) -> Optional[Dict[int, float]]:
    """Leave-one-window-out: {group: RMS of its rows' held-out residual (log)}; None
    with fewer than two groups."""
    groups = sorted({r.group for r in rows})
    if len(groups) < 2:
        return None
    out = {}
    for g in groups:
        train = [r for r in rows if r.group != g]
        test = [r for r in rows if r.group == g]
        if not any(r.absolute for r in train):
            continue
        x = fit(train, knots, lam)
        e = np.array([predict(x, knots, r) - r.y for r in test])
        out[g] = float(np.sqrt(np.mean(e ** 2)))
    return out or None


def choose_lambda(rows: Sequence[Row], knots: np.ndarray, grid: Sequence[float]
                  ) -> Tuple[float, Optional[float]]:
    """(λ with the lowest mean leave-one-window-out RMS, that RMS); the grid's middle
    value when the rows cannot be held out."""
    best, best_e = None, None
    for lam in grid:
        per = loo(rows, knots, lam)
        if per is None:
            continue
        e = float(np.mean(list(per.values())))
        if best_e is None or e < best_e:
            best, best_e = float(lam), e
    return (best if best is not None else float(grid[len(grid) // 2])), best_e


def measure_sigma(rows: List[Row], knots: np.ndarray, grid: Sequence[float]) -> float:
    """The instrument's own scatter: 1.4826 × MAD of its HELD-OUT residuals (each
    window predicted without it — in-sample residuals shrink as the curve bends to
    the noise), fitted alone on unit weights; each row's σ becomes max(own, that).
    Returns the scatter."""
    own = [r.sigma if np.isfinite(r.sigma) and r.sigma > 0 else None for r in rows]
    for r in rows:                      # the scatter is measured on unit weights
        r.sigma = 1.0
    lam, _ = choose_lambda(rows, knots, grid)
    res = loo_residuals(rows, knots, lam)
    if res is None:                     # one window: nothing to hold out, in-sample it is
        x = fit(rows, knots, lam)
        res = np.array([predict(x, knots, r) - r.y for r in rows])
    scatter = float(MAD_TO_SIGMA * np.median(np.abs(res - np.median(res))))
    for r, o in zip(rows, own):
        r.sigma = max(o, scatter) if o is not None else scatter
    if not all(r.sigma > 0 for r in rows):            # a perfect instrument: its fit floor
        floor = float(np.finfo(float).eps) * max(1.0, float(np.max(np.abs([r.y for r in rows]))))
        for r in rows:
            r.sigma = max(r.sigma, floor)
    return scatter


def solve(rows: Sequence[Row], c_max: float, gcfg, log: Callable = print) -> Dict[str, Any]:
    """Every absolute instrument judged alone, the winner fitted with the relative
    rows. Returns the model record."""
    _vendor_path()
    from loop_utils.metric_lock import heldout_change      # vendor/VGGT-Long
    knots = knots_for(c_max, gcfg.knot_walk_m)
    grid = list(gcfg.smooth_grid)
    rel = [r for r in rows if not r.absolute]
    by_inst: Dict[str, List[Row]] = {}
    for r in rows:
        if r.absolute:
            by_inst.setdefault(r.instrument, []).append(r)
    sigma_by, held_by, per_group_by, lam_by = {}, {}, {}, {}
    for name, rs in by_inst.items():
        sigma_by[name] = measure_sigma(rs, knots, grid)
        lam, e = choose_lambda(rs, knots, grid)
        lam_by[name], held_by[name] = lam, e
        per_group_by[name] = loo(rs, knots, lam) or {}
    judged = {k: v for k, v in held_by.items() if v is not None}
    if not judged:
        raise GaugeError("no absolute instrument spans two I3 windows — nothing can be held "
                         "out, the gauge cannot choose (instruments present: "
                         f"{sorted(by_inst) or 'none'})")
    # the configured order decides unless the held-out says otherwise BEYOND its noise:
    # the default is the first instrument of ``gauge.instruments`` that was judged; the one
    # with the lowest held-out error replaces it only when heldout_change IMPROVES (a
    # lower RMS inside the sample's noise is not evidence — no noise-driven switch)
    order = [k for k in gcfg.instruments if k in judged] + sorted(set(judged) - set(gcfg.instruments))
    default = order[0]
    lowest = min(order, key=lambda k: judged[k])          # ties → the configured order
    chosen, verdict = default, None

    def _versus(before: str, after: str) -> Dict[str, Any]:
        """Paired over the windows both instruments were held out on (none → neither)."""
        common = sorted(set(per_group_by[before]) & set(per_group_by[after]))
        return heldout_change([per_group_by[before][g] for g in common],
                              [per_group_by[after][g] for g in common],
                              confidence=float(gcfg.heldout_confidence))

    if lowest != default:
        verdict = _versus(default, lowest)
        verdict.update({"against": default, "candidate": lowest})
        if verdict["improves"]:
            chosen = lowest
            verdict["decision"] = "lowest_heldout_beyond_noise"
        else:
            verdict["decision"] = "default_kept_within_noise"
    elif len(order) > 1:
        runner = min(order[1:], key=lambda k: judged[k])
        verdict = _versus(runner, default)
        verdict.update({"against": runner, "candidate": default,
                        "decision": "default_is_lowest"})
    final_rows = list(by_inst[chosen]) + rel
    lam, e = choose_lambda(final_rows, knots, grid)
    x = fit(final_rows, knots, lam)
    log(f"{LOG_TAG} instruments held-out (log RMS): "
        + ", ".join(f"{k} {v:.4f}" for k, v in sorted(judged.items(), key=lambda kv: kv[1]))
        + f" → {chosen}" + (f" ({verdict['decision']}: {verdict['candidate']} vs "
                            f"{verdict['against']} "
                            f"{'better beyond the noise' if verdict['improves'] else 'within the noise'})"
                            if verdict else "")
        + f"; {len(rel)} relative row(s); λ {lam:g}; s along the walk "
          f"{math.exp(x.min()):.4f}–{math.exp(x.max()):.4f}")
    return {"knots_m": knots.tolist(), "x": x.tolist(), "lambda": lam,
            "lambda_by_instrument": lam_by, "heldout_final": e,
            "sigma_by_instrument": sigma_by, "heldout_by_instrument": held_by,
            "applied_instrument": chosen, "choice_verdict": verdict,
            "n_rows": {k: len(v) for k, v in by_inst.items()} | {"relative": len(rel)}}


def continuous_transforms(centres: np.ndarray, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(k_kf, t_kf): depth factor s_k per keyframe and the camera motion that keeps
    the walk continuous — every step of the walk rescaled by the mean factor of its
    two ends, the first camera fixed."""
    centres = np.asarray(centres, np.float64)
    s = np.asarray(s, np.float64)
    new = centres.copy()
    for k in range(1, len(centres)):
        new[k] = new[k - 1] + 0.5 * (s[k] + s[k - 1]) * (centres[k] - centres[k - 1])
    return s.copy(), new - centres


# ── rows ─────────────────────────────────────────────────────────────────

class Gain(NamedTuple):
    log: float                  # the Huber M-estimate of log(inst / omega)
    converged: bool             # the IRLS stopped by ``huber_tol``, not by ``huber_max_iter``
    iterations: int
    mad_zero: bool              # more than half the pixels agree exactly (see huber_location)


def huber_location(r: np.ndarray, w0: np.ndarray, k: float, tol: float,
                   max_iter: int) -> Gain:
    """Huber M-estimate of the location of ``r`` (prior weights ``w0``), the scale
    re-measured every step as 1.4826 × MAD about the current estimate, iterated until
    one step moves it less than ``tol`` × max(1, |μ|); ``max_iter`` only bounds a
    non-converging run, which is REPORTED (converged False), never used as an answer.
    MAD = 0: more than half the samples sit exactly on the estimate — the Huber
    estimate's limit as its scale → 0 is that value (L1 → the median), returned as
    converged and flagged; no stand-in scale is invented."""
    mu = float(np.median(r))
    for it in range(1, int(max_iter) + 1):
        mad = float(np.median(np.abs(r - mu)))
        if mad == 0.0:
            # the scale vanished: the Huber limit is the WEIGHTED median (the prior
            # weights decide which value more than half the weight shares)
            return Gain(_weighted_median(r, w0), True, it - 1, True)
        z = np.abs(r - mu) / (MAD_TO_SIGMA * mad)
        w = w0 * np.where(z <= k, 1.0, k / np.maximum(z, 1e-12))
        new = float(np.sum(w * r) / np.sum(w))
        if abs(new - mu) <= float(tol) * max(1.0, abs(mu)):
            return Gain(new, True, it, False)
        mu = new
    return Gain(mu, False, int(max_iter), False)


def _log_gain(inst: np.ndarray, omega: np.ndarray, conf: Optional[np.ndarray],
              gcfg) -> Optional[Gain]:
    """Huber M-estimate (``huber_k``, to ``huber_tol``) of log(inst/omega) over the pixels
    valid in both, weighted by ``conf`` (the instrument's confidence), at omega's grid.
    None with fewer than four common pixels; a non-converged estimate comes back flagged
    — the caller excludes it."""
    H, W = omega.shape

    def _fit(a):
        if a is None or a.shape[:2] == (H, W):
            return a
        yi = (np.arange(H) * a.shape[0] / H).astype(int).clip(0, a.shape[0] - 1)
        xi = (np.arange(W) * a.shape[1] / W).astype(int).clip(0, a.shape[1] - 1)
        return a[yi][:, xi]

    inst, conf = _fit(inst), _fit(conf)
    m = np.isfinite(inst) & np.isfinite(omega) & (inst > 0) & (omega > 0)
    if conf is not None:
        m &= np.isfinite(conf) & (conf > 0)
    if m.sum() < 4:
        return None
    r = np.log(inst[m].astype(np.float64)) - np.log(omega[m].astype(np.float64))
    w0 = conf[m].astype(np.float64) if conf is not None else np.ones(int(m.sum()))
    return huber_location(r, w0, float(gcfg.huber_k), float(gcfg.huber_tol),
                          int(gcfg.huber_max_iter))


def _weighted_median(v: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(v)
    cw = np.cumsum(w[o])
    return float(v[o][np.searchsorted(cw, 0.5 * cw[-1])])


def omega_depths(output_dir: Path) -> Dict[int, np.ndarray]:
    d = Path(output_dir) / "omega_run" / "results_output"
    out = {}
    for p in d.glob("frame_*.npz"):
        with np.load(p) as z:
            out[int(p.stem.split("_")[1])] = z["depth"].astype(np.float32)
    if not out:
        raise GaugeError(f"no Omega per-frame depth in {d} — the gauge compares against "
                         f"epoch 0's depth (map_worker _emit_omega_depth)")
    return out


def applied_global_scale(output_dir: Path) -> float:
    m = Path(output_dir) / ".metric_scale_applied"
    if not m.exists():
        return 1.0
    return float(m.read_text().strip().split("=")[-1])


def da3_rows(session_dir: Path, chainage: Dict[int, float], log_s0: float, gcfg
             ) -> Tuple[List[Row], List[Row], List[dict]]:
    """(da3_windows rows, da3_mono rows, per-window report incl. the near-band variant).
    A frame whose Huber gain did not converge gives no gain: the window's report lists
    it (``huber.not_converged`` / ``huber.mono_not_converged``; ``huber.mad_zero`` the
    frames whose gain is the agreeing majority's exact value)."""
    from reconstruction.scale_align import _ratio
    from intake.walk import WINDOWS_DIRNAME
    out = Path(session_dir) / "output"
    omega = omega_depths(out)
    wdir = out / WINDOWS_DIRNAME
    paths = sorted(wdir.glob("window_*.npz"))
    if not paths:
        raise GaugeError(f"no I3 window in {wdir} — run the DA3 windows first "
                         f"(python -m intake.walk --session <dir>)")
    rng = np.random.default_rng(BOOT_SEED)
    # claude_stac.txt §4-F6: once the session's confidence calibration exists (F6,
    # measured against the tier-0 depth), a DA3 pixel weighs 1/q² — q its calibrated
    # |error| quantile at its confidence and distance — instead of the raw confidence
    from precision import confidence as CAL
    cal = CAL.load_calibration(out, reference="tier0")
    cal_da3 = (cal or {}).get("models", {}).get("da3")
    if not (cal_da3 and "abs_err_quantile" in cal_da3):
        cal_da3 = None
    win_rows, mono_best, report = [], {}, []
    for gi, p in enumerate(paths):
        with np.load(p) as z:
            frames = [int(f) for f in z["frames"]]
            depth, conf = z["depth"], z["conf"]
            mono = z["depth_mono"] if "depth_mono" in z.files else None
        if cal_da3 is not None:
            q = CAL.lookup(cal_da3, np.nan_to_num(conf, nan=0.0), np.where(depth > 0, depth, 1.0))
            conf = np.where((conf > 0) & (q > 0), 1.0 / np.maximum(q, 1e-12) ** 2, 0.0)
        gains, weights, near, cs = [], [], [], []
        huber = {"not_converged": [], "mono_not_converged": [], "mad_zero": []}
        n = len(frames)
        for i, f in enumerate(frames):
            if f not in omega or f not in chainage:
                continue
            g = _log_gain(depth[i], omega[f], conf[i], gcfg)
            if g is None:
                continue
            if mono is not None:
                gm = _log_gain(mono[i], omega[f], None, gcfg)
                centrality = abs(i - (n - 1) / 2.0)
                if gm is not None and not gm.converged:
                    huber["mono_not_converged"].append(f)
                elif gm is not None and (f not in mono_best or centrality < mono_best[f][0]):
                    mono_best[f] = (centrality, gm.log - log_s0, gi)
            if not g.converged:
                huber["not_converged"].append(f)
                continue
            if g.mad_zero:
                huber["mad_zero"].append(f)
            gains.append(g.log - log_s0)
            weights.append(float(np.nansum(conf[i])))
            cs.append(chainage[f])
            nr = _ratio(depth[i], omega[f], conf=conf[i])
            near.append(math.log(nr) - log_s0 if nr and nr > 0 else float("nan"))
        if len(gains) < 2:
            if any(huber.values()):             # a window the non-converged gains emptied
                report.append({"window": p.name, "row": False, "n_frames": len(gains),
                               "huber": huber})
            continue
        g_arr, w_arr = np.array(gains), np.array(weights)
        s_w = _weighted_median(g_arr, w_arr)
        idx = rng.integers(0, len(g_arr), size=(BOOT_N, len(g_arr)))
        boots = [_weighted_median(g_arr[b], w_arr[b]) for b in idx]
        sig = float(np.std(boots))
        win_rows.append(Row("da3_windows", s_w, sig, gi, c=float(np.mean(cs)),
                            meta={"window": p.name, "n_frames": len(g_arr)}))
        report.append({"window": p.name, "row": True, "c_m": float(np.mean(cs)), "log_s": s_w,
                       "sigma": sig, "log_s_near_band": float(np.nanmedian(near)),
                       "n_frames": len(g_arr), "huber": huber,
                       "pixel_weights": ("calibrated (confidence_calibration.json, epoch "
                                         f"{cal.get('geometry_epoch')})" if cal_da3 is not None
                                         else "da3 confidence")})
    mono_rows = [Row("da3_mono", v, float("nan"), gi, c=chainage[f], meta={"frame": f})
                 for f, (_c, v, gi) in sorted(mono_best.items())]
    return win_rows, mono_rows, report


def vio_rows(session_dir: Path, frames: List[int], chainage: Dict[int, float],
             group_of: Callable[[float], int]) -> Optional[List[Row]]:
    """Per VIO segment (reconstruction/vio_scale.py): the VIO arc length over the
    CURRENT cameras' (already metric) arc length → log of the factor the cloud still
    needs, at the chainage of the segment's middle keyframe. None when the session has
    no VIO; a VIO that is present but unusable RAISES (vio_scale's own doctrine)."""
    from ingestors.vio_detector import detect_vio_data
    det = detect_vio_data(Path(session_dir))
    if not det.get("has_vio"):
        return None
    from reconstruction.scale_align import _kf_centres_and_times
    from reconstruction.vio_scale import estimate_vio_scale, load_vio_trajectory
    vt, vp, _hint = load_vio_trajectory(det["vio_path"])
    out = Path(session_dir) / "output"
    kf_t, kf_c = _kf_centres_and_times(out, Path(session_dir))
    info = estimate_vio_scale(vt, vp, kf_t, kf_c)
    t_by_frame = dict(zip(frames, kf_t)) if len(frames) == len(kf_t) else {}
    rows = []
    for seg in info["segments"]:
        mid = 0.5 * (seg["t_start"] + seg["t_end"])
        f = min(t_by_frame, key=lambda k: abs(t_by_frame[k] - mid)) if t_by_frame else None
        if f is None or f not in chainage:
            continue
        rows.append(Row("vio", math.log(float(seg["ratio"])), float("nan"),
                        group_of(chainage[f]), c=chainage[f],
                        meta={"t_start": seg["t_start"], "t_end": seg["t_end"]}))
    return rows


def stray_rows(session_dir: Path, omega: Dict[int, np.ndarray], chainage: Dict[int, float],
               group_of: Callable[[float], int], log_s0: float, gcfg,
               not_converged: Optional[List[int]] = None) -> Optional[List[Row]]:
    """Per keyframe: Stray LiDAR depth (``depth/<frame:06d>.png``, millimetres, weighted
    by ``confidence/<frame:06d>.png``) over Omega's. None without Stray depth. A frame
    whose Huber gain did not converge gives no row (appended to ``not_converged``)."""
    import cv2
    sd = Path(session_dir)
    ddir, cdir = sd / "depth", sd / "confidence"
    if not ddir.is_dir():
        return None
    rows = []
    for f, om in sorted(omega.items()):
        dp = ddir / f"{f:06d}.png"
        if f not in chainage or not dp.exists():
            continue
        d = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED)
        if d is None:
            continue
        cp = cdir / f"{f:06d}.png"
        c = cv2.imread(str(cp), cv2.IMREAD_UNCHANGED) if cp.exists() else None
        # millimetres → metres (a unit, not a parameter)
        g = _log_gain(d.astype(np.float32) / 1000.0, om,
                      c.astype(np.float32) if c is not None else None, gcfg)
        if g is not None and not g.converged:
            if not_converged is not None:
                not_converged.append(f)
        elif g is not None:
            rows.append(Row("stray", g.log - log_s0, float("nan"), group_of(chainage[f]),
                            c=chainage[f], meta={"frame": f, "huber_mad_zero": g.mad_zero}))
    return rows or None


def known_dim_rows(output_dir: Path, chainage: Dict[int, float],
                   group_of: Callable[[float], int]) -> List[Row]:
    """F8's known dimensions with role "gauge": log(true / measured-in-cloud) at the
    chainage of the frames that saw them; σ = the entry's own measurement σ over its
    length."""
    p = Path(output_dir) / "known_dimensions.json"
    if not p.exists():
        return []
    rows = []
    for d in json.loads(p.read_text()).get("dimensions", []):
        if d.get("role") != "gauge":
            continue
        fr = [chainage[f] for f in d.get("frames", []) if f in chainage]
        if not fr or not d.get("measured_m") or not d.get("true_m"):
            continue
        c = float(np.mean(fr))
        rows.append(Row("known_dims", math.log(float(d["true_m"]) / float(d["measured_m"])),
                        float(d.get("sigma_m", 0.0)) / float(d["true_m"]), group_of(c), c=c,
                        meta={"id": d.get("id")}))
    return rows


def visit_rows(output_dir: Path, frames: List[int], chainage: Dict[int, float],
               group_of: Callable[[float], int], epoch: int) -> List[Row]:
    """visit_drift's closures (scale_loop_rows.json) measured on THIS epoch: x(c_j) −
    x(c_i) = log k_b, σ = the closure's residual over its ray distance."""
    p = Path(output_dir) / "scale_loop_rows.json"
    if not p.exists():
        return []
    doc = json.loads(p.read_text())
    if int(doc.get("measured_on_epoch", -1)) != int(epoch):
        return []
    rows = []
    for r in doc.get("rows", []):
        i, j = int(r["i"]), int(r["j"])
        if not (0 <= i < len(frames) and 0 <= j < len(frames)):
            continue
        ci, cj = chainage.get(frames[i]), chainage.get(frames[j])
        if ci is None or cj is None or not r.get("D_b_m"):
            continue
        rows.append(Row("visit_drift", math.log(float(r["k_b"])),
                        float(r["residual_m"]) / float(r["D_b_m"]), group_of(cj),
                        c_ab=(ci, cj), meta={"instance_id": r.get("instance_id")}))
    return rows


# ── run ──────────────────────────────────────────────────────────────────

def _vendor_path() -> None:
    p = str(Path(__file__).resolve().parents[2] / "vendor" / "VGGT-Long")
    if p not in sys.path:
        sys.path.insert(0, p)


def run_gauge(session_dir: Path, gcfg, *, apply: bool = True, log: Callable = print) -> Dict[str, Any]:
    """Rows → model → (optionally) the epoch. Writes output/gauge.json and the v2 block
    of scale_diagnostics.json. Runs on epoch 0 only (the rows are measured against
    Omega's own depth)."""
    _vendor_path()
    from correction.epoch import current_epoch
    from intake.walk import load_walk
    session_dir = Path(session_dir)
    out = session_dir / "output"
    epoch = int(current_epoch(out))
    if epoch != 0:
        raise GaugeError(f"the session is at geometry epoch {epoch}: the gauge measures "
                         f"against Omega's epoch-0 depth — select epoch 0 first")
    walk = load_walk(session_dir)
    if walk is None:
        raise GaugeError(f"{session_dir / 'intake' / 'walk.json'} is missing — run the walk "
                         f"(python -m intake.walk --session <dir>)")
    # the window depth files are deleted once the chain is through (gauge.delete_windows_after_chain);
    # a re-run from F0 brings them back from the same plan (windows.json) — this interpreter is the
    # da3 env's, the one the windows were extracted with
    from intake.walk import WINDOWS_DIRNAME, run_da3_windows
    if not any((out / WINDOWS_DIRNAME).glob("window_*.npz")):
        log(f"{LOG_TAG} the I3 window depth files are gone — regenerating them from their plan")
        run_da3_windows(session_dir, gcfg, sys.executable, log=log)
    chainage = {int(c["frame"]): float(c["chainage_m"]) for c in walk["chainage"]}
    win_bounds = [(w["frames"][0], w["frames"][1]) for w in walk["windows"]]
    win_c = []
    for a, b in win_bounds:
        cs = [c for f, c in chainage.items() if a <= f <= b]
        win_c.append(float(np.mean(cs)) if cs else 0.0)

    def group_of(c: float) -> int:
        return int(np.argmin([abs(c - wc) for wc in win_c]))

    log_s0 = math.log(applied_global_scale(out))
    rows: List[Row] = []
    wins, mono, win_report = [], [], []
    inst = set(gcfg.instruments)
    if {"da3_windows", "da3_mono"} & inst:
        wins, mono, win_report = da3_rows(session_dir, chainage, log_s0, gcfg)
    if "da3_windows" in inst:
        rows += wins
    if "da3_mono" in inst:
        rows += mono
    absent = []
    if "known_dims" in inst:
        kd = known_dim_rows(out, chainage, group_of)
        rows += kd
        if not kd:
            absent.append("known_dims")
    from precision.poses_epoch import load_poses
    frames, poses = load_poses(out)          # no cloud: the core needs none until F7
    if "vio" in inst:
        vr = vio_rows(session_dir, frames, chainage, group_of)
        rows += vr or []
        if not vr:
            absent.append("vio")
    stray_not_converged: List[int] = []
    if "stray" in inst:
        sr = stray_rows(session_dir, omega_depths(out), chainage, group_of, log_s0,
                        gcfg, stray_not_converged)
        rows += sr or []
        if not sr:
            absent.append("stray")
    rows += visit_rows(out, frames, chainage, group_of, epoch)
    model = solve(rows, max(chainage.values()), gcfg, log=log)
    knots = np.array(model["knots_m"])
    x = np.array(model["x"])
    c_kf = np.array([chainage.get(f, np.nan) for f in frames])
    if np.isnan(c_kf).any():
        missing = [f for f, c in zip(frames, c_kf) if np.isnan(c)][:5]
        raise GaugeError(f"{int(np.isnan(c_kf).sum())} keyframe(s) of the cloud have no "
                         f"chainage in walk.json (e.g. {missing}) — the walk and the "
                         f"reconstruction were made from different keyframe sets")
    s_kf = np.exp([float(hat(c, knots) @ x) for c in c_kf])
    k_kf, t_kf = continuous_transforms(poses[:, :3, 3], s_kf)
    doc = {"version": GAUGE_VERSION, "provenance": PROVENANCE, "geometry_epoch": epoch,
           "params": {"knot_walk_m": gcfg.knot_walk_m, "smooth_grid": list(gcfg.smooth_grid),
                      "huber_k": gcfg.huber_k, "huber_tol": gcfg.huber_tol,
                      "huber_max_iter": gcfg.huber_max_iter,
                      "heldout_confidence": gcfg.heldout_confidence,
                      "instruments": list(gcfg.instruments)},
           "log_s_applied_global": log_s0, "walk_length_m": walk["walk_length_m"],
           **model, "instruments_absent": absent,
           "windows": win_report, "stray_huber_not_converged": stray_not_converged,
           "seams_reported_not_rows": _seams(out),
           "s_keyframes": {"min": float(s_kf.min()), "max": float(s_kf.max())},
           "camera_shift_max_m": float(np.linalg.norm(t_kf, axis=1).max()),
           "applied": False}
    if apply:
        from precision.poses_epoch import apply_pose_epoch
        R = np.tile(np.eye(3), (len(k_kf), 1, 1))
        res = apply_pose_epoch(out, R, t_kf, k_kf, "gauge",
                                    [{"stage": "gauge", "instrument": model["applied_instrument"],
                                      "s_min": float(s_kf.min()), "s_max": float(s_kf.max())}],
                                    log=log)
        doc["applied"] = bool(res)
        doc["epoch_to"] = (res or {}).get("epoch_to")
    (out / GAUGE_NAME).write_text(json.dumps(doc, indent=1, default=float))
    _write_diagnostics_v2(out, doc)
    log(f"{LOG_TAG} s per keyframe {s_kf.min():.4f}–{s_kf.max():.4f}, cameras moved ≤ "
        f"{doc['camera_shift_max_m'] * 100:.1f} cm; "
        + (f"applied → epoch {doc.get('epoch_to')}" if doc["applied"] else "not applied")
        + f" → {out / GAUGE_NAME}")
    return doc


def _seams(output_dir: Path) -> Dict[str, Any]:
    p = Path(output_dir) / "maplong_run" / "metric_lock.json"
    if not p.exists():
        return {}
    return (json.loads(p.read_text()) or {}).get("seams") or {}


def _write_diagnostics_v2(output_dir: Path, doc: Dict[str, Any]) -> None:
    p = Path(output_dir) / "scale_diagnostics.json"
    diag = json.loads(p.read_text()) if p.exists() else {}
    diag["v2"] = {k: doc.get(k) for k in (
        "version", "applied_instrument", "sigma_by_instrument", "heldout_by_instrument",
        "choice_verdict", "knots_m", "x", "lambda", "s_keyframes", "applied", "epoch_to")}
    p.write_text(json.dumps(diag, indent=2, default=float))


def gauge_applied(output_dir: Path) -> bool:
    p = Path(output_dir) / GAUGE_NAME
    return p.exists() and bool(json.loads(p.read_text()).get("applied"))


def main(argv: Optional[List[str]] = None) -> int:
    from precision.config import load_precision_config
    ap = argparse.ArgumentParser(prog="python -m precision.gauge",
                                 description="P2 — continuous metric gauge along the walk "
                                             "(runs I3 + the walk first when missing).")
    ap.add_argument("--session", required=True)
    ap.add_argument("--no-apply", action="store_true", help="measure and report only")
    args = ap.parse_args(argv)
    g = load_precision_config().gauge
    sd = Path(args.session)
    from intake.walk import load_walk, run_da3_windows, measure_walk
    if load_walk(sd) is None:
        run_da3_windows(sd, g, sys.executable)
        measure_walk(sd, g)
    run_gauge(sd, g, apply=not args.no_apply)
    return 0


if __name__ == "__main__":
    sys.exit(main())
