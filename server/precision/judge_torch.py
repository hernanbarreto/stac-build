"""THE USER'S RULE (``loop_utils.metric_lock.decide_change``, the fork's) with its bootstrap medians on the card
(USER 2026-11-01: "todo debe correr en GPU … en el menor tiempo posible, siempre determinista").

MEASURED on pccr (F6's per-keyframe verification, 296 calls: 15 s; F5's rung judges: ~1 s each over 129 k held-out
tracks): the cost is the median of every resample — ``np.median`` of a (rows × n) gather, or a pooled median per
cluster resample. Here the SAME resamples are drawn by the same seeded numpy generator in the SAME calls (block by
block, ``_BOOT_BLOCK_ELEMS`` per block — the generator's 32-bit buffering makes the block shapes part of the
stream), the medians are taken on the device by sorting (the middle element, or the mean of the two middle ones,
as numpy's), and everything after them — the percentiles, the verdict, the margins, the reason — is the fork's
own arithmetic on the same numbers. The fork is never modified: ``decide_change_t`` is a twin, and
tests/test_judge_torch.py holds its result equal to the fork's dict, key for key, on random inputs of both paths.
"""
from __future__ import annotations



import numpy as np


def _vendor():
    import sys
    from pathlib import Path
    p = str(Path(__file__).resolve().parents[2] / "vendor" / "VGGT-Long")
    if p not in sys.path:
        sys.path.insert(0, p)
    from loop_utils import metric_lock as ML
    return ML


def _rows_median_t(M):
    """np.median along axis 1 of the (r, n) tensor ``M`` (no NaN): the middle element, or the mean of the two
    middle ones — the same selection and the same (a + b) / 2."""
    import torch
    n = int(M.shape[1])
    s = torch.sort(M, dim=1).values
    lo, hi = (n - 1) >> 1, n >> 1
    if lo == hi:
        return s[:, lo]
    return (s[:, lo] + s[:, hi]) / 2.0


def decide_change_t(before, after, *, error, error_factor, confidence, min_judges=None, clusters=None,
                    n_boot=2000, seed=0, device: str = "cuda") -> dict:
    """``metric_lock.decide_change`` with the bootstrap medians on ``device`` (see the module docstring). The
    validation, the draws, the statistic and the verdict are the fork's; ``device`` 'numpy' calls the fork."""
    ML = _vendor()
    if str(device) == "numpy":
        return ML.decide_change(before, after, error=error, error_factor=error_factor, confidence=confidence,
                                min_judges=min_judges, clusters=clusters, n_boot=n_boot, seed=seed)
    import torch
    b = np.asarray(before, np.float64).ravel()
    a = np.asarray(after, np.float64).ravel()
    if b.size != a.size:
        raise ValueError(f"decide_change: {b.size} 'before' values for {a.size} 'after' values — "
                         f"the comparison is paired, judge by judge")
    if not (np.all(np.isfinite(b)) and np.all(np.isfinite(a))):
        raise ValueError("decide_change: non-finite disagreements — discard the observations that "
                         "are not valid in both states (and count them) before judging")
    err = float(error)
    fac = float(error_factor)
    conf = float(confidence)
    if not (np.isfinite(err) and err >= 0.0):
        raise ValueError(f"decide_change: error {error!r} must be a finite measured value >= 0")
    if not (np.isfinite(fac) and fac > 0.0):
        raise ValueError(f"decide_change: error_factor {error_factor!r} must be finite and > 0 "
                         f"(correction_graph.graph.improvement_error_factor)")
    if not (0.0 < conf < 1.0):
        raise ValueError(f"decide_change: confidence {confidence!r} must lie in (0, 1)")
    if min_judges is None:
        from loop_utils.loop_judge import min_judge_closures
        min_judges = min_judge_closures(conf)
    min_judges = int(min_judges)
    if min_judges < 1:
        raise ValueError(f"decide_change: min_judges {min_judges} must be >= 1")
    n_boot = int(n_boot)
    if n_boot < 1:
        raise ValueError(f"decide_change: n_boot {n_boot} must be >= 1")
    d = b - a
    n_obs = int(d.size)
    if clusters is None:
        inv, n_judges = None, n_obs
    else:
        cl = np.asarray(clusters).ravel()
        if cl.size != n_obs:
            raise ValueError(f"decide_change: {cl.size} cluster labels for {n_obs} observations")
        _labels, inv = np.unique(cl, return_inverse=True)
        n_judges = int(_labels.size)
    required = fac * err
    out = {"rule": "USER 2026-10-07: significant AND >= min_judges AND median >= factor x error",
           "statistic": "median of paired (before - after); positive = improves",
           "n_obs": n_obs, "n_judges": int(n_judges), "min_judges": min_judges,
           "clustered": clusters is not None, "confidence": conf, "n_boot": n_boot,
           "seed": int(seed), "error": err, "error_factor": fac, "required_delta": float(required)}
    if n_obs == 0:
        out.update({"median_delta": 0.0, "ci_low": 0.0, "ci_high": 0.0})
    else:
        dev = str(device)
        med = float(np.median(d))
        rng = np.random.default_rng(int(seed))
        meds = np.empty(n_boot, np.float64)
        block = int(ML._BOOT_BLOCK_ELEMS)
        if inv is None:
            d_t = torch.as_tensor(d, device=dev)
            rows = max(1, block // n_obs)
            for s in range(0, n_boot, rows):
                r = min(rows, n_boot - s)
                idx = torch.as_tensor(rng.integers(0, n_obs, size=(r, n_obs)), device=dev)
                meds[s:s + r] = _rows_median_t(d_t[idx]).cpu().numpy()
        else:
            order = np.argsort(d, kind="stable")
            d_sorted = torch.as_tensor(d[order], device=dev)
            inv_sorted = torch.as_tensor(inv[order], device=dev)
            k = int(n_judges)
            rows = max(1, block // k)
            for s in range(0, n_boot, rows):
                r = min(rows, n_boot - s)
                draw = torch.as_tensor(rng.integers(0, k, size=(r, k)), device=dev)
                counts = torch.zeros((r, k), dtype=torch.int64, device=dev)
                counts.scatter_add_(1, draw, torch.ones_like(draw))              # integer adds: exact
                for q0 in range(0, r, 64):                                       # bound the (rows × n) cumsum
                    w = counts[q0:q0 + 64][:, inv_sorted]
                    cw = torch.cumsum(w, dim=1)
                    m = cw[:, -1]
                    i_lo = torch.searchsorted(cw, ((m - 1) // 2)[:, None], right=True)[:, 0]
                    i_hi = torch.searchsorted(cw, (m // 2)[:, None], right=True)[:, 0]
                    v_lo, v_hi = d_sorted[i_lo], d_sorted[i_hi]
                    val = torch.where(i_lo == i_hi, v_lo, (v_lo + v_hi) / 2.0)
                    meds[s + q0:s + q0 + int(w.shape[0])] = val.cpu().numpy()
        alpha = (1.0 - conf) / 2.0
        lo = float(np.percentile(meds, 100.0 * alpha))
        hi = float(np.percentile(meds, 100.0 * (1.0 - alpha)))
        out.update({"median_delta": med, "ci_low": lo, "ci_high": hi})
    enough = bool(n_judges >= min_judges)
    significant = bool(n_obs > 0 and out["ci_low"] > 0.0)
    beyond = bool(n_obs > 0 and out["median_delta"] >= required)
    failed = []
    if not enough:
        failed.append("judges")
    if not significant:
        failed.append("significance")
    if not beyond:
        failed.append("error")
    pct = 100.0 * conf
    reason = (f"median improvement {out['median_delta']:.6g} vs {fac:g} x error {err:.6g} = "
              f"{required:.6g} (margin {out['median_delta'] - required:+.6g}); {pct:g} % CI "
              f"[{out['ci_low']:.6g}, {out['ci_high']:.6g}]; {n_judges} judge(s) for {min_judges} "
              f"required")
    out.update({"improves": bool(enough and significant and beyond), "significant": significant,
                "enough_judges": enough, "beyond_error": beyond,
                "worsens": bool(n_obs > 0 and out["ci_high"] < 0.0),
                "ci_margin": float(out["ci_low"]), "judges_margin": int(n_judges - min_judges),
                "error_margin": float(out["median_delta"] - required), "failed": failed,
                "reason": ("IMPROVES — " if not failed else
                           "does NOT pass (" + ", ".join(failed) + ") — ") + reason})
    return out


def cluster_bootstrap_se_t(groups, n_boot: int, seed: int, device: str) -> float:
    """``precision.depth_on_f5.cluster_bootstrap_se`` with the pooled medians on ``device``: the standard error
    of the POOLED MEDIAN of ``groups`` under a bootstrap that resamples whole groups — the same
    ``default_rng(seed).integers(0, k, size=k)`` call per resample (the reference's), the median of the
    concatenation with multiplicities taken as the fork's _pooled_median does (sorted values, cumulative
    weights, the middle element(s)), the sample standard deviation on the CPU."""
    import torch
    groups = [np.asarray(g, np.float64) for g in groups if len(g)]
    if len(groups) < 2:
        return 0.0
    rng = np.random.default_rng(int(seed))
    k = len(groups)
    vals = np.concatenate(groups)
    gid = np.repeat(np.arange(k), [len(g) for g in groups])
    order = np.argsort(vals, kind="stable")
    v_sorted = torch.as_tensor(vals[order], device=device)
    g_sorted = torch.as_tensor(gid[order], device=device)
    picks = np.stack([rng.integers(0, k, size=k) for _ in range(int(n_boot))])          # the reference's calls
    meds = np.empty(int(n_boot), np.float64)
    T = int(v_sorted.shape[0])
    step = max(1, (1 << 26) // max(T, 1))
    for s in range(0, int(n_boot), step):
        p = torch.as_tensor(picks[s:s + step], device=device)
        counts = torch.zeros((int(p.shape[0]), k), dtype=torch.int64, device=device)
        counts.scatter_add_(1, p, torch.ones_like(p))
        cw = torch.cumsum(counts[:, g_sorted], dim=1)
        m = cw[:, -1]
        i_lo = torch.searchsorted(cw, ((m - 1) // 2)[:, None], right=True)[:, 0]
        i_hi = torch.searchsorted(cw, (m // 2)[:, None], right=True)[:, 0]
        v_lo, v_hi = v_sorted[i_lo], v_sorted[i_hi]
        meds[s:s + int(p.shape[0])] = torch.where(i_lo == i_hi, v_lo, (v_lo + v_hi) / 2.0).cpu().numpy()
    return float(np.std(meds, ddof=1))
