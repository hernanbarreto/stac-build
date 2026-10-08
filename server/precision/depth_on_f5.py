"""F6-bend — the depth on F5: Omega's depth BENT to F5's landmarks, then a multi-view vote (the product).

USER 2026-10-01: *"integrá ya mismo el punto 6, la profundidad sobre F5, porque es lo que cambió
drásticamente el resultado"* — pccr epoch 7, judged the best cloud so far (*"piso perfecto, voladores
perfectos, la corrección de profundidad de una sutileza excepcional"*). It replaces F6's plane sweep and
the F7 corrected cloud on the default chain (`precision.cloud.source: omega_bent`).

Why: Omega measured each keyframe's depth with ITS OWN poses and camera. Put on F5's corrected poses
and camera unchanged, a third of it contradicted the other views (F6 on pccr: 32 % of the pixels — 13 %
of the floor, 49 % of the walls) and the vote that removes flyers eroded every object. Bent first to the
landmarks F5 itself triangulated, it agrees: 10.2 % contradicted on pccr, held-out |dz|/z 7.25 → 2.80 %.

Per keyframe i (cameras and poses = the session's, i.e. F5's after f5_refine):
 1. landmarks: F5's FIT tracks triangulated with these poses + camera (precision.refine helpers); the
    HELD-OUT tracks only judge;
 2. bend: z_i(u, v) · k_i(u, v), k_i = c0 + c1·u + c2·v fitted (robust IRLS, Huber `gauge.huber_k`) on
    the landmarks of keyframes i−w … i+w; w chosen among `bend.windows` by half A of the HELD-OUT
    (median |dz|/z). A keyframe whose window holds fewer than `bend.min_rows` landmark rows keeps
    Omega's depth (k = 1), exactly as epoch 7 did (the 'borrow the nearest k' variant belonged to the
    first, discarded epoch 8). REPRODUCED 2026-10-04 on pccr's F5 files with this code, steps 1-4,
    nothing published: held-out 7.25 → 2.80 % at ±0, tau 1.93 %, kept 63.1 %, contradicted 15.1 %,
    repaired 8.5 %, admitted 0.4 %, coverage 71.0 % — the hand-built epoch 8's own numbers.
    JUDGED since 2026-10-07 (docs/plan_determinismo.md, USER's DECIDIDO lines; pccr 2026-08-31 run B:
    keyframes 720 / 723 got c0 2.16 / 2.01 with c1, c2 of ±1.5 on a handful of rows — k from 0.74 to
    3.6 across the image — through a fit gated by a row COUNT alone, and ±0 beat ±1 by 1 % of the
    median with no tie margin):
      point 48  c1 and c2 enter only if |c| >= improvement_error_factor × their measured fit error
                (the IRLS covariance); c0 is VERIFIED on >= min_judge_closures(confidence) held-out
                rows of that keyframe by THE USER'S RULE (metric_lock.decide_change: significant,
                enough judges, improvement >= factor × the fit's own error of c0); a keyframe whose
                fit does not verify takes the pooled fit of its neighbours — the next wider window
                that verifies, else the widest (declared) — and every keyframe records what it got
                and why (`per_frame`);
      point 49  the window is the SMOOTHEST (widest) whose half-A held-out error is within factor ×
                the measured error of the best (the standard error of the best's median under a
                bootstrap by keyframe, fixed seed) — the curve and the bar are in the report;
      point 60  landmark tracks that do not round-trip through F5's lens are dropped and counted.
 3. validity: Omega's ONE confidence floor (`reconstruction.simple.conf_min_norm`, min-max per Omega
    chunk) and not sky (epoch0_cloud.SKY_CONF);
 4. vote over the keyframe offsets `bend.neighbors`: a pixel LEAVES when more neighbours see free space
    through it than agree; otherwise its depth is the MEDIAN of its own and the agreeing views carried
    onto its ray. τ = the `bend.tau_quantile` percentile of the session's own neighbour disagreement;
 5. chunks (Omega's) → the cloud stage's cleaner (voxel + SOR) → the new-cloud epoch through
    corrected_cloud.publish (the same transaction as F7). The `confidence` column carries the vote's
    agree count. The camera that built the cloud is written next to it (intrinsic.txt, one row per
    keyframe) so the viewer frames every keyframe with it.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
import types
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

LOG_TAG = "[depth-on-f5]"
REPORT = "depth_on_f5.json"
TX_TMP = "_tx_depth_on_f5"
MAD_TO_SIGMA = 1.0 / 0.6744897501960817          # 1.4826: σ of a normal from its MAD
IDENTITY = np.array([1.0, 0.0, 0.0])             # k = 1: Omega's depth as it is (epoch 7's fallback)


class DepthOnF5Error(RuntimeError):
    pass


def _vendor_path() -> None:
    from precision.refine import _vendor_path as vp
    vp()


# ── pure helpers (tested on synthetic data) ───────────────────────────────

def bilinear(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    H, W = img.shape
    u = np.clip(u, 0, W - 1 - 1e-6); v = np.clip(v, 0, H - 1 - 1e-6)
    u0 = np.floor(u).astype(np.int64); v0 = np.floor(v).astype(np.int64)
    du, dv = u - u0, v - v0
    return (img[v0, u0] * (1 - du) * (1 - dv) + img[v0, u0 + 1] * du * (1 - dv)
            + img[v0 + 1, u0] * (1 - du) * dv + img[v0 + 1, u0 + 1] * du * dv)


def design(u: np.ndarray, v: np.ndarray, W: int, H: int) -> np.ndarray:
    """Columns of the ratio model k = c0 + c1·u + c2·v (u, v centred and normalised by the image)."""
    return np.c_[np.ones(len(u)), (u - W / 2) / W, (v - H / 2) / H]


def irls_huber_fit(A: np.ndarray, r: np.ndarray, k: float, iterations: int) -> Tuple[np.ndarray, np.ndarray]:
    """pccr epoch 7's robust fit, exactly (omega_bent_epoch7.py `irls`; USER 2026-10-01 "exactamente el
    mismo ajuste que la 7"): `iterations` Huber IRLS steps from the least-squares start, weights at k·σ,
    σ = 1.4826·MAD of the residual. Iterated to convergence instead, the re-estimated MAD keeps shrinking
    and the bend fits worse: 3.90 % held-out on pccr against 2.80 %.

    Returns (c, σ_c): the coefficients (bit for bit what `irls_huber` returned before 2026-10-07) and
    their MEASURED fit error — the IRLS covariance at the last weights, σ_c² = s²·diag((AᵀW²A)⁻¹),
    s² the weighted residual sum of squares per degree of freedom (docs/plan_determinismo.md point 48:
    "su error medido del ajuste"). A singular normal matrix gives σ = inf: nothing is significant."""
    w = np.ones(len(r))
    c = np.zeros(A.shape[1])
    for _ in range(iterations):
        c = np.linalg.lstsq(A * w[:, None], r * w, rcond=None)[0]
        e = r - A @ c
        s = MAD_TO_SIGMA * np.median(np.abs(e)) + 1e-12
        w = np.sqrt(np.minimum(1.0, k * s / np.maximum(np.abs(e), 1e-12)))
    Aw = A * w[:, None]
    p = A.shape[1]
    # no error is MEASURABLE without a degree of freedom left or with a rank-deficient design (one
    # pixel repeated, every landmark on one line): σ = inf, so nothing passes a significance bar
    if len(r) <= p or np.linalg.matrix_rank(Aw) < p:
        return c, np.full(p, np.inf)
    s2 = float(np.sum((w * (r - A @ c)) ** 2)) / (len(r) - p)
    try:
        cov = s2 * np.linalg.inv(Aw.T @ Aw)
        sigma = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        sigma = np.full(p, np.inf)
    return c, sigma


def irls_huber(A: np.ndarray, r: np.ndarray, k: float, iterations: int) -> np.ndarray:
    """Epoch 7's coefficients alone (see `irls_huber_fit`)."""
    return irls_huber_fit(A, r, k, iterations)[0]


def significant_bend(A: np.ndarray, r: np.ndarray, huber_k: float, iterations: int,
                     error_factor: float) -> Tuple[np.ndarray, dict]:
    """Point 48 on one fit: the full model k = c0 + c1·u + c2·v is fitted; c1 and c2 ENTER only when
    |c| >= error_factor × their measured fit error, and the model is REFITTED on the kept columns
    (c0 always; a dropped term is exactly 0). Returns (c (3,), info): the full fit's σ, which terms
    were kept with their margins |c| − factor·σ, and the σ of the fit actually used (σ_c0 is what
    the keyframe's c0 verification is held to)."""
    c_full, s_full = irls_huber_fit(A, r, huber_k, iterations)
    fac = float(error_factor)
    keep = [True] + [bool(abs(c_full[j]) >= fac * s_full[j]) for j in (1, 2)]
    if all(keep):
        c, sig = c_full, s_full
    else:
        cols = [j for j in range(3) if keep[j]]
        cr, sr = irls_huber_fit(A[:, cols], r, huber_k, iterations)
        c = np.zeros(3); sig = np.full(3, np.nan)
        c[cols] = cr; sig[cols] = sr
    info = {"rows": int(len(r)), "sigma_full": [float(x) for x in s_full],
            "kept": {"c1": keep[1], "c2": keep[2]},
            "margins": {"c1": float(abs(c_full[1]) - fac * s_full[1]),
                        "c2": float(abs(c_full[2]) - fac * s_full[2])},
            "sigma": [float(x) for x in sig], "fallback": None}
    return c, info


def bend_coefficients(rows: Dict[int, tuple], n: int, window: int, min_rows: int, huber_k: float,
                      iterations: int, error_factor: float) -> Tuple[Dict[int, np.ndarray], Dict[int, dict]]:
    """({i: (c0, c1, c2)}, {i: info}) — each keyframe fitted on the rows of i−w … i+w with the judged
    terms of `significant_bend` (point 48); one with fewer than `min_rows` rows there keeps Omega's
    depth (k = 1), as epoch 7 did (info['fallback'] = 'identity')."""
    out: Dict[int, np.ndarray] = {}
    infos: Dict[int, dict] = {}
    for i in range(n):
        js = range(max(0, i - window), min(n, i + window + 1))
        A = [rows[j][0] for j in js if j in rows and len(rows[j][1])]
        r = [rows[j][1] for j in js if j in rows and len(rows[j][1])]
        n_rows = sum(len(x) for x in r)
        if A and n_rows >= min_rows:
            out[i], infos[i] = significant_bend(np.vstack(A), np.concatenate(r), huber_k, iterations,
                                                error_factor)
        else:
            out[i] = IDENTITY.copy()
            infos[i] = {"rows": int(n_rows), "sigma_full": None, "kept": None, "margins": None,
                        "sigma": None, "fallback": "identity"}
        infos[i]["window"] = int(window)
    return out, infos


def heldout_rows(obs_held: np.ndarray, zmap: np.ndarray, min_depth_m: float, half: Optional[int]):
    """(u, v, z_landmark, z_omega) of a keyframe's held-out landmark rows on Omega's depth; ``half``
    0 = the even rows (half A: selection — the window and the per-keyframe verification), 1 = the
    odd rows (half B: the report), None = all. Rows under ``min_depth_m`` of Omega depth are out."""
    h = np.asarray(obs_held, np.float64).reshape(-1, 3)
    if half is not None:
        h = h[(np.arange(len(h)) % 2) == int(half)]
    if not len(h):
        return np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)
    zz = bilinear(zmap, h[:, 0], h[:, 1])
    ok = zz > float(min_depth_m)
    return h[ok, 0], h[ok, 1], h[ok, 2], zz[ok]


def rel_errors(u, v, z_lm, z_omega, coef: np.ndarray, W: int, H: int) -> np.ndarray:
    """|z_omega·k(u, v) − z_landmark| / z_landmark per row (k = 1 with the identity coefficients)."""
    if not len(u):
        return np.zeros(0)
    return np.abs(z_omega * (design(u, v, W, H) @ coef) - z_lm) / z_lm


def cluster_bootstrap_se(groups: Sequence[np.ndarray], n_boot: int, seed: int) -> float:
    """The standard error of the POOLED MEDIAN of ``groups`` (one array of errors per keyframe)
    under a bootstrap that resamples whole keyframes (rows of one keyframe share its pose and
    camera — point 46's clustering, applied to the window choice of point 49). Fixed seed."""
    groups = [np.asarray(g, np.float64) for g in groups if len(g)]
    if len(groups) < 2:
        return 0.0
    rng = np.random.default_rng(int(seed))
    k = len(groups)
    meds = np.empty(int(n_boot))
    for b in range(int(n_boot)):
        pick = rng.integers(0, k, size=k)
        meds[b] = np.median(np.concatenate([groups[j] for j in pick]))
    return float(np.std(meds, ddof=1))


def choose_window(score: Dict[int, float], errs: Dict[int, Dict[int, np.ndarray]], error_factor: float,
                  n_boot: int, seed: int) -> Tuple[int, dict]:
    """Point 49 (USER 2026-10-07): the SMOOTHEST (widest) window whose half-A held-out error is
    within ``error_factor`` × the measured error of the best — the standard error of the best's
    pooled median under the keyframe bootstrap. ``score[w]`` = the pooled median per window,
    ``errs[w]`` = {keyframe: its rows' errors}. A strict argmin flipped on a 1 % near-tie (pccr
    ±0 2.85 % vs ±1 2.88 %). Returns (window, the rule with every margin)."""
    best = min(score, key=lambda w: (score[w], w))
    se = cluster_bootstrap_se(list(errs[best].values()), n_boot, seed)
    bar = float(score[best]) + float(error_factor) * se
    within = [w for w in score if score[w] <= bar]
    pick = max(within)
    return int(pick), {"rule": "smoothest window within error_factor x the standard error of the best's "
                               "median (bootstrap by keyframe)",
                       "window_best": int(best), "heldout_best": float(score[best]), "heldout_best_se": se,
                       "error_factor": float(error_factor), "bar": bar, "window": int(pick),
                       "margin": float(bar - score[pick]),
                       "curve": {str(w): {"heldout_median": float(score[w]), "margin_to_bar": float(bar - score[w]),
                                          "within": bool(score[w] <= bar)} for w in sorted(score)}}


def verify_keyframes(windows: Sequence[int], wb: int, coefs: Dict[int, Dict[int, np.ndarray]],
                     infos: Dict[int, Dict[int, dict]], rows_A: Dict[int, tuple], N: int, W: int, H: int,
                     error_factor: float, confidence: float, n_boot: int, seed: int,
                     min_rows: int) -> Tuple[Dict[int, np.ndarray], Dict[int, dict]]:
    """Point 48's c0 verification per keyframe. The candidates for keyframe i are the chosen window
    ``wb`` then every wider window of ``windows`` (the pooled fits of its neighbours). A candidate is
    VERIFIED when THE USER'S RULE (metric_lock.decide_change) says the bent depth improves the
    keyframe's own half-A held-out rows over Omega's unbent depth — significant, >= min_judge_closures
    (confidence) rows, median improvement >= error_factor × σ_c0 of that fit. The first verified
    candidate is the keyframe's; none → the widest candidate that has a fit (declared 'unverified
    pooled'); fewer judge rows than required → the first pooled candidate wider than ``wb`` with a
    fit (declared 'unverifiable'), else ``wb``'s own. Returns ({i: c}, {i: provenance})."""
    _vendor_path()
    from loop_utils.loop_judge import min_judge_closures
    from loop_utils.metric_lock import decide_change
    need = int(min_judge_closures(float(confidence)))
    order = [int(wb)] + sorted(int(w) for w in windows if int(w) > int(wb))
    out: Dict[int, np.ndarray] = {}
    prov: Dict[int, dict] = {}
    for i in range(N):
        u, v, z_lm, zo = rows_A[i]
        before = rel_errors(u, v, z_lm, zo, IDENTITY, W, H)
        cands = [(w, coefs[w][i], infos[w][i]) for w in order if infos[w][i].get("fallback") != "identity"]
        trials = []
        chosen = None
        if len(before) < need:
            reason = f"unverifiable: {len(before)} held-out row(s) of this keyframe, {need} required"
            pooled = [c for c in cands if c[0] > int(wb)]
            pick = pooled[0] if pooled else (cands[0] if cands else None)
            status = "unverifiable_pooled" if pooled else ("unverifiable_own" if cands else "identity")
        else:
            for w, c, info in cands:
                sig0 = float(info["sigma"][0]) if info.get("sigma") else float("nan")
                if not np.isfinite(sig0):
                    trials.append({"window": w, "verdict": None, "reason": "sigma_c0 not finite"})
                    continue
                after = rel_errors(u, v, z_lm, zo, c, W, H)
                d = decide_change(before, after, error=sig0, error_factor=float(error_factor),
                                  confidence=float(confidence), n_boot=int(n_boot), seed=int(seed))
                trials.append({"window": w, "verdict": {k: d[k] for k in ("improves", "significant", "enough_judges",
                                                                          "beyond_error", "median_delta", "ci_low",
                                                                          "ci_high", "n_judges", "min_judges",
                                                                          "error", "required_delta", "error_margin")},
                               "reason": d["reason"]})
                if d["improves"]:
                    chosen = (w, c, info)
                    break
            if chosen is not None:
                pick, status, reason = chosen, "verified", f"c0 verified on {len(before)} held-out rows at ±{chosen[0]}"
            else:
                pick = cands[-1] if cands else None
                status = "unverified_pooled" if cands else "identity"
                reason = ("no candidate window verified c0 on this keyframe's held-out rows — the widest "
                          "pooled fit stands" if cands else f"no window holds {min_rows} landmark rows")
        if pick is None:
            out[i] = IDENTITY.copy()
            prov[i] = {"status": "identity", "window": None, "reason": reason, "fit": None, "trials": trials,
                       "heldout_rows_A": int(len(before))}
        else:
            w, c, info = pick
            out[i] = np.asarray(c, np.float64)
            prov[i] = {"status": status, "window": int(w), "reason": reason, "fit": info, "trials": trials,
                       "heldout_rows_A": int(len(before))}
    return out, prov


def confidence_floor(values: np.ndarray, floor_norm: float) -> Tuple[float, float]:
    """THE ONE CONFIDENCE FLOOR of the pipeline (USER 2026-09-23, reconstruction.simple.conf_min_norm),
    per Omega chunk: ``thr = min + floor_norm × (max − min)`` over the chunk's valid confidences and
    ``cmax = max`` — the min-max fraction the viewer slider runs (ui PotreeLoader.ts: (c − confMin) /
    confRange). KEPT as is by docs/plan_determinismo.md point 52 (it rests on the chunk's two most
    extreme samples, deterministic with the same input); tests pin this arithmetic so a change is
    noticed, never silent."""
    v = np.asarray(values, np.float64).ravel()
    if v.size == 0:
        raise DepthOnF5Error("no valid confidence in this Omega chunk — the floor cannot be set")
    lo, hi = float(v.min()), float(v.max())
    return lo + float(floor_norm) * (hi - lo), hi


def interior(passed: np.ndarray) -> np.ndarray:
    """Floor-passing pixels whose whole 3x3 window passed too — Omega's confidence collapses at
    contours, so tau is measured where it does not (pccr epoch 8)."""
    from scipy.ndimage import binary_erosion
    return binary_erosion(passed, structure=np.ones((3, 3), bool), border_value=0)


def masks_stamp_check(output_dir: Path, masks) -> Tuple[bool, str]:
    """Point 51: SAM3 masks enter F6 only when they are STAMPED for THIS reconstruction — the npz
    carries a ``reconstruction_id`` member, or ``segmentation.json`` next to it carries the key,
    equal to ``correction.epoch.reconstruction_id`` of the session. A store left by an earlier
    segmentation (SAM3 runs after the cloud since 2026-10-05; a resumed run or a `--from f6_bend`
    finds the previous one) is matched by keyframe number only and would snap pixels or not
    depending on the session's history. Returns (taken, reason)."""
    from correction.epoch import RECONSTRUCTION_ID_KEY, reconstruction_id_or_none, same_reconstruction
    out = Path(output_dir)
    rid = reconstruction_id_or_none(out)
    doc = None
    if RECONSTRUCTION_ID_KEY in getattr(masks, "files", []):
        doc = {RECONSTRUCTION_ID_KEY: str(np.asarray(masks[RECONSTRUCTION_ID_KEY]).reshape(-1)[0])}
        where = f"seg_masks.npz['{RECONSTRUCTION_ID_KEY}']"
    else:
        sj = out / "segmentation.json"
        where = f"segmentation.json['{RECONSTRUCTION_ID_KEY}']"
        if sj.exists():
            try:
                d = json.loads(sj.read_text())
                doc = d if isinstance(d, dict) and RECONSTRUCTION_ID_KEY in d else None
            except ValueError:
                doc = None
    taken, why = same_reconstruction(doc, rid)
    return taken, f"{where}: {why}"


def mask_labels(output_dir: Path, H: int, W: int, log: Callable = print):
    """``position -> label map`` of the SAM3 masks (masklet id + 1, 0 = no mask, -1 = two masks
    overlap) on Omega's grid, or None when the session holds no masks on that grid, or holds masks
    NOT stamped for this reconstruction (point 51) — then no mixed pixel is snapped, declared."""
    from precision.silhouette_filter import masks_by_keyframe
    p = Path(output_dir) / "seg_masks.npz"
    if not p.exists():
        log(f"{LOG_TAG} no seg_masks.npz — mixed pixels are not snapped (they go to the vote as they are)")
        return None
    masks = np.load(p)
    taken, why = masks_stamp_check(output_dir, masks)
    if not taken:
        log(f"{LOG_TAG} seg_masks.npz IGNORED — not stamped for this reconstruction ({why}); mixed pixels "
            f"are not snapped (point 51)")
        return None
    by_kf = masks_by_keyframe(Path(output_dir), masks)
    probe = next((key for lst in by_kf.values() for _, key in lst), None)
    if probe is None or tuple(np.asarray(masks[probe]).shape) != (H, W):
        log(f"{LOG_TAG} the SAM3 masks are not on Omega's grid {W}x{H} — mixed pixels are not snapped")
        return None
    log(f"{LOG_TAG} SAM3 masks: {sum(len(v) for v in by_kf.values()):,} over {len(by_kf)} keyframes")

    def labels(i: int) -> np.ndarray:
        L = np.zeros((H, W), np.int64); cnt = np.zeros((H, W), np.uint8)
        for oid, key in by_kf.get(i, []):
            m = np.asarray(masks[key]) > 0
            L[m] = np.where(cnt[m] == 0, oid + 1, -1)
            cnt[m] = np.minimum(cnt[m] + 1, 2)
        return L
    return labels


def labels_on_undistorted(labels: Callable, maps) -> Callable:
    """The SAM3 label maps (made on the ORIGINAL frames) carried onto the undistorted native frame
    through F0's maps (nearest); where the frame has no original pixel: 0 (no mask)."""
    import cv2

    def fn(i: int) -> np.ndarray:
        L = labels(i)
        return cv2.remap(L.astype(np.int32), maps[0], maps[1], cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.int64)
    return fn


def edge_keeping_vote(frames: List[int], dep: Dict[int, np.ndarray], valid: Dict[int, np.ndarray],
                      passed: Dict[int, np.ndarray], K: np.ndarray, c2w: Dict[int, np.ndarray], labels,
                      neighbors, tau_quantile: float, repair_min_views: int, log: Callable = print,
                      bars_out: Optional[dict] = None):
    """pccr EPOCH 8 (USER 2026-10-01: "es la mejor, incorporar al pipeline"), on the bent depth
    `dep` (every `valid` pixel; `passed` = above the ONE confidence floor):
     1. tau = the `tau_quantile` percentile of the neighbour disagreement on INTERIOR pixels;
     2. EDGE pixels = the 3x3 window spans a depth step (on the depth before snapping);
        MIXED pixels (on neither surface of the step) snapped to the side their SAM3 mask says;
     3. the TWO-SIDED vote over `neighbors`, judged by floor-passing neighbour pixels only:
        floor-passing pixels stay with contra ≤ agree (median of the agreeing views), a
        contradicted one is repaired to the median of the neighbours when ≥ `repair_min_views`
        agree, else it leaves; a below-floor EDGE pixel enters when agree ≥ 1 and contra ≤ agree.
    Returns ({frame: (depth map, agree count map)}, tau, totals). ``bars_out`` (a dict) receives the
    vote's bar τ and every keyframe's pixel margins to it (point 53) — kept out of ``totals``, which
    stays numeric for its readers."""
    from precision import corrected_cloud as CC
    H, W = next(iter(dep.values())).shape
    inner = {f: interior(passed[f]) for f in frames}
    tau = CC.measured_tau(dep, inner, K, c2w, frames, neighbors, tau_quantile)
    del inner
    edge = {}
    n_mixed = n_snap = 0
    for i, f in enumerate(frames):
        edge[f] = CC.depth_steps(dep[f], valid[f], tau)
        if labels is not None:
            d, mixed, sn = CC.snap_mixed(dep[f], valid[f], labels(i), tau)
            dep[f] = d.astype(np.float32); n_mixed += int(mixed.sum()); n_snap += int(sn.sum())
    w2c = {f: np.linalg.inv(c2w[f]) for f in frames}
    final, tot = {}, {}
    tau_margins: Dict[int, Optional[dict]] = {}
    for i, f in enumerate(frames):
        cand = valid[f] & (passed[f] | edge[f])           # below-floor interior pixels stay out
        v = CC.two_sided_vote(i, frames, dep, passed, cand, K, c2w, w2c, neighbors, tau)
        rr, cc, agree, contra = v["rr"], v["cc"], v["agree"], v["contra"]
        # point 53: every judged pixel's margin to τ — the nearest neighbour surface on its ray
        # against its own depth, (τ − |Δz|/z) / τ (positive = agrees with its nearest witness)
        s = v["splats"]
        if s.size:
            with np.errstate(invalid="ignore", divide="ignore"):
                rel = np.abs(s - v["z"][None, :]) / np.where(s > 0, s, np.nan)
                dmin = np.nanmin(np.where(np.isfinite(rel), rel, np.inf), axis=0)
            seen = np.isfinite(dmin)
            from precision.mono_detail import margin_quantiles
            tau_margins[f] = margin_quantiles((tau - dmin[seen]) / max(tau, 1e-12)) if seen.any() else None
        else:
            tau_margins[f] = None
        pas, edg = passed[f][rr, cc], edge[f][rr, cc]
        contradicted = pas & (contra > agree)
        rep_ok = np.zeros(len(rr), bool); zrep = np.full(len(rr), np.nan); nrep = np.zeros(len(rr), np.int32)
        if contradicted.any():
            ok, z_, n_ = CC.agreeing_median(v["splats"][:, contradicted], tau, int(repair_min_views))
            rep_ok[contradicted] = ok; zrep[contradicted] = z_; nrep[contradicted] = n_
        keep, repair, admit = CC.edge_vote_decision(pas, edg, agree, contra, rep_ok)
        zmap = np.zeros((H, W), np.float32); amap = np.zeros((H, W), np.int16)
        m = keep | admit
        zmap[rr[m], cc[m]] = v["zmed"][m]; amap[rr[m], cc[m]] = agree[m]
        zmap[rr[repair], cc[repair]] = zrep[repair]; amap[rr[repair], cc[repair]] = nrep[repair]
        final[f] = (zmap, amap)
        for k, n in (("valid", int(valid[f].sum())), ("edge", int(edge[f].sum())), ("kept", int(keep.sum())),
                     ("contradicted", int(contradicted.sum())), ("repaired", int(repair.sum())),
                     ("admitted", int(admit.sum())), ("edge_below", int((~pas & edg).sum())),
                     ("out", int((zmap > 0).sum()))):
            tot[k] = tot.get(k, 0) + n
    tot.update(mixed=n_mixed, snapped=n_snap, tau=tau)
    # the bars of the vote and the margins to them (point 53): τ itself (the session's own
    # tau_quantile percentile) and, per keyframe, the distribution of the pixels' margins to it
    if bars_out is not None:
        med = [m["p50"] for m in tau_margins.values() if m]
        bars_out.update({"tau": tau, "tau_quantile": float(tau_quantile),
                         "tau_is": "this percentile of the session's own neighbour disagreement on interior pixels",
                         "tau_margin_rel": {"per_frame": {str(f): m for f, m in tau_margins.items()},
                                            "median_of_frame_medians": float(np.median(med)) if med else None}})
    nv = max(tot["valid"], 1)
    log(f"{LOG_TAG} tau {tau * 100:.2f} % (interior); edge pixels {tot['edge'] / nv * 100:.2f} % of the valid; "
        f"mixed {n_mixed:,} ({n_mixed / nv * 100:.2f} %), {n_snap:,} snapped to their mask's side")
    log(f"{LOG_TAG} vote: {tot['kept'] / nv * 100:.1f} % of the valid pixels kept, {tot['contradicted'] / nv * 100:.1f} % "
        f"contradicted, {tot['repaired'] / nv * 100:.1f} % repaired, {tot['admitted'] / nv * 100:.1f} % admitted below "
        f"the floor at edges; coverage {tot['out'] / float(len(frames) * H * W) * 100:.1f} % of all pixels")
    return final, tau, tot


def camera_travels(tmp: Path, params, n_kf: int, log: Callable = print) -> Path:
    """The camera travels with the cloud (pccr 2026-10-01: the viewer framed F5's poses with Omega's
    camera — standing at a keyframe, the scene did not match the image). The viewer reads
    output/intrinsic.txt (one fx fy cx cy row per keyframe). Written into the TRANSACTION, so
    corrected_cloud.publish registers it as an epoch artifact: the previous epoch keeps its own file
    (filed with its delta, restored when it is selected again) and the new one carries the camera it
    was built with — a copy written after the swap left the next epoch without one."""
    from repro import write_intrinsics_exact
    # float64 round-trip exact (docs/plan_determinismo.md point 45): '.10g' rounded the camera
    p = write_intrinsics_exact(tmp / "intrinsic.txt",
                               np.tile(np.asarray(params[:4], np.float64)[None], (int(n_kf), 1)))
    log(f"{LOG_TAG} intrinsic.txt = the camera of this cloud ({params[0]:.1f} / {params[1]:.1f}), "
        f"an artifact of the epoch")
    return p


# ── mono detail (claude_stac.txt 2026-10-04) ──────────────────────────────

# the cloud's `source` column: 1 = Omega bent (epoch 8's pixel), 2 = lowpass(Omega) + PointDiT detail,
# 3 / 4 = a mixed pixel resolved to the front / back surface. mixed_unresolved never becomes a point.
SRC_CLOUD_NAMES = {1: "omega_bent", 2: "mono_detail", 3: "band_front", 4: "band_back"}
MONO_LAYERS = ("mono_detail", "mixed_unresolved")


def confidence_weight(conf: np.ndarray, floor: float, cmax: float, passed: np.ndarray) -> np.ndarray:
    """The session's calibrated confidence as a fit weight: 0 under the chunk's own floor, rising
    linearly to 1 at the chunk's maximum (the same min-max arithmetic as the ONE confidence floor)."""
    span = max(cmax - floor, 1e-9)
    w = np.clip((conf.astype(np.float64) - floor) / span, 0.0, 1.0)
    return np.where(passed, w, 0.0)


def apply_mono_detail(pcfg, frames, dep, valid, passed, weight, inp, K, c2w, out: Path, log, _p):
    """The hook between the bend and the vote. With ``mono_detail.enabled`` false it returns its
    inputs untouched (epoch 8 bit for bit); otherwise PointDiT's detail refines every bent map, the
    unresolved mixed pixels leave the measurement tier (depth 0, not passed, not valid) and the
    per-pixel provenance comes back for the cloud's `source` column."""
    md = pcfg.mono_detail
    if not md.enabled:
        return dep, valid, passed, None, None
    from precision import corrected_cloud as CC
    from precision.mono_detail import run_stage
    from precision.pointdit_runner import PointDiTRunner
    _p(42, f"mono detail: PointDiT-{md.model} refines the bent depth (tiles, affine per tile, detail, band)")
    runner = PointDiTRunner(md, log=log)
    inner = {f: interior(passed[f]) for f in frames}
    # the session's own agreement tolerance on the BENT maps (the vote measures its own again after)
    tau0 = CC.measured_tau(dep, inner, K, c2w, frames, pcfg.bend.neighbors, pcfg.bend.tau_quantile)
    del inner
    grid = getattr(inp.cam, "omega_grid", None)
    scale = float(getattr(grid, "scale_x", 1.0) or 1.0) if grid is not None else 1.0
    image_of = lambda f: CC._rgb_undistorted(inp.frames_dir, f, inp.maps)   # noqa: E731
    overlay_dir = (out / "precision" / "mono_overlays") if bool(md.overlays) else None
    dep2, src, rep = run_stage(frames, dep, valid, weight, image_of, runner, md, tau0, scale,
                               pcfg.gauge.huber_k, log=log, progress=lambda pct, m: _p(42 + pct * 8 / 100, m),
                               overlay_dir=overlay_dir)
    for f in frames:
        alive = dep2[f] > 0
        valid[f] = valid[f] & alive
        passed[f] = passed[f] & alive
    rep.params["tau_bent"] = tau0
    rep.params["footprint"] = runner.footprint()          # device, model, steps, seed, card, numerics (point 54)
    rep.params["timing"] = runner.timing()                # peak VRAM: the timing sidecar's, not the report's
    return dep2, valid, passed, src, rep


def source_column(data, src_maps) -> np.ndarray:
    """Per cleaned point its provenance byte from the refined maps (1 = Omega bent when the stage did
    not run); looked up by (frame, pixel) because the cleaner carries only the origin columns."""
    n = len(data)
    if src_maps is None:
        return np.ones(n, np.uint8)
    from precision.mono_detail import SRC_UNRESOLVED
    fg = np.asarray(data["frame_global"], np.int64)
    r = np.asarray(data["pixel_row"], np.int64); c = np.asarray(data["pixel_col"], np.int64)
    out = np.ones(n, np.uint8)
    for f in np.unique(fg):
        m = fg == f
        sm = src_maps.get(int(f))
        if sm is None:
            continue
        v = sm[r[m], c[m]].astype(np.int64)
        if (v == SRC_UNRESOLVED).any():
            raise DepthOnF5Error("a mixed_unresolved pixel reached the cloud — the measurement tier must not hold it")
        out[m] = (v + 1).astype(np.uint8)
    return out


def mono_report(rep, md) -> dict:
    """The mono-detail section of depth_on_f5.json — no timing in it (that is `mono_timing`)."""
    tiles = [dict(frame=int(f), **t) for f, per in rep.per_frame.items() for t in per["tiles"]]
    frames = {str(f): {k: v for k, v in per.items() if k != "tiles"} for f, per in rep.per_frame.items()}
    params = {k: v for k, v in rep.params.items() if k != "timing"}
    return {"enabled": True, "model": md.model, "steps": int(md.steps), "params": params,
            "tiles": {"total": rep.tiles, "accepted": rep.accepted, "rejected_support": rep.rejected_support,
                      "rejected_residual": rep.rejected_residual, "residual_bar_rel": rep.residual_bar},
            "pixels": rep.totals, "per_frame": frames, "per_tile": tiles}


def mono_timing(rep) -> dict:
    return {"seconds_pointdit": rep.seconds_pointdit, "seconds_total": rep.seconds_total,
            **(rep.params.get("timing") or {})}


def write_mono_layers(pdir: Path, data, source: np.ndarray, rep, K, c2w, log) -> None:
    """Two viewer layers (GLB point clouds): the points PointDiT's detail or band resolution wrote
    (`mono_detail`: grey detail, green front, blue back; a subsample bounded by LAYER_MAX_POINTS) and the
    mixed_unresolved pixels at Omega's depth (`mixed_unresolved`, red) — what left the measurement tier."""
    import trimesh
    pdir.mkdir(parents=True, exist_ok=True)
    xyz = np.stack([np.asarray(data["x"]), np.asarray(data["y"]), np.asarray(data["z"])], 1).astype(np.float32)
    m = source >= 2
    idx = np.nonzero(m)[0]
    if len(idx) > LAYER_MAX_POINTS:
        idx = np.random.default_rng(0).choice(idx, LAYER_MAX_POINTS, replace=False)
    pal = {2: (150, 150, 150, 255), 3: (0, 200, 0, 255), 4: (0, 90, 255, 255)}
    col = np.array([pal[int(v)] for v in source[idx]], np.uint8) if len(idx) else np.zeros((0, 4), np.uint8)
    pts = xyz[idx] if len(idx) else np.zeros((1, 3), np.float32)
    if not len(idx):
        col = np.array([[0, 0, 0, 0]], np.uint8)
    trimesh.PointCloud(pts, colors=col).export(str(pdir / "layer_mono_detail.glb"), file_type="glb")
    P = []
    for f, (rr, cc, zz) in rep.unresolved.items():
        X = np.stack([(cc - K[0, 2]) / K[0, 0] * zz, (rr - K[1, 2]) / K[1, 1] * zz, zz], 1) @ c2w[f][:3, :3].T + c2w[f][:3, 3]
        P.append(X.astype(np.float32))
    U = np.concatenate(P) if P else np.zeros((1, 3), np.float32)
    uc = np.tile(np.array([[255, 0, 0, 255]], np.uint8), (len(U), 1)) if P else np.array([[0, 0, 0, 0]], np.uint8)
    trimesh.PointCloud(U, colors=uc).export(str(pdir / "layer_mixed_unresolved.glb"), file_type="glb")
    log(f"{LOG_TAG} viewer layers: {len(idx):,} mono-detail point(s), {len(U) if P else 0:,} mixed_unresolved "
        f"pixel(s) → {pdir}/layer_*.glb")


LAYER_MAX_POINTS = 2_000_000


# ── the step ─────────────────────────────────────────────────────────────

def _landmark_rows(session_dir: Path, pcfg, frames: List[int], w2c: np.ndarray, params,
                   log: Callable = print) -> Tuple[Dict[int, dict], dict]:
    """({split: {keyframe: (u, v, z) rows}}, {split: dropped}) — F5's tracks triangulated with
    THESE poses and camera; a track whose observations do not round-trip through the lens is
    dropped and counted (point 60), never a failure of the stage."""
    from precision.camera import undistort_solver
    from precision.tracks import load_tracks_v2
    from precision import refine as RF
    solver = undistort_solver(pcfg.camera)
    tr = load_tracks_v2(session_dir)
    split_of = dict(zip(tr["track_query_id"].tolist(), tr["track_split"].tolist()))
    split = np.array([split_of[int(t)] for t in tr["obs_track"]], np.int8)
    out, dropped = {}, {}
    for sp in (0, 1):
        m = split == sp
        g = RF.group_tracks(tr["obs_track"][m], tr["obs_frame"][m], tr["obs_uv_native"][m], frames)
        dropped[sp] = {"tracks": 0, "observations": 0}
        X = RF.triangulate_tracks(g, w2c, params, solver, pcfg.refine.min_tri_deg, dropped=dropped[sp])
        rows = {i: [] for i in range(len(frames))}
        for t, lst in g.items():
            if t in X:
                for i, uv in lst:
                    z = w2c[i][2, :3] @ X[t] + w2c[i][2, 3]
                    if z > 0:
                        rows[i].append((uv[0], uv[1], z))
        out[sp] = {i: np.array(v, float).reshape(-1, 3) for i, v in rows.items()}
    if any(d["tracks"] for d in dropped.values()):
        log(f"{LOG_TAG} landmark tracks dropped (no round trip through F5's lens — point 60): fit "
            f"{dropped[0]['tracks']}, held-out {dropped[1]['tracks']}")
    return out, {"fit": dropped[0], "held_out": dropped[1]}


def compute(session_dir: Path, pcfg, log: Callable = print,
            progress: Optional[Callable[[float, str], None]] = None, inp=None) -> types.SimpleNamespace:
    """Steps 1-4 (landmarks, bend, mono detail, vote) — nothing written into the session. ``inp`` (a
    depth_sweep.SweepInputs) may be injected: the A/B (precision/mono_ab.py) feeds F5's files directly.
    Returns everything ``publish_cloud`` needs, plus the held-out landmark rows for the judges."""
    from config import cfg as raw_cfg
    from precision import depth_sweep as DS
    from precision.epoch0_cloud import SKY_CONF
    t0 = time.time()
    session_dir = Path(session_dir)
    out = session_dir / "output"
    bc, gc = pcfg.bend, pcfg.gauge

    def _p(pct, msg):
        log(f"{LOG_TAG} {msg}")
        if progress:
            progress(pct, msg)

    from precision import corrected_cloud as CC
    inp = inp if inp is not None else DS.load_inputs(session_dir, pcfg)   # F5's camera + poses (guards F5's epoch)
    params = list(inp.cam.params)
    K = np.asarray(inp.K, np.float64)
    W, H = int(inp.wh[0]), int(inp.wh[1])
    frames = [int(f) for f in inp.kf]
    N = len(frames)
    w2c = np.asarray(inp.kf_w2c, np.float64)
    c2w = {f: np.linalg.inv(w2c[i]) for i, f in enumerate(frames)}

    # THE GRID OF THE BEND is the UNDISTORTED NATIVE frame of F5's camera (K_F5 on F0's maps) — the
    # chain's own convention (corrected_cloud, silhouette_filter, provenance: pixel_u_und / v_und).
    # Omega's record is the ORIGINAL frame on Omega's grid. With no lens and a record on the camera's
    # grid (pccr: 464x832, k = 0) the two frames are one and the record enters as it is — epoch 8's
    # arithmetic, bit for bit. Otherwise (zaragoza 2026-10-05: records 1920x1088 for 1920x1080 frames
    # and F5's rung R2 with k1 -0.0035) the record, the landmark pixels and the SAM3 masks are CARRIED
    # onto that frame — the lens, then the record grid (corrected_cloud.record_on_native, nearest) —
    # so PointDiT (on the undistorted frame), the vote (K), the colours and the cloud's pixel columns
    # all live on one grid. The camera's lens is neither ignored nor refused.
    lens = bool(np.any(inp.cam.dist()))
    with np.load(inp.records_dir / f"frame_{frames[0]}.npz") as z:
        rec_hw = tuple(int(x) for x in np.asarray(z["depth"]).shape)
    carry = lens or rec_hw != (H, W)
    if carry:
        grid_note = (f"Omega's record {rec_hw[1]}x{rec_hw[0]}"
                     + (" + lens k1 %.5f k2 %.5f p1 %.5f p2 %.5f" % tuple(params[4:8]) if lens else "")
                     + f" carried onto the undistorted native grid {W}x{H} (F0's maps, nearest)")
    else:
        grid_note = f"Omega's record on the camera grid {W}x{H}, no lens — read as it is"
    _p(3, f"{N} keyframes, camera fx {K[0, 0]:.1f} fy {K[1, 1]:.1f} ({W}x{H}); {grid_note}; F5's landmarks")
    obs, dropped_tracks = _landmark_rows(session_dir, pcfg, frames, w2c, params, log=log)
    if lens:
        # the tracks were observed where the lens put them (the original frame) → the undistorted frame
        from precision.camera import undistort_points, undistort_solver
        solver = undistort_solver(pcfg.camera)
        for sp in obs:
            for i, o in obs[sp].items():
                if len(o):
                    o[:, :2] = undistort_points(o[:, :2], inp.cam, **solver)

    zo, conf, chunk = {}, {}, {}
    for f in frames:
        with np.load(inp.records_dir / f"frame_{f}.npz") as z:
            d = np.asarray(z["depth"], np.float32)
            cf = np.asarray(z["conf"], np.float32)
            if tuple(d.shape) != rec_hw or tuple(cf.shape) != rec_hw:
                raise DepthOnF5Error(f"Omega's record of frame {f} is {d.shape[1]}x{d.shape[0]} (conf "
                                     f"{cf.shape[1]}x{cf.shape[0]}), the first record {rec_hw[1]}x{rec_hw[0]} "
                                     f"— one record grid per session")
            if carry:
                d = CC.record_on_native(d, inp.cam, inp.maps)       # NaN where the record does not cover
                cf = CC.record_on_native(cf, inp.cam, inp.maps)
            zo[f] = d; conf[f] = cf; chunk[f] = int(z["chunk"]) if "chunk" in z.files else 0
    floor_norm = float(raw_cfg["reconstruction"]["simple"]["conf_min_norm"])
    thr, cmax = {}, {}
    for k in sorted(set(chunk.values())):
        v = np.concatenate([conf[f][np.isfinite(conf[f]) & (conf[f] > SKY_CONF)].ravel()
                            for f in frames if chunk[f] == k])
        thr[k], cmax[k] = confidence_floor(v, floor_norm)           # THE ONE floor (point 52: kept, pinned)

    # 2. the bend
    _p(25, "bending Omega's depth to F5's landmarks")
    from reconstruction.loops.config import improvement_error_factor
    fac = float(improvement_error_factor(raw_cfg))                   # the user's 2 (point 1), ONE place
    rows = {}
    for i, f in enumerate(frames):
        o = obs[0][i]
        zz = bilinear(zo[f], o[:, 0], o[:, 1]) if len(o) else np.zeros(0)
        ok = zz > bc.min_depth_m
        rows[i] = (design(o[ok, 0], o[ok, 1], W, H), o[ok, 2] / zz[ok])
    # the window is chosen on half A of the held-out (its even rows), half B reports — epoch 7
    rows_A = {i: heldout_rows(obs[1][i], zo[f], bc.min_depth_m, 0) for i, f in enumerate(frames)}
    rows_B = {i: heldout_rows(obs[1][i], zo[f], bc.min_depth_m, 1) for i, f in enumerate(frames)}
    score, coefs, infos, errs_A = {}, {}, {}, {}
    for w in bc.windows:
        cw, inf = bend_coefficients(rows, N, int(w), bc.min_rows, gc.huber_k, bc.irls_iterations, fac)
        per_kf = {i: rel_errors(*rows_A[i], cw[i], W, H) for i in range(N)}
        per_kf = {i: e for i, e in per_kf.items() if len(e)}
        if not per_kf:
            raise DepthOnF5Error("no held-out landmark row on Omega's depth — the bend cannot be judged")
        coefs[int(w)] = cw; infos[int(w)] = inf; errs_A[int(w)] = per_kf
        score[int(w)] = float(np.median(np.concatenate(list(per_kf.values()))))
    raw_err = []
    for i, f in enumerate(frames):
        h = obs[1][i]
        if len(h):
            zz = bilinear(zo[f], h[:, 0], h[:, 1]); ok = zz > 0
            raw_err.append(np.abs(zz[ok] - h[ok, 2]) / h[ok, 2])
    # point 49: the smoothest window within error_factor x the measured error of the best
    wb, window_rule = choose_window(score, errs_A, fac, bc.bootstrap, bc.seed)
    # point 48: every keyframe's c0 verified on its own held-out rows; the pooled fit otherwise
    final_coefs, provenance = verify_keyframes(bc.windows, wb, coefs, infos, rows_A, N, W, H, fac,
                                               bc.heldout_confidence, bc.bootstrap, bc.seed, bc.min_rows)
    status_counts: Dict[str, int] = {}
    for p_ in provenance.values():
        status_counts[p_["status"]] = status_counts.get(p_["status"], 0) + 1
    kept_c1 = sum(1 for p_ in provenance.values() if p_["fit"] and p_["fit"]["kept"] and p_["fit"]["kept"]["c1"])
    kept_c2 = sum(1 for p_ in provenance.values() if p_["fit"] and p_["fit"]["kept"] and p_["fit"]["kept"]["c2"])
    # half B (the report): the final coefficients' held-out error, never used to choose anything
    errs_B = [rel_errors(*rows_B[i], final_coefs[i], W, H) for i in range(N)]
    errs_B = [e for e in errs_B if len(e)]
    score_B = float(np.median(np.concatenate(errs_B))) if errs_B else float("nan")
    c0 = np.array([final_coefs[i][0] for i in range(N)])
    _p(40, f"held-out |dz|/z (half A): unbent {np.median(np.concatenate(raw_err)) * 100:.2f} %, "
           + ", ".join(f"±{w} {score[w] * 100:.2f} %" for w in sorted(score))
           + f" → ±{wb} (best ±{window_rule['window_best']}, bar {window_rule['bar'] * 100:.2f} % = best + "
           f"{fac:g} x se {window_rule['heldout_best_se'] * 100:.3f} %); per keyframe: "
           + ", ".join(f"{k} {v}" for k, v in sorted(status_counts.items()))
           + f"; c1 kept on {kept_c1}, c2 on {kept_c2} of {N}; half B {score_B * 100:.2f} %; scale "
           f"{np.median(c0):.4f} [{c0.min():.4f}, {c0.max():.4f}], largest consecutive jump {np.abs(np.diff(c0)).max():.3f}")
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    Dm = design(uu.ravel(), vv.ravel(), W, H)
    dep, valid, passed, weight = {}, {}, {}, {}
    md = pcfg.mono_detail
    for i, f in enumerate(frames):
        valid[f] = np.isfinite(zo[f]) & (zo[f] > 0) & np.isfinite(conf[f]) & (conf[f] > SKY_CONF)
        passed[f] = valid[f] & (conf[f] >= thr[chunk[f]])
        # the bent map in float32 first, then the mask — epoch 8's arithmetic, bit for bit
        bent = (zo[f] * (Dm @ final_coefs[i]).reshape(H, W)).astype(np.float32)
        dep[f] = np.where(valid[f], bent, 0).astype(np.float32)
        if md.enabled:
            # the session's calibrated confidence as a weight in 0..1 above its own floor (per chunk)
            weight[f] = confidence_weight(conf[f], thr[chunk[f]], float(cmax[chunk[f]]), passed[f])
    cmax = None
    del zo, conf

    # 3b. mono detail (claude_stac.txt 2026-10-04): PointDiT refines the bent maps before the vote
    dep, valid, passed, src_maps, md_rep = apply_mono_detail(
        pcfg, frames, dep, valid, passed, weight, inp, K, c2w, out, log, _p)

    # 4. the edge-keeping vote (pccr epoch 8)
    _p(50, "edge-keeping multi-view vote")
    labels = mask_labels(out, H, W, log)
    if labels is not None and lens:
        labels = labels_on_undistorted(labels, inp.maps)       # the masks live on the original frame
    vote_bars: dict = {}
    voted, tau, vst = edge_keeping_vote(frames, dep, valid, passed, K, c2w, labels,
                                        bc.neighbors, bc.tau_quantile, int(pcfg.cloud.repair_min_views), log,
                                        bars_out=vote_bars)
    cover = vst["out"] / float(N * H * W)
    _p(65, f"vote done: coverage {cover * 100:.1f} %")
    del dep, valid, passed
    return types.SimpleNamespace(session_dir=session_dir, out=out, inp=inp, frames=frames, N=N, W=W, H=H, K=K,
                                 c2w=c2w, params=params, chunk=chunk, voted=voted, tau=tau, vst=vst, cover=cover,
                                 wb=wb, score=score, score_B=score_B, window_rule=window_rule, coefs=final_coefs,
                                 vote_bars=vote_bars,
                                 provenance=provenance, status_counts=status_counts, c0=c0, obs=obs,
                                 dropped_tracks=dropped_tracks, error_factor=fac,
                                 floor={str(k): {"threshold": thr[k]} for k in sorted(thr)}, floor_norm=floor_norm,
                                 src_maps=src_maps, md_rep=md_rep, md=md, uu=uu, vv=vv, t0=t0, _p=_p,
                                 seconds_compute=round(time.time() - t0, 1),
                                 rec_hw=rec_hw, carry=carry, lens=lens, grid_note=grid_note)


def publish_cloud(C: types.SimpleNamespace, pcfg, log: Callable = print) -> dict:
    """Step 5: the voted maps → chunks → the cloud stage's cleaner → the new-cloud epoch (transaction)."""
    from config import cfg as raw_cfg
    from precision import corrected_cloud as CC
    from precision.epoch0_cloud import _write_ply_xyzrgb
    from correction.session import read_ply
    session_dir, out, inp, frames, W, H, K, c2w = (C.session_dir, C.out, C.inp, C.frames, C.W, C.H, C.K, C.c2w)
    params, chunk, voted, tau, vst, cover, wb, score, coefs, c0 = (C.params, C.chunk, C.voted, C.tau, C.vst, C.cover,
                                                                   C.wb, C.score, C.coefs, C.c0)
    src_maps, md_rep, md, uu, vv, t0, _p = C.src_maps, C.md_rep, C.md, C.uu, C.vv, C.t0, C._p
    t_publish = time.time()

    # 5. the cloud
    tmp = out / TX_TMP
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "chunks").mkdir(parents=True)
    by_chunk: Dict[int, List[int]] = {}
    for f in frames:
        by_chunk.setdefault(chunk[f], []).append(f)
    n_raw = 0
    try:
        for k, fl in sorted(by_chunk.items()):
            L = {x: [] for x in ("xyz", "rgb", "fg", "pr", "pc", "cf")}
            for f in fl:
                zmap, amap = voted[f][0], voted[f][1]
                m = zmap > 0
                r, c = vv[m], uu[m]; zz = zmap[m].astype(np.float64)
                X = np.stack([(c - K[0, 2]) / K[0, 0] * zz, (r - K[1, 2]) / K[1, 1] * zz, zz], 1) @ c2w[f][:3, :3].T + c2w[f][:3, 3]
                img = CC._rgb_undistorted(inp.frames_dir, f, inp.maps)
                L["xyz"].append(X.astype(np.float32)); L["rgb"].append(img[r, c])
                L["fg"].append(np.full(len(r), f, np.int32)); L["pr"].append(r.astype(np.int16))
                L["pc"].append(c.astype(np.int16)); L["cf"].append(amap[m].astype(np.float32))
            xyz = np.concatenate(L["xyz"]); n_raw += len(xyz)
            _write_ply_xyzrgb(tmp / "chunks" / f"chunk_{k:03d}.ply", xyz, np.concatenate(L["rgb"]))
            np.savez(tmp / "chunks" / f"chunk_{k:03d}_origins.npz", frame_global=np.concatenate(L["fg"]),
                     pixel_row=np.concatenate(L["pr"]), pixel_col=np.concatenate(L["pc"]),
                     confidence=np.concatenate(L["cf"]))
        _p(72, f"{n_raw:,} raw points → the cloud stage's cleaner")
        cleaned = tmp / "cleaned_cloud.ply"
        CC.clean(raw_cfg, tmp / "chunks", cleaned, log)
        shutil.rmtree(tmp / "chunks", ignore_errors=True)
        _, data = read_ply(cleaned)
        cols = {"pixel_u_und": np.asarray(data["pixel_col"]), "pixel_v_und": np.asarray(data["pixel_row"]),
                "n_consistent": np.clip(np.asarray(data["confidence"]), 0, 255).astype(np.uint8),
                "source": source_column(data, src_maps)}
        report = {"version": 1, "stage": "depth_on_f5", "provenance": "tool_measured",
                  "source_of_depth": "Omega's depth bent to F5's landmarks + edge-keeping multi-view vote (pccr epoch 8)",
                  "camera": params, "grid": [W, H],
                  # the grid the bend worked on: the undistorted native frame; Omega's record (its own
                  # grid, the original frame) carried onto it through the lens + grid when they differ
                  "grid_of_the_bend": {"undistorted_native": [W, H], "record": [C.rec_hw[1], C.rec_hw[0]],
                                       "lens": bool(C.lens), "carried": bool(C.carry), "note": C.grid_note},
                  "bend": {"window": int(wb), "held_out": {str(w): v for w, v in sorted(score.items())},
                           "held_out_half_B_final": C.score_B, "window_rule": C.window_rule,
                           "improvement_error_factor": C.error_factor,
                           "heldout_confidence": float(pcfg.bend.heldout_confidence),
                           "keyframe_status": C.status_counts,
                           "landmark_tracks_dropped": C.dropped_tracks,
                           "scale": {"median": float(np.median(c0)), "min": float(c0.min()),
                                     "max": float(c0.max())}},
                  "confidence_floor": {"conf_min_norm": C.floor_norm, "per_chunk": C.floor,
                                       "rule": "min + conf_min_norm x (max - min) of the chunk's valid "
                                               "confidences (THE ONE floor, shared with the viewer; point 52)"},
                  # what each keyframe's depth was multiplied by — the chunk check (f6_check) measures
                  # the PUBLISHED cloud with it: s_k = c0, bend = (c1, c2) of precision.depth_on_f5.design —
                  # and WHY it got that fit (point 48: status, window, the terms kept, the margins)
                  "per_frame": {str(f): {"s_k": float(coefs[i][0]),
                                         "bend": [float(coefs[i][1]), float(coefs[i][2])],
                                         **C.provenance[i]}
                                for i, f in enumerate(frames)},
                  "vote": {"tau": tau, "coverage": cover, "bars": C.vote_bars,
                           **{k: (float(v) / max(vst["valid"], 1) if k not in ("tau", "valid") else v)
                              for k, v in vst.items() if k != "tau"}},
                  "raw_points": n_raw}
        if md_rep is not None:
            report["mono_detail"] = mono_report(md_rep, md)
            report["source_counts"] = {SRC_CLOUD_NAMES[int(v)]: int(c) for v, c in
                                       zip(*np.unique(cols["source"], return_counts=True))}
            write_mono_layers(out / "precision", data, cols["source"], md_rep, K, c2w, log)
        camera_travels(tmp, params, len(frames), log)
        _p(85, "publishing the epoch (octree, atomic swap)")
        rep = CC.publish(session_dir, tmp, report, log, columns=cols)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    (out / "precision").mkdir(exist_ok=True)
    (out / "precision" / REPORT).write_text(json.dumps(rep, indent=1, default=float))
    # the clocks live next to the report, never in it (points 36 / 56)
    times = {"seconds_compute": C.seconds_compute, "seconds_publish": round(time.time() - t_publish, 1),
             "seconds": round(time.time() - t0, 1)}
    if md_rep is not None:
        times["mono_detail"] = mono_timing(md_rep)
    CC.write_timing(out / "precision" / REPORT, times)
    _p(100, f"depth on F5 published epoch {rep['epoch_to']} ({rep['n_points']:,} pts, {times['seconds']} s)")
    return rep


def run_depth_on_f5(session_dir: Path, pcfg, log: Callable = print,
                    progress: Optional[Callable[[float, str], None]] = None) -> dict:
    return publish_cloud(compute(session_dir, pcfg, log=log, progress=progress), pcfg, log=log)




def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--session", required=True)
    args = ap.parse_args(argv)
    from precision.config import load_precision_config
    run_depth_on_f5(Path(args.session), load_precision_config())
    return 0


if __name__ == "__main__":
    sys.exit(main())
