"""F6 — the depth on F5: Omega's depth at the session's scale, then a multi-view vote (the product).

USER 2026-10-01: *"integrá ya mismo el punto 6, la profundidad sobre F5, porque es lo que cambió
drásticamente el resultado"*. It is the cloud of the default chain (`precision.cloud.source: omega_bent`).

The cameras and poses are the session's (F5's after f5_refine); the depth of a keyframe is Omega's record:
 1. landmarks: F5's FIT tracks triangulated with these poses + camera (precision.refine helpers), within the range;
    tracks that do not round-trip through F5's lens are dropped and counted;
 2. THE DEPTH'S FACTOR — ONE FOR THE SESSION (USER 2026-11-20, :func:`session_factor`): Omega's records are in
    the cube world's units, the cameras were rescaled since; the factor that brings the depth to them is the
    robust fit of one constant to z_landmark / z_Omega over every landmark row of the session. Until that date
    the factor was fitted keyframe by keyframe on a window of neighbours and "verified" on held-out landmarks
    of the same F5: it followed F5's local errors (0.10 … 2.18 on pccr 2026-08-24) and displaced what a
    keyframe sees in proportion to its distance (docs/pipeline_audit_2026-11-20.md). F4's held-out tracks
    only REPORT what the depth at that factor leaves against them (:func:`heldout_error`);
 3. validity: Omega's ONE confidence floor (`reconstruction.simple.conf_min_norm`, min-max per Omega
    chunk) and not sky (epoch0_cloud.SKY_CONF);
 4. vote over the keyframe offsets `bend.neighbors`: a pixel LEAVES when more neighbours see free space
    through it than agree; otherwise its depth is the MEDIAN of its own and the agreeing views carried
    onto its ray. τ = the `bend.tau_quantile` percentile of the session's own neighbour disagreement.
    The turn frames vote after the neighbours, and the keyframes of OTHER moments of the walk that see the
    same surface (USER 2026-11-07, :func:`covisible_witnesses`: two moments that disagree contradict each
    other instead of both standing);
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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

LOG_TAG = "[depth-on-f5]"
REPORT = "depth_on_f5.json"
TX_TMP = "_tx_depth_on_f5"
MAD_TO_SIGMA = 1.0 / 0.6744897501960817          # 1.4826: σ of a normal from its MAD


RECORD_THREADS = 8            # the records' inflation (I/O and zlib: no arithmetic of the product)


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


def session_factor(ratios: Sequence[np.ndarray], huber_k: float, iterations: int) -> Tuple[float, dict]:
    """THE DEPTH'S FACTOR — ONE FOR THE WHOLE SESSION (USER 2026-11-20). ``ratios``: per keyframe, z_landmark / z_Omega
    of F5's landmarks on that keyframe's depth map. The factor is the robust fit (Huber IRLS) of ONE constant to every
    row of the session; every keyframe's depth is multiplied by it.

    Why one, and not one per keyframe (what this replaced): the factor makes Omega's depth agree with the scale of the
    session's cameras, and that is a property of the SESSION. Decided keyframe by keyframe it followed every local
    accident of F5 — pccr 2026-08-24: from 0.10 to 2.18, x0.87 on the stretch around a door where F5 had shortened the
    camera steps, which published a cabinet seen from 5.9 m half a metre short of the same cabinet seen from 1 m. A
    depth factor the cameras do not match displaces what a keyframe sees in proportion to its distance
    (docs/pipeline_audit_2026-11-20.md, Lemma 1). Returns (factor, its record)."""
    r = [np.asarray(x, np.float64) for x in ratios if len(x)]
    if not r:
        raise DepthOnF5Error("no landmark row on Omega's depth in the whole session — its depth factor cannot be "
                             "measured")
    rr = np.concatenate(r)
    c, sig = irls_huber_fit(np.ones((len(rr), 1)), rr, huber_k, iterations)
    c0 = float(c[0])
    if not np.isfinite(c0) or c0 <= 0:
        raise DepthOnF5Error(f"the session's depth factor came out {c0!r} from {len(rr):,} landmark rows")
    return c0, {"factor": c0, "sigma": float(sig[0]), "rows": int(len(rr)), "keyframes_with_rows": int(len(r)),
                "median_of_the_rows": float(np.median(rr))}


def heldout_rows(obs_held: np.ndarray, zmap: np.ndarray, min_depth_m: float) -> Tuple[np.ndarray, np.ndarray]:
    """(z_landmark, z_Omega) of a keyframe's held-out landmark rows on Omega's depth (rows under ``min_depth_m`` of
    Omega depth are out). The held-out tracks are F4's own split: the factor never saw them."""
    h = np.asarray(obs_held, np.float64).reshape(-1, 3)
    if not len(h):
        return np.zeros(0), np.zeros(0)
    zz = bilinear(zmap, h[:, 0], h[:, 1])
    ok = zz > float(min_depth_m)
    return h[ok, 2], zz[ok]


def heldout_error(rows: Sequence[Tuple[np.ndarray, np.ndarray]], factor: float) -> dict:
    """What the depth at the session's factor leaves against the landmarks the factor never saw, pooled over the
    keyframes: the median |z_Omega·factor − z_landmark| / z_landmark, and the same rows in METRES along the ray with
    the range they sit at. The distance is the session's repeatability when nothing more direct measured it
    (reconstruction.certify.repeatability, its fourth source: a session of one cube has no shared frame). A REPORT:
    nothing is decided with it."""
    e, z = [], []
    for z_lm, z_om in rows:
        if len(z_lm):
            e.append(np.abs(np.asarray(z_om, np.float64) * float(factor) - z_lm)); z.append(np.asarray(z_lm, np.float64))
    if not e:
        return {"median_rel": None, "median_m": None, "range_m": None, "rows": 0}
    e, z = np.concatenate(e), np.concatenate(z)
    return {"median_rel": float(np.median(e / z)), "median_m": float(np.median(e)), "range_m": float(np.median(z)),
            "rows": int(len(e))}


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
    from precision import f6_torch as FT
    return FT._np(FT.interior(FT._t(passed, FT.torch.bool)))


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
                      neighbors, tau_quantile: float, repair_min_views: int, rescue_below_floor: bool,
                      log: Callable = print, bars_out: Optional[dict] = None,
                      extra: Optional[Dict[int, List[int]]] = None, min_parallax_deg: Optional[float] = None,
                      min_support: int = 0, fusion: Optional[dict] = None, no_birth=(), vote_batch: int = 0):
    """pccr EPOCH 8 (USER 2026-10-01: "es la mejor, incorporar al pipeline"), on the bent depth
    `dep` (every `valid` pixel; `passed` = above the ONE confidence floor):
     1. tau = the `tau_quantile` percentile of the neighbour disagreement on INTERIOR pixels;
     2. EDGE pixels = the 3x3 window spans a depth step (on the depth before snapping);
        MIXED pixels (on neither surface of the step) snapped to the side their SAM3 mask says;
     3. the TWO-SIDED vote over `neighbors`, judged by floor-passing neighbour pixels only:
        floor-passing pixels stay with contra ≤ agree (median of the agreeing views), a
        contradicted one is repaired to the median of the neighbours when ≥ `repair_min_views`
        agree, else it leaves; a below-floor EDGE pixel enters when agree ≥ 1 and contra ≤ agree;
     4. ``rescue_below_floor`` (USER 2026-11-04, night): any other below-floor pixel enters with the
        depth ≥ `repair_min_views` floor-passing views agree on along its ray (the repair's rule,
        their median — never its own depth), unless its own depth lies in front of that surface
        beyond tau (something nearer no view measured: the surface behind would wear its colour).
    Returns ({frame: (depth map, agree count map)}, tau, totals). ``bars_out`` (a dict) receives the
    vote's bar τ and every keyframe's pixel margins to it (point 53) — kept out of ``totals``, which
    stays numeric for its readers. ``extra``: {keyframe: [witness frames]} — the turn frames and the
    keyframes of other moments that see the same surface, voting on that keyframe after its neighbours
    (USER 2026-11-04, 2026-11-07; their maps and poses are in ``dep`` / ``passed`` / ``c2w``); τ is
    measured on the walk neighbours alone, as before.
     5. THE SUPPORT (USER 2026-11-07: "si pedimos piso de 10 vistas … quiero puntos válidos sin ruido"; a vote
        counts when it TESTS the depth): with ``min_parallax_deg``, a pixel's support is the number of views that
        agree with the depth it is published at AND see that point from at least ``min_parallax_deg`` away from
        this keyframe's camera (F5's bound for a landmark, refine.min_tri_deg — a view from almost the same place
        agrees with any depth along the ray, and a turn or interval frame next to its keyframe repeats it): the
        agreeing witnesses for a kept or admitted pixel, the agreeing surface's views for a repaired or rescued
        one. A pixel the vote keeps, repairs, admits or rescues enters only with a support of at least
        ``min_support``; the support is the agree map returned (the cloud's ``confidence`` column).
        ``min_parallax_deg`` None = every agreeing view counts and no floor (before 2026-11-07).
    ``vote_batch`` (cloud.vote_batch): 0 = THE RECORD — one witness at a time, its maps uploaded at every pair;
    N ≥ 1 = the same vote with the session's maps resident on the device and the witnesses N at a time
    (f6_torch.two_sided_vote_fast: every map the same bit for bit — tests/test_f6_vote_fast.py)."""
    import torch
    from precision import corrected_cloud as CC
    from precision import f6_torch as FT
    H, W = next(iter(dep.values())).shape
    inner = {f: interior(passed[f]) for f in frames}
    rng_m = float((fusion or {}).get("witness_max_m", 0.0))
    if rng_m > 0:
        # THE RANGE (reconstruction.view_range_m, USER 2026-11-09): τ is measured only where a pixel can enter
        from precision.camera import depth_map_distance
        inner = {f: inner[f] & (depth_map_distance(dep[f], K) <= rng_m) for f in frames}
    tau = CC.measured_tau(dep, inner, K, c2w, frames, neighbors, tau_quantile)
    del inner
    if fusion is not None and float(fusion.get("max_depth_rel", 0.0)) <= 0.0:
        fusion = dict(fusion, max_depth_rel=float(tau))        # the session's measured τ, never a fixed number
    edge = {}
    n_mixed = n_snap = 0
    for i, f in enumerate(frames):
        edge[f] = CC.depth_steps(dep[f], valid[f], tau)
        if labels is not None:
            d, mixed, sn = CC.snap_mixed(dep[f], valid[f], labels(i), tau)
            dep[f] = d.astype(np.float32); n_mixed += int(mixed.sum()); n_snap += int(sn.sum())
    w2c = {f: np.linalg.inv(c2w[f]) for f in c2w}
    extra = extra or {}
    final, tot = {}, {}
    tau_margins: Dict[int, Optional[dict]] = {}
    res = None
    if int(vote_batch) > 0:
        res = FT.Resident(dep, passed)               # (after the snap: the maps the vote reads, once)
        log(f"{LOG_TAG} vote: {len(dep)} maps resident on {FT.device()}, witnesses {int(vote_batch)} at a time, "
            f"z-buffer by {FT.zbuffer_method()} (cloud.vote_batch; the record's bits)")
    # USER 2026-10-08: the vote on the card — dense [H, W] tensors per keyframe (f6_torch), the
    # decisions below are the same rules on the candidate pixels
    for i, f in enumerate(frames):
        cand = valid[f] & (passed[f] | edge[f])           # below-floor interior pixels stay out
        if res is not None:
            v = FT.two_sided_vote_fast(i, frames, res, cand, K, c2w, w2c, neighbors, tau,
                                       extra=extra.get(f, ()), min_parallax_deg=min_parallax_deg, fusion=fusion,
                                       batch=int(vote_batch))
        else:
            v = FT.two_sided_vote(i, frames, dep, passed, cand, K, c2w, w2c, neighbors, tau,
                                  extra=extra.get(f, ()), min_parallax_deg=min_parallax_deg, fusion=fusion)
        candt = FT._t(cand, torch.bool)
        if fusion is not None and str(fusion.get("mode", "keep")) == "adjust":
            # ADJUST (USER 2026-11-08: "no es un filtro, es un ajuste … buscar la posición correcta de los
            # puntos"): every qualified witness ≤ witness_max_m puts ITS surface's depth on this ray
            # (two_sided_vote's ``cons``), the strongest cluster decides the position (strongest_consensus) —
            # a pixel whose own camp loses is MOVED onto the winner (the displaced wall copy collapses onto
            # the wall), and only a pixel with no cluster of ``min_support`` dies. The published depth is the
            # winner's median; the support is the winner's size.
            z_cons, n_cons, own_in = FT.strongest_consensus(v["cons"], float(tau), int(min_support))
            keep = candt & (n_cons > 0) & v["in_range"]
            if f in no_birth:
                keep = torch.zeros_like(keep)
            zero = torch.zeros_like(v["z"])
            zmap = torch.where(keep, z_cons.to(v["z"].dtype), zero)
            amap = torch.where(keep, n_cons, torch.zeros_like(n_cons))
            moved = keep & ~own_in
            pas = FT._t(passed[f], torch.bool) & candt
            zmap_np = FT._np(zmap, np.float32); amap_np = FT._np(amap, np.int16)
            final[f] = (zmap_np, amap_np)
            tau_margins[f] = None
            for k, n in (("valid", int(valid[f].sum())), ("edge", int(edge[f].sum())), ("kept", int(keep.sum())),
                         ("contradicted", int((pas & ~keep).sum())), ("repaired", int(moved.sum())),
                         ("admitted", 0), ("edge_below", 0),
                         ("below_floor", int((valid[f] & ~passed[f]).sum())), ("rescued", 0),
                         ("rescue_refused_nearer", 0), ("below_support", int((candt & ~keep).sum())),
                         ("no_birth", int(candt.sum()) if f in no_birth else 0),
                         ("out", int((zmap_np > 0).sum()))):
                tot[k] = tot.get(k, 0) + n
            continue
        if fusion is not None:
            # COLMAP's fusion rule (USER 2026-11-08): a pixel enters with ≥ min_support witnesses that CONFIRM it
            # (forward-backward reprojection ≤ max_reproj_px, depths within max_depth_rel, triangulation angle in
            # the band); nothing is repaired, rescued or admitted — a pixel is kept or dropped. A keyframe without
            # parallax (``no_birth``) votes as a witness and gives birth to nothing.
            n_conf = v["agree_q"]
            keep = candt & (n_conf >= int(min_support))
            # THE RANGE (USER 2026-11-09: "nadie debe aportar puntos a más de 10m de distancia"): no keyframe
            # gives birth to a point farther than witness_max_m from its camera
            beyond = candt & ~v["in_range"]
            keep = keep & ~beyond
            n_minor = 0
            if bool(fusion.get("majority")):
                # THE MAJORITY (USER 2026-11-09: "si tenemos muchas vistas del mismo punto 5 puede significar poco,
                # pero si tiene pocas vistas puede significar mucho"): the witnesses in range that CONFIRM the point
                # must outnumber those in range and in the band that see THROUGH it (pccr 3108, the reference cloud:
                # 94.1 % of its kept pixels pass; zaragoza 74.6 % — census of 2026-11-09)
                minority = keep & ~(n_conf > v["thru_q"])
                n_minor = int(minority.sum())
                keep = keep & ~minority
            n_veto = 0
            if bool(fusion.get("occlusion_veto")):
                # the Merrell duel (ICCV07, researched 2026-11-08): more witnesses measured surfaces IN FRONT of
                # this pixel than confirm it — it sits behind a better-supported surface (the hidden duplicate)
                occluders = (v["contra"] - v["contra_fwd"]).clamp(min=0)
                veto = keep & (occluders > n_conf)
                n_veto = int(veto.sum())
                keep = keep & ~veto
            inpl = torch.zeros_like(keep)
            if bool(fusion.get("in_place")) and v["splats"].shape[0]:
                # IN PLACE (USER 2026-11-09: "los recuperados que están cerca de la cámara, ése es el problema" / "aplicá
                # esta nueva regla"): a pixel the vote did not keep whose witnesses' surfaces on its ray (in range)
                # agree — ≥ min_support within τ of their median — AT ITS OWN DEPTH (that median within τ of it) is
                # published WHERE IT IS: never moved, so it keeps its own colour at its own depth (the repair's moved
                # pixels painted their birth colour elsewhere — zaragoza 2026-11-09). pccr 3108 census: 79 % of the
                # near surfaces the validated run had and the COLMAP vote lost come back, all inside that cloud
                ok_s, z_s, n_s = FT.agreeing_median(v["splats"], tau, int(min_support))
                own = torch.abs(z_s - v["z"].to(z_s.dtype)) <= tau * v["z"].to(z_s.dtype)   # NaN → False
                inpl = candt & ~keep & ok_s & own & v["in_range"]
            rep = torch.zeros_like(keep)
            if bool(fusion.get("repair")) and v["splats"].shape[0]:
                # USER 2026-11-08 evening: a pixel SHORT of support whose witnesses' surfaces agree among
                # themselves on its ray is published AT THEIR MEDIAN — coverage of the GOOD kind (A6)
                rep_ok, zrep, nrep = FT.agreeing_median(v["splats"], tau, int(min_support))
                rep = candt & ~keep & ~inpl & rep_ok & v["in_range"]
            if f in no_birth:
                keep = torch.zeros_like(keep)
                rep = torch.zeros_like(rep)
                inpl = torch.zeros_like(inpl)
            zero = torch.zeros_like(v["z"])
            zmap = torch.where(keep, v["zmed"], zero)
            amap = torch.where(keep, n_conf, torch.zeros_like(n_conf))
            if bool(inpl.any()):
                zmap = torch.where(inpl, v["z"].to(zmap.dtype), zmap)
                amap = torch.where(inpl, n_s.to(amap.dtype), amap)
            if bool(rep.any()):
                zmap = torch.where(rep, zrep.to(zmap.dtype), zmap)
                amap = torch.where(rep, nrep.to(amap.dtype), amap)
            pas = FT._t(passed[f], torch.bool) & candt
            zmap_np = FT._np(zmap, np.float32); amap_np = FT._np(amap, np.int16)
            final[f] = (zmap_np, amap_np)
            tau_margins[f] = None
            for k, n in (("valid", int(valid[f].sum())), ("edge", int(edge[f].sum())), ("kept", int(keep.sum())),
                         ("contradicted", int((pas & ~keep & ~rep & ~inpl).sum())), ("repaired", int(rep.sum())),
                         ("in_place", int(inpl.sum())),
                         ("admitted", 0), ("occlusion_veto", n_veto),
                         ("beyond_range", int(beyond.sum())), ("minority", n_minor),
                         ("edge_below", 0), ("below_floor", int((valid[f] & ~passed[f]).sum())), ("rescued", 0),
                         ("rescue_refused_nearer", 0),
                         ("below_support", int((candt & ~keep & ~rep & ~inpl).sum())),
                         ("no_birth", int(candt.sum()) if f in no_birth else 0),
                         ("out", int((zmap_np > 0).sum()))):
                tot[k] = tot.get(k, 0) + n
            continue
        agree, contra, z, s = v["agree"], v["contra"], v["z"], v["splats"]
        # point 53: every judged pixel's margin to τ — the nearest neighbour surface on its ray
        # against its own depth, (τ − |Δz|/z) / τ (positive = agrees with its nearest witness)
        if s.shape[0]:
            nan = torch.full_like(s, float("nan"))
            rel = torch.abs(s - z[None]) / torch.where(s > 0, s, nan)
            dmin = torch.where(torch.isfinite(rel), rel, torch.full_like(rel, float("inf"))).amin(0)
            seen = torch.isfinite(dmin) & candt
            tau_margins[f] = FT.margin_quantiles((tau - dmin[seen]) / max(tau, 1e-12)) if bool(seen.any()) else None
            rep_ok, zrep, nrep = FT.agreeing_median(s, tau, int(repair_min_views))
        else:
            tau_margins[f] = None
            rep_ok = torch.zeros((H, W), dtype=torch.bool, device=z.device)
            zrep = torch.full((H, W), float("nan"), dtype=torch.float64, device=z.device)
            nrep = torch.zeros((H, W), dtype=torch.int32, device=z.device)
        pas = FT._t(passed[f], torch.bool) & candt
        edg = FT._t(edge[f], torch.bool) & candt
        contradicted = pas & (contra > agree)
        agreed = rep_ok                      # ≥ repair_min_views floor-passing views on one surface
        rep_ok = rep_ok & contradicted
        keep, repair, admit = CC.edge_vote_decision(pas, edg, agree, contra, rep_ok)
        below = FT._t(valid[f], torch.bool) & ~FT._t(passed[f], torch.bool)
        offer = below & agreed & ~admit      # a below-floor pixel the floor-passing views agree on
        nearer = offer & (z < zrep * (1.0 - tau))       # its own depth in front of their surface
        if not rescue_below_floor:
            offer = nearer = torch.zeros_like(offer)
        rescue = offer & ~nearer
        m = keep | admit
        zero = torch.zeros_like(z)
        zmap = torch.where(m, v["zmed"], zero)
        zmap = torch.where(repair | rescue, zrep, zmap)
        if min_parallax_deg is None:
            amap = torch.where(m, agree, torch.zeros_like(agree))
            amap = torch.where(repair | rescue, nrep, amap)
            low = torch.zeros_like(m)
        else:
            # 5. the support: only the views that test the depth the pixel is published at
            nrep_q = (FT.agreeing_count(s, tau, v["par_splat"]) if s.shape[0]
                      else torch.zeros_like(agree))
            amap = torch.where(m, v["agree_q"], torch.zeros_like(agree))
            amap = torch.where(repair | rescue, nrep_q, amap)
            low = (m | repair | rescue) & (amap < int(min_support))
            zmap = torch.where(low, zero, zmap)
            amap = torch.where(low, torch.zeros_like(amap), amap)
        zmap_np = FT._np(zmap, np.float32); amap_np = FT._np(amap, np.int16)
        final[f] = (zmap_np, amap_np)
        for k, n in (("valid", int(valid[f].sum())), ("edge", int(edge[f].sum())), ("kept", int(keep.sum())),
                     ("contradicted", int(contradicted.sum())), ("repaired", int(repair.sum())),
                     ("admitted", int(admit.sum())), ("edge_below", int((~pas & edg).sum())),
                     ("below_floor", int(below.sum())), ("rescued", int(rescue.sum())),
                     ("rescue_refused_nearer", int(nearer.sum())), ("below_support", int(low.sum())),
                     ("out", int((zmap_np > 0).sum()))):
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
        f"contradicted, {tot['repaired'] / nv * 100:.1f} % repaired, {tot.get('in_place', 0) / nv * 100:.1f} % in place, "
        f"{tot['admitted'] / nv * 100:.1f} % admitted below "
        f"the floor at edges, "
        + (f"{tot['rescued'] / nv * 100:.1f} % rescued below the floor ({tot['rescued'] / max(tot['below_floor'], 1) * 100:.1f} % "
           f"of the {tot['below_floor'] / nv * 100:.1f} % under it; {tot['rescue_refused_nearer'] / nv * 100:.1f} % refused, "
           f"their own depth in front)" if rescue_below_floor else "no rescue below the floor (rescue_below_floor false)")
        + (f"; {tot['below_support'] / nv * 100:.1f} % left under the support of {int(min_support)} views that test "
           f"the depth (≥ {float(min_parallax_deg):g}° from this keyframe's camera at the point)"
           if min_parallax_deg is not None else "")
        + f"; coverage {tot['out'] / float(len(frames) * H * W) * 100:.1f} % of all pixels")
    if min_parallax_deg is not None:
        tot.update(min_support=int(min_support), min_parallax_deg=float(min_parallax_deg))
    if fusion is not None:
        tot.update(min_support=int(min_support), no_birth_keyframes=int(len(no_birth)))   # numeric only (the report)
        if bars_out is not None:
            bars_out["fusion"] = dict(fusion)
        if str(fusion.get("mode", "keep")) == "adjust":
            log(f"{LOG_TAG} fusion rule: ADJUST (USER 2026-11-08 'no es un filtro, es un ajuste') — every "
                f"qualified witness at [{fusion['tri_min_deg']:g}°, {fusion['tri_max_deg']:g}°] within "
                f"{fusion.get('witness_max_m', 0):g} m puts its surface on the ray; the strongest cluster "
                f"(± τ {float(tau) * 100:.2f} %) of ≥ {int(min_support)} views decides the position: "
                f"{tot.get('repaired', 0):,} pixel(s) MOVED onto the winner ({tot.get('repaired', 0) / nv * 100:.1f} % "
                f"of the valid), {tot.get('below_support', 0) / nv * 100:.1f} % with no consensus died; "
                f"{len(no_birth)} keyframe(s) without parallax give birth to nothing")
        else:
            log(f"{LOG_TAG} fusion rule: COLMAP — ≥ {int(min_support)} witnesses at [{fusion['tri_min_deg']:g}°, "
                f"{fusion['tri_max_deg']:g}°] confirming within {fusion['max_reproj_px']:g} px and "
                f"{fusion['max_depth_rel'] * 100:g} % of depth; "
                + (f"repair ON: {tot.get('repaired', 0):,} short-of-support pixel(s) published at their witnesses' "
                   f"median; " if bool(fusion.get("repair")) else "no repair; ")
                + (f"in place ON: {tot.get('in_place', 0):,} pixel(s) whose witnesses agree with their own depth "
                   f"published where they are; " if bool(fusion.get("in_place")) else "")
                + (f"range {float(fusion.get('witness_max_m', 0)):g} m: {tot.get('beyond_range', 0):,} pixel(s) "
                   f"farther from their camera born by nobody, no witness beyond it votes; "
                   if float(fusion.get("witness_max_m", 0)) > 0 else "")
                + (f"majority ON: {tot.get('minority', 0):,} pixel(s) seen through by more witnesses than "
                   f"confirm them left; " if bool(fusion.get("majority")) else "")
                + (f"occlusion veto ON: {tot.get('occlusion_veto', 0):,} pixel(s) behind a better-supported "
                   f"surface left; " if bool(fusion.get("occlusion_veto")) else "")
                + f"{len(no_birth)} keyframe(s) without "
                f"parallax give birth to nothing ({tot.get('no_birth', 0) / nv * 100:.1f} % of the valid pixels)")
    return final, tau, tot


def rotation_keyframes(session_dir: Path, frames: List[int], log: Callable = print) -> set:
    """The keyframes I1 made by ROTATION (frames/selected_frames.json ``windows[].rotation``), as frame
    numbers among ``frames`` — they vote as witnesses and give birth to no point (USER 2026-11-08: COLMAP
    registers an image without parallax by PnP and triangulates nothing from it)."""
    p = Path(session_dir) / "frames" / "selected_frames.json"
    if not p.exists():
        log(f"{LOG_TAG} no frames/selected_frames.json — no keyframe marked without parallax")
        return set()
    doc = json.loads(p.read_text())
    files = [str(x) for x in (doc.get("selected_files") or [])]
    stems = {}
    for i, name in enumerate(files):
        try:
            stems[i] = int(Path(name).stem)
        except ValueError:
            pass
    have = set(int(f) for f in frames)
    rot = set()
    for w in doc.get("windows") or []:
        # the mark is ``closed_by == "rotation"`` (the window closed on the rotation quantum, not on parallax);
        # ``rotation`` itself is a dict of margins present on EVERY window (pccr 2026-11-08: reading it as the
        # mark made 294 of 313 keyframes birthless — coverage 1.2 %)
        if str(w.get("closed_by", "")) != "rotation":
            continue
        k = w.get("keyframe")
        cands = []
        if isinstance(k, str):
            try:
                cands.append(int(Path(k).stem))
            except ValueError:
                pass
        elif isinstance(k, (int, float)):
            cands.append(int(k))
            if int(k) in stems:
                cands.append(stems[int(k)])
        for c in cands:
            if c in have:
                rot.add(c)
                break
    log(f"{LOG_TAG} {len(rot)} of {len(frames)} keyframes were made by rotation (no parallax): witnesses only")
    return rot


def load_turn_witnesses(out: Path, frames: List[int], chunk: Dict[int, int], thr: Dict[int, float],
                        zmed: Dict[int, float], depth_factor: float, c2w: Dict[int, np.ndarray],
                        inp, carry: bool, rec_hw, H: int, W: int, neighbors,
                        log: Callable = print):
    """The TURN FRAMES and the INTERVAL WITNESSES as witnesses of the vote (USER 2026-11-04, "hacé 1
    y 2" … "que solo entren para afirmar o no en el F6"): the fork kept every context view's (c*)
    and every witness view's (w*, a pass of their own after the chunk) Omega depth, confidence and
    pose relative to the keyframe before it (its anchor, output/maplong_run/context_views, vendor
    loop_utils/context_views.context_records). Each one
    takes its anchor's chain: the depth into the record's units by the ratio of the two medians in
    that pass (Omega's chunk units → the record's), the anchor's BEND (its coefficients at the
    view's pixels), the anchor's confidence floor (its chunk's), and the anchor's F5 pose composed
    with the relative pose (its translation in the record's units times the bend's scale c0). It
    votes on the keyframes whose neighbour window holds its anchor — more views agreeing on a turn
    keep and repair more of its pixels; the points stay the keyframes' (their birth pixels, what
    the masks and the corrections read). A view of another fork run, an anchor that is not a
    keyframe of this run, another record grid: ignored, counted. Returns (dep, valid, passed, c2w,
    {keyframe: [witnesses]}, report)."""
    import json as _json
    from precision import corrected_cloud as CC
    from precision.epoch0_cloud import SKY_CONF
    d = Path(out) / "maplong_run" / "context_views"
    rep: Dict[str, Any] = {"dir": "maplong_run/context_views", "used": 0, "ignored": {}}
    empty = ({}, {}, {}, {}, {}, rep)
    if not d.is_dir():
        rep["why_none"] = "no context views (turn frames off, or a run before them)"
        return empty
    fsp = Path(out) / "maplong_run" / "fork_stamp.json"
    stamp = _json.loads(fsp.read_text()).get("sha256") if fsp.exists() else None
    idx = {f: i for i, f in enumerate(frames)}

    def _ign(why):
        rep["ignored"][why] = rep["ignored"].get(why, 0) + 1
    chosen: Dict[int, Tuple[Path, int]] = {}
    # c<chunk>_<frame>: the turn frames, inferred WITH their chunk; w<chunk>_<frame>: the interval
    # frames, inferred in a pass of their own after it (USER 2026-11-04: they only affirm or deny
    # here, the chunks never see them) — the same record, the same carry
    for p in sorted(list(d.glob("c*_*.npz")) + list(d.glob("w*_*.npz"))):
        with np.load(p) as z:
            ft, a, ck, st = int(z["frame"]), int(z["anchor_frame"]), int(z["chunk"]), str(z["fork_stamp"])
        if stamp is None or st != stamp:
            _ign("another fork run's"); continue
        if a not in idx:
            _ign("anchor not a keyframe of this run"); continue
        if ft in idx:
            _ign("a keyframe"); continue
        prev = chosen.get(ft)
        # the anchor's own chunk first; else a chunk with a confidence floor (on CUBES a view lives in every cube
        # holding its two keyframes, and a cube may own no record — USER 2026-11-06), the earliest file among them
        if (prev is None or (ck == chunk.get(a) and prev[1] != chunk.get(a))
                or (prev[1] != chunk.get(a) and prev[1] not in thr and ck in thr)):
            chosen[ft] = (p, ck)
    dep, valid, passed, cw, extra = {}, {}, {}, {}, {}
    lo, hi = min(int(x) for x in neighbors), max(int(x) for x in neighbors)
    for ft, (p, ck) in sorted(chosen.items()):
        with np.load(p) as z:
            raw = np.asarray(z["depth"], np.float32); cf = np.asarray(z["conf"], np.float32)
            rel = np.asarray(z["rel_pose"], np.float64); a = int(z["anchor_frame"])
            am = float(z["anchor_depth_median"])
        if tuple(raw.shape) != tuple(rec_hw):
            _ign("another record grid"); continue
        if ck not in thr:
            _ign("a chunk without a confidence floor"); continue
        if not (am > 0 and zmed.get(a, 0) > 0):
            _ign("no anchor depth to carry the scale"); continue
        ia = idx[a]
        factor = float(zmed[a]) / am
        dt = raw * factor
        if carry:
            dt = CC.record_on_native(dt, inp.cam, inp.maps)
            cf = CC.record_on_native(cf, inp.cam, inp.maps)
        v = np.isfinite(dt) & (dt > 0) & np.isfinite(cf) & (cf > SKY_CONF)
        bent = (dt.astype(np.float64) * float(depth_factor)).astype(np.float32)
        dep[ft] = np.where(v, bent, 0).astype(np.float32)
        valid[ft] = v
        passed[ft] = v & (cf >= thr[ck])
        T = rel.copy()
        T[:3, 3] *= factor * float(depth_factor)
        cw[ft] = np.asarray(c2w[a], np.float64) @ T
        for i in range(max(0, ia + lo), min(len(frames), ia + hi + 1)):
            extra.setdefault(frames[i], []).append(ft)
        rep["used"] += 1
    rep["keyframes_with_witnesses"] = len(extra)
    log(f"{LOG_TAG} turn witnesses: {rep['used']} turn frame(s) vote on {len(extra)} keyframe(s)"
        + (f"; ignored {rep['ignored']}" if rep["ignored"] else ""))
    return dep, valid, passed, cw, extra, rep


def covisible_witnesses(session_dir: Path, frames: List[int], neighbors, c2w: Dict[int, np.ndarray],
                        log: Callable = print) -> Tuple[Dict[int, List[int]], Dict[str, Any]]:
    """The keyframes of OTHER MOMENTS of the walk that see what a keyframe sees, as witnesses of its vote (USER
    2026-11-07, on observatorio's duplicated wooden wall and signs: "me gusta tu enfoque de corrección … aplícalo a
    conciencia y que se corrija en serio"). The vote heard the walk around a keyframe only (``neighbors``, its turn
    frames): two moments that saw one wall from two cubes that disagree each kept their copy, and nothing contradicted
    either. What each keyframe sees of every other in AMB3R's map (the cube plan's ``kf_covisibility``,
    reconstruction/cubes.py 7 — which keyframe sees what, which share a zone): for keyframe a, the keyframes b that see
    ANY PART of what a saw (c(a→b) > 0, and ≥ the plan's τ when it keeps one), AT ANY DISTANCE ALONG THE WALK but its own
    offsets (which vote already) — EVERY ONE of them (USER 2026-11-08: "que voten todos … siempre"; "ven lo mismo es
    ver total o parcialmente el sector"), listed the widest baseline first (the farthest camera from a's, ``c2w``,
    F5's poses; ties by walk order). Each votes on the pixels it sees, by the vote's own rules. USER 2026-11-07 (pccr's racks): walking sideways or turning in front of a target, the keyframes at
    ±3/6/12 look elsewhere while ±1, ±2, ±4 … see it — the fixed offsets plus six keyframes beyond ±12 gave the racks'
    pixels a median of one confirming view; chosen by what they see, a well-seen keyframe's pixels with ≥ 10
    confirmations went 25 → 44 % (#100) and 41 → 75 % (#150). USER 2026-11-08 (pccr's door at the end of a corridor,
    gone): the most co-visible are the NEAREST keyframes of the walk — 20 cm from a's camera at 686 keyframes, 37 cm at
    295 — and 7 m away they test no depth (under ``refine.min_tri_deg``): 7 of the 93 keyframes that see that door could
    gather the support floor's 10 views; the widest among the same co-visible keyframes give 82 of 93 (a table at 2 m:
    64 of 64 either way).
    They vote after the neighbours and the turn frames by the same rules (agree within τ, free space either way); the
    vote's τ stays the walk neighbours' own disagreement. Returns ({keyframe: [keyframes]}, report)."""
    rep: Dict[str, Any] = {"sources": {}, "used": 0}
    p = Path(session_dir) / "intake" / "cubes.json"
    plan_cv = json.loads(p.read_text()).get("kf_covisibility") if p.exists() else None
    offsets = {int(x) for x in neighbors}
    idx = {int(f): i for i, f in enumerate(frames)}
    best: Dict[Tuple[int, int], float] = {}            # (a, b) → c(a→b)
    for name, cv in (("structure probe (intake/cubes.json)", plan_cv),):
        if not cv:
            rep["sources"][name] = {"pairs": 0, "why_none": "no cube plan" if not p.exists() else
                                    "the cube plan holds no keyframe co-visibility (a plan before version 5)"}
            continue
        tau = float(cv["tau"])
        n_in, unknown = 0, 0
        for a, b, cab, cba in cv["pairs"]:
            a, b = int(a), int(b)
            if a not in idx or b not in idx:
                unknown += 1
                continue
            if a == b or (idx[b] - idx[a]) in offsets:      # itself, or one of its offsets (they vote already)
                continue
            for x, y, c in ((a, b, float(cab)), (b, a, float(cba))):
                if c > 0 and c >= tau and c > best.get((x, y), -1.0):
                    best[(x, y)] = c
                    n_in += 1
        rep["sources"][name] = {"pairs": n_in, "tau": tau, "band_rel": cv.get("band_rel"),
                                "not_keyframes_of_this_run": unknown}
    centre = {int(f): np.asarray(c2w[f], np.float64)[:3, 3] for f in frames}
    cand: Dict[int, List[Tuple[float, int]]] = {}
    for (a, b), c in best.items():
        cand.setdefault(a, []).append((-float(np.linalg.norm(centre[b] - centre[a])), idx[b]))
    extra: Dict[int, List[int]] = {}
    for f in frames:
        if f in cand:
            extra[f] = [frames[j] for _c, j in sorted(cand[f])]
    n = [len(v) for v in extra.values()]
    base = [float(np.linalg.norm(centre[b] - centre[a])) for a, ws in extra.items() for b in ws]
    rep.update({"any_distance_but_the_offsets": sorted(offsets), "per_keyframe_cap": None, "used": int(sum(n)),
                "per_keyframe": ({"median": float(np.median(n)), "max": int(max(n))} if n else None),
                "keyframes_with_witnesses": len(extra), "candidates": len(best), "order": "widest baseline first",
                "baseline_m": ({"median": float(np.median(base)), "min": float(np.min(base)), "max": float(np.max(base))}
                               if base else None)})
    log(f"{LOG_TAG} co-visible witnesses: {len(extra)} of {len(frames)} keyframes hear {sum(n)} keyframe(s) that see "
        f"any part of what they see — every one"
        + (f", a median of {np.median(n):.0f} per keyframe (max {max(n)})" if n else "")
        + (f", at a median of {np.median(base):.2f} m from the keyframe's camera" if base else "")
        + f" (any distance along the walk but their offsets, from "
        f"{len(best)} candidate(s); "
        + "; ".join(f"{k}: {v['pairs']}" + (f" ({v['why_none']})" if v.get("why_none") else "")
                    for k, v in rep["sources"].items()) + ")")
    return extra, rep


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


# ── the cloud's provenance column ────────────────────────────────────────

SRC_OMEGA_BENT = 1          # precision.provenance (schema v2) `source`: Omega's depth at the session's scale, voted


def source_column(data) -> np.ndarray:
    """Per cleaned point its provenance byte: every point of this product is Omega's own pixel."""
    return np.full(len(data), SRC_OMEGA_BENT, np.uint8)


# ── the step ─────────────────────────────────────────────────────────────

def landmark_rows_by_keyframe(g: Dict[int, list], X: Dict[int, np.ndarray], w2c: np.ndarray,
                              n_frames: int) -> Dict[int, np.ndarray]:
    """{keyframe index: (u, v, z) rows} of every observation of a triangulated track, z the landmark's
    depth in that keyframe, rows with z ≤ 0 left out — as arrays (2026-11-01: the per-observation
    Python loop over 16 M observations of 2.6 M tracks on pccr), in the loop's order: tracks in the
    groups' order, observations in their order (:func:`_landmark_rows_by_keyframe_reference`)."""
    from precision.refine import _track_arrays
    tracks, counts, _starts, ii, uv = _track_arrays(g, min_views=1)       # every track, single observations included
    out = {i: np.zeros((0, 3), float) for i in range(int(n_frames))}
    if not tracks:
        return out
    has = np.fromiter((t in X for t in tracks), bool, len(tracks))
    if not has.any():
        return out
    tid = np.repeat(np.arange(len(tracks)), counts)
    keep = has[tid]
    Xo = np.stack([X[t] for t, h in zip(tracks, has) if h])[np.cumsum(has)[tid[keep]] - 1]
    ii_k, uv_k = ii[keep], uv[keep]
    R2 = w2c[ii_k, 2, :3]
    z = R2[:, 0] * Xo[:, 0] + R2[:, 1] * Xo[:, 1] + R2[:, 2] * Xo[:, 2] + w2c[ii_k, 2, 3]
    pos = z > 0
    ii_k, uv_k, z = ii_k[pos], uv_k[pos], z[pos]
    order = np.argsort(ii_k, kind="stable")                   # per keyframe, the loop's order kept
    ii_s = ii_k[order]
    rows = np.column_stack([uv_k[order].astype(np.float64), z[order]])
    bounds = np.searchsorted(ii_s, np.arange(int(n_frames) + 1))
    for i in range(int(n_frames)):
        out[i] = rows[bounds[i]:bounds[i + 1]].reshape(-1, 3)
    return out


def _landmark_rows_by_keyframe_reference(g, X, w2c, n_frames):
    """The per-observation loop :func:`landmark_rows_by_keyframe` replaced (kept for its test)."""
    rows = {i: [] for i in range(int(n_frames))}
    for t, lst in g.items():
        if t in X:
            for i, uv in lst:
                z = w2c[i][2, :3] @ X[t] + w2c[i][2, 3]
                if z > 0:
                    rows[i].append((uv[0], uv[1], z))
    return {i: np.array(v, float).reshape(-1, 3) for i, v in rows.items()}


def _landmark_rows(session_dir: Path, pcfg, frames: List[int], w2c: np.ndarray, params,
                   log: Callable = print) -> Tuple[Dict[int, dict], dict]:
    """({split: {keyframe: (u, v, z) rows}}, {split: dropped}) — F5's tracks triangulated with
    THESE poses and camera; a track whose observations do not round-trip through the lens is
    dropped and counted (point 60), never a failure of the stage."""
    from precision.camera import undistort_solver
    from precision.tracks import load_tracks_v2
    from precision import refine as RF
    solver = undistort_solver(pcfg.camera)
    RF.use_device(pcfg.refine.triangulation_device)      # F6's landmark DLTs on the declared device
    tr = load_tracks_v2(session_dir, members=("track_query_id", "track_split", "obs_track", "obs_frame",
                                              "obs_uv_native"), threads=pcfg.tracks.npz_threads)
    split = RF.split_of_observations(tr["track_query_id"], tr["track_split"], tr["obs_track"])
    out, dropped = {}, {}
    for sp in (0, 1):
        m = split == sp
        g = RF.group_tracks(tr["obs_track"][m], tr["obs_frame"][m], tr["obs_uv_native"][m], frames, lists=False)
        dropped[sp] = {"tracks": 0, "observations": 0}
        X = RF.triangulate_tracks(g, w2c, params, solver, pcfg.refine.min_tri_deg, dropped=dropped[sp])
        out[sp] = landmark_rows_by_keyframe(g, X, w2c, len(frames))
    if any(d["tracks"] for d in dropped.values()):
        log(f"{LOG_TAG} landmark tracks dropped (no round trip through F5's lens — point 60): fit "
            f"{dropped[0]['tracks']}, held-out {dropped[1]['tracks']}")
    return out, {"fit": dropped[0], "held_out": dropped[1]}


def load_records(records_dir: Path, frames: Sequence[int], rec_hw: Tuple[int, int], carry: Optional[Callable] = None):
    """({frame: Omega's depth}, {frame: its confidence}, {frame: its cube}) of every keyframe's record, float32 on
    the record grid ``rec_hw`` — or carried onto the camera's grid by ``carry`` (NaN where the record does not
    cover). Each record is its frame's alone: inflated on threads (zlib releases the GIL — pccr 2026-08-24: 2765
    records read one after another with the card idle) and taken in the keyframes' order — the same arrays, and
    the first bad record in that order is the one that fails."""
    from concurrent.futures import ThreadPoolExecutor
    records_dir = Path(records_dir)

    def one(f):
        with np.load(records_dir / f"frame_{f}.npz") as z:
            d = np.asarray(z["depth"], np.float32)
            cf = np.asarray(z["conf"], np.float32)
            ch = int(z["chunk"]) if "chunk" in z.files else 0
        if tuple(d.shape) != tuple(rec_hw) or tuple(cf.shape) != tuple(rec_hw):
            raise DepthOnF5Error(f"Omega's record of frame {f} is {d.shape[1]}x{d.shape[0]} (conf "
                                 f"{cf.shape[1]}x{cf.shape[0]}), the first record {rec_hw[1]}x{rec_hw[0]} "
                                 f"— one record grid per session")
        if carry is not None:
            d = carry(d)
            cf = carry(cf)
        return d, cf, ch
    zo, conf, chunk = {}, {}, {}
    with ThreadPoolExecutor(max_workers=RECORD_THREADS) as ex:
        for f, (d, cf, ch) in zip(frames, ex.map(one, [int(f) for f in frames])):
            zo[f] = d; conf[f] = cf; chunk[f] = ch
    return zo, conf, chunk


def compute(session_dir: Path, pcfg, log: Callable = print,
            progress: Optional[Callable[[float, str], None]] = None, inp=None) -> types.SimpleNamespace:
    """Steps 1-4 (landmarks, bend, vote) — nothing written into the session. ``inp`` (a
    depth_sweep.SweepInputs) may be injected.
    Returns everything ``publish_cloud`` needs, plus the held-out landmark rows for the judges."""
    from precision.config import raw_of
    from precision import depth_sweep as DS
    from precision.epoch0_cloud import SKY_CONF
    raw_cfg = raw_of(pcfg)                   # the job's frozen configuration (point 69), never the disk
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
    # so the vote (K), the colours and the cloud's pixel columns
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
    # THE RANGE (reconstruction.view_range_m, USER 2026-11-09: "esos puntos tampoco deben ser considerados en etapas
    # anteriores … ni de nada"): a landmark farther than it from the keyframe neither bends that keyframe's depth nor
    # judges the bend
    from precision.camera import pixel_distance
    n_far = 0
    for sp in obs:
        for i, o in list(obs[sp].items()):
            if len(o):
                near = pixel_distance(o[:, 0], o[:, 1], o[:, 2], K) <= float(pcfg.view_range_m)
                n_far += int((~near).sum())
                obs[sp][i] = o[near]
    log(f"{LOG_TAG} landmark rows: {n_far:,} beyond the {float(pcfg.view_range_m):g} m range left out of the bend")

    zo, conf, chunk = load_records(inp.records_dir, frames, rec_hw,
                                   (lambda a: CC.record_on_native(a, inp.cam, inp.maps)) if carry else None)
    floor_norm = float(raw_cfg["reconstruction"]["simple"]["conf_min_norm"])
    thr = {}
    for k in sorted(set(chunk.values())):
        v = np.concatenate([conf[f][np.isfinite(conf[f]) & (conf[f] > SKY_CONF)].ravel()
                            for f in frames if chunk[f] == k])
        thr[k], _cmax = confidence_floor(v, floor_norm)             # THE ONE floor (point 52: kept, pinned)

    # 2. the depth's factor: ONE for the session (USER 2026-11-20)
    _p(25, "the session's depth factor on F5's landmarks")
    if bc.enabled:
        ratios = []
        for i, f in enumerate(frames):
            o = obs[0][i]
            if len(o):
                zz = bilinear(zo[f], o[:, 0], o[:, 1])
                ok = zz > bc.min_depth_m
                ratios.append(o[ok, 2] / zz[ok])
        factor, factor_info = session_factor(ratios, gc.huber_k, bc.irls_iterations)
        said = (f"depth factor {factor:.5f} (± {factor_info['sigma']:.5f}) for every keyframe, on "
                f"{factor_info['rows']:,} landmark rows of {factor_info['keyframes_with_rows']} keyframes")
    else:
        factor = 1.0
        factor_info = {"factor": 1.0, "disabled": True,
                       "reason": "reconstruction.precision.bend.enabled: false — Omega's depth stands as it is on "
                                 "F5's poses"}
        said = "bend.enabled: false — Omega's depth stands as it is on F5's poses"
    # the landmarks the factor never saw (a report; in metres, the session's repeatability when nothing else measured it)
    heldout = heldout_error([heldout_rows(obs[1][i], zo[f], bc.min_depth_m) for i, f in enumerate(frames)], factor)
    _p(40, said + (f"; held-out |dz|/z {heldout['median_rel'] * 100:.2f} % = {heldout['median_m'] * 100:.2f} cm along "
                   f"the ray at {heldout['range_m']:.2f} m over {heldout['rows']:,} rows"
                   if heldout["rows"] else "; no held-out landmark row on Omega's depth"))
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    dep, valid, passed = {}, {}, {}
    zmed: Dict[int, float] = {}
    for i, f in enumerate(frames):
        valid[f] = np.isfinite(zo[f]) & (zo[f] > 0) & np.isfinite(conf[f]) & (conf[f] > SKY_CONF)
        if valid[f].any():
            zmed[f] = float(np.median(zo[f][valid[f]]))   # the turn witnesses' scale (record units)
        passed[f] = valid[f] & (conf[f] >= thr[chunk[f]])
        # the depth at the session's factor, in float32 first, then the mask
        bent = (zo[f].astype(np.float64) * factor).astype(np.float32)
        dep[f] = np.where(valid[f], bent, 0).astype(np.float32)
    del zo, conf

    # 4. the edge-keeping vote (pccr epoch 8), with the turn frames as witnesses (USER 2026-11-04)
    t_dep, t_valid, t_passed, t_c2w, t_extra, t_rep = load_turn_witnesses(
        out, frames, chunk, thr, zmed, factor, c2w, inp, carry, rec_hw, H, W, bc.neighbors, log)
    dep.update(t_dep); valid.update(t_valid); passed.update(t_passed)
    c2w_vote = dict(c2w); c2w_vote.update(t_c2w)
    # the keyframes of other moments of the walk that see the same surface (USER 2026-11-07)
    cv_extra, cv_rep = covisible_witnesses(session_dir, frames, bc.neighbors, c2w, log)
    for f, ws in cv_extra.items():
        t_extra.setdefault(f, []).extend(ws)
    _p(50, "edge-keeping multi-view vote")
    labels = mask_labels(out, H, W, log)
    if labels is not None and lens:
        labels = labels_on_undistorted(labels, inp.maps)       # the masks live on the original frame
    vote_bars: dict = {}
    fusion = None
    if str(pcfg.cloud.fusion_rule) in ("colmap", "adjust"):
        # the depth tolerance: the session's OWN neighbour disagreement τ (measured in the vote) when the key is
        # 0 — COLMAP's 1 % is for its PatchMatch maps, already geometrically consistent; Omega's neighbours
        # disagree by τ (pccr 1.7 %), so 1 % refused honest pairs (39.5 % contradicted, coverage 37.9 %)
        from precision.camera import angle_px
        # USER 2026-11-09: the reprojection bound is an ANGLE, in px on this grid by F5's focal
        fusion = {"max_reproj_px": angle_px(float(K[0, 0]), float(pcfg.cloud.fusion_max_reproj_deg)),
                  "max_reproj_deg": float(pcfg.cloud.fusion_max_reproj_deg),
                  "max_depth_rel": float(pcfg.cloud.fusion_max_depth_rel),
                  "tri_min_deg": float(pcfg.cloud.fusion_tri_min_deg),
                  "tri_max_deg": float(pcfg.cloud.fusion_tri_max_deg),
                  # ADJUST (USER 2026-11-08): the vote finds the position; witnesses beyond this range have none
                  "mode": "adjust" if str(pcfg.cloud.fusion_rule) == "adjust" else "keep",
                  "witness_max_m": float(pcfg.view_range_m),
                  # the two selectable F6 additions (USER 2026-11-08 evening, OFF = the validated recipe)
                  "repair": bool(pcfg.cloud.fusion_repair),
                  "occlusion_veto": bool(pcfg.cloud.fusion_occlusion_veto),
                  "majority": bool(pcfg.cloud.fusion_majority),
                  "in_place": bool(pcfg.cloud.fusion_in_place)}
    no_birth = rotation_keyframes(session_dir, frames, log) if bool(pcfg.cloud.no_birth_without_parallax) else set()
    voted, tau, vst = edge_keeping_vote(frames, dep, valid, passed, K, c2w_vote, labels,
                                        bc.neighbors, bc.tau_quantile, int(pcfg.cloud.repair_min_views),
                                        bool(pcfg.cloud.rescue_below_floor), log,
                                        bars_out=vote_bars, extra=t_extra,
                                        min_parallax_deg=float(pcfg.refine.min_tri_deg),
                                        min_support=int(pcfg.cloud.min_support_views),
                                        fusion=fusion, no_birth=no_birth,
                                        vote_batch=int(getattr(pcfg.cloud, "vote_batch", 0)))
    cover = vst["out"] / float(N * H * W)
    _p(65, f"vote done: coverage {cover * 100:.1f} %")
    del dep, valid, passed
    return types.SimpleNamespace(session_dir=session_dir, out=out, inp=inp, frames=frames, N=N, W=W, H=H, K=K,
                                 c2w=c2w, params=params, chunk=chunk, voted=voted, tau=tau, vst=vst, cover=cover,
                                 factor=factor, factor_info=factor_info, heldout=heldout,
                                 vote_bars=vote_bars, turn_witnesses=t_rep, covisible_witnesses=cv_rep,
                                 obs=obs, dropped_tracks=dropped_tracks,
                                 floor={str(k): {"threshold": thr[k]} for k in sorted(thr)}, floor_norm=floor_norm,
                                 uu=uu, vv=vv, t0=t0, _p=_p,
                                 seconds_compute=round(time.time() - t0, 1),
                                 rec_hw=rec_hw, carry=carry, lens=lens, grid_note=grid_note)


def publish_cloud(C: types.SimpleNamespace, pcfg, log: Callable = print) -> dict:
    """Step 5: the voted maps → chunks → the cloud stage's cleaner → the new-cloud epoch (transaction)."""
    from precision.config import raw_of
    from precision import corrected_cloud as CC
    from precision.epoch0_cloud import _write_ply_xyzrgb
    from correction.session import read_ply
    raw_cfg = raw_of(pcfg)                   # the job's frozen configuration (point 69), never the disk
    session_dir, out, inp, frames, W, H, K, c2w = (C.session_dir, C.out, C.inp, C.frames, C.W, C.H, C.K, C.c2w)
    params, chunk, voted, tau, vst, cover = C.params, C.chunk, C.voted, C.tau, C.vst, C.cover
    uu, vv, t0, _p = C.uu, C.vv, C.t0, C._p
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
                "source": source_column(data)}
        report = {"version": 1, "stage": "depth_on_f5", "provenance": "tool_measured",
                  "source_of_depth": "Omega's depth at the session's one factor (fitted on F5's landmarks) + "
                                     "edge-keeping multi-view vote",
                  "camera": params, "grid": [W, H],
                  # the grid the bend worked on: the undistorted native frame; Omega's record (its own
                  # grid, the original frame) carried onto it through the lens + grid when they differ
                  "grid_of_the_bend": {"undistorted_native": [W, H], "record": [C.rec_hw[1], C.rec_hw[0]],
                                       "lens": bool(C.lens), "carried": bool(C.carry), "note": C.grid_note},
                  # ONE depth factor for the session (USER 2026-11-20) and what it leaves against the landmarks it
                  # never saw. ``held_out_half_B_final`` / ``_m`` keep the names reconstruction.certify.repeatability
                  # reads (its fourth source); since that date they are over EVERY held-out row.
                  "bend": {"scope": "session", **C.factor_info,
                           "held_out_half_B_final": C.heldout["median_rel"],
                           "held_out_half_B_final_m": C.heldout["median_m"],
                           "held_out_half_B_range_m": C.heldout["range_m"],
                           "held_out_half_B_rows": C.heldout["rows"],
                           "landmark_tracks_dropped": C.dropped_tracks},
                  "confidence_floor": {"conf_min_norm": C.floor_norm, "per_chunk": C.floor,
                                       "rule": "min + conf_min_norm x (max - min) of the chunk's valid "
                                               "confidences (THE ONE floor, shared with the viewer; point 52)"},
                  "turn_witnesses": getattr(C, "turn_witnesses", None),
                  "covisible_witnesses": getattr(C, "covisible_witnesses", None),
                  "vote": {"tau": tau, "coverage": cover, "bars": C.vote_bars,
                           **{k: (float(v) / max(vst["valid"], 1)
                                  if k not in ("tau", "valid", "min_support", "min_parallax_deg") else v)
                              for k, v in vst.items() if k != "tau"}},
                  "raw_points": n_raw}
        from precision import f6_torch as FT
        report["maps_numerics"] = FT.record()          # the device and torch numerics of every map here (USER 2026-10-08)
        camera_travels(tmp, params, len(frames), log)
        _p(85, "publishing the epoch (octree, atomic swap)")
        # what the publish removed OUTSIDE the new product's lineage (point 136: the previous
        # certification's epochs, another reconstruction's leftovers) depends on the session's
        # history, not on the inputs: CC.publish writes it into ``history`` and it goes to the
        # timing sidecar below, never into the compared report (the same as corrected_cloud's
        # own step; wave-2 cross item 136)
        history: dict = {}
        rep = CC.publish(session_dir, tmp, report, log, columns=cols, history=history)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    (out / "precision").mkdir(exist_ok=True)
    (out / "precision" / REPORT).write_text(json.dumps(rep, indent=1, default=float))
    # the clocks and the session's history live next to the report, never in it (points 36 / 56 / 136)
    times = {"seconds_compute": C.seconds_compute, "seconds_publish": round(time.time() - t_publish, 1),
             "seconds": round(time.time() - t0, 1), "history": history}
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
    from precision import f6_torch as FT
    pcfg = load_precision_config(session_dir=Path(args.session))   # the job's frozen config (point 69)
    # USER 2026-10-08: the maps run on the card, strict deterministic torch for the whole process; no card, no
    # stage. Nothing here is drawn at random: the seed (the bend's own) only fixes torch's generator.
    rec = FT.require_cuda(int(pcfg.bend.seed))
    print(f"{LOG_TAG} maps on {rec['device']} ({rec['card']}), torch {rec['torch']}, deterministic strict")
    run_depth_on_f5(Path(args.session), pcfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
