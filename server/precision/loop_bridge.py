"""The loop bridge — wide-baseline correspondences across the two visits of every loop (audit
2026-10-23, phase 3d; USER: "si existen mejores … para los cierres de loop hay que implementarlo").

WHY. A loop window (F4) tracks the queries of keyframe i through the keyframes around i AND around
j with the VGGSfM tracker — a tracker built for consecutive frames. Across the revisit (another
heading, another distance, another light) it loses most of them, and a track seen on ONE side only
constrains nothing between the visits: the loop closes only through the few tracks that survive the
jump. The drift between the visits is exactly what F5 cannot see then (precision.loop_gap measures
it as the gap that stays).

WHAT. A dense wide-baseline matcher (RoMa, MASt3R — ``tracks.loop_bridge``) matches keyframe i to
keyframe j directly; every match (p_i ↔ p_j) verified by the session camera's own geometry (an
essential matrix by RANSAC with F5's K — the pair must be a rigid revisit, never a lookalike) is
added as a QUERY on each side: p_i on i, p_j on j. The VGGSfM tracker then tracks p_i through i's
side and p_j through j's side, each on its own familiar frames, and the two tracks are MERGED into
one — observations on both visits, triangulated from either, judged by F5 like any other, held out
by the same hash of its query. The matcher only bridges; every observation F5 sees is the tracker's
own sub-pixel one, or the match itself at the two query pixels (σ = the matcher's declared one).

Backends (``MatchFn(img_i, img_j, frame_i, frame_j) -> (uv_i, uv_j, conf)`` on native RGB frames): ``roma`` (MIT, the
DINOv2 backbone Apache-2.0) and ``mast3r`` (naver — its checkpoint is CC BY-NC-SA 4.0: NOT for a
commercial run unless licensed; the config says so and the default is none). A missing backend
fails by name; nothing falls back.

Reported per loop pair in tracks.json: matches, inliers, the bridged tracks, and the verification's
verdict; a pair with fewer inliers than ``loop_bridge_min_inliers`` is not bridged and says why.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

MatchFn = Callable[[np.ndarray, np.ndarray, int, int], Tuple[np.ndarray, np.ndarray, np.ndarray]]
BACKENDS = ("none", "roma", "mast3r")
LOG_TAG = "[bridge]"


class LoopBridgeError(RuntimeError):
    pass


@dataclass(frozen=True)
class PairMatches:
    i: int                      # keyframe INDEX of the first visit
    j: int                      # keyframe index of the second
    uv_i: np.ndarray            # (N, 2) native pixels on keyframe i (verified inliers)
    uv_j: np.ndarray            # (N, 2) on keyframe j
    conf: np.ndarray            # (N,) the matcher's confidence
    n_raw: int                  # matches before the verification
    inlier_ratio: float
    reason: str                 # "" when bridged, else why not


def verify_pair(uv_i: np.ndarray, uv_j: np.ndarray, conf: np.ndarray, K: np.ndarray, dist: np.ndarray,
                ransac_px: float, min_inliers: int, max_matches: int, ransac_prob: float,
                seed: int, min_inlier_ratio: float = 0.0) -> Tuple[np.ndarray, str]:
    """The session camera's own geometry as the judge of a pair: the matches undistorted through
    the lens, an essential matrix by RANSAC (OpenCV, seeded through its own RNG), the inliers kept
    (the ``max_matches`` most confident). Returns (inlier mask, reason — "" when the pair stands)."""
    import cv2
    n = int(len(uv_i))
    if n < int(min_inliers):
        return np.zeros(n, bool), f"{n} matches, fewer than loop_bridge_min_inliers {int(min_inliers)}"
    order = np.argsort(-np.asarray(conf, np.float64), kind="stable")[:int(max_matches)]
    a = np.asarray(uv_i, np.float64)[order].reshape(-1, 1, 2)
    b = np.asarray(uv_j, np.float64)[order].reshape(-1, 1, 2)
    Kf = np.asarray(K, np.float64)
    d = np.asarray(dist, np.float64).reshape(-1)
    an = cv2.undistortPoints(a, Kf, d).reshape(-1, 2)
    bn = cv2.undistortPoints(b, Kf, d).reshape(-1, 2)
    cv2.setRNGSeed(int(seed))
    f = float(0.5 * (Kf[0, 0] + Kf[1, 1]))
    E, mask = cv2.findEssentialMat(an, bn, np.eye(3), method=cv2.RANSAC, prob=float(ransac_prob),
                                   threshold=float(ransac_px) / f)
    keep = np.zeros(n, bool)
    if E is None or mask is None:
        return keep, "no essential matrix — the pair is not a rigid revisit"
    inl = mask.reshape(-1).astype(bool)
    keep[order[inl]] = True
    k = int(keep.sum())
    ratio = k / max(len(order), 1)
    if k < int(min_inliers):
        return np.zeros(n, bool), f"{k} inliers of {len(order)} (ratio {ratio:.2f}), fewer than {int(min_inliers)}"
    if ratio < float(min_inlier_ratio):
        return np.zeros(n, bool), (f"{k} inliers of {len(order)}: ratio {ratio:.2f} under loop_bridge_min_inlier_ratio "
                                   f"{float(min_inlier_ratio):g} — the matches do not share one rigid geometry")
    return keep, ""


def match_pairs(pairs: Sequence[Tuple[int, int]], kf: Sequence[int], image_native: Callable[[int], np.ndarray],
                match_fn: MatchFn, K: np.ndarray, dist: np.ndarray, *, ransac_px: float, min_inliers: int,
                max_matches: int, ransac_prob: float, seed: int, min_inlier_ratio: float = 0.0,
                log: Callable = print) -> List[PairMatches]:
    """Every loop pair matched and verified (deduplicated: one entry per unordered pair)."""
    out: List[PairMatches] = []
    seen = set()
    for i, j in pairs:
        key = (min(i, j), max(i, j))
        if key in seen or i == j:
            continue
        seen.add(key)
        i, j = key
        uv_i, uv_j, conf = match_fn(image_native(kf[i]), image_native(kf[j]), int(kf[i]), int(kf[j]))
        uv_i = np.asarray(uv_i, np.float64).reshape(-1, 2); uv_j = np.asarray(uv_j, np.float64).reshape(-1, 2)
        conf = np.asarray(conf, np.float64).reshape(-1)
        if not (len(uv_i) == len(uv_j) == len(conf)):
            raise LoopBridgeError(f"the matcher returned {len(uv_i)} / {len(uv_j)} / {len(conf)} rows for pair {kf[i]}-{kf[j]}")
        keep, why = verify_pair(uv_i, uv_j, conf, K, dist, ransac_px, min_inliers, max_matches, ransac_prob, seed,
                                min_inlier_ratio)
        out.append(PairMatches(i, j, uv_i[keep], uv_j[keep], conf[keep], int(len(uv_i)),
                               float(keep.sum() / max(min(len(uv_i), int(max_matches)), 1)), why))
        log(f"{LOG_TAG} pair {kf[i]}-{kf[j]}: {len(uv_i)} matches, {int(keep.sum())} verified"
            + (f" — NOT bridged: {why}" if why else ""))
    return out


def merge_bridged(obs: Dict[str, list], tq: Dict[str, list], links: Sequence[Tuple[int, int]],
                  side_a: set, side_b: set) -> Dict[str, Any]:
    """The merge of the window's bridged tracks: for every link (ta, tb) — ta queried on keyframe i,
    tb on keyframe j — tb's observations on side B join ta (ta keeps its own on side A; where both
    saw a frame, the track queried on that frame's side decides), tb's query entry is dropped.
    ``obs`` / ``tq`` are the WINDOW's lists (mutated in place). Returns the counts."""
    tr = np.asarray(obs["track"], np.int64)
    fr = np.asarray(obs["frame"], np.int64)
    drop = np.zeros(len(tr), bool)
    new_tid = tr.copy()
    for ta, tb in links:
        ma = tr == ta; mb = tr == tb
        if not ma.any() or not mb.any():
            continue
        # frames of ta on side B are tb's business, frames of tb on side A are ta's
        fa_B = np.isin(fr, list(side_b)) & ma
        fb_A = np.isin(fr, list(side_a)) & mb
        drop |= fa_B | fb_A
        # a frame both still hold (outside either side — witnesses): keep ta's
        keep_b = mb & ~fb_A
        dup = np.isin(fr[keep_b], fr[ma & ~fa_B])
        idx_b = np.flatnonzero(keep_b)
        drop[idx_b[dup]] = True
        new_tid[mb] = ta
    n_links = 0
    for ta, tb in links:
        sa = set(fr[(new_tid == ta) & ~drop & np.isin(fr, list(side_a))].tolist())
        sb = set(fr[(new_tid == ta) & ~drop & np.isin(fr, list(side_b))].tolist())
        if len(sa) >= 1 and len(sb) >= 1:
            n_links += 1
    keep = ~drop
    for k in list(obs.keys()):
        vals = obs[k]
        obs[k] = [v for v, kp in zip(vals, keep) if kp]
    obs["track"] = new_tid[keep].tolist()
    drop_q = {tb for _ta, tb in links}
    keep_q = [k for k, t in enumerate(tq["id"]) if t not in drop_q]
    for k in list(tq.keys()):
        vals = tq[k]
        tq[k] = [vals[q] for q in keep_q]
    return {"links": int(len(links)), "bridged_tracks": int(n_links),
            "observations_dropped": int(drop.sum())}


# ── backends ─────────────────────────────────────────────────────────────────

def roma_match_fn(weights_url: str, weights_sha256: str, device: str = "cuda", log: Callable = print) -> MatchFn:
    """RoMa (Edstedt et al., MIT; DINOv2 backbone Apache-2.0), the outdoor/indoor checkpoint at a
    pinned revision verified by its sha256. Dense warp → sampled matches with their certainty."""
    import torch
    from precision.tracks import verified_checkpoint
    import importlib.util
    if importlib.util.find_spec("romatch") is None:
        raise LoopBridgeError("tracks.loop_bridge: roma needs the `romatch` package (vendor/RoMa, "
                              "pip install -e vendor/RoMa) — no fallback")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise LoopBridgeError("torch sees no CUDA device — the matcher runs on the card, never the CPU")
    from romatch import roma_outdoor as _roma
    ckpt = verified_checkpoint(weights_url, weights_sha256, log=log)
    model = _roma(device=torch.device(device), weights=torch.load(str(ckpt), map_location="cpu"))
    model.eval()

    def fn(img_i: np.ndarray, img_j: np.ndarray, _fi: int = -1, _fj: int = -1):
        from PIL import Image
        A = Image.fromarray(np.ascontiguousarray(img_i)); B = Image.fromarray(np.ascontiguousarray(img_j))
        with torch.no_grad():
            warp, certainty = model.match(A, B, device=device)
            m, c = model.sample(warp, certainty)
            kA, kB = model.to_pixel_coordinates(m, A.size[1], A.size[0], B.size[1], B.size[0])
        return kA.cpu().numpy(), kB.cpu().numpy(), c.cpu().numpy()
    return fn


def mast3r_match_fn(weights_url: str, weights_sha256: str, device: str = "cuda", log: Callable = print) -> MatchFn:
    """MASt3R (naver; the checkpoint is CC BY-NC-SA 4.0 — a licence decision the config records).
    Fast reciprocal NN matching on the two descriptor maps, confidence = the matching score."""
    import torch
    from precision.tracks import verified_checkpoint
    import importlib.util
    if importlib.util.find_spec("mast3r") is None or importlib.util.find_spec("dust3r") is None:
        raise LoopBridgeError("tracks.loop_bridge: mast3r needs vendor/mast3r (and its dust3r) on the path — "
                              "no fallback")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise LoopBridgeError("torch sees no CUDA device — the matcher runs on the card, never the CPU")
    from mast3r.model import AsymmetricMASt3R as _M
    from mast3r.fast_nn import fast_reciprocal_NNs
    from dust3r.inference import inference
    from dust3r.utils.image import load_images
    ckpt = verified_checkpoint(weights_url, weights_sha256, log=log)
    model = _M.from_pretrained(str(ckpt)).to(device).eval()

    def fn(img_i: np.ndarray, img_j: np.ndarray, _fi: int = -1, _fj: int = -1):
        import tempfile
        from PIL import Image
        with tempfile.TemporaryDirectory() as td:
            pa, pb = str(Path(td) / "a.png"), str(Path(td) / "b.png")
            Image.fromarray(np.ascontiguousarray(img_i)).save(pa); Image.fromarray(np.ascontiguousarray(img_j)).save(pb)
            ims = load_images([pa, pb], size=512, verbose=False)
        with torch.no_grad():
            out = inference([tuple(ims)], model, device, batch_size=1, verbose=False)
        v1, v2 = out["view1"], out["view2"]
        d1, d2 = out["pred1"]["desc"][0], out["pred2"]["desc1"][0]
        m1, m2 = fast_reciprocal_NNs(d1, d2, subsample_or_initxy1=8, device=device, dist="dot", block_size=2 ** 13)
        # back to native pixels: the loader's crop/resize is recorded per view
        H0a, W0a = img_i.shape[:2]; H0b, W0b = img_j.shape[:2]
        sa = np.array([W0a / v1["true_shape"][0][1].item(), H0a / v1["true_shape"][0][0].item()])
        sb = np.array([W0b / v2["true_shape"][0][1].item(), H0b / v2["true_shape"][0][0].item()])
        uv_i = (np.asarray(m1, np.float64) + 0.5) * sa - 0.5
        uv_j = (np.asarray(m2, np.float64) + 0.5) * sb - 0.5
        conf = np.ones(len(uv_i))
        return uv_i, uv_j, conf
    return fn


def make_match_fn(backend: str, tcfg, log: Callable = print) -> Optional[MatchFn]:
    if backend == "none":
        return None
    if backend == "roma":
        return roma_match_fn(tcfg.loop_bridge_weights_url, tcfg.loop_bridge_weights_sha256, log=log)
    if backend == "mast3r":
        return mast3r_match_fn(tcfg.loop_bridge_weights_url, tcfg.loop_bridge_weights_sha256, log=log)
    raise LoopBridgeError(f"tracks.loop_bridge {backend!r} is not one of {BACKENDS}")
