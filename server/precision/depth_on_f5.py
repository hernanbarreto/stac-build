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
    repaired 8.5 %, admitted 0.4 %, coverage 71.0 % — the hand-built epoch 8's own numbers;
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
from typing import Callable, Dict, List, Optional

import numpy as np

LOG_TAG = "[depth-on-f5]"
REPORT = "depth_on_f5.json"
TX_TMP = "_tx_depth_on_f5"
MAD_TO_SIGMA = 1.0 / 0.6744897501960817          # 1.4826: σ of a normal from its MAD


class DepthOnF5Error(RuntimeError):
    pass


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


def irls_huber(A: np.ndarray, r: np.ndarray, k: float, iterations: int) -> np.ndarray:
    """pccr epoch 7's robust fit, exactly (omega_bent_epoch7.py `irls`; USER 2026-10-01 "exactamente el
    mismo ajuste que la 7"): `iterations` Huber IRLS steps from the least-squares start, weights at k·σ,
    σ = 1.4826·MAD of the residual. Iterated to convergence instead, the re-estimated MAD keeps shrinking
    and the bend fits worse: 3.90 % held-out on pccr against 2.80 %."""
    w = np.ones(len(r))
    c = np.zeros(A.shape[1])
    for _ in range(iterations):
        c = np.linalg.lstsq(A * w[:, None], r * w, rcond=None)[0]
        e = r - A @ c
        s = MAD_TO_SIGMA * np.median(np.abs(e)) + 1e-12
        w = np.sqrt(np.minimum(1.0, k * s / np.maximum(np.abs(e), 1e-12)))
    return c


def bend_coefficients(rows: Dict[int, tuple], n: int, window: int, min_rows: int, huber_k: float,
                      iterations: int) -> Dict[int, np.ndarray]:
    """{i: (c0, c1, c2)} — each keyframe fitted on the rows of i−w … i+w; one with fewer than `min_rows`
    rows there keeps Omega's depth (k = 1), as epoch 7 did."""
    out: Dict[int, np.ndarray] = {}
    for i in range(n):
        js = range(max(0, i - window), min(n, i + window + 1))
        A = [rows[j][0] for j in js if j in rows and len(rows[j][1])]
        r = [rows[j][1] for j in js if j in rows and len(rows[j][1])]
        if A and sum(len(x) for x in r) >= min_rows:
            out[i] = irls_huber(np.vstack(A), np.concatenate(r), huber_k, iterations)
        else:
            out[i] = np.array([1.0, 0.0, 0.0])
    return out


def interior(passed: np.ndarray) -> np.ndarray:
    """Floor-passing pixels whose whole 3x3 window passed too — Omega's confidence collapses at
    contours, so tau is measured where it does not (pccr epoch 8)."""
    from scipy.ndimage import binary_erosion
    return binary_erosion(passed, structure=np.ones((3, 3), bool), border_value=0)


def mask_labels(output_dir: Path, H: int, W: int, log: Callable = print):
    """``position -> label map`` of the SAM3 masks (masklet id + 1, 0 = no mask, -1 = two masks
    overlap) on Omega's grid, or None when the session holds no masks on that grid (then no mixed
    pixel is snapped — declared)."""
    from precision.silhouette_filter import masks_by_keyframe
    p = Path(output_dir) / "seg_masks.npz"
    if not p.exists():
        log(f"{LOG_TAG} no seg_masks.npz — mixed pixels are not snapped (they go to the vote as they are)")
        return None
    masks = np.load(p)
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


def edge_keeping_vote(frames: List[int], dep: Dict[int, np.ndarray], valid: Dict[int, np.ndarray],
                      passed: Dict[int, np.ndarray], K: np.ndarray, c2w: Dict[int, np.ndarray], labels,
                      neighbors, tau_quantile: float, repair_min_views: int, log: Callable = print):
    """pccr EPOCH 8 (USER 2026-10-01: "es la mejor, incorporar al pipeline"), on the bent depth
    `dep` (every `valid` pixel; `passed` = above the ONE confidence floor):
     1. tau = the `tau_quantile` percentile of the neighbour disagreement on INTERIOR pixels;
     2. EDGE pixels = the 3x3 window spans a depth step (on the depth before snapping);
        MIXED pixels (on neither surface of the step) snapped to the side their SAM3 mask says;
     3. the TWO-SIDED vote over `neighbors`, judged by floor-passing neighbour pixels only:
        floor-passing pixels stay with contra ≤ agree (median of the agreeing views), a
        contradicted one is repaired to the median of the neighbours when ≥ `repair_min_views`
        agree, else it leaves; a below-floor EDGE pixel enters when agree ≥ 1 and contra ≤ agree.
    Returns ({frame: (depth map, agree count map)}, tau, totals)."""
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
    for i, f in enumerate(frames):
        cand = valid[f] & (passed[f] | edge[f])           # below-floor interior pixels stay out
        v = CC.two_sided_vote(i, frames, dep, passed, cand, K, c2w, w2c, neighbors, tau)
        rr, cc, agree, contra = v["rr"], v["cc"], v["agree"], v["contra"]
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
    p = tmp / "intrinsic.txt"
    p.write_text("".join(f"{params[0]:.10g} {params[1]:.10g} {params[2]:.10g} {params[3]:.10g}\n"
                         for _ in range(n_kf)))
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
    rep.params["footprint"] = runner.footprint()
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
    tiles = [dict(frame=int(f), **t) for f, per in rep.per_frame.items() for t in per["tiles"]]
    frames = {str(f): {k: v for k, v in per.items() if k != "tiles"} for f, per in rep.per_frame.items()}
    return {"enabled": True, "model": md.model, "steps": int(md.steps), "params": rep.params,
            "tiles": {"total": rep.tiles, "accepted": rep.accepted, "rejected_support": rep.rejected_support,
                      "rejected_residual": rep.rejected_residual, "residual_bar_rel": rep.residual_bar},
            "pixels": rep.totals, "seconds_pointdit": rep.seconds_pointdit, "seconds_total": rep.seconds_total,
            "per_frame": frames, "per_tile": tiles}


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

def _landmark_rows(session_dir: Path, pcfg, frames: List[int], w2c: np.ndarray, params) -> Dict[int, dict]:
    from precision.camera import undistort_solver
    from precision.tracks import load_tracks_v2
    from precision import refine as RF
    solver = undistort_solver(pcfg.camera)
    tr = load_tracks_v2(session_dir)
    split_of = dict(zip(tr["track_query_id"].tolist(), tr["track_split"].tolist()))
    split = np.array([split_of[int(t)] for t in tr["obs_track"]], np.int8)
    out = {}
    for sp in (0, 1):
        m = split == sp
        g = RF.group_tracks(tr["obs_track"][m], tr["obs_frame"][m], tr["obs_uv_native"][m], frames)
        X = RF.triangulate_tracks(g, w2c, params, solver, pcfg.refine.min_tri_deg)
        rows = {i: [] for i in range(len(frames))}
        for t, lst in g.items():
            if t in X:
                for i, uv in lst:
                    z = w2c[i][2, :3] @ X[t] + w2c[i][2, 3]
                    if z > 0:
                        rows[i].append((uv[0], uv[1], z))
        out[sp] = {i: np.array(v, float).reshape(-1, 3) for i, v in rows.items()}
    return out


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

    inp = inp if inp is not None else DS.load_inputs(session_dir, pcfg)   # F5's camera + poses (guards F5's epoch)
    params = list(inp.cam.params)
    if any(abs(float(x)) > 0 for x in params[4:]):
        raise DepthOnF5Error(f"the session camera carries lens distortion {params[4:]} — the bend reads "
                             f"Omega's record pixels as the camera's pixels; undistort first")
    K = np.asarray(inp.K, np.float64)
    W, H = int(inp.wh[0]), int(inp.wh[1])
    frames = [int(f) for f in inp.kf]
    N = len(frames)
    w2c = np.asarray(inp.kf_w2c, np.float64)
    c2w = {f: np.linalg.inv(w2c[i]) for i, f in enumerate(frames)}

    _p(3, f"{N} keyframes, camera fx {K[0, 0]:.1f} fy {K[1, 1]:.1f} ({W}x{H}); F5's landmarks")
    obs = _landmark_rows(session_dir, pcfg, frames, w2c, params)

    zo, conf, chunk = {}, {}, {}
    for f in frames:
        with np.load(inp.records_dir / f"frame_{f}.npz") as z:
            d = np.asarray(z["depth"], np.float32)
            if d.shape != (H, W):
                raise DepthOnF5Error(f"Omega's record of frame {f} is {d.shape[1]}x{d.shape[0]}, the camera "
                                     f"grid {W}x{H} — the bend needs Omega's depth on the camera's grid")
            zo[f] = d; conf[f] = np.asarray(z["conf"], np.float32); chunk[f] = int(z["chunk"]) if "chunk" in z.files else 0
    floor_norm = float(raw_cfg["reconstruction"]["simple"]["conf_min_norm"])
    thr, cmax = {}, {}
    for k in sorted(set(chunk.values())):
        v = np.concatenate([conf[f][np.isfinite(conf[f]) & (conf[f] > SKY_CONF)].ravel()
                            for f in frames if chunk[f] == k])
        thr[k] = float(v.min() + floor_norm * (v.max() - v.min()))
        cmax[k] = float(v.max())

    # 2. the bend
    _p(25, "bending Omega's depth to F5's landmarks")
    rows = {}
    for i, f in enumerate(frames):
        o = obs[0][i]
        zz = bilinear(zo[f], o[:, 0], o[:, 1]) if len(o) else np.zeros(0)
        ok = zz > bc.min_depth_m
        rows[i] = (design(o[ok, 0], o[ok, 1], W, H), o[ok, 2] / zz[ok])
    # the window is chosen on half A of the held-out (its even rows), half B reports — epoch 7
    score, coefs = {}, {}
    for w in bc.windows:
        cw = bend_coefficients(rows, N, int(w), bc.min_rows, gc.huber_k, bc.irls_iterations)
        errs = []
        for i, f in enumerate(frames):
            h = obs[1][i]
            h = h[(np.arange(len(h)) % 2) == 0]
            if len(h):
                zz = bilinear(zo[f], h[:, 0], h[:, 1]); ok = zz > bc.min_depth_m
                errs.append(np.abs(zz[ok] * (design(h[ok, 0], h[ok, 1], W, H) @ cw[i]) - h[ok, 2]) / h[ok, 2])
        coefs[w] = cw; score[w] = float(np.median(np.concatenate(errs)))
    raw_err = []
    for i, f in enumerate(frames):
        h = obs[1][i]
        if len(h):
            zz = bilinear(zo[f], h[:, 0], h[:, 1]); ok = zz > 0
            raw_err.append(np.abs(zz[ok] - h[ok, 2]) / h[ok, 2])
    wb = min(score, key=score.get)
    c0 = np.array([coefs[wb][i][0] for i in range(N)])
    _p(40, f"held-out |dz|/z: unbent {np.median(np.concatenate(raw_err)) * 100:.2f} %, "
           + ", ".join(f"±{w} {score[w] * 100:.2f} %" for w in bc.windows) + f" → ±{wb}; scale "
           f"{np.median(c0):.4f} [{c0.min():.4f}, {c0.max():.4f}], largest consecutive jump {np.abs(np.diff(c0)).max():.3f}")
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    Dm = design(uu.ravel(), vv.ravel(), W, H)
    dep, valid, passed, weight = {}, {}, {}, {}
    md = pcfg.mono_detail
    for i, f in enumerate(frames):
        valid[f] = np.isfinite(zo[f]) & (zo[f] > 0) & np.isfinite(conf[f]) & (conf[f] > SKY_CONF)
        passed[f] = valid[f] & (conf[f] >= thr[chunk[f]])
        # the bent map in float32 first, then the mask — epoch 8's arithmetic, bit for bit
        bent = (zo[f] * (Dm @ coefs[wb][i]).reshape(H, W)).astype(np.float32)
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
    voted, tau, vst = edge_keeping_vote(frames, dep, valid, passed, K, c2w, mask_labels(out, H, W, log),
                                        bc.neighbors, bc.tau_quantile, int(pcfg.cloud.repair_min_views), log)
    cover = vst["out"] / float(N * H * W)
    _p(65, f"vote done: coverage {cover * 100:.1f} %")
    del dep, valid, passed
    return types.SimpleNamespace(session_dir=session_dir, out=out, inp=inp, frames=frames, N=N, W=W, H=H, K=K,
                                 c2w=c2w, params=params, chunk=chunk, voted=voted, tau=tau, vst=vst, cover=cover,
                                 wb=wb, score=score, coefs=coefs, c0=c0, obs=obs, src_maps=src_maps, md_rep=md_rep,
                                 md=md, uu=uu, vv=vv, t0=t0, _p=_p, seconds_compute=round(time.time() - t0, 1))


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
                  "bend": {"window": int(wb), "held_out": score,
                           "scale": {"median": float(np.median(c0)), "min": float(c0.min()),
                                     "max": float(c0.max())}},
                  # what each keyframe's depth was multiplied by — the chunk check (f6_check) measures
                  # the PUBLISHED cloud with it: s_k = c0, bend = (c1, c2) of precision.depth_on_f5.design
                  "per_frame": {str(f): {"s_k": float(coefs[wb][i][0]),
                                         "bend": [float(coefs[wb][i][1]), float(coefs[wb][i][2])]}
                                for i, f in enumerate(frames)},
                  "vote": {"tau": tau, "coverage": cover,
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
    rep["seconds"] = round(time.time() - t0, 1)
    (out / "precision").mkdir(exist_ok=True)
    (out / "precision" / REPORT).write_text(json.dumps(rep, indent=1, default=float))
    _p(100, f"depth on F5 published epoch {rep['epoch_to']} ({rep['n_points']:,} pts, {rep['seconds']} s)")
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
