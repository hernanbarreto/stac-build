"""Epoch 8 — Omega bent to F5 with its EDGES kept (USER 2026-10-01: *"lo más importante es que los objetos
deben tener mucha definición, corte en los filos, las aristas"*, and *"la que consideres mejor, olvidate de
las reglas que pusimos en su momento, estamos en otra etapa"*). Epoch 7's recipe (omega_bent_epoch7.py) with
the four fixes of the edge audit (#9, #6, #1, #14); no DA3. Per keyframe:
  1. landmarks: F5's FIT tracks triangulated with F5 (R1) poses + camera; HELD-OUT tracks only judge.
  2. bend: Omega's depth × k(u,v) = c0 + c1 u + c2 v, fitted on the landmarks of keyframes i-w..i+w with
     EPOCH 7'S estimator (Huber IRLS, 10 steps — irls7). The product's converged IRLS
     (precision.depth_on_f5.irls_huber, tol 1e-12) reads a WORSE held-out on the same pccr rows: 3.84 % vs
     2.80 % median |dz|/z (scratch probe 2026-10-01), so the estimator the user validated on epoch 7 stays.
     A keyframe with fewer than refine.min_witness_corr rows in its window BORROWS the nearest fitted
     keyframe's k (epoch 7 gave it identity [1,0,0] next to bent neighbours; on pccr no keyframe was under
     its 20 rows, so that fallback did NOT cause the 0.359 jump — the per-keyframe fit did).
     THE WINDOW (#9): epoch 7 took the lowest median held-out error (±0), which cannot see the few keyframes
     where c0 jumps (0.359 between two consecutive keyframes). Now every candidate window is scored on
     CONSECUTIVE-keyframe agreement: per consecutive pair, the median ratio of the two keyframes' bent depths
     over the pixels both see (F5 camera + poses, symmetric a->b / b->a so occlusions cancel). The jump a
     window introduces = that ratio over the UNBENT one; a window is accepted only when every jump stays
     within Omega's own consecutive-pair spread (p1-p99 of the unbent ratios, measured here). Among the
     accepted, the lowest held-out (half A) error wins; half B reports it independently. The configured
     windows (precision.bend.windows, powers of two) continue doubling up to the whole walk, where every
     keyframe shares one bend and no jump exists — so a candidate always passes.
  3. validity: Omega's ONE confidence floor (reconstruction.simple.conf_min_norm, min-max per Omega chunk),
     sky out. tau = the bend.tau_quantile percentile of the session's own neighbour disagreement, measured on
     INTERIOR pixels only (floor-passing with their whole 3x3 window floor-passing: the confidence collapses
     at contours, so the mixed pixels never set their own tolerance).
  4. edges (#1, USER decision): EDGE pixels = the pixels whose 3x3 window spans a depth step on the bent depth
     (corrected_cloud.depth_steps: extremes far enough apart to hold a mixed pixel — the ramp plus one pixel of
     each side). A band measured from the floor's pass rate per ring away from a step does not exist (pccr:
     9.7 % at the step, 50.8 % at 5 rings, 80 % at 25, 100 % only at 193 rings: it would make 97 % of the pixels
     "edge"), so the user's own 3x3 window decides. MIXED pixels (more than tau from both extremes of their
     3x3 window) are then snapped to the side their SAM3 mask says (output/seg_masks.npz, keyframe positions,
     832x464): that side's nearest valid depth.
  5. the vote, TWO-SIDED (#6) over bend.neighbors, judged ONLY by neighbour pixels that passed the floor: a
     contradiction is the pixel's point in front of a neighbour's surface by > tau OR the neighbour's surface
     (splatted into this view) in front of the pixel's depth by > tau. Floor-passing: contra <= agree stays
     (median of the agreeing views; 0/0 stays only here); contradicted -> repaired to the median of the
     neighbours' splats when >= cloud.repair_min_views agree, else it leaves. Below the floor: an EDGE pixel
     enters when agree >= 1 and contra <= agree; every other below-floor pixel stays out (interior keeps the
     floor).
  6. cloud (Omega chunks as units) with the vote's AGREE COUNT in the 'confidence' column (#14), voxel + SOR,
     octree, publish epoch 8, masks, certify -> renumbered 8.

--dry-run N: steps 1-4 on the whole session (the window and tau are session measurements), the vote
on N keyframes spread over the walk, epoch 7's recipe on the same keyframes as BEFORE; prints, writes NOTHING."""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

SERVER = Path("/workspace/stac-build/server"); sys.path.insert(0, str(SERVER))
S = SERVER / "projects/pccr/scans/2026-08-31/src_default"; O = S / "output"
TMP, DST, EPOCH = O / "_tx_omega_edges", O / "_epoch_8", 8
SPREAD_PCT = (1.0, 99.0)          # USER/audit 2026-10-01: Omega's own consecutive-pair spread is its p1-p99
                                  # (pccr, audit: 0.973-1.021)
EPOCH7_WINDOWS, EPOCH7_MIN_ROWS = (0, 1, 2, 4, 8), 20   # epoch 7's own choices, ONLY to rebuild it as the
                                                        # dry-run's BEFORE (omega_bent_epoch7.py:19, :110)


def log(m):
    print(f"[edges {time.strftime('%H:%M:%S')}] {m}", flush=True)


def live():
    return int(json.loads((O / "geometry_epoch.json").read_text())["epoch"])


def irls7(A, r):
    """Epoch 7's robust fit (omega_bent_epoch7.py:90-96): Huber weights at 1.345 x 1.4826 MAD, 10 steps."""
    w = np.ones(len(r))
    for _ in range(10):
        c = np.linalg.lstsq(A * w[:, None], r * w, rcond=None)[0]
        e = r - A @ c; sc = 1.4826 * np.median(np.abs(e)) + 1e-12
        w = np.sqrt(np.minimum(1.0, 1.345 * sc / np.maximum(np.abs(e), 1e-12)))
    return c


def bend_coefficients7(rows, N, w, min_rows):
    """{i: (c0, c1, c2)}: keyframe i fitted (irls7) on the rows of i-w..i+w; a keyframe with fewer than
    `min_rows` there BORROWS its nearest fitted keyframe's k (the lower one on a tie) — never identity next
    to bent neighbours; identity only when no keyframe at all is fitted."""
    out = {}
    for i in range(N):
        js = [j for j in range(max(0, i - w), min(N, i + w + 1)) if len(rows[j][1])]
        if js and sum(len(rows[j][1]) for j in js) >= min_rows:
            out[i] = irls7(np.vstack([rows[j][0] for j in js]), np.concatenate([rows[j][1] for j in js]))
    fitted = sorted(out)
    for i in range(N):
        if i not in out:
            out[i] = out[min(fitted, key=lambda j: abs(j - i))].copy() if fitted else np.array([1.0, 0.0, 0.0])
    return out


def mem_gb():
    try:                                                     # CLAUDE.md: the pod's real limit is the cgroup's
        return int(Path("/sys/fs/cgroup/memory/memory.usage_in_bytes").read_text()) / 1e9
    except OSError:
        return float("nan")


# ── 0. the session ──────────────────────────────────────────────────────────

def load_session():
    from config import cfg
    from precision.config import load_precision_config
    from precision import refine as RF
    from precision.epoch0_cloud import SKY_CONF
    pcfg = load_precision_config()
    cam = json.loads((O / "precision" / "camera_f5_r1.json").read_text())
    p = cam["params"]
    if any(abs(float(x)) > 0 for x in p[4:]):
        raise SystemExit(f"F5's camera carries lens distortion {p[4:]} — this recipe reads Omega's record "
                         f"pixels as the camera's pixels")
    K = RF.K_of(p)
    frames = [int(float(x)) for x in (O / "camera_frames.txt").read_text().split()]
    F5 = np.loadtxt(O / "precision" / "f5_camera_poses.txt").reshape(-1, 4, 4)
    if len(F5) != len(frames):
        raise SystemExit(f"{len(F5)} F5 poses for {len(frames)} keyframes")
    w2c_a = np.linalg.inv(F5)
    rec = O / "omega_run" / "results_output"
    zo, conf, chunk = {}, {}, {}
    for f in frames:
        with np.load(rec / f"frame_{f}.npz") as z:
            zo[f] = np.asarray(z["depth"], np.float32); conf[f] = np.asarray(z["conf"], np.float32)
            chunk[f] = int(z["chunk"])
    H, W = zo[frames[0]].shape
    if (W, H) != (int(cam["width"]), int(cam["height"])):
        raise SystemExit(f"Omega's record grid {W}x{H} is not F5's camera grid {cam['width']}x{cam['height']}")
    floor_norm = float(cfg["reconstruction"]["simple"]["conf_min_norm"])
    thr = {}
    for k in sorted(set(chunk.values())):
        v = np.concatenate([conf[f][np.isfinite(conf[f]) & (conf[f] > SKY_CONF)].ravel() for f in frames
                            if chunk[f] == k])
        thr[k] = v.min() + floor_norm * (v.max() - v.min())
    valid, passed = {}, {}
    for f in frames:
        valid[f] = np.isfinite(zo[f]) & (zo[f] > 0) & np.isfinite(conf[f]) & (conf[f] > SKY_CONF)
        passed[f] = valid[f] & (conf[f] >= thr[chunk[f]])
    del conf
    return dict(cfg=cfg, pcfg=pcfg, params=p, K=K, frames=frames, N=len(frames), H=H, W=W, w2c_a=w2c_a,
                c2w={f: F5[i] for i, f in enumerate(frames)}, w2c={f: w2c_a[i] for i, f in enumerate(frames)},
                zo=zo, chunk=chunk, valid=valid, passed=passed, floor_norm=floor_norm, thr=thr)


# ── 1. landmarks ────────────────────────────────────────────────────────────

def landmarks(s, cache=None):
    """F5's FIT (0) and HELD-OUT (1) landmark rows per keyframe; `cache` (a path OUTSIDE the session, for
    repeated dry runs: the triangulation is ~7 min on 2 threads) is read when present, written when not."""
    if cache is not None:
        cache = Path(cache).resolve()
        if S.resolve() in cache.parents:
            raise SystemExit(f"--landmarks-cache {cache} lies inside the session — the dry run writes nothing there")
        if cache.exists():
            with np.load(cache) as z:
                obs = {sp: {i: z[f"s{sp}_{i}"] for i in range(s["N"])} for sp in (0, 1)}
            log(f"landmark rows read from {cache}")
            return obs
    obs = _triangulate(s)
    if cache is not None:
        np.savez(cache, **{f"s{sp}_{i}": obs[sp][i] for sp in (0, 1) for i in range(s["N"])})
        log(f"landmark rows cached in {cache}")
    return obs


def _triangulate(s):
    from precision.camera import undistort_solver
    from precision.tracks import load_tracks_v2
    from precision import refine as RF
    pcfg, frames, N, w2c = s["pcfg"], s["frames"], s["N"], s["w2c_a"]
    solver = undistort_solver(pcfg.camera)
    tr = load_tracks_v2(S)
    split_of = dict(zip(tr["track_query_id"].tolist(), tr["track_split"].tolist()))
    split = np.array([split_of[int(t)] for t in tr["obs_track"]], np.int8)
    obs = {}
    for sp in (0, 1):
        m = split == sp
        g = RF.group_tracks(tr["obs_track"][m], tr["obs_frame"][m], tr["obs_uv_native"][m], frames)
        X = RF.triangulate_tracks(g, w2c, s["params"], solver, pcfg.refine.min_tri_deg)
        rows = {i: [] for i in range(N)}
        for t, lst in g.items():
            if t in X:
                for i, uv in lst:
                    z = w2c[i][2, :3] @ X[t] + w2c[i][2, 3]
                    if z > 0:
                        rows[i].append((uv[0], uv[1], z))
        obs[sp] = {i: np.array(v, float).reshape(-1, 3) for i, v in rows.items()}
        log(f"{'fit' if sp == 0 else 'held-out'}: {len(X):,} landmarks with F5 R1")
    return obs


# ── 2. the bend, window chosen by consecutive agreement ─────────────────────

def heldout_err(s, obs_ho, coefs, half):
    """Median |dz|/z of the held-out landmarks of `half` (0 = A, the even rows; 1 = B); coefs None = unbent."""
    from precision.depth_on_f5 import bilinear, design
    errs = []
    for i, f in enumerate(s["frames"]):
        h = obs_ho[i]
        h = h[(np.arange(len(h)) % 2) == half]
        if len(h):
            z_ = bilinear(s["zo"][f], h[:, 0], h[:, 1]); ok = z_ > 0
            k = design(h[ok, 0], h[ok, 1], s["W"], s["H"]) @ coefs[i] if coefs is not None else 1.0
            errs.append(np.abs(z_[ok] * k - h[ok, 2]) / h[ok, 2])
    return float(np.median(np.concatenate(errs)))


def bent(s, f, c, D):
    z = s["zo"][f]
    return z if c is None else (z * (D @ c).reshape(z.shape)).astype(np.float32)


def pair_ratios(s, coefs, D, stride):
    """Per consecutive pair (i, i+1): sqrt(r_ab / r_ba) of the bent depths over the floor-passing pixels both
    see — the occlusion bias (always > 1 in both directions) cancels, a scale jump c_a / c_b survives."""
    from precision import corrected_cloud as CC
    fr, N, K, c2w, w2c, ok = s["frames"], s["N"], s["K"], s["c2w"], s["w2c"], s["passed"]
    out = np.full(N - 1, np.nan)
    db = bent(s, fr[0], None if coefs is None else coefs[0], D)
    for i in range(N - 1):
        a, b = fr[i], fr[i + 1]
        da, db = db, bent(s, b, None if coefs is None else coefs[i + 1], D)
        r_ab = CC.consecutive_ratio(da, ok[a], db, ok[b], K, c2w[a], w2c[b], stride)
        r_ba = CC.consecutive_ratio(db, ok[b], da, ok[a], K, c2w[b], w2c[a], stride)
        out[i] = np.sqrt(r_ab / r_ba)
    return out


def candidate_windows(configured, N):
    """The configured windows (precision.bend.windows), then their powers of two continued to the whole walk
    (N - 1: one bend for every keyframe, no consecutive jump by construction). The continuation is only
    walked while no window has been accepted (choose_window), so a candidate always passes."""
    ws = sorted(set(int(w) for w in configured))
    n_conf = len(ws)
    w = ws[-1]
    while w < N - 1:
        w = min(2 * w if w > 0 else 1, N - 1)
        ws.append(w)
    return ws, n_conf


def choose_window(s, obs):
    from precision.depth_on_f5 import bilinear, design
    pcfg, frames, N, W, H = s["pcfg"], s["frames"], s["N"], s["W"], s["H"]
    rows = {}
    for i, f in enumerate(frames):
        o = obs[0][i]
        z_ = bilinear(s["zo"][f], o[:, 0], o[:, 1]) if len(o) else np.zeros(0)
        ok = z_ > 0
        rows[i] = (design(o[ok, 0], o[ok, 1], W, H), o[ok, 2] / z_[ok])
    n_rows = np.array([len(rows[i][1]) for i in range(N)])
    min_rows = int(pcfg.refine.min_witness_corr)          # declared: the product's bar (precision/depth_on_f5.py)
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    D = design(uu.ravel(), vv.ravel(), W, H)
    stride = int(pcfg.chunk_check.pixel_stride)           # declared sampling of the depth maps (BOUND: cost)
    t = time.time()
    m0 = pair_ratios(s, None, D, stride)
    fin = np.isfinite(m0)
    lo, hi = np.percentile(m0[fin], SPREAD_PCT)
    log(f"Omega's own consecutive agreement (unbent, {int(fin.sum())} of {N - 1} pairs, every {stride}th "
        f"pixel): median {np.median(m0[fin]):.4f}, p{SPREAD_PCT[0]:g}-p{SPREAD_PCT[1]:g} [{lo:.4f}, {hi:.4f}] "
        f"({time.time() - t:.0f} s)")
    held0 = heldout_err(s, obs[1], None, 0)
    table = {}
    cands, n_conf = candidate_windows(pcfg.bend.windows, N)
    for n_done, w in enumerate(cands):
        if n_done >= n_conf and any(T["accepted"] for T in table.values()):
            break                                       # past the configured windows only until one passes
        t = time.time()
        cw = bend_coefficients7(rows, N, w, min_rows)
        J = pair_ratios(s, cw, D, stride) / m0
        ok = np.isfinite(J)
        out = ok & ((J < lo) | (J > hi))
        c0 = np.array([cw[i][0] for i in range(N)])
        win_rows = np.array([n_rows[max(0, i - w):min(N, i + w + 1)].sum() for i in range(N)])
        dev = np.where(ok, np.abs(np.log(np.where(ok, J, 1.0))), -1.0)
        table[w] = dict(coefs=cw, held=heldout_err(s, obs[1], cw, 0), J=J, n_out=int(out.sum()),
                        worst=int(np.argmax(dev)), c0=c0, accepted=not out.any(),
                        borrowed=np.nonzero(win_rows < min_rows)[0].tolist())
        T = table[w]
        log(f"  ±{w}: held-out A {T['held'] * 100:.2f} %, jumps [{np.nanmin(J):.4f}, {np.nanmax(J):.4f}] "
            f"(worst pair {T['worst']}->{T['worst'] + 1}: {J[T['worst']]:.4f}; {T['n_out']} pair(s) outside), "
            f"largest |dc0| {np.abs(np.diff(c0)).max():.3f}, {len(T['borrowed'])} keyframe(s) borrow -> "
            f"{'ACCEPTED' if T['accepted'] else 'rejected'} ({time.time() - t:.0f} s)")
    acc = [w for w in table if table[w]["accepted"]]
    if not acc:
        raise SystemExit("no window keeps consecutive keyframes within Omega's own spread")
    wb = min(acc, key=lambda w: table[w]["held"])
    T = table[wb]
    log(f"bend window ±{wb} (lowest held-out among the accepted {acc}); held-out |dz|/z unbent "
        f"{held0 * 100:.2f} % -> {T['held'] * 100:.2f} % (half A), half B {heldout_err(s, obs[1], None, 1) * 100:.2f}"
        f" % -> {heldout_err(s, obs[1], T['coefs'], 1) * 100:.2f} %; scale median {np.median(T['c0']):.4f} "
        f"[{T['c0'].min():.4f}, {T['c0'].max():.4f}], largest consecutive jump {np.nanmax(np.abs(np.log(T['J']))):.4f}"
        f" (log) inside the spread [{lo:.4f}, {hi:.4f}]; keyframes borrowing: {T['borrowed'] or 'none'}")
    return wb, T["coefs"], D, dict(spread=(float(lo), float(hi)), table=table, n_rows=n_rows, min_rows=min_rows)


def epoch7_bend(s, obs):
    """Epoch 7's bend exactly as omega_bent_epoch7.py:79-125 wrote it — the dry-run's BEFORE."""
    frames, N, W, H, zo = s["frames"], s["N"], s["W"], s["H"], s["zo"]

    def bil(img, u, v):
        u = np.clip(u, 0, W - 1.001); v = np.clip(v, 0, H - 1.001)
        u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int); du, dv = u - u0, v - v0
        return (img[v0, u0] * (1 - du) * (1 - dv) + img[v0, u0 + 1] * du * (1 - dv) + img[v0 + 1, u0] * (1 - du) * dv
                + img[v0 + 1, u0 + 1] * du * dv)

    def des(u, v):
        return np.c_[np.ones(len(u)), (u - W / 2) / W, (v - H / 2) / H]

    fitrows = {}
    for i, f in enumerate(frames):
        o = obs[0][i]; z_ = bil(zo[f], o[:, 0], o[:, 1]) if len(o) else np.zeros(0); ok = z_ > 0.05
        fitrows[i] = (des(o[ok, 0], o[ok, 1]), o[ok, 2] / z_[ok])
    score, coefs, ident = {}, {}, {}
    for wnd in EPOCH7_WINDOWS:
        cw, errs, idn = {}, [], []
        for i, f in enumerate(frames):
            js = range(max(0, i - wnd), min(N, i + wnd + 1))
            A = np.vstack([fitrows[j][0] for j in js]); r = np.concatenate([fitrows[j][1] for j in js])
            if len(r) >= EPOCH7_MIN_ROWS:
                cw[i] = irls7(A, r)
            else:
                cw[i] = np.array([1.0, 0, 0]); idn.append(i)
            h = obs[1][i]; h = h[(np.arange(len(h)) % 2) == 0]
            if len(h):
                z_ = bil(zo[f], h[:, 0], h[:, 1]); ok = z_ > 0.05
                errs.append(np.abs(z_[ok] * (des(h[ok, 0], h[ok, 1]) @ cw[i]) - h[ok, 2]) / h[ok, 2])
        coefs[wnd] = cw; score[wnd] = float(np.median(np.concatenate(errs))); ident[wnd] = idn
    wb = min(score, key=score.get)
    c0 = np.array([coefs[wb][i][0] for i in range(N)])
    jumps = np.abs(np.diff(c0)); jw = int(np.argmax(jumps))
    near = [i for i in ident[wb] if abs(i - jw) <= 1 or abs(i - jw - 1) <= 1]
    log(f"BEFORE (epoch 7's bend): ±{wb} by held-out A ({', '.join(f'±{w} {score[w] * 100:.2f} %' for w in EPOCH7_WINDOWS)}); "
        f"{len(ident[wb])} keyframe(s) at identity [1,0,0] (< {EPOCH7_MIN_ROWS} rows): {ident[wb][:20]}; largest "
        f"|dc0| {jumps[jw]:.3f} at pair {jw}->{jw + 1} (c0 {c0[jw]:.3f} -> {c0[jw + 1]:.3f}), "
        f"{'NEXT TO an identity keyframe ' + str(near) if near else 'not next to an identity keyframe'}")
    return wb, coefs[wb], ident[wb]


# ── 3-4. tau on the interior, edge pixels, mixed pixels ─────────────────────

def interior(passed_f):
    """Floor-passing pixels whose whole 3x3 window passed too — Omega's confidence collapses at contours, so
    this excludes the contour band without a tolerance of its own (tau is measured HERE; it cannot define it)."""
    from scipy.ndimage import binary_erosion
    return binary_erosion(passed_f, structure=np.ones((3, 3), bool), border_value=0)


def label_loader(s):
    """Per keyframe POSITION the SAM3 label map: masklet id + 1 (0 = no mask, -1 = two masks overlap)."""
    from precision.silhouette_filter import masks_by_keyframe
    masks = np.load(O / "seg_masks.npz")
    by_kf = masks_by_keyframe(O, masks)
    H, W = s["H"], s["W"]

    def labels(i):
        L = np.zeros((H, W), np.int64); cnt = np.zeros((H, W), np.uint8)
        for oid, key in by_kf.get(i, []):
            m = np.asarray(masks[key]) > 0
            if m.shape != (H, W):
                raise SystemExit(f"mask {key} is {m.shape[1]}x{m.shape[0]}, Omega's grid {W}x{H}")
            L[m] = np.where(cnt[m] == 0, oid + 1, -1)
            cnt[m] = np.minimum(cnt[m] + 1, 2)
        return L
    return labels, sum(len(v) for v in by_kf.values()), len(by_kf)


def edges_and_snap(s, dep, tau, labels):
    """The EDGE pixels of every keyframe (3x3 window spans a depth step, on the bent depth BEFORE snapping), then
    the mixed pixels snapped to their mask's side (every keyframe: the judges carry them too)."""
    from precision import corrected_cloud as CC
    frames, valid, passed = s["frames"], s["valid"], s["passed"]
    n_mixed = n_snap = n_edge = n_edge_below = 0
    edge = {}
    for i, f in enumerate(frames):
        edge[f] = CC.depth_steps(dep[f], valid[f], tau)
        n_edge += int(edge[f].sum()); n_edge_below += int((edge[f] & ~passed[f]).sum())
        d, mixed, sn = CC.snap_mixed(dep[f], valid[f], labels(i), tau)
        dep[f] = d.astype(np.float32); n_mixed += int(mixed.sum()); n_snap += int(sn.sum())
    n_valid = sum(int(v.sum()) for v in valid.values())
    n_below = n_valid - sum(int(v.sum()) for v in passed.values())
    log(f"edge pixels (3x3 window spans a depth step): {n_edge / n_valid * 100:.2f} % of the valid, "
        f"{n_edge_below / max(n_edge, 1) * 100:.1f} % of them below the floor ({n_edge_below / max(n_below, 1) * 100:.1f} % "
        f"of all below-floor pixels)")
    log(f"mixed pixels: {n_mixed:,} ({n_mixed / n_valid * 100:.2f} % of the valid), {n_snap:,} snapped to their "
        f"mask's side ({n_snap / max(n_mixed, 1) * 100:.1f} %), the rest go to the vote as they are")
    return edge


# ── 5. the vote ─────────────────────────────────────────────────────────────

def vote_frame(s, i, dep, edges, tau):
    from precision import corrected_cloud as CC
    pcfg, frames, valid, passed = s["pcfg"], s["frames"], s["valid"], s["passed"]
    f = frames[i]
    H, W = s["H"], s["W"]
    edge = edges[f]
    cand = valid[f] & (passed[f] | edge)                   # below-floor interior pixels stay out without a vote
    v = CC.two_sided_vote(i, frames, dep, passed, cand, s["K"], s["c2w"], s["w2c"], pcfg.bend.neighbors, tau)
    rr, cc, agree, contra = v["rr"], v["cc"], v["agree"], v["contra"]
    pas, edg = passed[f][rr, cc], edge[rr, cc]
    contradicted = pas & (contra > agree)
    rep_ok = np.zeros(len(rr), bool); zrep = np.full(len(rr), np.nan); nrep = np.zeros(len(rr), np.int32)
    if contradicted.any():
        ok, z_, n_ = CC.agreeing_median(v["splats"][:, contradicted], tau, int(pcfg.cloud.repair_min_views))
        rep_ok[contradicted] = ok; zrep[contradicted] = z_; nrep[contradicted] = n_
    keep, repair, admit = CC.edge_vote_decision(pas, edg, agree, contra, rep_ok)
    zmap = np.zeros((H, W), np.float32); amap = np.zeros((H, W), np.int16)
    m = keep | admit
    zmap[rr[m], cc[m]] = v["zmed"][m]; amap[rr[m], cc[m]] = agree[m]
    zmap[rr[repair], cc[repair]] = zrep[repair]; amap[rr[repair], cc[repair]] = nrep[repair]
    st = dict(valid=int(valid[f].sum()), passed=int(passed[f].sum()), edge=int(edge.sum()),
              edge_below=int((~pas & edg).sum()), kept=int(keep.sum()),
              unjudged=int((keep & (agree == 0) & (contra == 0)).sum()), contradicted=int(contradicted.sum()),
              by_second_side=int((contradicted & (v["contra_fwd"] <= agree)).sum()), repaired=int(repair.sum()),
              admitted=int(admit.sum()), out=int((zmap > 0).sum()),
              edge_out=int((edge & (zmap > 0)).sum()))
    return zmap, amap, st, edge


def vote_epoch7(s, i, dep7, tau7, edge):
    """Epoch 7's one-sided vote (omega_bent_epoch7.py:136-161) on keyframe i: floor-passing pixels only,
    contradiction = the pixel's point in front of a neighbour's surface; 0/0 stays."""
    from precision import corrected_cloud as CC
    f = s["frames"][i]
    v = CC.two_sided_vote(i, s["frames"], dep7, s["passed"], s["passed"][f], s["K"], s["c2w"], s["w2c"],
                          s["pcfg"].bend.neighbors, tau7)
    keep = v["contra_fwd"] <= v["agree"]
    rr, cc = v["rr"], v["cc"]
    return dict(kept=int(keep.sum()), contradicted=int((~keep).sum()),
                unjudged=int((keep & (v["agree"] == 0) & (v["contra_fwd"] == 0)).sum()),
                edge_out=int((edge[rr, cc] & keep).sum()))


# ── 6. cloud -> octree -> epoch 8 -> masks -> certify -> renumbered 8 ───────

def publish(s, final, wb, tau):
    from PIL import Image
    from precision import corrected_cloud as CC
    from precision.epoch0_cloud import _write_ply_xyzrgb
    from correction.run import run_select
    from potree_converter import convert_ply_to_potree
    frames, K, c2w, chunk, H, W, cfg = s["frames"], s["K"], s["c2w"], s["chunk"], s["H"], s["W"], s["cfg"]
    t0 = time.time()
    if live() == EPOCH:                    # replacing epoch 8: show 7 again, the old 8 is deleted below
        run_select(O, EPOCH - 1, "auto", log=log)
    prev = live()
    for pth in (TMP, DST):
        shutil.rmtree(pth, ignore_errors=True)
    (TMP / "chunks").mkdir(parents=True)
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    by_chunk = {}
    for f in frames:
        by_chunk.setdefault(chunk[f], []).append(f)
    for k, fl in sorted(by_chunk.items()):
        L = {x: [] for x in ("xyz", "rgb", "fg", "pr", "pc", "cf")}
        for f in fl:
            z, agree = final[f]; m = z > 0; r, c = vv[m], uu[m]; zz = z[m].astype(np.float64)
            X = np.stack([(c - K[0, 2]) / K[0, 0] * zz, (r - K[1, 2]) / K[1, 1] * zz, zz], 1) @ c2w[f][:3, :3].T + c2w[f][:3, 3]
            img = np.asarray(Image.open(S / "frames" / f"{f:06d}.jpg").convert("RGB").resize((W, H)))
            L["xyz"].append(X.astype(np.float32)); L["rgb"].append(img[r, c]); L["fg"].append(np.full(len(r), f, np.int32))
            L["pr"].append(r.astype(np.int16)); L["pc"].append(c.astype(np.int16))
            L["cf"].append(agree[m].astype(np.float32))      # #14: the vote's agree count, not a constant
        _write_ply_xyzrgb(TMP / "chunks" / f"chunk_{k:03d}.ply", np.concatenate(L["xyz"]), np.concatenate(L["rgb"]))
        np.savez(TMP / "chunks" / f"chunk_{k:03d}_origins.npz", frame_global=np.concatenate(L["fg"]),
                 pixel_row=np.concatenate(L["pr"]), pixel_col=np.concatenate(L["pc"]), confidence=np.concatenate(L["cf"]))
    cleaned = TMP / "cleaned_cloud.ply"
    CC.clean(cfg, TMP / "chunks", cleaned, log)
    if not convert_ply_to_potree(S, force=True, ply_override=cleaned, potree_dir_override=TMP / "potree"):
        log("octree failed"); sys.exit(1)
    DST.mkdir()
    shutil.move(str(cleaned), str(DST / "cleaned_cloud.ply")); shutil.move(str(TMP / "potree"), str(DST / "potree"))
    shutil.copy(O / "precision" / "f5_camera_poses.txt", DST / "camera_poses.txt")
    # the camera that built the cloud travels WITH it (d4fcccd): the stored epochs list intrinsic.txt, so a
    # manifest without it files the live one away with the previous epoch and leaves the session with none —
    # the certify crashed on exactly that (pccr 2026-10-01, restored by hand)
    from precision.depth_on_f5 import camera_travels
    camera_travels(DST, s["params"], len(frames), log)
    (DST / "_manifest.json").write_text(json.dumps({"epoch": EPOCH, "epoch_from": EPOCH, "epoch_to": EPOCH, "kind": "new_cloud",
        "note": f"Omega bent to F5 with EPOCH 7's bend (±{wb}, lowest held-out), mixed pixels snapped by SAM3, "
                f"two-sided vote (tau {tau * 100:.2f} % interior), below-floor edge pixels (3x3 window spans a step) admitted when "
                f"confirmed, confidence = agree count; no DA3",
        "artifacts": [{"rel": x, "existed_before": True}
                      for x in ("cleaned_cloud.ply", "potree", "camera_poses.txt", "intrinsic.txt")]}))
    shutil.rmtree(TMP, ignore_errors=True)
    log(f"cloud built ({(time.time() - t0) / 60:.1f} min) — epoch {EPOCH} live")
    run_select(O, EPOCH, "auto", log=log)
    Ep = O / f"_epoch_{prev}"                                 # the previous epoch's segmentation travels with it
    mp = json.loads((Ep / "_manifest.json").read_text()); have = {a["rel"] for a in mp["artifacts"]}
    for r_ in ("segmentation_result.json", "scene_r.db", "classification.npy", "class_map.json"):
        if (O / r_).exists():
            shutil.copy2(O / r_, Ep / r_)
            if r_ not in have:
                mp["artifacts"].append({"rel": r_, "existed_before": True})
    (Ep / "_manifest.json").write_text(json.dumps(mp))
    for r_ in ("segmentation_result.json", "classification.npy", "class_map.json"):
        if (O / r_).exists():
            (O / r_).rename(O / "precision" / f"before_e{EPOCH}_{r_}")
    from segmentation.pipeline import map_segmentation_to_cloud
    d = map_segmentation_to_cloud(O)
    log(f"  {len(d.get('instances', []))} instances, coverage {d.get('coverage')} ({(time.time() - t0) / 60:.1f} min)")
    from reconstruction.loops.config import load_loops_config
    from reconstruction.certify.run import certify_session
    certify_session(S, cfg=load_loops_config(cfg), operator="pipeline", log=log, progress=lambda a, b: None)
    e_fin = live()
    if e_fin != EPOCH:
        shutil.rmtree(O / f"_epoch_{EPOCH}", ignore_errors=True)
        recp = O / "geometry_epoch.json"; r_ = json.loads(recp.read_text()); r_.update(epoch=EPOCH, renumbered_from=e_fin)
        recp.write_text(json.dumps(r_, indent=2))
    log(f"DONE in {(time.time() - t0) / 60:.1f} min — live {live()}, stored {sorted(x.name for x in O.glob('_epoch_*'))}")


# ── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", type=int, default=0, metavar="N",
                    help="vote on N keyframes spread over the walk, print the stats, write NOTHING")
    ap.add_argument("--landmarks-cache", default=None, metavar="NPZ",
                    help="dry runs only: F5's landmark rows read from / written to this file outside the session")
    args = ap.parse_args()
    if args.landmarks_cache and not args.dry_run:
        raise SystemExit("--landmarks-cache is for dry runs only: the real run triangulates its own landmarks")
    dry = args.dry_run > 0
    if not dry and live() == EPOCH:
        raise SystemExit(f"the live epoch is already {EPOCH}: run_select({EPOCH}) would change nothing, the masks and "
                         f"the certify would run on the OLD cloud and the renumbering would delete the new one — "
                         f"select the parent epoch first (correction.run.run_select)")
    t0 = time.time()
    from precision import corrected_cloud as CC
    s = load_session()
    pcfg, frames, N = s["pcfg"], s["frames"], s["N"]
    log(f"{N} keyframes, {s['W']}x{s['H']}, floor {s['floor_norm']} per chunk ({len(s['thr'])} chunks); "
        f"cgroup {mem_gb():.1f} GB")
    obs = landmarks(s, args.landmarks_cache)
    # USER 2026-10-01: "reconstruí la epoch 8 pero con el exactamente mismo ajuste que la 7" — the bend is
    # epoch 7's, unchanged (10-step IRLS, z > 0.05, < 20 rows = identity, window by lowest held-out A);
    # the consecutive-agreement window (±16) gave the floor that was judged worse.
    from precision.depth_on_f5 import design
    wb, coefs, _ident = epoch7_bend(s, obs)
    _uu, _vv = np.meshgrid(np.arange(s["W"]), np.arange(s["H"]))
    D = design(_uu.ravel(), _vv.ravel(), s["W"], s["H"])
    brep = {"spread": (float("nan"), float("nan"))}
    e7 = epoch7_bend(s, obs) if dry else None
    dep = {f: np.where(s["valid"][f], bent(s, f, coefs[i], D), 0).astype(np.float32) for i, f in enumerate(frames)}
    inner = {f: interior(s["passed"][f]) for f in frames}
    tau = CC.measured_tau(dep, inner, s["K"], s["c2w"], frames, pcfg.bend.neighbors, pcfg.bend.tau_quantile)
    tau_all = CC.measured_tau(dep, s["passed"], s["K"], s["c2w"], frames, pcfg.bend.neighbors, pcfg.bend.tau_quantile)
    log(f"tau {tau * 100:.2f} % (p{pcfg.bend.tau_quantile:g} of the neighbour disagreement on INTERIOR pixels; "
        f"on every floor-passing pixel it reads {tau_all * 100:.2f} %); cgroup {mem_gb():.1f} GB")
    del inner
    labels, n_masks, n_kf_masks = label_loader(s)
    log(f"SAM3 masks: {n_masks:,} over {n_kf_masks} keyframes")
    edges = edges_and_snap(s, dep, tau, labels)
    sel = (sorted(set(int(round(x)) for x in np.linspace(0, N - 1, args.dry_run + 2)[1:-1])) if dry
           else list(range(N)))
    final, tot = {}, {}
    for i in sel:
        zmap, amap, st, edge = vote_frame(s, i, dep, edges, tau)
        final[frames[i]] = (zmap, amap)
        for k, v in st.items():
            tot[k] = tot.get(k, 0) + v
        if dry:
            st["edge_mask"] = edge
            final[frames[i]] = (zmap, amap, st)
    nv = max(tot["valid"], 1)
    log(f"vote: {tot['kept'] / nv * 100:.1f} % of the valid pixels kept ({tot['unjudged'] / nv * 100:.1f} % unjudged, "
        f"floor-passing), {tot['contradicted'] / nv * 100:.1f} % contradicted ({tot['by_second_side'] / nv * 100:.1f} % "
        f"by the second side alone), {tot['repaired'] / nv * 100:.1f} % repaired, {tot['admitted'] / nv * 100:.1f} % "
        f"admitted below the floor at edges (of {tot['edge_below'] / nv * 100:.1f} % candidates); coverage "
        f"{tot['out'] / (len(sel) * s['H'] * s['W']) * 100:.1f} % of all pixels; cgroup {mem_gb():.1f} GB")
    if dry:
        wb7, coefs7, _ = e7
        dep7 = {f: np.where(s["passed"][f], bent(s, f, coefs7[i], D), 0).astype(np.float32) for i, f in enumerate(frames)}
        tau7 = CC.measured_tau(dep7, s["passed"], s["K"], s["c2w"], frames, pcfg.bend.neighbors, pcfg.bend.tau_quantile)
        log(f"BEFORE: tau {tau7 * 100:.2f} % (epoch 7: every floor-passing pixel, its own bend)")
        print(f"\n{'keyframe':>9} | {'valid':>7} | BEFORE kept  contra  edge-kept | AFTER kept  contra  repaired  "
              f"admitted  edge-kept  out  | 2nd-side  unjudged")
        b_tot = {}
        for i in sel:
            f = frames[i]
            zmap, amap, st = final[f]
            b = vote_epoch7(s, i, dep7, tau7, st["edge_mask"])
            for k, v in b.items():
                b_tot[k] = b_tot.get(k, 0) + v
            V = max(st["valid"], 1)
            print(f"{i:>4} ({f:>4}) | {st['valid']:>7} | {b['kept'] / V * 100:10.1f}% {b['contradicted'] / V * 100:6.1f}% "
                  f"{b['edge_out']:>9} | {st['kept'] / V * 100:9.1f}% {st['contradicted'] / V * 100:6.1f}% "
                  f"{st['repaired'] / V * 100:8.1f}% {st['admitted'] / V * 100:8.1f}% {st['edge_out']:>10} "
                  f"{st['out'] / V * 100:5.1f}% | {st['by_second_side'] / V * 100:7.1f}% {st['unjudged'] / V * 100:7.1f}%")
        print(f"{'total':>9} | {tot['valid']:>7} | {b_tot['kept'] / nv * 100:10.1f}% {b_tot['contradicted'] / nv * 100:6.1f}% "
              f"{b_tot['edge_out']:>9} | {tot['kept'] / nv * 100:9.1f}% {tot['contradicted'] / nv * 100:6.1f}% "
              f"{tot['repaired'] / nv * 100:8.1f}% {tot['admitted'] / nv * 100:8.1f}% {tot['edge_out']:>10} "
              f"{tot['out'] / nv * 100:5.1f}% | {tot['by_second_side'] / nv * 100:7.1f}% {tot['unjudged'] / nv * 100:7.1f}%")
        print(f"(edge pixels: {tot['edge']:,} px = {tot['edge'] / nv * 100:.1f} % of the valid; BEFORE kept {b_tot['edge_out']:,} "
              f"of them, AFTER {tot['edge_out']:,}; before = epoch 7 ±{wb7}, after = ±{wb}, spread "
              f"[{brep['spread'][0]:.4f}, {brep['spread'][1]:.4f}])")
        log(f"DRY RUN done in {(time.time() - t0) / 60:.1f} min — nothing written")
        return
    log("DA3 holes: off (epoch 8 = Omega bent to F5 + the edge-keeping vote, nothing else)")
    del dep
    del edges
    publish(s, final, wb, tau)


if __name__ == "__main__":
    main()
