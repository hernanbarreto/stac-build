"""Photometric bundle adjustment over the keyframes — "minimise the red".

USER 2026-09-09: stand at every keyframe, reproject what the OTHER keyframes
wrote (with their own colours) into its photo, and compare with the photo.
Where the synthetic photo does not sit on the real one the poses are wrong,
and the optical flow between the two is the error, pixel by pixel, without
mixing surfaces (a pixel of the photo is one surface). Validated on pccr
31/08: consecutive keyframes and chunk seams 2–4 px (1.5–2.6 cm), the
revisit 34–47 px (27–33 cm).

Correction = dense photometric bundle adjustment (BundleFusion-style):
  * correspondences: for pair (j → k), the 3-D points written by j that
    land in k's photo; their target = landing pixel + flow (real → synthetic
    inverted), weight = image gradient at the target (a flat wall gives no
    information, no threshold needed);
  * unknowns: one SE(3) correction per keyframe, kf 0 fixed;
  * residual: reprojection error of X_j p in the camera of k moved by X_k,
    minus the target pixel; Levenberg-Marquardt with closed-form Jacobians;
  * outer loop: re-render with the solved poses, re-measure the flow, solve
    again (correspondences improve with the poses).

Read-only: ``corrections/photobundle.json`` + ``photobundle_poses.npz`` +
PNGs; the apply is a separate, transactional step.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation

from correction.config import CorrectionConfig
from correction.revisit import _frame_path, _project, load_intrinsics, provenance_grid
from correction.session import CorrectionSession
from correction.units import chunks_of_keyframe, load_chunk_plan

REPORT_NAME = "photobundle.json"
POSES_NAME = "photobundle_poses.npz"


class PhotoContext:
    """Frames, intrinsics on the IMAGE grid, per-keyframe point index."""

    def __init__(self, session: CorrectionSession, cfg: CorrectionConfig):
        self.s = session
        self.cfg = cfg
        n = session.n_kf
        K = load_intrinsics(session.output_dir, n)
        H, W = provenance_grid(session)
        fp = _frame_path(session, 0)
        if fp is None:
            raise RuntimeError("no frames/ directory next to output/ — the "
                               "photometric check needs the real photos")
        im = cv2.imread(str(fp))
        self.Hi, self.Wi = im.shape[:2]
        sx, sy = self.Wi / W, self.Hi / H
        self.K = K * np.array([sx, sy, sx, sy])       # fx fy cx cy in image px
        order = np.argsort(session.ks, kind="stable")
        bounds = np.searchsorted(session.ks[order], np.arange(-1, n + 1))
        self.own = [order[bounds[k + 1]:bounds[k + 2]] for k in range(n)]
        self.rgb = np.stack([session.data["red"], session.data["green"],
                             session.data["blue"]], 1).astype(np.uint8)
        self._gray: Dict[int, np.ndarray] = {}
        self._grad: Dict[int, np.ndarray] = {}
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    def gray(self, k: int) -> np.ndarray:
        if k not in self._gray:
            im = cv2.imread(str(_frame_path(self.s, k)))
            g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            self._gray[k] = g
            gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
            self._grad[k] = np.sqrt(gx * gx + gy * gy)
        return self._gray[k]

    def grad(self, k: int) -> np.ndarray:
        self.gray(k)
        return self._grad[k]


def _cam(P: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    return P @ w2c[:3, :3].T + w2c[:3, 3]


def _pix(Q: np.ndarray, k4: np.ndarray):
    z = Q[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = k4[0] * Q[:, 0] / z + k4[2]
        v = k4[1] * Q[:, 1] / z + k4[3]
    return u, v, z


def render_pair(ctx: PhotoContext, k: int, j: int, R: np.ndarray, t: np.ndarray,
                w2c_k: np.ndarray, pc: "PhotoCfg") -> Optional[dict]:
    """Points of j (corrected by X_j) splatted into k's camera (corrected by
    X_k). Returns correspondences: point idx, target pixel, weight, depth;
    plus the median flow (px) and coverage."""
    s = ctx.s
    idx = ctx.own[j]
    if not len(idx):
        return None
    P = s.xyz[idx] @ R[j].T + t[j]
    Q = _cam(P, w2c_k)
    u, v, z = _pix(Q, ctx.K[k])
    ok = (z > pc.depth_min_m) & (u >= 0) & (u < ctx.Wi - 1) & (v >= 0) & (v < ctx.Hi - 1)
    if ok.sum() < pc.pair_min_points:
        return None
    idx, u, v, z = idx[ok], u[ok], v[ok], z[ok]
    ui, vi = u.astype(np.int64), v.astype(np.int64)
    order = np.argsort(-z)
    syn = np.zeros((ctx.Hi, ctx.Wi), np.uint8)
    zb = np.full((ctx.Hi, ctx.Wi), np.inf)
    owner = np.full((ctx.Hi, ctx.Wi), -1, np.int64)
    gcol = cv2.cvtColor(ctx.rgb[idx][None], cv2.COLOR_RGB2GRAY)[0]
    r_ = pc.splat_radius
    for dr in range(-r_, r_ + 1):
        for dc in range(-r_, r_ + 1):
            uu = np.clip(ui + dc, 0, ctx.Wi - 1)
            vv = np.clip(vi + dr, 0, ctx.Hi - 1)
            syn[vv[order], uu[order]] = gcol[order]
            zb[vv[order], uu[order]] = z[order]
            owner[vv[order], uu[order]] = order
    valid = np.isfinite(zb)
    cov = float(valid.mean())
    if valid.sum() < pc.pair_min_points:
        return None
    real = ctx.gray(k)
    synf = syn.copy()
    synf[~valid] = real[~valid]
    flow = ctx.dis.calc(real, synf, None)            # real → synthetic
    # the content at real pixel x is at x + flow(x) in the synthetic image;
    # inverting: a synthetic pixel y holds content that belongs at y − flow.
    # Sample at the point's landing pixel (y): target = y − flow(y) (first
    # order; the outer loop refines).
    er = cv2.erode(valid.astype(np.uint8), np.ones((2 * r_ + 3,) * 2, np.uint8)).astype(bool)
    sel = er[vi, ui]
    if sel.sum() < pc.pair_min_points:
        return None
    # simpler: keep points whose depth equals the z-buffer at their pixel
    own_pix = np.abs(zb[vi, ui] - z) < 1e-9
    sel &= own_pix
    if sel.sum() < pc.pair_min_points:
        return None
    ptr = np.where(sel)[0]
    if len(ptr) > pc.points_per_pair:
        ptr = np.random.default_rng(pc.seed + k * 1000 + j).choice(ptr, pc.points_per_pair, replace=False)
    fl = flow[vi[ptr], ui[ptr]]
    target = np.stack([u[ptr] - fl[:, 0], v[ptr] - fl[:, 1]], 1)
    mag = np.linalg.norm(fl, axis=1)
    w = ctx.grad(k)[np.clip(target[:, 1].astype(int), 0, ctx.Hi - 1),
                    np.clip(target[:, 0].astype(int), 0, ctx.Wi - 1)]
    fmag = np.linalg.norm(flow[er], axis=1)
    med_px = float(np.median(fmag))
    med_cm = float(np.median(fmag * zb[er] / ctx.K[k][0])) * 100
    return {"k": k, "j": j, "idx": idx[ptr], "target": target, "w": w,
            "z": z[ptr], "flow_px": mag, "coverage": cov,
            "med_px": med_px, "med_cm": med_cm, "n": int(len(ptr))}


class PhotoCfg:
    def __init__(self, cfg: CorrectionConfig):
        p = cfg.photo
        self.depth_min_m = cfg.revisit.depth_min_m
        self.pair_min_points = p.pair_min_points
        self.points_per_pair = p.points_per_pair
        self.splat_radius = p.splat_radius
        self.seed = cfg.solve.seed
        self.outer_iters = p.outer_iters
        self.candidate_sample = p.candidate_sample
        self.candidate_min = p.candidate_min


def candidate_pairs(ctx: PhotoContext, pc: PhotoCfg, log=print) -> List[Tuple[int, int]]:
    """(k, j) pairs where a sample of j's points lands in k's image, in
    front of the camera and in k's depth window."""
    s = ctx.s
    n = s.n_kf
    rng = np.random.default_rng(pc.seed)
    samp = []
    owner = []
    for j in range(n):
        o = ctx.own[j]
        if not len(o):
            continue
        pick = o if len(o) <= pc.candidate_sample else rng.choice(o, pc.candidate_sample, replace=False)
        samp.append(pick)
        owner.append(np.full(len(pick), j))
    samp = np.concatenate(samp)
    owner = np.concatenate(owner)
    P = s.xyz[samp]
    w2cs = np.linalg.inv(s.poses)
    pairs = []
    for k in range(n):
        u, v, z = _pix(_cam(P, w2cs[k]), ctx.K[k])
        ok = (z > pc.depth_min_m) & (z < ctx.cfg.revisit.depth_max_m) & \
             (u >= 0) & (u < ctx.Wi) & (v >= 0) & (v < ctx.Hi)
        cnt = np.bincount(owner[ok], minlength=n)
        for j in np.where(cnt >= pc.candidate_min)[0]:
            if j != k:
                pairs.append((k, int(j)))
    log(f"  {len(pairs)} candidate (k ← j) pairs")
    return pairs


def solve_poses(n_kf: int, corrs: List[dict], ctx: PhotoContext, R0, t0,
                cfg: CorrectionConfig, log=print):
    """LM over one SE(3) per keyframe (world-frame left perturbation,
    (t, θ)); residual = π_k(X_k⁻¹ X_j p) − target, weight = gradient."""
    kc = cfg.kfgraph
    huber = cfg.photo.huber_px
    s = ctx.s
    w2cs = np.linalg.inv(s.poses)
    R = R0.copy(); t = t0.copy()
    free = list(range(1, n_kf))
    col = {k: 6 * (k - 1) for k in free}
    N = 6 * (n_kf - 1)

    def residuals(R, t):
        rs, Js = [], []
        cost = 0.0
        for c in corrs:
            k, j = c["k"], c["j"]
            p = s.xyz[c["idx"]] @ R[j].T + t[j]                # world, corrected j
            # camera k moved by X_k: w2c' = w2c_k ∘ X_k⁻¹
            Rk_inv = R[k].T
            pw = (p - t[k]) @ Rk_inv.T                           # X_k⁻¹ p
            Q = _cam(pw, w2cs[k])
            u, v, z = _pix(Q, ctx.K[k])
            r = np.stack([u, v], 1) - c["target"]
            # Huber (IRLS): a residual beyond huber_px — a flow locked on the
            # neighbouring tile joint, a hole border, a reflection — enters
            # with weight δ/|r| instead of pulling quadratically
            rn = np.linalg.norm(r, axis=1)
            hub = np.where(rn <= huber, 1.0, huber / np.maximum(rn, 1e-9))
            w = c["w"] * hub
            cost += float((c["w"] * np.where(rn <= huber, rn ** 2,
                                              2 * huber * rn - huber ** 2)).sum())
            # Jacobians: dπ/dQ (2×3), dQ/dpw = R_w2c, dpw/dδ_j = Rk_inv [I, −[p]×],
            # dpw/dδ_k = −Rk_inv [I, −[p]×]
            fx, fy = ctx.K[k][0], ctx.K[k][1]
            iz = 1.0 / z
            dpi = np.zeros((len(z), 2, 3))
            dpi[:, 0, 0] = fx * iz; dpi[:, 0, 2] = -fx * Q[:, 0] * iz * iz
            dpi[:, 1, 1] = fy * iz; dpi[:, 1, 2] = -fy * Q[:, 1] * iz * iz
            A = np.einsum('nij,jk->nik', dpi, w2cs[k][:3, :3] @ Rk_inv)   # (n,2,3)
            skew = np.zeros((len(z), 3, 3))
            skew[:, 0, 1] = -p[:, 2]; skew[:, 0, 2] = p[:, 1]
            skew[:, 1, 0] = p[:, 2]; skew[:, 1, 2] = -p[:, 0]
            skew[:, 2, 0] = -p[:, 1]; skew[:, 2, 1] = p[:, 0]
            G = np.concatenate([np.tile(np.eye(3), (len(z), 1, 1)), -skew], 2)  # (n,3,6)
            Jj = np.einsum('nik,nkl->nil', A, G)                 # (n,2,6)
            rs.append((k, j, r, w, Jj))
        return rs, cost

    lam = kc.damping
    hist = []
    rs, cost = residuals(R, t)
    for it in range(kc.gn_iters):
        A = sparse.lil_matrix((N, N)); b = np.zeros(N)
        for k, j, r, w, Jj in rs:
            for k_, sgn in ((j, 1.0), (k, -1.0)):
                if k_ == 0:
                    continue
                c0 = col[k_]
                b[c0:c0 + 6] += -sgn * np.einsum('ni,nij,n->j', r, Jj, w)
                for k2, sgn2 in ((j, 1.0), (k, -1.0)):
                    if k2 == 0:
                        continue
                    c1 = col[k2]
                    A[c0:c0 + 6, c1:c1 + 6] = A[c0:c0 + 6, c1:c1 + 6] + \
                        sgn * sgn2 * np.einsum('nij,nik,n->jk', Jj, Jj, w)
        A = A.tocsr(); diag = A.diagonal()
        accepted = False
        for _try in range(kc.lm_tries):
            dx = spsolve(A + sparse.diags(lam * diag + lam * diag.max()), b)
            R2, t2 = R.copy(), t.copy(); step = 0.0
            for k in free:
                c0 = col[k]
                dt, dth = dx[c0:c0 + 3], dx[c0 + 3:c0 + 6]
                Rd = Rotation.from_rotvec(dth).as_matrix()
                R2[k] = Rd @ R[k]; t2[k] = Rd @ t[k] + dt
                step = max(step, float(np.linalg.norm(dt)), float(np.linalg.norm(dth)))
            rs2, cost2 = residuals(R2, t2)
            if cost2 < cost:
                R, t, rs, cost = R2, t2, rs2, cost2
                lam = max(lam / kc.lm_factor, kc.damping_min)
                accepted = True
                break
            lam *= kc.lm_factor
        hist.append({"iter": it, "cost": cost, "max_step": step, "lambda": lam,
                     "accepted": accepted})
        log(f"  LM iter {it}: cost {cost:.0f}, max step {step:.4f}, λ {lam:.1e}"
            f"{'' if accepted else ' (no step accepted)'}")
        if not accepted or step < kc.gn_converge:
            break
    return R, t, hist


def run_photobundle(session: CorrectionSession, cfg: CorrectionConfig,
                    log=print, progress=None) -> dict:
    t0 = time.time()
    ctx = PhotoContext(session, cfg)
    pc = PhotoCfg(cfg)
    n = session.n_kf

    def _p(pct, msg):
        log(msg)
        if progress:
            progress(pct, msg)

    _p(2, "candidate pairs (whose points land in whose photo)...")
    pairs = candidate_pairs(ctx, pc, log=log)
    R = np.tile(np.eye(3), (n, 1, 1)); t = np.zeros((n, 3))
    rounds = []
    base_w2c = np.linalg.inv(session.poses)
    for rnd in range(pc.outer_iters):
        _p(5 + 45 * rnd / pc.outer_iters, f"round {rnd}: rendering + flow for {len(pairs)} pairs...")
        corrs = []
        cache = session.output_dir / "corrections" / f"photobundle_corrs_round{rnd}.npz"
        if rnd == 0 and cache.exists():
            z = np.load(cache, allow_pickle=True)
            corrs = list(z["corrs"])
            log(f"  round 0 correspondences loaded from {cache.name} ({len(corrs)} pairs)")
        for n_done, (k, j) in enumerate(pairs if not corrs else []):
            # camera k moved by X_k: c2w' = X_k c2w  →  w2c' = w2c X_k⁻¹
            Xk_inv = np.eye(4); Xk_inv[:3, :3] = R[k].T; Xk_inv[:3, 3] = -R[k].T @ t[k]
            c = render_pair(ctx, k, j, R, t, base_w2c[k] @ Xk_inv, pc)
            if c is not None:
                corrs.append(c)
            if (n_done + 1) % 500 == 0:
                log(f"  round {rnd}: {n_done + 1}/{len(pairs)} pairs")
        if rnd == 0 and not cache.exists():
            np.savez(cache, corrs=np.array(corrs, dtype=object))
        med = np.array([c["med_px"] for c in corrs])
        gap = np.array([abs(c["k"] - c["j"]) for c in corrs])
        summ = {"round": rnd, "n_pairs": len(corrs),
                "median_px_all": round(float(np.median(med)), 2),
                "median_px_near": round(float(np.median(med[gap <= cfg.consistency.near_kf])), 2)
                if (gap <= cfg.consistency.near_kf).any() else None,
                "median_px_far": round(float(np.median(med[gap > cfg.consistency.near_kf])), 2)
                if (gap > cfg.consistency.near_kf).any() else None,
                "p90_px_far": round(float(np.percentile(med[gap > cfg.consistency.near_kf], 90)), 2)
                if (gap > cfg.consistency.near_kf).any() else None}
        log(f"  round {rnd} flow: {summ}")
        rounds.append({**summ, "pairs": [{"k": c["k"], "j": c["j"], "med_px": round(c["med_px"], 2),
                                         "med_cm": round(c["med_cm"], 1), "coverage": round(c["coverage"], 3)}
                                        for c in corrs]})
        _p(50 + 45 * rnd / pc.outer_iters, f"round {rnd}: solving {n} poses on {sum(c['n'] for c in corrs)} correspondences...")
        R, t, hist = solve_poses(n, corrs, ctx, R, t, cfg, log=log)
        rounds[-1]["solver"] = hist
    # final measurement with the solved poses
    _p(95, "final flow with the solved poses...")
    final = []
    for k, j in pairs:
        Xk_inv = np.eye(4); Xk_inv[:3, :3] = R[k].T; Xk_inv[:3, 3] = -R[k].T @ t[k]
        c = render_pair(ctx, k, j, R, t, base_w2c[k] @ Xk_inv, pc)
        if c is not None:
            final.append({"k": k, "j": j, "med_px": round(c["med_px"], 2),
                          "med_cm": round(c["med_cm"], 1), "coverage": round(c["coverage"], 3)})
    med = np.array([c["med_px"] for c in final]); gap = np.array([abs(c["k"] - c["j"]) for c in final])
    near = gap <= cfg.consistency.near_kf
    plan = load_chunk_plan(session.output_dir)
    per_kf = []
    for k in range(n):
        dR = np.eye(3) if k == 0 else R[k] @ R[k - 1].T
        dt = np.zeros(3) if k == 0 else t[k] - dR @ t[k - 1]
        per_kf.append({"kf": k, "chunks": chunks_of_keyframe(plan, k),
                       "rot_deg": round(float(np.degrees(np.linalg.norm(Rotation.from_matrix(R[k]).as_rotvec()))), 3),
                       "t_cm": round(float(np.linalg.norm(t[k])) * 100, 1),
                       "step_rot_deg": round(float(np.degrees(np.linalg.norm(Rotation.from_matrix(dR).as_rotvec()))), 3),
                       "step_t_cm": round(float(np.linalg.norm(dt)) * 100, 1)})
    report = {"n_keyframes": n, "n_pairs": len(pairs), "rounds": rounds,
              "final": {"n_pairs": len(final),
                        "median_px_near": round(float(np.median(med[near])), 2) if near.any() else None,
                        "median_px_far": round(float(np.median(med[~near])), 2) if (~near).any() else None,
                        "p90_px_far": round(float(np.percentile(med[~near], 90)), 2) if (~near).any() else None,
                        "pairs": final},
              "per_keyframe": per_kf,
              "elapsed_s": round(time.time() - t0, 1), "provenance": "tool_measured"}
    out = session.output_dir / "corrections"
    out.mkdir(exist_ok=True)
    (out / REPORT_NAME).write_text(json.dumps(report, indent=1))
    np.savez(out / POSES_NAME, R_kf=R, t_kf=t)
    _p(100, f"photobundle done: far pairs {rounds[0]['median_px_far']} → "
            f"{report['final']['median_px_far']} px, near {rounds[0]['median_px_near']} → "
            f"{report['final']['median_px_near']} px, {report['elapsed_s']} s")
    return report
