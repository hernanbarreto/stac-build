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
    the landmarks of keyframes i−w … i+w; w chosen among `bend.windows` by the held-out (median |dz|/z).
    A keyframe with fewer than `refine.min_witness_corr` landmark rows borrows the NEAREST fitted
    keyframe's k (never identity next to bent neighbours);
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


def irls_huber(A: np.ndarray, r: np.ndarray, k: float, tol: float, max_iter: int) -> np.ndarray:
    """Robust least squares: Huber weights at k·σ, σ = 1.4826·MAD of the residual."""
    w = np.ones(len(r))
    c = np.linalg.lstsq(A, r, rcond=None)[0]
    for _ in range(max_iter):
        sw = np.sqrt(w)
        c_new = np.linalg.lstsq(A * sw[:, None], r * sw, rcond=None)[0]
        e = r - A @ c_new
        s = MAD_TO_SIGMA * np.median(np.abs(e)) + 1e-12
        w = np.minimum(1.0, k * s / np.maximum(np.abs(e), 1e-12))
        done = np.max(np.abs(c_new - c)) <= tol * max(1.0, float(np.max(np.abs(c_new))))
        c = c_new
        if done:
            break
    return c


def bend_coefficients(rows: Dict[int, tuple], n: int, window: int, min_rows: int, huber_k: float,
                      tol: float, max_iter: int) -> Dict[int, np.ndarray]:
    """{i: (c0, c1, c2)} — each keyframe fitted on the rows of i−w … i+w; one with fewer than `min_rows`
    rows there borrows the nearest fitted keyframe's coefficients (identity only when none is fitted)."""
    out: Dict[int, np.ndarray] = {}
    for i in range(n):
        js = range(max(0, i - window), min(n, i + window + 1))
        A = [rows[j][0] for j in js if j in rows and len(rows[j][1])]
        r = [rows[j][1] for j in js if j in rows and len(rows[j][1])]
        if A and sum(len(x) for x in r) >= min_rows:
            out[i] = irls_huber(np.vstack(A), np.concatenate(r), huber_k, tol, max_iter)
    fitted = sorted(out)
    for i in range(n):
        if i not in out:
            out[i] = (out[min(fitted, key=lambda j: abs(j - i))].copy() if fitted
                      else np.array([1.0, 0.0, 0.0]))
    return out


def vote(dep: Dict[int, np.ndarray], ent: Dict[int, np.ndarray], K: np.ndarray, c2w: Dict[int, np.ndarray],
         order: List[int], neighbors, tau: float):
    """{frame: (depth map, agree count map)} after the vote: a pixel leaves when more neighbours see free
    space through it (its point lies in FRONT of their surface by more than τ) than agree within τ;
    a kept pixel takes the median of its own depth and the agreeing views' depths along its ray."""
    w2c = {f: np.linalg.inv(c2w[f]) for f in order}
    out = {}
    for i, f in enumerate(order):
        H, W = dep[f].shape
        rr, cc = np.nonzero(ent[f])
        z = dep[f][rr, cc].astype(np.float64)
        ray = np.stack([(cc - K[0, 2]) / K[0, 0], (rr - K[1, 2]) / K[1, 1], np.ones(len(rr))], 1) @ c2w[f][:3, :3].T
        C = c2w[f][:3, 3]
        agree = np.zeros(len(z), np.int32); contra = np.zeros(len(z), np.int32); cand = [z]
        for d in neighbors:
            j = i + int(d)
            if not 0 <= j < len(order):
                continue
            g = order[j]
            a = w2c[g][2, :3] @ C + w2c[g][2, 3]; b = ray @ w2c[g][2, :3]
            X = C + z[:, None] * ray
            Xg = X @ w2c[g][:3, :3].T + w2c[g][:3, 3]; zg = Xg[:, 2]
            ok = zg > 0
            zs = np.where(ok, zg, 1.0)
            u = np.rint(K[0, 0] * Xg[:, 0] / zs + K[0, 2]).astype(np.int64)
            v = np.rint(K[1, 1] * Xg[:, 1] / zs + K[1, 2]).astype(np.int64)
            ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
            dg = np.zeros(len(z))
            dg[ok] = dep[g][v[ok], u[ok]]
            ok &= dg > 0
            e = np.zeros(len(z))
            e[ok] = (zg[ok] - dg[ok]) / dg[ok]
            ag = ok & (np.abs(e) <= tau) & (np.abs(b) > 0)
            agree += ag
            contra += ok & (e < -tau)
            cand.append(np.where(ag, (dg - a) / np.where(np.abs(b) > 0, b, 1.0), np.nan))
        keep = contra <= agree
        zf = np.nanmedian(np.vstack(cand), 0)
        zmap = np.zeros((H, W), np.float32); amap = np.zeros((H, W), np.int32)
        zmap[rr[keep], cc[keep]] = zf[keep]; amap[rr[keep], cc[keep]] = agree[keep]
        out[f] = (zmap, amap, int(len(z)), int((~keep).sum()))
    return out


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


def run_depth_on_f5(session_dir: Path, pcfg, log: Callable = print,
                    progress: Optional[Callable[[float, str], None]] = None) -> dict:
    from config import cfg as raw_cfg
    from precision import depth_sweep as DS
    from precision import corrected_cloud as CC
    from precision.epoch0_cloud import SKY_CONF, _write_ply_xyzrgb
    from correction.session import read_ply
    t0 = time.time()
    session_dir = Path(session_dir)
    out = session_dir / "output"
    bc, gc = pcfg.bend, pcfg.gauge

    def _p(pct, msg):
        log(f"{LOG_TAG} {msg}")
        if progress:
            progress(pct, msg)

    inp = DS.load_inputs(session_dir, pcfg)                 # F5's camera + poses (guards F5's epoch)
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
    thr = {}
    for k in sorted(set(chunk.values())):
        v = np.concatenate([conf[f][np.isfinite(conf[f]) & (conf[f] > SKY_CONF)].ravel()
                            for f in frames if chunk[f] == k])
        thr[k] = float(v.min() + floor_norm * (v.max() - v.min()))

    # 2. the bend
    _p(25, "bending Omega's depth to F5's landmarks")
    rows = {}
    for i, f in enumerate(frames):
        o = obs[0][i]
        zz = bilinear(zo[f], o[:, 0], o[:, 1]) if len(o) else np.zeros(0)
        ok = zz > 0
        rows[i] = (design(o[ok, 0], o[ok, 1], W, H), o[ok, 2] / zz[ok])
    score, coefs = {}, {}
    for w in bc.windows:
        cw = bend_coefficients(rows, N, int(w), int(pcfg.refine.min_witness_corr), gc.huber_k, gc.huber_tol,
                               gc.huber_max_iter)
        errs = []
        for i, f in enumerate(frames):
            h = obs[1][i]
            if len(h):
                zz = bilinear(zo[f], h[:, 0], h[:, 1]); ok = zz > 0
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
    dep, ent = {}, {}
    for i, f in enumerate(frames):
        k = (design(uu.ravel(), vv.ravel(), W, H) @ coefs[wb][i]).reshape(H, W)
        valid = (np.isfinite(zo[f]) & (zo[f] > 0) & np.isfinite(conf[f]) & (conf[f] > SKY_CONF)
                 & (conf[f] >= thr[chunk[f]]))
        dep[f] = np.where(valid, zo[f] * k, 0).astype(np.float32)
        ent[f] = valid

    # 4. the vote
    _p(50, "multi-view vote")
    tau = CC.measured_tau(dep, ent, K, c2w, frames, bc.neighbors, bc.tau_quantile)
    voted = vote(dep, ent, K, c2w, frames, bc.neighbors, tau)
    n_in = sum(v[2] for v in voted.values()); n_out = sum(v[3] for v in voted.values())
    cover = sum(int((voted[f][0] > 0).sum()) for f in frames) / float(N * H * W)
    _p(65, f"vote: τ {tau * 100:.2f} % (p{bc.tau_quantile:g} of the session's own disagreement); "
           f"{n_out / max(n_in, 1) * 100:.1f} % of the valid pixels contradicted; coverage {cover * 100:.1f} %")
    del dep, ent, zo, conf

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
                "source": np.ones(len(data), np.uint8)}
        report = {"version": 1, "stage": "depth_on_f5", "provenance": "tool_measured",
                  "source_of_depth": "Omega's depth bent to F5's landmarks + multi-view vote",
                  "camera": params, "grid": [W, H],
                  "bend": {"window": int(wb), "held_out": score,
                           "scale": {"median": float(np.median(c0)), "min": float(c0.min()),
                                     "max": float(c0.max())}},
                  # what each keyframe's depth was multiplied by — the chunk check (f6_check) measures
                  # the PUBLISHED cloud with it: s_k = c0, bend = (c1, c2) of precision.depth_on_f5.design
                  "per_frame": {str(f): {"s_k": float(coefs[wb][i][0]),
                                         "bend": [float(coefs[wb][i][1]), float(coefs[wb][i][2])]}
                                for i, f in enumerate(frames)},
                  "vote": {"tau": tau, "contradicted_frac": n_out / max(n_in, 1), "coverage": cover},
                  "raw_points": n_raw}
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
