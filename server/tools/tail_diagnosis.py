"""The heavy held-out tail of F5, measured (USER 2026-11-08: station's witness bound 37 px against 2.96 px of the
01/02-11 station — "si termina en mayor cobertura de la buena claro que interesa"; 2026-11-09: "diagnóstico: la cola
de errores grandes de station y zaragoza (testigos 20–37 px)").

F5's witness bound (refine.json ``keyframe_heldout_bound_px``) is the bootstrap upper bound of the keyframes'
leave-one-view-out held-out RMS quantile. This tool RE-MEASURES that per keyframe on the state F5 applied (the
session's camera_poses.txt and camera.json, F5's own held-out tracks, F5's own batched leave-one-view-out
arithmetic), checks it reproduces the bound F5 recorded, and breaks the tail down by what could explain it: the
keyframe's cube, its place along the walk, the camera's turn rate and speed there, how many held-out observations it
has, the image region of the large residuals (radius from the principal point, row), the landmark's distance and
the track's length. READ-ONLY: nothing in the session is written. Usage:
  python -m precision.tail_diagnosis --session <scan dir> [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict

import numpy as np


def per_observation_lovo(held_g, w2c: np.ndarray, params, solver: dict, min_tri_deg: float):
    """F5's batched leave-one-view-out (precision.refine.heldout_leave_one_view_out), every observation's residual
    kept: (keyframe index, residual px, uv, landmark distance m, track length)."""
    from precision.camera import undistort_normalized_tracks
    from precision.refine import K_of, _TRI_BATCH_OBS, _dlt_stack, _track_arrays, project_obs
    tracks, counts, starts, ii, uv = _track_arrays(held_g, min_views=3)
    if not tracks:
        return {k: np.zeros(0) for k in ("frame", "res", "u", "v", "dist", "len", "key")}
    xn, ok_track = undistort_normalized_tracks(uv, K_of(params), np.asarray(params[4:8], np.float64),
                                               starts, counts, **solver)
    out = {k: [] for k in ("frame", "res", "u", "v", "dist", "len", "key")}
    for k in np.unique(counts):
        sel_all = np.nonzero((counts == k) & ok_track)[0]
        step = max(1, _TRI_BATCH_OBS // int(k))
        for b0 in range(0, len(sel_all), step):
            sel = sel_all[b0:b0 + step]
            idx = starts[sel][:, None] + np.arange(int(k))[None, :]
            for j in range(int(k)):
                rest = np.concatenate([np.arange(j), np.arange(j + 1, int(k))])
                X, good = _dlt_stack(xn[idx[:, rest]], w2c[ii[idx[:, rest]]], float(min_tri_deg))
                if not good.any():
                    continue
                i_left = ii[idx[good, j]]
                pred = project_obs(X[good], w2c[i_left], np.asarray(params, np.float64))
                obs = uv[idx[good, j]]
                C = np.linalg.inv(w2c[i_left])[:, :3, 3]
                out["frame"].append(i_left)
                out["res"].append(np.linalg.norm(pred - obs, axis=1))
                out["u"].append(obs[:, 0]); out["v"].append(obs[:, 1])
                out["dist"].append(np.linalg.norm(X[good] - C, axis=1))
                out["len"].append(np.full(int(good.sum()), int(k)))
                out["key"].append(sel[good] * int(k) + j)
    return {k: np.concatenate(v) if v else np.zeros(0) for k, v in out.items()}


def per_frame_rms(obs: dict, max_px: float) -> Dict[int, float]:
    """F5's per-keyframe reduction: per frame, in the loop's order, the RMS of the finite residuals ≤ max_px."""
    if not len(obs["frame"]):
        return {}
    order = np.lexsort((obs["key"], obs["frame"]))
    fr, res = obs["frame"][order], obs["res"][order]
    bounds = np.r_[0, np.flatnonzero(np.diff(fr)) + 1, len(fr)]
    out = {}
    for a, b in zip(bounds[:-1], bounds[1:]):
        v = res[a:b]
        v = v[np.isfinite(v) & (v >= 0.0) & (v <= max_px)]
        if v.size:
            out[int(fr[a])] = float(np.sqrt(np.mean(np.square(v))))
    return out


def diagnose(session_dir: Path, log=print) -> dict:
    from precision.camera import load_camera_json, undistort_solver
    from precision.config import load_precision_config
    from precision.refine import group_tracks, heldout_bound, split_of_observations
    from precision.tracks import load_tracks_v2
    S = Path(session_dir)
    out = S / "output"
    try:
        pcfg = load_precision_config(session_dir=S, log=lambda *a: None)
    except Exception as exc:          # a run frozen before a key existed: the server's configuration
        log(f"[tail] frozen configuration refused ({exc}); the server's config.yaml")
        pcfg = load_precision_config(log=lambda *a: None)
    cfg = pcfg.refine
    ref = json.loads((out / "precision" / "refine.json").read_text())
    cam = load_camera_json(out / "camera.json")
    params = list(cam.params)
    solver = undistort_solver(pcfg.camera)
    poses = np.loadtxt(out / "camera_poses.txt").reshape(-1, 4, 4)
    kf = [int(float(x)) for x in (out / "camera_frames.txt").read_text().split()]
    w2c = np.linalg.inv(poses)
    tr = load_tracks_v2(S)
    split = split_of_observations(tr["track_query_id"], tr["track_split"], tr["obs_track"])
    h = split == 1
    held_g = group_tracks(tr["obs_track"][h], tr["obs_frame"][h], tr["obs_uv_native"][h], kf)
    max_px = float(np.hypot(cam.width, cam.height))
    obs = per_observation_lovo(held_g, w2c, params, solver, cfg.min_tri_deg)
    pf = per_frame_rms(obs, max_px)
    bound = heldout_bound(pf, cfg)
    rec = float(ref["keyframe_heldout_bound_px"])
    rep: dict = {"session": str(S), "keyframes": len(kf), "heldout_observations": int(len(obs["res"])),
                 "bound_recomputed_px": bound, "bound_recorded_px": rec,
                 "reproduces_f5": bool(abs(bound - rec) <= 1e-6 * max(1.0, rec))}
    v = np.array(list(pf.values())); fi = np.array(list(pf.keys()))
    q = float(cfg.heldout_confidence)
    rep["per_keyframe_rms_px"] = {"p50": float(np.median(v)), "p90": float(np.percentile(v, 90)),
                                  "p95": float(np.percentile(v, 95)), "max": float(v.max()),
                                  "quantile_of_the_bound": q}
    thr = float(np.percentile(v, 100.0 * q))
    tail = fi[v > thr]
    rep["tail_threshold_px"] = thr
    # cube owner, walk position, turn rate and speed per keyframe
    owner = None
    cp = S / "intake" / "cubes.json"
    if cp.exists():
        owner = np.asarray(json.loads(cp.read_text()).get("owner") or [], int)
        if len(owner) != len(kf):
            owner = None
    C = poses[:, :3, 3]
    N = len(kf)
    ang = np.zeros(N); spd = np.zeros(N)
    for i in range(N):
        a, b = max(0, i - 1), min(N - 1, i + 1)
        if a == b:
            continue
        R = w2c[b, :3, :3] @ w2c[a, :3, :3].T
        ang[i] = math.degrees(math.acos(float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))) / (b - a)
        spd[i] = float(np.linalg.norm(C[b] - C[a])) / (b - a)
    n_obs = np.bincount(obs["frame"].astype(np.int64), minlength=N)

    def _cmp(name, arr):
        a_t, a_r = arr[tail], arr[np.setdiff1d(fi, tail)]
        return {"tail_median": float(np.median(a_t)) if len(a_t) else None,
                "rest_median": float(np.median(a_r)) if len(a_r) else None}
    rep["tail_keyframes"] = [{"index": int(i), "frame": int(kf[i]), "rms_px": round(pf[int(i)], 2),
                              "cube": (int(owner[i]) if owner is not None else None),
                              "turn_deg_per_kf": round(float(ang[i]), 2), "step_m_per_kf": round(float(spd[i]), 3),
                              "heldout_obs": int(n_obs[i])}
                             for i in sorted(tail, key=lambda i: -pf[int(i)])]
    rep["tail_vs_rest"] = {"turn_deg_per_kf": _cmp("turn", ang), "step_m_per_kf": _cmp("step", spd),
                           "heldout_obs": _cmp("obs", n_obs.astype(float))}
    if owner is not None:
        by = {}
        for c in sorted(set(owner.tolist())):
            m = np.isin(fi, np.flatnonzero(owner == c))
            if m.any():
                by[str(c)] = {"keyframes": int(m.sum()), "median_rms_px": round(float(np.median(v[m])), 3),
                              "in_tail": int(np.isin(fi[m], tail).sum())}
        rep["by_cube"] = by
    # where in the image and at what distance the large residuals sit (observations of every keyframe)
    res = obs["res"]
    big = res > thr
    r_img = np.hypot(obs["u"] - params[2], obs["v"] - params[3]) / np.hypot(cam.width / 2, cam.height / 2)

    def _bins(x, edges, label):
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (x >= lo) & (x < hi)
            if m.sum():
                rows.append({label: f"{lo:g}-{hi:g}", "observations": int(m.sum()),
                             "median_px": round(float(np.median(res[m])), 3),
                             "share_beyond_tail_px": round(float(big[m].mean()), 4)})
        return rows
    rep["by_image_radius"] = _bins(r_img, [0, 0.25, 0.5, 0.75, 1.0, 2.0], "radius_of_half_diagonal")
    rep["by_image_row"] = _bins(obs["v"] / cam.height, [0, 0.2, 0.4, 0.6, 0.8, 1.01], "row_of_height")
    rep["by_landmark_distance_m"] = _bins(obs["dist"], [0, 2, 4, 6, 8, 10, 15, 1e9], "metres")
    rep["by_track_length"] = _bins(obs["len"].astype(float), [3, 4, 6, 10, 20, 1e9], "views")
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--session", required=True)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    rep = diagnose(Path(a.session))
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
