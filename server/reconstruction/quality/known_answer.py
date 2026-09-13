"""§10.10 known-answer experiment and §10.11 collapse experiment.

Known answer: on a COPY of a session, one reconstruction chunk's keyframes
receive a known rigid (yaw + translation) and scale perturbation — applied
exactly the way an epoch would (cloud along the rays, poses, depth sidecar)
— then the certification loop runs and the recovery error is measured
against the unperturbed geometry. The instrument is verified in the field
(a real session) or on the bench (a synthetic one) by the same code.

Envelope: the perturbation grows level by level until the loop's gates fail
or the recovery leaves tolerance; the last recovered level is the session's
DECLARED operating envelope (max correctable drift in m and %, at each loop
density). Industrial grade is knowing when the system is out of
specification — and saying so.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


def _yaw(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def copy_session(session_dir, work: Optional[Path] = None) -> Path:
    """A working copy of the session (output/ + frames/ metadata)."""
    session_dir = Path(session_dir)
    work = Path(work) if work else Path(tempfile.mkdtemp(prefix="stac_ka_"))
    dst = work / session_dir.name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(session_dir, dst, symlinks=True)
    return dst


def perturb_chunk(output_dir, chunk: str, yaw_deg: float, t_m: float, scale: float,
                  log: Callable = print) -> dict:
    """Perturb the keyframes of one chunk in place ("first" | "middle" |
    "last" | an index): rigid yaw about the chunk's first camera + translation
    along its walking direction, scale about each keyframe's camera (depth ×
    s, chain-continuous shift), through the same per-keyframe machinery an
    epoch uses (warp_full_cloud / transform_poses / the depth sidecar).
    Returns the injected per-keyframe transform (for the recovery check)."""
    from correction.session import load_session, write_ply, write_poses
    from correction.apply import warp_full_cloud, transform_poses
    from reconstruction.certify.scale_stage import chunk_of_keyframes, scale_transforms
    from segmentation.session_io import DEPTH_CORRECTION_NAME, load_depth_affine
    output_dir = Path(output_dir)
    session = load_session(output_dir)
    N = session.n_kf
    ranges, owner = chunk_of_keyframes(output_dir, N)
    if chunk == "first":
        k = 0
    elif chunk == "last":
        k = len(ranges) - 1
    elif chunk == "middle":
        k = len(ranges) // 2
    else:
        k = int(chunk)
    sel = np.flatnonzero(owner == k)
    a, b = ranges[k]
    c0 = session.poses[int(a)][:3, 3]
    fwd = session.poses[min(int(b) - 1, N - 1)][:3, 3] - c0
    fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
    R = _yaw(yaw_deg)
    # scale first (depth about each camera + chain-continuous shift), then the
    # rigid about the chunk's first camera: p' = R (p_s − c0) + c0 + t·fwd — the
    # chunk AND everything the walk wrote after it (a seam error propagates)
    r = np.ones(len(ranges)); r[k] = float(scale)
    k_kf, t_s = scale_transforms(session, ranges, owner, r)
    R_kf = np.tile(np.eye(3), (N, 1, 1)); t_kf = t_s.copy()
    for g in np.flatnonzero(owner >= k):
        R_kf[g] = R
        t_kf[g] = R @ t_s[g] + (c0 - R @ c0) + t_m * fwd
    xyz, _ = warp_full_cloud(session, R_kf, t_kf, k_kf, log=lambda m: None)
    data = session.data.copy()
    data["x"] = xyz[:, 0].astype(data.dtype["x"]); data["y"] = xyz[:, 1].astype(data.dtype["y"])
    data["z"] = xyz[:, 2].astype(data.dtype["z"])
    write_ply(output_dir / "cleaned_cloud.ply", session.header, data)
    if session.raw_data is not None:
        raw_xyz = np.stack([session.raw_data["x"], session.raw_data["y"], session.raw_data["z"]], 1).astype(np.float64)
        import dataclasses
        raw_sess = dataclasses.replace(session, data=session.raw_data, xyz=raw_xyz,
                                       fg=session.raw_data["frame_global"].astype(np.int64))
        rxyz, _ = warp_full_cloud(raw_sess, R_kf, t_kf, k_kf, log=lambda m: None)
        rd = session.raw_data.copy()
        rd["x"] = rxyz[:, 0].astype(rd.dtype["x"]); rd["y"] = rxyz[:, 1].astype(rd.dtype["y"])
        rd["z"] = rxyz[:, 2].astype(rd.dtype["z"])
        write_ply(output_dir / "cleaned_cloud_raw.ply", session.raw_header, rd)
    write_poses(output_dir / "camera_poses.txt", transform_poses(session.poses, R_kf, t_kf))
    old = load_depth_affine(output_dir) or {}
    kb = dict(old)
    for g in sel:
        f = session.frames[g]
        k0, b0 = kb.get(f, (1.0, 0.0))
        kb[f] = (k0 * float(scale), float(scale) * b0)
    (output_dir / DEPTH_CORRECTION_NAME).write_text(json.dumps(
        {"version": 2, "epoch": 0, "k": {str(f): v[0] for f, v in sorted(kb.items())},
         "b": {str(f): v[1] for f, v in sorted(kb.items())}}, indent=1))
    # the DA3 anchors are independent of the cloud: their agreement with the
    # (now scaled) depth changes by 1/s on the perturbed keyframes
    from correction.diagnose import regenerate_scale_diagnostics
    diag = regenerate_scale_diagnostics(output_dir, {int(session.frames[g]): float(scale) for g in sel},
                                        0, "known_answer_injection")
    if diag is not None:
        (output_dir / "scale_diagnostics.json").write_text(json.dumps(diag, indent=1))
    log(f"[known-answer] chunk {k} ({len(sel)} keyframes) perturbed: yaw {yaw_deg}°, t {t_m} m, "
        f"scale {scale}")
    return {"chunk": int(k), "keyframes": [int(g) for g in np.flatnonzero(owner >= k)],
            "chunk_keyframes": [int(g) for g in sel], "yaw_deg": float(yaw_deg),
            "t_m": float(t_m), "scale": float(scale), "R_kf": R_kf, "t_kf": t_kf, "k_kf": k_kf,
            "poses_true": session.poses.copy()}


def recovery_error(output_dir, injected: dict) -> dict:
    """Per-keyframe error of the CURRENT poses vs the unperturbed ones over
    the perturbed chunk (rotation deg, translation m) and the residual scale
    (sidecar k of the chunk's keyframes × the injected scale)."""
    from correction.session import read_poses
    from segmentation.session_io import load_depth_affine
    output_dir = Path(output_dir)
    poses = read_poses(output_dir / "camera_poses.txt")
    true = injected["poses_true"]
    errs_t, errs_r = [], []
    for g in injected["keyframes"]:
        dT = np.linalg.inv(true[g]) @ poses[g]
        errs_t.append(float(np.linalg.norm(dT[:3, 3])))
        c = np.clip((np.trace(dT[:3, :3]) - 1) / 2, -1, 1)
        errs_r.append(float(np.degrees(np.arccos(c))))
    kb = load_depth_affine(output_dir) or {}
    from correction.session import load_session
    session = load_session(output_dir)
    ks = [kb.get(session.frames[g], (1.0, 0.0))[0] for g in injected["chunk_keyframes"]]
    return {"t_m_median": float(np.median(errs_t)), "t_m_max": float(np.max(errs_t)),
            "deg_median": float(np.median(errs_r)), "deg_max": float(np.max(errs_r)),
            "scale_residual": float(np.median(ks)) if ks else 1.0}


def known_answer(session_dir, cfg=None, correction_cfg=None, chunk: Optional[str] = None,
                 perturb: Optional[Tuple[float, float, float]] = None, work: Optional[Path] = None,
                 log: Callable = print, **certify_kw) -> dict:
    """Run the experiment on a copy of the session. Returns the report."""
    from reconstruction.loops.config import load_loops_config
    from reconstruction.certify.run import certify_session
    cfg = cfg or load_loops_config()
    ka = cfg.certify.known_answer
    chunk = chunk if chunk is not None else ka.chunk
    yaw, t_m, s = perturb if perturb else (ka.yaw_deg, ka.t_m, ka.scale)
    t0 = time.time()
    copy = copy_session(session_dir, work)
    out = copy / "output"
    inj = perturb_chunk(out, chunk, yaw, t_m, s, log=log)
    before = recovery_error(out, inj)
    acta = certify_session(copy, cfg, operator="known_answer", log=log, correction_cfg=correction_cfg,
                           **certify_kw)
    after = recovery_error(out, inj)
    within = (after["t_m_max"] <= ka.tol_t_m and after["deg_max"] <= ka.tol_deg
              and abs(after["scale_residual"] - 1.0) <= ka.tol_scale)
    rep = {"version": 1, "chunk": inj["chunk"], "n_keyframes": len(inj["keyframes"]),
           "injected": {"yaw_deg": yaw, "t_m": t_m, "scale": s},
           "error_before": before, "error_after": after,
           "tolerance": {"t_m": ka.tol_t_m, "deg": ka.tol_deg, "scale": ka.tol_scale},
           "recovered_within_tolerance": bool(within),
           "loop": {"stopped_at": acta.get("stopped_at"), "stop_reason": acta.get("stop_reason"),
                    "epoch_final": acta.get("epoch_final"),
                    "iterations": [{"verdict": it.get("verdict"), "reason": it.get("reason"),
                                    "objective": it.get("objective")} for it in acta.get("iterations", [])]},
           "work_dir": str(copy), "elapsed_s": round(time.time() - t0, 1), "provenance": "tool_measured"}
    log(f"[known-answer] t {before['t_m_max'] * 100:.1f} → {after['t_m_max'] * 100:.1f} cm, rot "
        f"{before['deg_max']:.2f} → {after['deg_max']:.2f}°, scale {before['scale_residual']:.4f} → "
        f"{after['scale_residual']:.4f} → {'RECOVERED' if within else 'NOT within tolerance'}")
    return rep


def envelope(session_dir, cfg=None, correction_cfg=None, chunk: Optional[str] = None,
             work: Optional[Path] = None, log: Callable = print, **certify_kw) -> dict:
    """Increase the injected perturbation level by level (translation, then
    scale) at each loop density until the loop fails or recovery leaves
    tolerance; the last recovered level is the declared envelope."""
    from reconstruction.loops.config import load_loops_config
    cfg = cfg or load_loops_config()
    env = cfg.certify.envelope
    ka = cfg.certify.known_answer
    t0 = time.time()
    results = {}
    for density in env.loop_densities:
        rows_t, rows_s = [], []
        last_t = None
        for lvl in env.levels_t_m:
            rep = known_answer(session_dir, cfg, correction_cfg, chunk, (ka.yaw_deg, float(lvl), 1.0),
                               work, log, loop_density=float(density), **certify_kw)
            rows_t.append({"level_t_m": float(lvl), "recovered": rep["recovered_within_tolerance"],
                           "error_after": rep["error_after"], "stop_reason": rep["loop"]["stop_reason"]})
            if not rep["recovered_within_tolerance"]:
                break
            last_t = float(lvl)
        last_s = None
        for pct in env.levels_scale_pct:
            rep = known_answer(session_dir, cfg, correction_cfg, chunk, (0.0, 0.0, 1.0 + float(pct) / 100.0),
                               work, log, loop_density=float(density), **certify_kw)
            rows_s.append({"level_scale_pct": float(pct), "recovered": rep["recovered_within_tolerance"],
                           "error_after": rep["error_after"], "stop_reason": rep["loop"]["stop_reason"]})
            if not rep["recovered_within_tolerance"]:
                break
            last_s = float(pct)
        results[str(density)] = {"loop_density": float(density), "translation": rows_t, "scale": rows_s,
                                 "max_correctable_t_m": last_t, "max_correctable_scale_pct": last_s}
        log(f"[envelope] loop density {density}: max correctable drift {last_t} m, scale {last_s} %")
    dens = sorted(results, key=float)
    mono_t = all((results[dens[i]]["max_correctable_t_m"] or 0.0) <= (results[dens[i + 1]]["max_correctable_t_m"] or 0.0)
                 for i in range(len(dens) - 1))
    mono_s = all((results[dens[i]]["max_correctable_scale_pct"] or 0.0) <= (results[dens[i + 1]]["max_correctable_scale_pct"] or 0.0)
                 for i in range(len(dens) - 1))
    return {"version": 1, "per_density": results, "monotone_with_loop_density": bool(mono_t and mono_s),
            "declared": {"max_correctable_t_m": results[dens[-1]]["max_correctable_t_m"] if dens else None,
                         "max_correctable_scale_pct": results[dens[-1]]["max_correctable_scale_pct"] if dens else None,
                         "min_loop_density_recovering": next((float(d) for d in dens
                                                              if results[d]["max_correctable_t_m"]), None)},
            "elapsed_s": round(time.time() - t0, 1), "provenance": "tool_measured"}
