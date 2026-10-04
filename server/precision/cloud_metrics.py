"""Cloud quality metrics of every run — the floor metric and the crease-profile edge metric as
automatic reports (pending A "run checks" of docs/pipeline_final.md, closed 2026-10-04).

FLOOR (ported from analysis/2026-09-30_depth_sources/floor_metric.py, the judge of epochs 4-8; every
number it used now lives in ``precision.cloud_metrics``): up = minus the mean camera y-axis (the
cameras are held upright); global floor = RANSAC plane among the lowest points (normal within
``floor_max_tilt_deg`` of up, ``camera_min_height_m`` under at least ``camera_above_frac`` of the
cameras), refined on its inliers; band = points within ``band_m`` of it; per ``cell_m`` cell with
at least ``cell_min_points`` band points: the dominant layer (histogram mode at ``mode_bin_m``),
a local plane on the points within ``mode_band_m`` of it (at least ``plane_min_points``),
  thickness   = p90 − p10 of the residuals to the cell's own plane   (layers INSIDE the cell)
  height      = the cell plane's offset at the cell centre vs the global plane
  undulation  = p90 − p10 of the cell heights after the tilt is fitted out   (waves BETWEEN cells)
pccr epoch 7 reference: thickness 5.2 cm, undulation 7.6 cm.

EDGES: ``precision/edge_metric.py`` (the crease profile per adjacent plane pair of every segmented
object) on the certified cloud + its segmentation, bounded by ``edge_max_objects`` (BOUND: cost; the
largest objects first) — it needs the projection, so it runs in the certify stage only.

Written to ``output/precision/cloud_metrics.json`` (stamped with the geometry epoch and the stage
that measured) and folded into chunk_check.json / the acta. Provenance ``tool_measured``; no verdict.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

LOG_TAG = "[cloud-metrics]"
REPORT = "cloud_metrics.json"
PROVENANCE = "tool_measured"


class CloudMetricsError(RuntimeError):
    pass


def floor_metric(X: np.ndarray, c2w: np.ndarray, cfg, seed: int = 0) -> dict:
    """The floor metric on points ``X`` (N x 3) with camera poses ``c2w`` (M x 4 x 4). Returns the
    global plane, the camera heights and the per-cell statistics; raises when no floor plane
    satisfies the declared conditions."""
    rng = np.random.default_rng(int(seed))
    X = np.asarray(X, np.float64)
    if len(X) > int(cfg.max_points):
        X = X[rng.choice(len(X), int(cfg.max_points), replace=False)]
    P = np.asarray(c2w, np.float64).reshape(-1, 4, 4)
    up = -P[:, :3, 1].mean(0); up /= max(np.linalg.norm(up), 1e-12)
    C = P[:, :3, 3]
    cos_max = np.cos(np.radians(float(cfg.floor_max_tilt_deg)))
    inl_m = float(cfg.inlier_m)
    best, bn, bd = 0, None, None
    sub = X[::int(cfg.score_stride)]
    for _ in range(int(cfg.ransac_iterations)):
        smp = X[rng.choice(len(X), 3, replace=False)]
        n = np.cross(smp[1] - smp[0], smp[2] - smp[0]); nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        if n @ up < 0:
            n = -n
        if n @ up < cos_max:
            continue
        d = -n @ smp[0]
        if np.mean(C @ n + d > float(cfg.camera_min_height_m)) < float(cfg.camera_above_frac):
            continue
        c = int((np.abs(sub @ n + d) < inl_m).sum())
        if c > best:
            best, bn, bd = c, n, d
    if bn is None:
        raise CloudMetricsError("no plane within the declared tilt with the cameras above it — no floor measured")
    inl = X[np.abs(X @ bn + bd) < inl_m]
    ctr = inl.mean(0); _, _, vt = np.linalg.svd(inl - ctr, full_matrices=False)
    bn = vt[2] if vt[2] @ up > 0 else -vt[2]; bd = -bn @ ctr
    cam_h = C @ bn + bd
    dist = X @ bn + bd
    B = X[np.abs(dist) < float(cfg.band_m)]; bdist = B @ bn + bd
    e1 = np.cross(bn, np.eye(3)[int(np.argmin(np.abs(bn)))]); e1 /= np.linalg.norm(e1); e2 = np.cross(bn, e1)
    cell = float(cfg.cell_m)
    uv = np.stack([B @ e1, B @ e2], 1) / cell
    ci = np.floor(uv).astype(np.int64)
    key = ci[:, 0] * 100000 + ci[:, 1]
    order = np.argsort(key, kind="stable"); key, uv, bdist = key[order], uv[order], bdist[order]
    starts = np.r_[0, np.nonzero(np.diff(key))[0] + 1, len(key)]
    half = float(cfg.band_m); bin_m = float(cfg.mode_bin_m)
    bins = np.arange(-half, half + bin_m / 2, bin_m)
    th, ht, cuv = [], [], []
    for a, b in zip(starts[:-1], starts[1:]):
        if b - a < int(cfg.cell_min_points):
            continue
        hist, _ = np.histogram(bdist[a:b], bins)
        hist = np.convolve(hist, np.ones(3), "same")
        mode = bins[int(np.argmax(hist))] + bin_m / 2            # the cell's dominant floor layer
        m = np.abs(bdist[a:b] - mode) < float(cfg.mode_band_m)
        if m.sum() < int(cfg.plane_min_points):
            continue
        A = np.c_[uv[a:b][m], np.ones(int(m.sum()))]
        coef = np.linalg.lstsq(A, bdist[a:b][m], rcond=None)[0]
        res = bdist[a:b][m] - A @ coef
        th.append(float(np.percentile(res, 90) - np.percentile(res, 10)))
        ht.append(float(mode)); cuv.append(np.floor(uv[a]) + 0.5)
    th = np.array(th); ht = np.array(ht)
    if len(th) == 0:
        raise CloudMetricsError("no floor cell holds enough band points — the floor cannot be measured")
    cu = np.array(cuv)
    A = np.c_[cu, np.ones(len(cu))]
    ht_flat = ht - A @ np.linalg.lstsq(A, ht, rcond=None)[0]      # tilt out: undulation only
    return {"plane": {"normal": bn.tolist(), "d": float(bd), "tilt_vs_camera_up_deg":
                      float(np.degrees(np.arccos(np.clip(bn @ up, -1, 1))))},
            "cameras_above_floor_m": {"median": float(np.median(cam_h)), "p10": float(np.percentile(cam_h, 10)),
                                      "p90": float(np.percentile(cam_h, 90))},
            "cells": int(len(th)), "thickness_m": {"median": float(np.median(th)), "p90": float(np.percentile(th, 90))},
            "undulation_m": float(np.percentile(ht_flat, 90) - np.percentile(ht_flat, 10)),
            "band_share": float(len(B) / max(len(X), 1)), "points_measured": int(len(X))}


def _epoch(out: Path) -> Optional[int]:
    p = out / "geometry_epoch.json"
    try:
        return int(json.loads(p.read_text()).get("epoch"))
    except (OSError, ValueError, TypeError):
        return None


def run_cloud_metrics(session_dir: Path, pcfg, stage: str, log: Callable = print,
                      edges: bool = False) -> dict:
    """Measure the live cloud of ``session_dir`` (floor always; edges when asked and a segmentation
    exists), write output/precision/cloud_metrics.json, return the report. A metric that cannot be
    measured is declared in the report, never silent, and never fails the caller."""
    from correction.session import read_ply
    from precision.depth_sweep import _read_poses
    t0 = time.time()
    session_dir = Path(session_dir); out = session_dir / "output"
    cfg = pcfg.cloud_metrics
    rep: dict = {"version": 1, "provenance": PROVENANCE, "stage": stage, "geometry_epoch": _epoch(out),
                 "measured_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    ply = out / "cleaned_cloud.ply"
    try:
        if not ply.exists():
            raise CloudMetricsError(f"{ply} is missing")
        _, data = read_ply(ply)
        X = np.stack([data["x"], data["y"], data["z"]], 1)
        frames, c2w = _read_poses(out / "camera_poses.txt", out / "camera_frames.txt")
        rep["floor"] = floor_metric(X, c2w, cfg, seed=int(cfg.seed))
        rep["floor"]["points_total"] = int(len(X))
        f = rep["floor"]
        log(f"{LOG_TAG} floor: {f['cells']} cells, thickness median {f['thickness_m']['median'] * 100:.1f} cm "
            f"(p90 {f['thickness_m']['p90'] * 100:.1f}), undulation {f['undulation_m'] * 100:.1f} cm, tilt vs camera-up "
            f"{f['plane']['tilt_vs_camera_up_deg']:.1f}°, cameras {f['cameras_above_floor_m']['median']:.2f} m above")
    except Exception as e:  # noqa: BLE001 — declared, never the caller's failure
        rep["floor"] = {"error": f"{type(e).__name__}: {e}"}
        log(f"{LOG_TAG} floor not measured: {e}")
    if edges:
        seg = out / "segmentation_result.json"
        try:
            if not seg.exists():
                raise CloudMetricsError("no segmentation_result.json — the edge metric needs the projected objects")
            from precision.edge_metric import run as edge_run
            doc = json.loads(seg.read_text())
            inst = sorted(doc.get("instances", []), key=lambda i: -int(i.get("total_points") or len(i.get("globalIndices") or [])))
            ids = [int(i.get("instance_id", i.get("id"))) for i in inst[:int(cfg.edge_max_objects)]]
            er = edge_run(out, ply, seg, out_path=out / "precision" / "edge_metric.json", instance_ids=ids, log=log)
            objs = er.get("reference", {}).get("objects", [])
            summ = [o["summary"] for o in objs if o.get("summary") and o["summary"].get("n_creases")]
            rep["edges"] = {"objects_measured": len(objs), "objects_with_creases": len(summ),
                            "objects_bound": int(cfg.edge_max_objects), "seconds": er.get("seconds"),
                            "r_hat_m_median": (float(np.median([s["r_hat_m"] for s in summ if s.get("r_hat_m") is not None]))
                                               if any(s.get("r_hat_m") is not None for s in summ) else None),
                            "report": str(out / "precision" / "edge_metric.json")}
            log(f"{LOG_TAG} edges: {len(summ)} object(s) with creases of {len(objs)} measured, r̂ median "
                f"{(rep['edges']['r_hat_m_median'] or 0) * 1000:.1f} mm ({er.get('seconds')} s)")
        except Exception as e:  # noqa: BLE001
            rep["edges"] = {"error": f"{type(e).__name__}: {e}"}
            log(f"{LOG_TAG} edges not measured: {e}")
    rep["seconds"] = round(time.time() - t0, 1)
    pdir = out / "precision"; pdir.mkdir(parents=True, exist_ok=True)
    (pdir / REPORT).write_text(json.dumps(rep, indent=1, default=float))
    return rep
