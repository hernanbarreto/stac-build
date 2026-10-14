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


def cloud_floor_levels(out: Path, fields: dict, frames, c2w: np.ndarray, raw_cfg: dict,
                       max_points: int) -> dict:
    """The floor of every Omega chunk of the published cloud — ``correction.floor.
    chunk_floor_levels``, the certification's own choice (USER 2026-10-08: one floor concept for
    the certification, the chunk check and this metric) — on the deterministic subsample of at
    most ``max_points`` points (:func:`stable_point_sample`), with the surface each chunk's floor
    is (``correction.floor.chosen_surfaces``). Returns {levels, surfaces, X, ks, ck}; raises
    CloudMetricsError when the cloud carries no keyframe provenance."""
    from correction.config import judge_of, load_correction_config
    from correction.floor import (chosen_surfaces, chunk_floor_levels, omega_chunk_of_frames,
                                  repeatability_of)
    if "frame_global" not in fields:
        raise CloudMetricsError("the cloud carries no frame_global — the floor of each chunk "
                                "cannot be measured")
    fr = np.asarray(list(frames), np.int64)
    order = np.argsort(fr, kind="stable")
    fg = np.asarray(fields["frame_global"], np.int64)
    at = np.clip(np.searchsorted(fr[order], fg), 0, len(fr) - 1)
    ks_all = np.where(fr[order][at] == fg, order[at], -1)
    rows = stable_point_sample(fields, np.flatnonzero(ks_all >= 0), int(max_points))
    X = np.stack([np.asarray(fields[k], np.float64)[rows] for k in ("x", "y", "z")], 1)
    ks = ks_all[rows]
    ck = omega_chunk_of_frames(out, fr.tolist())
    cams = np.asarray(c2w, np.float64)[:, :3, 3]
    ccfg = load_correction_config(raw_cfg)
    fac, conf = judge_of(ccfg)
    levels = chunk_floor_levels(X, ks, cams, ck, ccfg.floor, repeatability_of(out, ccfg), fac, conf)
    return {"levels": levels, "surfaces": chosen_surfaces(levels, cams), "X": X, "ks": ks, "ck": ck}


def floor_by_surface(X: np.ndarray, c2w: np.ndarray, cfg, fl: Optional[dict]) -> dict:
    """The floor metric per FLOOR SURFACE (USER 2026-10-08: a session can hold several floors —
    a platform and the track bed below it — and each is measured against its own plane, never the
    other one's): one surface, or no chunk floors measured (``fl`` None) → :func:`floor_metric`
    on the whole cloud, exactly as before; several → :func:`floor_metric` on each surface's points
    (every point of its chunks within ``band_m`` of that chunk's floor, the cameras of those
    chunks), the surface with the most cells reported at the top level and every surface under
    ``floor_surfaces``."""
    surfaces = sorted(set((fl or {}).get("surfaces", {}).values()))
    if len(surfaces) <= 1:
        res = floor_metric(X, c2w, cfg, seed=int(cfg.seed))
        res["floor_surfaces"] = ({"n": len(surfaces), "chunks": {str(c): s for c, s in fl["surfaces"].items()}}
                                 if fl is not None else {"n": None, "reason": "the chunk floors were not measured"})
        return res
    lv, names = fl["levels"], fl["surfaces"]
    per: dict = {}
    for s in surfaces:
        chunks_s = sorted(c for c, nm in names.items() if nm == s)
        sel = np.zeros(len(fl["X"]), bool)
        for c in chunks_s:
            q = lv["chosen"][c]
            rows = np.flatnonzero(np.isin(fl["ks"], np.flatnonzero(fl["ck"] == c)))
            near = np.abs((fl["X"][rows] - q["centroid"]) @ q["normal"]) < float(cfg.band_m)
            sel[rows[near]] = True
        kfs = np.flatnonzero(np.isin(fl["ck"], chunks_s))
        try:
            per[s] = floor_metric(fl["X"][sel], np.asarray(c2w)[kfs], cfg, seed=int(cfg.seed))
        except CloudMetricsError as e:
            per[s] = {"error": str(e)}
        per[s]["chunks"] = [int(c) for c in chunks_s]
    measured = [s for s in surfaces if "cells" in per[s]]
    if not measured:
        raise CloudMetricsError("no floor surface could be measured: " +
                                "; ".join(f"{s}: {per[s]['error']}" for s in surfaces))
    top = max(measured, key=lambda s: (per[s]["cells"], -surfaces.index(s)))
    res = {k: v for k, v in per[top].items() if k != "chunks"}
    res["floor_surfaces"] = {"n": len(surfaces), "reported_at_top": top, "per_surface": per,
                             "chunks": {str(c): s for c, s in names.items()}}
    return res


EDGE_REPORT = "edge_metric.json"
EDGE_OBJECTS = "edge_metric_objects.json"      # the sampled instances handed to the subprocess (transient)
_KEY_FIELDS = ("frame_global", "pixel_row", "pixel_col")


def select_edge_objects(instances: list, n_max: int) -> list:
    """The ``n_max`` LARGEST instances by point count, ties broken by instance id — a stable rule
    (docs/plan_determinismo.md point 138), not the order a JSON happened to list them in.
    Returns [(n_points, instance_id, instance)]."""
    rows = []
    for inst in instances:
        gi = inst.get("globalIndices") or []
        n = int(inst.get("total_points") or len(gi))
        iid = int(inst.get("instance_id", inst.get("id")))
        rows.append((n, iid, inst))
    rows.sort(key=lambda t: (-t[0], t[1]))
    return rows[:int(n_max)]


def _splitmix64(x: np.ndarray) -> np.ndarray:
    """A fixed 64-bit mix (SplitMix64's finaliser) — a point's rank in the sample never depends
    on its neighbours or on the cloud's size."""
    z = x.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def stable_point_keys(fields: dict, rows: np.ndarray) -> np.ndarray:
    """One int64 per row that names the POINT, not its position: (frame_global, pixel_row,
    pixel_col) when the cloud carries its provenance (the key precision/edge_metric.py pairs
    epochs with), else the float32 bits of x, y, z."""
    rows = np.asarray(rows, np.int64)
    if all(k in fields for k in _KEY_FIELDS):
        fg = np.asarray(fields["frame_global"])[rows].astype(np.int64)
        pr = np.asarray(fields["pixel_row"])[rows].astype(np.int64) & 0xFFFF
        pc = np.asarray(fields["pixel_col"])[rows].astype(np.int64) & 0xFFFF
        return (fg << 32) | (pr << 16) | pc
    bits = [np.ascontiguousarray(np.asarray(fields[k], np.float32)[rows]).view(np.uint32).astype(np.uint64)
            for k in ("x", "y", "z")]
    return ((bits[0] << np.uint64(32)) | bits[1] ^ (bits[2] << np.uint64(16))).astype(np.int64)


def stable_point_sample(fields: dict, rows: np.ndarray, n_max: int) -> np.ndarray:
    """At most ``n_max`` of ``rows``, the ones whose mixed stable key ranks lowest (ties — identical
    points — by row order, stable sort), returned in row order: the same points every run, on
    every machine (point 138; the construction of point 11's stable pixel keys)."""
    rows = np.asarray(rows, np.int64)
    if len(rows) <= int(n_max):
        return rows
    h = _splitmix64(stable_point_keys(fields, rows).view(np.uint64))
    order = np.argsort(h, kind="stable")[:int(n_max)]
    return np.sort(rows[order])


def _edge_metric_subprocess(out: Path, ply: Path, seg: Path, ids, log: Callable) -> dict:
    """precision/edge_metric.py in its own process (pccr 2026-10-04: one object held the certify
    stage for over half an hour in the in-process call). No wall clock bounds it (point 138): the
    work is bounded by the objects and the points per object handed to it; a hung process is the
    cancel path's to kill, never a verdict on the metric. The previous report is deleted first, so
    a failed run never leaves another run's edges on disk. Returns the parsed report."""
    import subprocess
    import sys as _sys
    out_path = out / "precision" / EDGE_REPORT
    if out_path.exists():
        out_path.unlink()
    cmd = [_sys.executable, "-m", "precision.edge_metric", "--session-output", str(out), "--ply", str(ply),
           "--seg", str(seg), "--out", str(out_path), "--instance-ids", *[str(i) for i in ids]]
    server_dir = Path(__file__).resolve().parents[1]
    proc = subprocess.run(cmd, cwd=str(server_dir), capture_output=True, text=True)
    for line in (proc.stdout or "").splitlines():
        if line.startswith("[edge]"):
            log(line)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-3:]
        raise CloudMetricsError(f"edge metric failed (exit {proc.returncode}): {' | '.join(tail)}")
    if not out_path.exists():
        raise CloudMetricsError("edge metric wrote no report")
    rep = json.loads(out_path.read_text())
    secs = rep.pop("seconds", None)                       # the clock leaves the compared report
    out_path.write_text(json.dumps(rep, indent=1, allow_nan=False))
    rep["_seconds"] = secs
    return rep


def edge_work(out: Path, ply: Path, seg: Path, cfg, log: Callable) -> tuple:
    """The WORK the edge metric gets (point 138): the ``edge_max_objects`` largest instances, each
    sampled to ``edge_max_points_per_object`` points by the stable key. Writes the sampled
    instances as ``precision/edge_metric_objects.json`` (what the subprocess measures; removed
    after the run) and returns (that path, the instance ids, the per-object record)."""
    from segmentation.perfect_object import _read_ply_fields
    doc = json.loads(seg.read_text())
    chosen = select_edge_objects(doc.get("instances", []), int(cfg.edge_max_objects))
    if not chosen:
        raise CloudMetricsError("no segmented instance to measure" if int(cfg.edge_max_objects) > 0
                                else "edge_max_objects is 0 — the edge metric is off")
    fields = _read_ply_fields(ply)
    n_rows = len(fields["x"])
    cap = int(cfg.edge_max_points_per_object)
    insts, record = [], []
    for n, iid, inst in chosen:
        gi = np.asarray(inst.get("globalIndices") or [], np.int64)
        gi = gi[(gi >= 0) & (gi < n_rows)]
        kept = stable_point_sample(fields, gi, cap)
        insts.append({"instance_id": iid, "label": inst.get("label", "segment"),
                      "globalIndices": [int(v) for v in kept], "total_points": int(len(gi))})
        record.append({"instance_id": iid, "label": inst.get("label", "segment"), "points": int(len(gi)),
                       "points_measured": int(len(kept)), "sampled": bool(len(kept) < len(gi))})
    p = out / "precision" / EDGE_OBJECTS
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"instances": insts, "rule": "largest edge_max_objects by point count (ties by "
                                                          "instance id); per object the edge_max_points_per_object "
                                                          "points of lowest stable key (point 138)"},
                            separators=(",", ":")))
    log(f"{LOG_TAG} edges: {len(insts)} object(s) of {len(doc.get('instances', []))}, "
        f"{sum(r['points_measured'] for r in record):,} points measured "
        f"({sum(1 for r in record if r['sampled'])} sampled to {cap:,}; keyed by "
        f"{'frame/pixel' if all(k in fields for k in _KEY_FIELDS) else 'xyz bits'})")
    return p, [iid for _, iid, _ in chosen], record


def _epoch(out: Path) -> Optional[int]:
    p = out / "geometry_epoch.json"
    try:
        return int(json.loads(p.read_text()).get("epoch"))
    except (OSError, ValueError, TypeError):
        return None


def run_cloud_metrics(session_dir: Path, pcfg, stage: str, log: Callable = print,
                      edges: bool = False, floor_levels: Optional[dict] = None) -> dict:
    """Measure the live cloud of ``session_dir`` (floor always; edges when asked and a segmentation
    exists), write output/precision/cloud_metrics.json, return the report. A metric that cannot be
    measured is declared in the report, never silent, and never fails the caller. ``floor_levels``:
    the chunk floors of :func:`cloud_floor_levels` when the caller measured them already."""
    from correction.epoch import RECONSTRUCTION_ID_KEY, reconstruction_id_or_none
    from correction.session import read_ply
    from precision.corrected_cloud import write_timing
    from precision.depth_sweep import _read_poses
    t0 = time.time()
    session_dir = Path(session_dir); out = session_dir / "output"
    cfg = pcfg.cloud_metrics
    # no clock in the report (point 36 / 56): when it was measured goes to the timing sidecar;
    # the reconstruction id names the geometry independently of the epoch history (point 63)
    rep: dict = {"version": 1, "provenance": PROVENANCE, "stage": stage, "geometry_epoch": _epoch(out),
                 RECONSTRUCTION_ID_KEY: reconstruction_id_or_none(out)}
    times: dict = {"stage": stage, "measured_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    ply = out / "cleaned_cloud.ply"
    try:
        if not ply.exists():
            raise CloudMetricsError(f"{ply} is missing")
        _, data = read_ply(ply)
        X = np.stack([data["x"], data["y"], data["z"]], 1)
        frames, c2w = _read_poses(out / "camera_poses.txt", out / "camera_frames.txt")
        fl = floor_levels
        if fl is None:
            try:
                from precision.config import raw_of
                fl = cloud_floor_levels(out, {k: data[k] for k in data.dtype.names}, frames, c2w,
                                        raw_of(pcfg), int(cfg.max_points))   # the job's config (point 69)
            except (RuntimeError, OSError) as e:
                # the measurement's own errors (a session that cannot be measured: no keyframe
                # provenance, no Omega record, no floor plane, a file missing) are declared and
                # ONE plane for the session stands in, as before; a TypeError / KeyError /
                # IndexError is a bug and reaches the outer catch, which NAMES its type in the
                # report instead of measuring a fallback over it
                log(f"{LOG_TAG} the floor of each chunk was not measured ({type(e).__name__}: {e}) "
                    f"— one floor plane for the session, DECLARED")
        rep["floor"] = floor_by_surface(X, c2w, cfg, fl)
        rep["floor"]["points_total"] = int(len(X))
        f = rep["floor"]
        log(f"{LOG_TAG} floor: {f['cells']} cells, thickness median {f['thickness_m']['median'] * 100:.1f} cm "
            f"(p90 {f['thickness_m']['p90'] * 100:.1f}), undulation {f['undulation_m'] * 100:.1f} cm, tilt vs camera-up "
            f"{f['plane']['tilt_vs_camera_up_deg']:.1f}°, cameras {f['cameras_above_floor_m']['median']:.2f} m above"
            + (f"; surface {f['floor_surfaces']['reported_at_top']} of {f['floor_surfaces']['n']} floor surfaces, "
               f"each measured on its own plane" if (f.get("floor_surfaces") or {}).get("n", 0) and
               f["floor_surfaces"]["n"] > 1 else ""))
    except Exception as e:  # noqa: BLE001 — declared, never the caller's failure
        rep["floor"] = {"error": f"{type(e).__name__}: {e}"}
        log(f"{LOG_TAG} floor not measured: {e}")
    if edges:
        seg = out / "segmentation_result.json"
        objects_path = None
        try:
            if not seg.exists():
                raise CloudMetricsError("no segmentation_result.json — the edge metric needs the projected objects")
            t_e = time.time()
            objects_path, ids, record = edge_work(out, ply, seg, cfg, log)
            er = _edge_metric_subprocess(out, ply, objects_path, ids, log)
            objs = er.get("reference", {}).get("objects", [])
            summ = [o["summary"] for o in objs if o.get("summary") and o["summary"].get("n_creases")]
            times["edges_seconds"] = er.get("_seconds")
            times["edges_seconds_total"] = round(time.time() - t_e, 1)
            rep["edges"] = {"objects_measured": len(objs), "objects_with_creases": len(summ),
                            "work": {"objects_bound": int(cfg.edge_max_objects),
                                     "points_per_object_bound": int(cfg.edge_max_points_per_object),
                                     "objects": record},
                            "r_hat_m_median": (float(np.median([s["r_hat_m"] for s in summ if s.get("r_hat_m") is not None]))
                                               if any(s.get("r_hat_m") is not None for s in summ) else None),
                            "report": str(out / "precision" / EDGE_REPORT)}
            log(f"{LOG_TAG} edges: {len(summ)} object(s) with creases of {len(objs)} measured, r̂ median "
                f"{(rep['edges']['r_hat_m_median'] or 0) * 1000:.1f} mm ({er.get('_seconds')} s)")
        except Exception as e:  # noqa: BLE001
            rep["edges"] = {"error": f"{type(e).__name__}: {e}"}
            log(f"{LOG_TAG} edges not measured: {e}")
        finally:
            if objects_path is not None and objects_path.exists():
                objects_path.unlink()
    times["seconds"] = round(time.time() - t0, 1)
    pdir = out / "precision"; pdir.mkdir(parents=True, exist_ok=True)
    (pdir / REPORT).write_text(json.dumps(rep, indent=1, default=float))
    write_timing(pdir / REPORT, times)
    return rep
