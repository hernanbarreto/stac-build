"""Fuse scans of a project (USER DESIGN 2026-09-06) — orchestrator (da3 env).

Pairs = segments carrying the SAME label in the reference scan and in the
scan to fuse (the user chose invariant objects on purpose); ≥2 per scan.

Pipeline per fuse run:
  1. `find_pairs`          — labels shared with the reference, point counts.
  2. extract each paired object's points (classification.npy / result
     indices) from both scans → npy files + a held-out sample of the scan
     (points NOT in any pair) and of the reference.
  3. CloudComPy job (`cloudcompy_register.py`): dominant planes, robust size
     ratio, coarse yaw/XZ, ICP with scale → similarity scan→reference.
  4. GUARDS (here): ≥2 usable pairs; plane-normal conditioning; scale within
     bounds; tilt (pitch/roll) small — yaw is free between days; pair
     residuals; HELD-OUT: the scan's unused points must land near the
     reference's geometry (median NN distance under a threshold). Any
     failure → REJECTED with the reason, nothing written.
  5. SYMMETRIC SCALE: s = ICP scale scan→ref. Reference points × 1/√s,
     scan points × √s (rotation/translation kept) — both meet in the middle.
  6. Merged cloud → `scans/fused/src_<stamp>/output/cleaned_cloud.ply` (+
     classification.npy namespaced per scan, segmentation_result.json
     metadata, identity floor transform, `scan` id per point) → Potree.
     Registered in project.json (scans list + composition transforms) so it
     shows in the scans list as a FUSED entry and opens as a tab.
The fused product is read-only (measure, mesh, chat); segmentation and
corrections keep living in each scan.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from project_paths import ProjectPaths
from project_scans import (FUSED_DAY, get_reference, load_meta, set_transform,
                           split_key, sync_scans)

SERVER_DIR = Path(__file__).resolve().parent


def _norm(label: str) -> str:
    return " ".join(str(label).strip().lower().split())


def _instances(output_dir: Path) -> List[dict]:
    p = output_dir / "segmentation_result.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text()).get("instances", [])
    except Exception:  # noqa: BLE001
        return []


def find_pairs(paths: ProjectPaths, scan_keys: List[str]) -> dict:
    ref = get_reference(paths)
    if not ref:
        raise ValueError("no composition reference set")
    rd, rs = split_key(ref)
    ref_inst = {_norm(i.get("label", "")): i
                for i in _instances(paths.for_source(rd, rs).output_dir)}
    out = {"reference": ref, "scans": {}}
    for key in scan_keys:
        d, s = split_key(key)
        inst = {_norm(i.get("label", "")): i
                for i in _instances(paths.for_source(d, s).output_dir)}
        shared = sorted(set(ref_inst) & set(inst))
        out["scans"][key] = {
            "pairs": [{"label": inst[l].get("label"),
                       "ref_points": int(ref_inst[l].get("total_points") or 0),
                       "scan_points": int(inst[l].get("total_points") or 0)}
                      for l in shared],
            "only_in_scan": sorted(inst[l].get("label") for l in set(inst) - set(ref_inst)),
            "only_in_ref": sorted(ref_inst[l].get("label") for l in set(ref_inst) - set(inst)),
        }
    return out


# ── cloud I/O (binary PLY, header preserved) ──────────────────────────────
_PLY_TYPE = {'float': '<f4', 'float32': '<f4', 'double': '<f8', 'float64': '<f8',
             'uchar': 'u1', 'uint8': 'u1', 'char': 'i1', 'int8': 'i1',
             'ushort': '<u2', 'uint16': '<u2', 'short': '<i2', 'int16': '<i2',
             'uint': '<u4', 'uint32': '<u4', 'int': '<i4', 'int32': '<i4'}
_PLY_NAME = {'<f4': 'float', '<f8': 'double', 'u1': 'uchar', 'i1': 'char',
             '<u2': 'ushort', '<i2': 'short', '<u4': 'uint', '<i4': 'int'}


def _read_ply(path: Path):
    props, n = [], 0
    with open(path, 'rb') as f:
        while True:
            s = f.readline().decode('ascii', 'ignore').strip()
            if s.startswith('element vertex'):
                n = int(s.split()[-1])
            elif s.startswith('property'):
                parts = s.split()
                props.append((parts[2], _PLY_TYPE[parts[1]]))
            elif s == 'end_header':
                break
        return np.frombuffer(f.read(), dtype=np.dtype(props), count=n)


def _ply_type_name(dt: np.dtype) -> str:
    s = dt.str
    if s.startswith('|'):
        s = s[1:]
    return _PLY_NAME[s]


def _write_ply(path: Path, data: np.ndarray):
    with open(path, 'wb') as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
        f.write(f"element vertex {len(data)}\n".encode())
        for name in data.dtype.names:
            f.write(f"property {_ply_type_name(data.dtype[name])} {name}\n".encode())
        f.write(b"end_header\n")
        f.write(np.ascontiguousarray(data).tobytes())


def _xyz(data) -> np.ndarray:
    return np.stack([data['x'], data['y'], data['z']], axis=1).astype(np.float64)


def _segment_points(output_dir: Path, xyz: np.ndarray, inst: dict) -> np.ndarray:
    """Points of an instance: classification.npy first, result indices
    (legacy) second."""
    iid = int(inst.get("instance_id", inst.get("id", -1)))
    cls_p = output_dir / "classification.npy"
    if cls_p.exists():
        cls = np.load(cls_p, mmap_mode="r")
        if len(cls) == len(xyz):
            idx = np.flatnonzero(cls == iid)
            if len(idx):
                return xyz[idx]
    gi = np.asarray(inst.get("globalIndices") or [], dtype=np.int64)
    gi = gi[(gi >= 0) & (gi < len(xyz))]
    return xyz[gi]


def _sub(P: np.ndarray, n: int, rng) -> np.ndarray:
    return P if len(P) <= n else P[rng.choice(len(P), n, replace=False)]


def _tilt_deg(R: np.ndarray) -> float:
    """Rotation of the Y axis away from vertical (yaw is free between days)."""
    up = R @ np.array([0.0, 1.0, 0.0])
    return float(np.degrees(np.arccos(np.clip(up[1] / max(np.linalg.norm(up), 1e-12), -1, 1))))


def _median_nn(P: np.ndarray, Q: np.ndarray, rng, n: int = 30000) -> float:
    from scipy.spatial import cKDTree
    P = _sub(P, n, rng)
    d, _ = cKDTree(Q).query(P, workers=8)
    return float(np.median(d))


def run_fuse(paths: ProjectPaths, scan_keys: List[str],
             exclude: Optional[Dict[str, List[str]]] = None,
             log=print, progress=None, params: Optional[dict] = None) -> dict:
    params = {"overlap": 0.8, "ransac_epsilon": 0.02, "size_tol": 0.10,
              "scale_min": 0.9, "scale_max": 1.1, "max_tilt_deg": 5.0,
              "min_conditioning_deg": 20.0, "max_pair_residual_cm": 6.0,
              "max_heldout_cm": 8.0, "pair_sample": 150000,
              "heldout_sample": 300000, **(params or {})}
    exclude = {k: {_norm(x) for x in v} for k, v in (exclude or {}).items()}
    rng = np.random.default_rng(0)

    def _p(pct, msg):
        log(msg)
        if progress:
            progress(pct, msg)

    ref = get_reference(paths)
    if not ref:
        raise ValueError("no composition reference set")
    rd, rs = split_key(ref)
    ref_ctx = paths.for_source(rd, rs)
    _p(3, f"loading reference {ref}…")
    ref_data = _read_ply(ref_ctx.output_dir / "cleaned_cloud.ply")
    ref_xyz = _xyz(ref_data)
    ref_inst = {_norm(i.get("label", "")): i for i in _instances(ref_ctx.output_dir)}

    work = Path(tempfile.mkdtemp(prefix="fuse_", dir=str(paths.project_dir)))
    spec = {"reference": {"pairs": {}}, "scans": [], "params": params}
    scan_states = []
    try:
        ref_pair_pts = {}
        for key in scan_keys:
            d, s = split_key(key)
            ctx = paths.for_source(d, s)
            _p(8, f"loading {key}…")
            data = _read_ply(ctx.output_dir / "cleaned_cloud.ply")
            xyz = _xyz(data)
            inst = {_norm(i.get("label", "")): i for i in _instances(ctx.output_dir)}
            labels = [l for l in sorted(set(ref_inst) & set(inst))
                      if l not in exclude.get(key, set())]
            if len(labels) < 2:
                raise ValueError(f"{key}: only {len(labels)} shared segment(s) "
                                 f"— at least 2 invariant objects with the same "
                                 f"label in both scans are required")
            pairs = {}
            used_mask = np.zeros(len(xyz), dtype=bool)
            for l in labels:
                A = _segment_points(ref_ctx.output_dir, ref_xyz, ref_inst[l])
                B = _segment_points(ctx.output_dir, xyz, inst[l])
                if len(A) < 500 or len(B) < 500:
                    log(f"  {key}/{l}: too few points ({len(A)}/{len(B)}) — skipped")
                    continue
                if l not in ref_pair_pts:
                    pa = work / f"ref_{l}.npy"
                    np.save(pa, _sub(A, params["pair_sample"], rng))
                    ref_pair_pts[l] = str(pa)
                pb = work / f"{d}_{s}_{l}.npy"
                np.save(pb, _sub(B, params["pair_sample"], rng))
                pairs[l] = str(pb)
                iid = int(inst[l].get("instance_id", inst[l].get("id", -1)))
                cls_p = ctx.output_dir / "classification.npy"
                if cls_p.exists():
                    cls = np.load(cls_p, mmap_mode="r")
                    if len(cls) == len(xyz):
                        used_mask |= (cls == iid)
            if len(pairs) < 2:
                raise ValueError(f"{key}: fewer than 2 usable pairs")
            spec["reference"]["pairs"].update(ref_pair_pts)
            spec["scans"].append({"key": key, "pairs": pairs})
            # held-out sample: the scan's points NOT in any pair
            heldout = _sub(xyz[~used_mask], params["heldout_sample"], rng)
            scan_states.append({"key": key, "data": data, "xyz": xyz,
                                "heldout": heldout, "labels": labels,
                                "inst": inst, "ctx": ctx})
        spec_p = work / "spec.json"
        spec_p.write_text(json.dumps(spec))
        out_p = work / "result.json"
        _p(25, "CloudComPy registration (planes + ICP with scale)…")
        proc = subprocess.run(
            ["bash", str(SERVER_DIR / "run_cloudcompy_script.sh"),
             str(SERVER_DIR / "cloudcompy_register.py"), str(spec_p), str(out_p)],
            capture_output=True, text=True, timeout=3600)
        for ln in (proc.stdout or "").splitlines():
            if "[register]" in ln:
                log(ln.strip())
        if proc.returncode != 0 or not out_p.exists():
            raise RuntimeError(f"CloudComPy registration failed: "
                               f"{(proc.stderr or proc.stdout)[-1500:]}")
        result = json.loads(out_p.read_text())

        # ── GUARDS + symmetric scale ─────────────────────────────────
        _p(55, "validating (guards + held-out)…")
        report = {"reference": ref, "accepted": True, "reason": None, "scans": []}
        ref_sample = _sub(ref_xyz, 400000, rng)
        for st, r in zip(scan_states, result["scans"]):
            entry = {"key": st["key"], "pairs": r.get("pairs", []),
                     "verdict": "accepted"}
            if r.get("error"):
                entry.update(verdict=f"rejected: {r['error']}")
                report["accepted"] = False; report["reason"] = entry["verdict"]
                report["scans"].append(entry); continue
            T = np.array(r["T"], dtype=np.float64).reshape(4, 4)
            s = float(r["scale"])
            R = T[:3, :3] / max(s, 1e-12)
            tilt = _tilt_deg(R)
            entry.update(scale=s, rms_cm=r["rms_cm"], rot_deg=r.get("coarse_yaw_deg"),
                         tilt_deg=tilt, conditioning_deg=r.get("conditioning_deg"),
                         t_m=float(np.linalg.norm(T[:3, 3])))
            why = []
            if not (params["scale_min"] <= s <= params["scale_max"]):
                why.append(f"scale {s:.4f} outside [{params['scale_min']}, {params['scale_max']}]")
            if tilt > params["max_tilt_deg"]:
                why.append(f"tilt {tilt:.2f}° > {params['max_tilt_deg']}°")
            if (r.get("conditioning_deg") or 0) < params["min_conditioning_deg"]:
                why.append(f"pairs' plane normals too parallel "
                           f"({r.get('conditioning_deg', 0):.1f}°) — add an invariant object "
                           f"facing another direction")
            # per-pair residuals (median NN scan→ref) before/after the solve
            T0 = np.array(r["T_coarse"], dtype=np.float64).reshape(4, 4)
            for p in r["pairs"]:
                A = np.load(spec["reference"]["pairs"][p["label"]])
                B = np.load(next(sc["pairs"][p["label"]] for sc in spec["scans"]
                                 if sc["key"] == st["key"]))
                Bh = np.c_[B, np.ones(len(B))]
                p["residual_cm_before"] = _median_nn((Bh @ T0.T)[:, :3], A, rng) * 100
                p["residual_cm_after"] = _median_nn((Bh @ T.T)[:, :3], A, rng) * 100
            bad = [p["label"] for p in r["pairs"]
                   if not p.get("suspect") and p.get("residual_cm_after", 0) > params["max_pair_residual_cm"]]
            if bad:
                why.append(f"pair residual > {params['max_pair_residual_cm']} cm: {bad}")
            # symmetric split: ref × 1/√s ; scan: √s·R·p + t/√s
            rs_ = np.sqrt(s)
            T_scan = np.eye(4); T_scan[:3, :3] = rs_ * R; T_scan[:3, 3] = T[:3, 3] / rs_
            T_ref = np.eye(4) * (1.0 / rs_); T_ref[3, 3] = 1.0
            H = (np.c_[st["heldout"], np.ones(len(st["heldout"]))] @ T_scan.T)[:, :3]
            ref_mid = ref_sample / rs_
            ho = _median_nn(H, ref_mid, rng)
            entry["heldout"] = {"after_cm": ho * 100,
                                "threshold_cm": params["max_heldout_cm"]}
            if ho * 100 > params["max_heldout_cm"]:
                why.append(f"held-out (unused points) median {ho*100:.1f} cm > {params['max_heldout_cm']} cm")
            if why:
                entry["verdict"] = "rejected: " + "; ".join(why)
                report["accepted"] = False
                report["reason"] = (report["reason"] or "") + f"{st['key']}: {entry['verdict']} "
            entry["T_scan"] = T_scan.reshape(-1).tolist()
            entry["T_ref"] = T_ref.reshape(-1).tolist()
            st["T_scan"], st["T_ref"] = T_scan, T_ref
            log(f"  {st['key']}: scale {s:.4f} (split ±√), tilt {tilt:.2f}°, rms {r['rms_cm']:.1f} cm, "
                f"held-out {ho*100:.1f} cm → {entry['verdict']}")
            report["scans"].append(entry)
        if not report["accepted"]:
            _p(100, f"REJECTED — {report['reason']}")
            report["report_path"] = None
            return report

        # ── merged cloud ────────────────────────────────────────────
        _p(65, "building the merged cloud…")
        # with several scans each brings its own √s; the reference takes the
        # geometric mean of their 1/√s (all scales meet in the middle)
        ref_scale = float(np.exp(np.mean([np.log(st["T_ref"][0, 0]) for st in scan_states])))
        names = list(ref_data.dtype.names)
        keep = [n for n in ("x", "y", "z", "red", "green", "blue", "confidence")
                if n in names and all(n in st["data"].dtype.names for st in scan_states)]
        dt = [(n, ref_data.dtype[n]) for n in keep] + [("scan", "u1")]
        parts, cls_parts, instances, offset = [], [], [], 0
        ref_cls_p = ref_ctx.output_dir / "classification.npy"
        ref_cls = np.load(ref_cls_p) if ref_cls_p.exists() else np.zeros(len(ref_xyz), np.uint8)

        def _block(data, xyz_t, scan_id, cls, base):
            out = np.empty(len(data), dtype=dt)
            out["x"], out["y"], out["z"] = xyz_t[:, 0], xyz_t[:, 1], xyz_t[:, 2]
            for n in keep:
                if n not in ("x", "y", "z"):
                    out[n] = data[n]
            out["scan"] = scan_id
            c = cls.astype(np.int32)
            c[c > 0] += base
            return out, c

        blk, c = _block(ref_data, ref_xyz * ref_scale, 0, ref_cls, offset)
        parts.append(blk); cls_parts.append(c)
        for i in ref_inst.values():
            instances.append({**{k: v for k, v in i.items() if k != "globalIndices"},
                              "scan": ref, "instance_id": int(i.get("instance_id", 0)) + offset,
                              "label": f"{i.get('label')} [{rd}]"})
        offset += max([int(i.get("instance_id", 0)) for i in ref_inst.values()] + [0])
        for si, st in enumerate(scan_states, start=1):
            xyz_t = (np.c_[st["xyz"], np.ones(len(st["xyz"]))] @ st["T_scan"].T)[:, :3]
            cls_p = st["ctx"].output_dir / "classification.npy"
            cls = np.load(cls_p) if cls_p.exists() else np.zeros(len(xyz_t), np.uint8)
            blk, c = _block(st["data"], xyz_t, si, cls, offset)
            parts.append(blk); cls_parts.append(c)
            for i in st["inst"].values():
                instances.append({**{k: v for k, v in i.items() if k != "globalIndices"},
                                  "scan": st["key"], "instance_id": int(i.get("instance_id", 0)) + offset,
                                  "label": f"{i.get('label')} [{split_key(st['key'])[0]}]"})
            offset += max([int(i.get("instance_id", 0)) for i in st["inst"].values()] + [0])
        merged = np.concatenate(parts)
        cls_all = np.concatenate(cls_parts)
        if cls_all.max() > 255:
            log("  namespaced instance ids exceed 255 — fused classification dropped (uint8 limit)")
            cls_all = np.zeros(len(merged), np.int32)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        fused_ctx = paths.for_source(FUSED_DAY, stamp)
        paths.ensure_source_dirs(FUSED_DAY, stamp)
        out_dir = fused_ctx.output_dir
        _p(80, f"writing merged cloud ({len(merged):,} pts)…")
        _write_ply(out_dir / "cleaned_cloud.ply", merged)
        np.save(out_dir / "classification.npy", cls_all.astype(np.uint8))
        np.savez(out_dir / "floor_transform.npz", s=np.float64(1.0), R=np.eye(3), t=np.zeros(3))
        (out_dir / "segmentation_result.json").write_text(json.dumps({
            "type": "segmentation", "version": "4.0-fused",
            "membership": "classification.npy", "total_points": int(len(merged)),
            "instances": instances, "fused_from": [ref] + scan_keys,
            "read_only": True}))
        (out_dir / "fused_meta.json").write_text(json.dumps({
            "reference": ref, "scans": scan_keys, "created_at": stamp,
            "reference_scale": ref_scale,
            "transforms": {st["key"]: st["T_scan"].reshape(-1).tolist() for st in scan_states},
            "report": report}, indent=1))
        _p(85, "Potree of the merged cloud…")
        from potree_converter import convert_ply_to_potree
        if not convert_ply_to_potree(fused_ctx.source_dir, force=True):
            raise RuntimeError("Potree of the merged cloud failed")
        # register in project.json: scans list (label) + composition transforms
        meta = sync_scans(paths)
        for sc in meta["scans"]:
            if sc["key"] == f"{FUSED_DAY}/{stamp}":
                sc["label"] = "Fused · " + " + ".join([rd] + [split_key(k)[0] for k in scan_keys])
        paths.save_project_meta(meta)
        for st in scan_states:
            set_transform(paths, st["key"], st["T_scan"], "objects",
                          extra={"scale_scan_to_ref": float(st["T_ref"][0, 0] ** -2),
                                 "pairs": st["labels"], "fused": f"{FUSED_DAY}/{stamp}"})
        meta = load_meta(paths)
        meta["composition"]["reference_scale"] = ref_scale
        paths.save_project_meta(meta)
        report["fused_key"] = f"{FUSED_DAY}/{stamp}"
        report["points"] = int(len(merged))
        (out_dir / "fuse_report.json").write_text(json.dumps(report, indent=1))
        _p(100, f"✅ merged cloud ready: {FUSED_DAY}/{stamp} ({len(merged):,} pts)")
        return report
    finally:
        shutil.rmtree(work, ignore_errors=True)
