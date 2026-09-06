"""Correction analysis — user-directed revisit/depth correction (2026-09-06).

USER DESIGN (his words, this date):
  * The USER marks which segments exhibit the parallel-copies error when he
    launches the analysis. He does NOT say which copies relate to which —
    "eso lo debe determinar el algoritmo, quienes con quienes en que chunks
    frames".
  * Diagnosis logic (his matrix): per copy, WHO shot it (chunk/frames) and
    from WHERE (camera→target distance). Early chunk + far shot → depth
    error likely real. Late frame + close shot → pose drift likely. Late +
    far → possibly both. The INTERNAL fingerprint separates them: if the
    relations between the selected objects' planes inside a displaced copy
    match the reference relations → pure POSE (rigid); if compressed /
    expanded → DEPTH (expand along the shooting rays) first, then rigid.
  * A correction that is confirmed good IS the cloud ("mas valida que la
    reconstruccion original porque esta basada en el ojo humano"). The next
    correction runs ON TOP of it — never on the discarded original.
  * Undo is ONE level: each correction is approved or rejected; approval
    leaves no remains of the previous state.
  * Validation criterion (his): the correction is computed from the marked
    segments but the REST of the scene is the exam — floor and the other
    surfaces of the same chunk must land correctly on their own.

Everything here is tool_measured; the cloud is only modified through the
apply step, with a one-level undo backup and a full report on disk.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

UNDO_DIR = "correction_undo"
STATE_FILE = "correction_state.json"
REPORT_FILE = "correction_report.json"
DEPTH_COMPRESS_TOL = 0.03  # |k-1| beyond this → depth (scale) correction
ICP_ITERS = 30
ICP_TRIM = 0.7


def _log_default(msg: str) -> None:
    print(f"[Correction] {msg}", flush=True)


def load_state(output_dir: Path) -> dict:
    p = output_dir / STATE_FILE
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            pass
    return {"status": "none"}


def _save_state(output_dir: Path, state: dict) -> None:
    (output_dir / STATE_FILE).write_text(json.dumps(state, indent=1))


# ── PLY I/O (binary little-endian, header preserved) ─────────────────────
_PLY_TYPE = {
    'float': '<f4', 'float32': '<f4', 'double': '<f8', 'float64': '<f8',
    'uchar': 'u1', 'uint8': 'u1', 'char': 'i1', 'int8': 'i1',
    'ushort': '<u2', 'uint16': '<u2', 'short': '<i2', 'int16': '<i2',
    'uint': '<u4', 'uint32': '<u4', 'int': '<i4', 'int32': '<i4',
}


def _read_ply(path: Path):
    header: List[bytes] = []
    props = []
    n = 0
    with open(path, 'rb') as f:
        while True:
            line = f.readline()
            header.append(line)
            s = line.decode('ascii', 'ignore').strip()
            if s.startswith('element vertex'):
                n = int(s.split()[-1])
            elif s.startswith('property'):
                parts = s.split()
                props.append((parts[2], _PLY_TYPE[parts[1]]))
            elif s == 'end_header':
                break
        data = np.frombuffer(f.read(), dtype=np.dtype(props),
                             count=n).copy()
    return header, data


def _write_ply(path: Path, header: List[bytes], data: np.ndarray) -> None:
    import tempfile, os
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".ply")
    os.close(fd)
    with open(tmp, 'wb') as f:
        for line in header:
            s = line.decode('ascii', 'ignore').strip()
            if s.startswith('element vertex'):
                f.write(f"element vertex {len(data)}\n".encode('ascii'))
            else:
                f.write(line)
        f.write(data.tobytes())
    Path(tmp).replace(path)


# ── geometry helpers ─────────────────────────────────────────────────────
def _fit_plane(P: np.ndarray, rng, tol: float = 0.02):
    best, bn = None, -1
    S = P if len(P) <= 60000 else P[rng.choice(len(P), 60000, replace=False)]
    for _ in range(300):
        a, b, c = S[rng.choice(len(S), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        cnt = int((np.abs((S - a) @ n) < tol).sum())
        if cnt > bn:
            bn, best = cnt, (n, a)
    n, a = best
    inl = np.abs((S - a) @ n) < tol * 2
    c0 = S[inl].mean(0)
    n = np.linalg.svd(S[inl] - c0, full_matrices=False)[2][2]
    return n, c0


def _icp(src: np.ndarray, tree, target: np.ndarray, rng,
         iters: int = ICP_ITERS, trim: float = ICP_TRIM):
    """Trimmed point-to-point ICP src→target. Returns R, t, trimmed rms."""
    R = np.eye(3)
    t = np.zeros(3)
    S = src.copy()
    rms = None
    for _ in range(iters):
        d, j = tree.query(S, workers=8)
        k = max(100, int(len(S) * trim))
        sel = np.argsort(d)[:k]
        P, Q = S[sel], target[j[sel]]
        mp, mq = P.mean(0), Q.mean(0)
        H = (P - mp).T @ (Q - mq)
        U, _sv, Vt = np.linalg.svd(H)
        D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
        Ri = Vt.T @ D @ U.T
        ti = mq - Ri @ mp
        S = S @ Ri.T + ti
        R = Ri @ R
        t = Ri @ t + ti
        rms = float(np.sqrt(
            (np.linalg.norm(S[sel] - Q, axis=1) ** 2).mean()))
    return R, t, rms


def _rot_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2,
                                              -1, 1))))


# ── main entry ───────────────────────────────────────────────────────────
def run_correction(output_dir: Path, instance_ids: List[int],
                   log=_log_default, progress=None) -> dict:
    """Analyze the user-marked segments, solve per-chunk corrections
    (depth and/or rigid per HIS matrix), validate on the rest of the
    scene, APPLY to cleaned_cloud.ply + camera_poses.txt, delete and
    rebuild the Potree. One-level undo backup. Returns the report."""
    output_dir = Path(output_dir)
    t0 = time.time()
    state = load_state(output_dir)
    if state.get("status") == "pending":
        raise RuntimeError("a correction is pending approval — approve or "
                           "undo it before running a new one")
    rng = np.random.default_rng(0)

    def _p(pct, msg):
        log(msg)
        if progress:
            progress(pct, msg)

    # 1) inputs -----------------------------------------------------------
    _p(2, "loading cloud + provenance...")
    ply_path = output_dir / "cleaned_cloud.ply"
    header, data = _read_ply(ply_path)
    names = data.dtype.names
    for req in ("frame_global", "pixel_row", "pixel_col"):
        if req not in names:
            raise RuntimeError(f"cleaned_cloud.ply lacks '{req}' — "
                               f"provenance is mandatory for correction")
    xyz = np.stack([data["x"], data["y"], data["z"]],
                   axis=1).astype(np.float64)
    fg = data["frame_global"].astype(np.int64)
    pr = data["pixel_row"].astype(np.int64)
    pc = data["pixel_col"].astype(np.int64)
    N = len(xyz)

    frames = [int(x) for x in
              (output_dir / "camera_frames.txt").read_text().split()]
    kf_index = {f: k for k, f in enumerate(frames)}
    poses = []
    for ln in (output_dir / "camera_poses.txt").read_text().splitlines():
        vals = [float(x) for x in ln.split()]
        if len(vals) == 16:
            poses.append(np.array(vals).reshape(4, 4))
    poses = np.stack(poses)
    cam_center = {f: poses[k][:3, 3] for f, k in kf_index.items()}

    fmax = int(fg.max())
    kf_arr = np.full(fmax + 1, -1, dtype=np.int64)
    for f, k in kf_index.items():
        if f <= fmax:
            kf_arr[f] = k
    ks = kf_arr[np.clip(fg, 0, fmax)]
    n_chunks = (len(frames) + 29) // 30
    chunk = np.where(ks >= 0, np.minimum(ks // 30, n_chunks - 1), -1)

    # 2) copies via the segment's BBOX (USER 2026-09-06: "lo que yo dejé
    # dentro del bbox es lo correcto que debe corregirse; lo que esté
    # afuera aunque esté en la máscara no es correcto" — masks can grab
    # other objects/floaters; the curated bbox is the truth). ALL cloud
    # points inside the OBB are evidence, grouped per chunk via
    # provenance — nothing inside the bbox is ever skipped.
    seg_res = json.loads(
        (output_dir / "segmentation_result.json").read_text())
    inst_by_iid = {int(i.get("instance_id", i.get("id"))): i
                   for i in seg_res.get("instances", [])}
    label_by_iid = {k: (v.get("label") or v.get("name") or str(k))
                    for k, v in inst_by_iid.items()}
    _p(8, f"extracting copies of {len(instance_ids)} marked segment(s) "
          f"via their curated BBOX (all cloud points inside)...")
    copies = []          # {iid, label, visit, chunks, idx (cloud)}
    OBB_MARGIN = 0.03
    for iid in instance_ids:
        inst = inst_by_iid.get(iid)
        if inst is None:
            raise RuntimeError(f"instance {iid} not in "
                               f"segmentation_result.json")
        gidx = np.asarray(inst.get("globalIndices") or [], dtype=np.int64)
        if not len(gidx):
            raise RuntimeError(f"instance {iid} "
                               f"({label_by_iid.get(iid)}) has no points")
        P = xyz[gidx]
        # OBB of the curated segment: PCA axes + extents (+margin)
        c_obb = P.mean(0)
        axes = np.linalg.svd(P - c_obb, full_matrices=False)[2]
        loc = (P - c_obb) @ axes.T
        lo = loc.min(0) - OBB_MARGIN
        hi = loc.max(0) + OBB_MARGIN
        rel = (xyz - c_obb) @ axes.T
        inside = np.all((rel >= lo) & (rel <= hi), axis=1)
        idx_all = np.where(inside)[0]
        log(f"  {label_by_iid.get(iid)}: bbox holds {len(idx_all):,} cloud "
            f"pts (segment itself: {len(gidx):,})")
        # group by TEMPORAL runs of chunks (a visit = consecutive chunks)
        chs_all = chunk[idx_all]
        uniq = sorted(set(chs_all.tolist()) - {-1})
        runs: List[List[int]] = []
        for c in uniq:
            if runs and c - runs[-1][-1] <= 1:
                runs[-1].append(c)
            else:
                runs.append([c])
        for vi, run in enumerate(runs):
            m = np.isin(chs_all, run)
            idx = idx_all[m]
            cen = np.median(xyz[idx], axis=0)
            vframes = sorted(set(int(f) for f in fg[idx]))
            dists = [float(np.linalg.norm(cam_center[f] - cen))
                     for f in vframes if f in cam_center]
            copies.append({
                "iid": iid, "label": label_by_iid.get(iid, str(iid)),
                "visit": vi, "frames": vframes, "chunks": run,
                "idx": idx, "centroid": cen,
                "kf_range": [kf_index.get(vframes[0], -1),
                             kf_index.get(vframes[-1], -1)],
                "shoot_dist_m": (float(np.median(dists))
                                 if dists else None),
            })
            log(f"  {label_by_iid.get(iid)} visit {vi}: {len(idx):,} pts, "
                f"chunks {run}, kf {copies[-1]['kf_range'][0]}.."
                f"{copies[-1]['kf_range'][1]}, shoot dist "
                f"{(copies[-1]['shoot_dist_m'] or -1):.2f} m")

    if not copies:
        raise RuntimeError("no copies extracted — empty bboxes")

    # 3) reference per instance = the copy with most points ---------------
    ref = {}
    for iid in instance_ids:
        cands = [c for c in copies if c["iid"] == iid]
        if cands:
            ref[iid] = max(cands, key=lambda c: len(c["idx"]))
    for iid, r in ref.items():
        log(f"  reference for {r['label']}: visit {r['visit']} "
            f"({len(r['idx']):,} pts, chunks {r['chunks']})")
    ref_chunks = set()
    for r in ref.values():
        ref_chunks.update(r["chunks"])

    displaced = [c for c in copies
                 if c is not ref.get(c["iid"])
                 and not set(c["chunks"]) <= ref_chunks]
    if not displaced:
        raise RuntimeError("every copy belongs to the reference chunks — "
                           "nothing to correct")

    # target cloud for ICP: all reference copies together
    tgt_idx = np.unique(np.concatenate([r["idx"] for r in ref.values()]))
    target = xyz[tgt_idx]
    from scipy.spatial import cKDTree
    tree = cKDTree(target)

    # reference inter-object fingerprint (centroid distances)
    ref_fp = {}
    iids = sorted(ref.keys())
    for i in range(len(iids)):
        for j in range(i + 1, len(iids)):
            ref_fp[(iids[i], iids[j])] = float(np.linalg.norm(
                ref[iids[i]]["centroid"] - ref[iids[j]]["centroid"]))

    # 4) per-chunk solve --------------------------------------------------
    solve_chunks = sorted({c for cp in displaced for c in cp["chunks"]
                           if c not in ref_chunks})
    _p(20, f"solving {len(solve_chunks)} displaced chunk(s): "
           f"{solve_chunks}")
    per_chunk = {}
    deferred = []        # chunks with too little evidence to solve ALONE —
    #                      they inherit the nearest solved chunk's transform
    #                      (USER: nothing inside the bbox is ever skipped)
    floor_ref = float(np.median(
        xyz[(np.abs(xyz[:, 1]) < 0.5) & ~np.isin(chunk, solve_chunks), 1]))
    for c in solve_chunks:
        entry = {"chunk": int(c)}
        cp_here = [cp for cp in displaced if c in cp["chunks"]]
        src_idx = np.unique(np.concatenate(
            [cp["idx"][chunk[cp["idx"]] == c] for cp in cp_here]))
        if len(src_idx) < 300:
            deferred.append((c, len(src_idx)))
            continue
        src = xyz[src_idx]
        # diagnosis: HIS matrix ------------------------------------------
        kfs_c = sorted({kf_index.get(f, -1) for cp in cp_here
                        for f in cp["frames"]
                        if kf_index.get(f, -1) // 30 == c})
        shoot = float(np.median([cp["shoot_dist_m"] for cp in cp_here
                                 if cp["shoot_dist_m"]]))
        entry["kf"] = [kfs_c[0], kfs_c[-1]] if kfs_c else None
        entry["shoot_dist_m"] = round(shoot, 2)
        entry["objects"] = sorted({cp["label"] for cp in cp_here})
        # internal fingerprint: needs ≥2 marked objects in this chunk
        k_scale = 1.0
        seen = {}
        for cp in cp_here:
            ii = cp["idx"][chunk[cp["idx"]] == c]
            if len(ii) >= 2000:
                seen[cp["iid"]] = np.median(xyz[ii], axis=0)
        fps = []
        for (a, b), dref in ref_fp.items():
            if a in seen and b in seen and dref > 0.3:
                fps.append(float(np.linalg.norm(seen[a] - seen[b])) / dref)
        if fps:
            k_scale = float(np.median([1.0 / f for f in fps]))
            entry["fingerprint_ratio"] = round(float(np.median(fps)), 4)
        depth_needed = fps and abs(k_scale - 1.0) > DEPTH_COMPRESS_TOL
        entry["diagnosis"] = ("depth+pose" if depth_needed else "pose"
                              ) if fps else "pose (single-object evidence)"
        log(f"  ch{c:02d}: kf {entry['kf']}, shoot {shoot:.2f} m, "
            f"objects {entry['objects']}, diagnosis: {entry['diagnosis']}"
            + (f" (compression {entry.get('fingerprint_ratio')})"
               if fps else ""))
        # depth expansion along shooting rays -----------------------------
        S = src.copy()
        if depth_needed:
            cams = np.stack([cam_center.get(int(f), np.zeros(3))
                             for f in fg[src_idx]])
            S = cams + (S - cams) * k_scale
            entry["depth_scale"] = round(k_scale, 4)
            log(f"  ch{c:02d}: depth expansion x{k_scale:.4f} applied "
                f"along shooting rays")
        # rigid: init along the dominant object plane, then trimmed ICP ---
        big = max(cp_here, key=lambda cp: len(cp["idx"]))
        n_pl, c_pl = _fit_plane(xyz[ref[big["iid"]]["idx"]], rng)
        off = float(np.median((S - c_pl) @ n_pl))
        init_t = -off * n_pl
        sub = S[rng.choice(len(S), min(50000, len(S)), replace=False)] \
            + init_t
        R, t, rms = _icp(sub, tree, target, rng)
        t_full = R @ init_t + t
        entry["rigid"] = {"rot_deg": round(_rot_deg(R), 3),
                          "t_m": round(float(np.linalg.norm(t_full)), 4),
                          "icp_rms_cm": round(rms * 100, 2)}
        log(f"  ch{c:02d}: rigid rot {_rot_deg(R):.2f}° "
            f"|t| {np.linalg.norm(t_full):.3f} m, rms {rms*100:.1f} cm")
        # validation numbers (objects = solve evidence; floor = held-out) -
        corr_src = (S @ R.T + t_full)
        d_before, _ = tree.query(
            src[rng.choice(len(src), min(20000, len(src)), replace=False)],
            workers=8)
        d_after, _ = tree.query(
            corr_src[rng.choice(len(corr_src), min(20000, len(corr_src)),
                                replace=False)], workers=8)
        entry["object_residual_cm"] = {
            "before": round(float(np.median(d_before)) * 100, 1),
            "after": round(float(np.median(d_after)) * 100, 1)}
        mc = chunk == c
        pch = xyz[mc]
        if depth_needed:
            camsc = np.stack([cam_center.get(int(f), np.zeros(3))
                              for f in fg[mc]])
            pch = camsc + (pch - camsc) * k_scale
        pch = pch @ R.T + t_full
        fb_b = np.abs(xyz[mc][:, 1] - floor_ref) < 0.5
        fb_a = np.abs(pch[:, 1] - floor_ref) < 0.5
        entry["floor_heldout_cm"] = {
            "reference": round(floor_ref * 100, 1),
            "before": (round(float(np.median(xyz[mc][fb_b, 1])) * 100, 1)
                       if fb_b.any() else None),
            "after": (round(float(np.median(pch[fb_a, 1])) * 100, 1)
                      if fb_a.any() else None)}
        log(f"  ch{c:02d}: object residual {entry['object_residual_cm']} | "
            f"floor held-out {entry['floor_heldout_cm']}")
        entry["_R"] = R
        entry["_t"] = t_full
        entry["_k"] = k_scale if depth_needed else 1.0
        per_chunk[c] = entry

    if not per_chunk:
        raise RuntimeError("no chunk gathered enough evidence to solve")
    # chunks with too few own points: inherit the NEAREST solved chunk's
    # transform (never skipped — their bbox points must be corrected too)
    for c, n_pts in deferred:
        nearest = min(per_chunk.keys(), key=lambda s: abs(s - c))
        src_e = per_chunk[nearest]
        entry = {"chunk": int(c),
                 "inherited_from": int(nearest),
                 "n_own_points": int(n_pts),
                 "diagnosis": f"inherited (only {n_pts} bbox pts of its "
                              f"own — takes ch{nearest:02d}'s solution)",
                 "rigid": src_e["rigid"],
                 "_R": src_e["_R"], "_t": src_e["_t"],
                 "_k": src_e["_k"]}
        per_chunk[c] = entry
        log(f"  ch{c:02d}: only {n_pts} bbox pts — INHERITS ch{nearest:02d}"
            f"'s transform (nothing inside the bbox is skipped)")

    # 5) APPLY: one-level undo backup, then bake --------------------------
    _p(55, "applying: backup (one-level undo) + rewrite cloud & poses...")
    undo = output_dir / UNDO_DIR
    if undo.exists():
        shutil.rmtree(undo)
    undo.mkdir()
    shutil.copy2(ply_path, undo / "cleaned_cloud.ply")
    shutil.copy2(output_dir / "camera_poses.txt", undo / "camera_poses.txt")

    xyz_new = xyz
    n_moved = 0
    for c, e in per_chunk.items():
        mc = chunk == c
        p = xyz_new[mc]
        if e["_k"] != 1.0:
            cams = np.stack([cam_center.get(int(f), np.zeros(3))
                             for f in fg[mc]])
            p = cams + (p - cams) * e["_k"]
        xyz_new[mc] = p @ e["_R"].T + e["_t"]
        n_moved += int(mc.sum())
    data["x"] = xyz_new[:, 0].astype(data.dtype["x"])
    data["y"] = xyz_new[:, 1].astype(data.dtype["y"])
    data["z"] = xyz_new[:, 2].astype(data.dtype["z"])
    _write_ply(ply_path, header, data)
    log(f"  cleaned_cloud.ply rewritten: {n_moved:,} pts moved "
        f"({len(per_chunk)} chunk(s))")

    # cameras move with their chunk (rigid only — depth doesn't move them)
    n_cam = 0
    for f, k in kf_index.items():
        c = min(k // 30, n_chunks - 1)
        if c in per_chunk:
            R, t = per_chunk[c]["_R"], per_chunk[c]["_t"]
            poses[k][:3, :3] = R @ poses[k][:3, :3]
            poses[k][:3, 3] = R @ poses[k][:3, 3] + t
            n_cam += 1
    (output_dir / "camera_poses.txt").write_text("\n".join(
        " ".join(f"{x:.9f}" for x in P.reshape(-1)) for P in poses) + "\n")
    log(f"  camera_poses.txt rewritten: {n_cam} cameras moved")

    # 6) Potree: DELETE and REBUILD fresh from the corrected cloud --------
    _p(65, "deleting Potree and rebuilding from the corrected cloud...")
    potree = output_dir / "potree"
    if potree.exists():
        shutil.rmtree(potree)
    import sys
    server_dir = str(Path(__file__).resolve().parents[1])
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)
    from potree_converter import convert_ply_to_potree
    ok = convert_ply_to_potree(output_dir.parent, force=True)
    if not ok:
        raise RuntimeError("Potree rebuild FAILED after correction — "
                           "undo is available (correction_undo/)")
    log("  Potree rebuilt ✓")

    # 7) report + state ---------------------------------------------------
    report = {
        "instance_ids": instance_ids,
        "copies": [{k: v for k, v in cp.items()
                    if k not in ("idx", "centroid", "frames")} | {
                        "n_points": len(cp["idx"]),
                        "n_frames": len(cp["frames"]),
                        "is_reference": cp is ref.get(cp["iid"])}
                   for cp in copies],
        "chunks": [{k: v for k, v in e.items()
                    if not k.startswith("_")}
                   for e in per_chunk.values()],
        "points_moved": n_moved,
        "cameras_moved": n_cam,
        "elapsed_s": round(time.time() - t0, 1),
        "provenance": "tool_measured",
    }
    (output_dir / REPORT_FILE).write_text(json.dumps(report, indent=1))
    _save_state(output_dir, {
        "status": "pending",
        "applied_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "chunks": sorted(per_chunk.keys()),
        "points_moved": n_moved,
    })
    _p(100, f"✅ correction applied ({report['elapsed_s']}s) — awaiting "
            f"your verdict: Approve or Undo")
    return report


def undo_correction(output_dir: Path, log=_log_default) -> dict:
    """Reject the pending correction: restore cloud + poses, rebuild
    Potree from the restored cloud."""
    output_dir = Path(output_dir)
    undo = output_dir / UNDO_DIR
    if not (undo / "cleaned_cloud.ply").exists():
        raise RuntimeError("no pending correction to undo")
    log("undo: restoring cleaned_cloud.ply + camera_poses.txt...")
    shutil.copy2(undo / "cleaned_cloud.ply", output_dir / "cleaned_cloud.ply")
    shutil.copy2(undo / "camera_poses.txt", output_dir / "camera_poses.txt")
    shutil.rmtree(undo)
    potree = output_dir / "potree"
    if potree.exists():
        shutil.rmtree(potree)
    import sys
    server_dir = str(Path(__file__).resolve().parents[1])
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)
    from potree_converter import convert_ply_to_potree
    ok = convert_ply_to_potree(output_dir.parent, force=True)
    log(f"undo complete — Potree rebuilt from the restored cloud "
        f"({'ok' if ok else 'FAILED'})")
    _save_state(output_dir, {"status": "none"})
    return {"ok": bool(ok)}


def approve_correction(output_dir: Path, log=_log_default) -> dict:
    """Confirm the pending correction: the corrected cloud IS the cloud.
    No remains of the previous state are kept (USER: one level, approved
    leaves nothing behind)."""
    output_dir = Path(output_dir)
    state = load_state(output_dir)
    if state.get("status") != "pending":
        raise RuntimeError("no pending correction to approve")
    undo = output_dir / UNDO_DIR
    if undo.exists():
        shutil.rmtree(undo)
    hist_p = output_dir / "correction_history.json"
    hist = []
    if hist_p.exists():
        try:
            hist = json.loads(hist_p.read_text())
        except Exception:  # noqa: BLE001
            hist = []
    hist.append({"approved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "chunks": state.get("chunks"),
                 "points_moved": state.get("points_moved")})
    hist_p.write_text(json.dumps(hist, indent=1))
    _save_state(output_dir, {"status": "approved",
                             "history": len(hist)})
    log(f"correction APPROVED — it is now the cloud "
        f"({len(hist)} approved so far)")
    return {"ok": True, "approved": len(hist)}
