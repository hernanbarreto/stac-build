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
ICP_ITERS = 100   # 30 stopped short (4.5° of a 6° yaw on a synthetic pair)
ICP_TRIM = 0.7
MAX_RESIDUAL_M = 0.15   # object copies must end within this (median NN)


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
         iters: int = ICP_ITERS, trim: float = ICP_TRIM,
         planar: bool = True):
    """Trimmed point-to-point ICP src→target. Returns R, t, trimmed rms.
    planar=True (default, USER 2026-09-06): the rotation is a YAW about the
    vertical axis only, translation is free in 3-D — a full 3-D rotation
    solved on two objects tilted chunk 6 and lifted its floor 11 cm."""
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
        if planar:
            # 2-D Kabsch in the XZ plane → yaw
            P2 = (P - mp)[:, [0, 2]]
            Q2 = (Q - mq)[:, [0, 2]]
            H2 = P2.T @ Q2
            U2, _s2, Vt2 = np.linalg.svd(H2)
            D2 = np.diag([1, np.sign(np.linalg.det(Vt2.T @ U2.T))])
            R2 = Vt2.T @ D2 @ U2.T          # maps XZ → XZ
            Ri = np.eye(3)
            Ri[0, 0], Ri[0, 2] = R2[0, 0], R2[0, 1]
            Ri[2, 0], Ri[2, 2] = R2[1, 0], R2[1, 1]
        else:
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
        # converged: this step barely moved anything
        if _rot_deg(Ri) < 1e-3 and np.linalg.norm(ti) < 1e-5:
            break
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
            # the object's OWN curated points in this visit = the matching
            # evidence (the bbox also holds floor/neighbours: on ccc1 68%
            # of the bbox points were floor, the ICP matched floor to floor
            # and the copies stayed 40-75 cm apart — 2026-09-06)
            seg_idx = np.intersect1d(idx, gidx)
            copies.append({
                "iid": iid, "label": label_by_iid.get(iid, str(iid)),
                "visit": vi, "frames": vframes, "chunks": run,
                "idx": idx, "seg_idx": seg_idx,
                "centroid": (np.median(xyz[seg_idx], axis=0)
                             if len(seg_idx) else cen),
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

    # 3) ONE reference visit for the whole run ------------------------------
    # The reference must be a single, consistent group of chunks: picking
    # "most points" per instance independently could name visit 1 (chunks
    # 6-7) for one object and visit 0 (chunk 0) for another, which made
    # EVERY chunk a reference chunk and left nothing to correct (bug seen
    # on pccr ccc1/ccc2, 2026-09-06). The reference is the EARLIEST visit
    # group (USER 2026-09-06: "el que habría que corregir es el último, el
    # que viene acumulando deriva fuerte") — drift accumulates along the
    # trajectory, so the first sighting is the anchor and the late
    # revisit is what moves. Each object's reference copy is its copy
    # inside that group (objects never seen in that group have no
    # reference and only act as displaced copies).
    groups: List[set] = []
    for cp in copies:
        s = set(cp["chunks"])
        merged = None
        for g in groups:
            if g & s:
                g |= s
                merged = g
                break
        if merged is None:
            groups.append(s)
    # union of overlapping groups (transitive)
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if groups[i] & groups[j]:
                    groups[i] |= groups.pop(j)
                    changed = True
                    break
            if changed:
                break

    def _score(g):
        return sum(len(cp["idx"]) for cp in copies if set(cp["chunks"]) & g)

    ref_group = min(groups, key=lambda g: min(g))
    log(f"  reference visit group (earliest): chunks {sorted(ref_group)} "
        f"({_score(ref_group):,} pts over {len(instance_ids)} object(s); "
        f"candidates {[sorted(g) for g in groups]})")
    ref = {}
    for iid in instance_ids:
        cands = [c for c in copies if c["iid"] == iid
                 and set(c["chunks"]) & ref_group]
        if cands:
            ref[iid] = max(cands, key=lambda c: len(c["idx"]))
        else:
            log(f"  {label_by_iid.get(iid)}: not seen in the reference "
                f"visit — no reference copy, its copies are displaced only")
    if not ref:
        raise RuntimeError("no reference copy in the reference visit")
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

    # target cloud for ICP: the reference copies' OWN object points
    tgt_idx = np.unique(np.concatenate([r["seg_idx"] for r in ref.values()]))
    if len(tgt_idx) < 300:
        raise RuntimeError("reference copies have too few object points")
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
            [cp["seg_idx"][chunk[cp["seg_idx"]] == c] for cp in cp_here]))
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
            ii = cp["seg_idx"][chunk[cp["seg_idx"]] == c]
            if len(ii) >= 500:
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
        # rigid: init from the object copies' centroid offsets (the copies
        # sit ~1 m apart — a plane-offset init + trimmed ICP converged to
        # a local minimum 25 cm along the way), then trimmed ICP on the
        # object points only
        offs = []
        for cp in cp_here:
            if cp["iid"] in ref and cp["iid"] in seen:
                offs.append(ref[cp["iid"]]["centroid"] - seen[cp["iid"]])
        if not offs:
            big = max(cp_here, key=lambda cp: len(cp["idx"]))
            n_pl, c_pl = _fit_plane(xyz[ref[big["iid"]]["seg_idx"]], rng)
            off = float(np.median((S - c_pl) @ n_pl))
            init_t = -off * n_pl
        else:
            init_t = np.mean(offs, axis=0)
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
        # GUARD (docs/MEJORAS_OBLIGATORIAS.md): the copies must COLLAPSE.
        # A solve that leaves the object copies apart is not a correction —
        # nothing is applied, the user gets the numbers instead.
        res_after = float(np.median(d_after))
        res_before = float(np.median(d_before))
        if res_after > MAX_RESIDUAL_M or res_after > 0.5 * res_before:
            raise RuntimeError(
                f"ch{c:02d}: the copies did NOT collapse — object residual "
                f"{res_before*100:.1f} → {res_after*100:.1f} cm (limit "
                f"{MAX_RESIDUAL_M*100:.0f} cm and half of before); rigid "
                f"{_rot_deg(R):.2f}° / {np.linalg.norm(t_full):.2f} m. "
                f"Nothing was applied.")
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
                 "diagnosis": f"inherited (only {n_pts} object pts of its "
                              f"own — takes ch{nearest:02d}'s solution)",
                 "rigid": src_e["rigid"],
                 "_R": src_e["_R"], "_t": src_e["_t"],
                 "_k": src_e["_k"]}
        per_chunk[c] = entry
        log(f"  ch{c:02d}: only {n_pts} bbox pts — INHERITS ch{nearest:02d}"
            f"'s transform (nothing inside the bbox is skipped)")

    # 4b) LOOP-CLOSURE DISTRIBUTION PER KEYFRAME (USER 2026-09-06, after
    # two failed variants: per-chunk interpolation multiplied the seam —
    # ~18 cm between every pair of neighbouring chunks — and moving only
    # the displaced chunks duplicated everything chunk 5 and 6 both saw).
    # The revisit's error accumulated along the trajectory (VGGT-Long
    # aligns chunk after chunk), so the correction is spread over
    # KEYFRAMES: identity up to the last keyframe of the reference visit,
    # the solved transform at each solved chunk's evidence keyframes
    # (anchor at the first of its copy frames), slerp(yaw)+lerp(t)
    # in between, the last anchor's transform from there on. Neighbouring
    # keyframes differ by millimetres — no seam anywhere; the copies still
    # land exactly on the reference. Every point/camera is warped by ITS
    # keyframe (provenance).
    from scipy.spatial.transform import Rotation, Slerp
    kf_ref_end = max(kf_index.get(f, -1) for r in ref.values()
                     for f in r["frames"])
    anchors = {kf_ref_end: (np.eye(3), np.zeros(3))}
    for c, e in per_chunk.items():
        if e.get("kf"):
            # anchor at the FIRST evidence keyframe: every frame that saw
            # the copy gets the full solved transform (copies land exact)
            a_kf = int(e["kf"][0])
        else:                                   # inherited: chunk middle
            a_kf = int(min(c * 30 + 15, len(frames) - 1))
        if a_kf <= kf_ref_end:
            continue
        anchors[a_kf] = (e["_R"], e["_t"])
    a_kfs = sorted(anchors)
    if len(a_kfs) < 2:
        raise RuntimeError("no anchor keyframe after the reference visit")
    rots = Rotation.from_matrix(np.stack([anchors[k][0] for k in a_kfs]))
    slerp = Slerp(a_kfs, rots)
    n_kf = len(frames)
    R_kf = np.tile(np.eye(3), (n_kf, 1, 1))
    t_kf = np.zeros((n_kf, 3))
    for k in range(n_kf):
        if k <= a_kfs[0]:
            continue
        if k >= a_kfs[-1]:
            R_kf[k], t_kf[k] = anchors[a_kfs[-1]]
            continue
        lo = max(a for a in a_kfs if a <= k)
        hi = min(a for a in a_kfs if a > k)
        w = (k - lo) / (hi - lo)
        R_kf[k] = slerp([k]).as_matrix()[0]
        t_kf[k] = (1 - w) * anchors[lo][1] + w * anchors[hi][1]
    step = [float(np.linalg.norm(t_kf[k] - t_kf[k - 1]))
            for k in range(1, n_kf)]
    distribution = {
        "identity_until_kf": int(kf_ref_end),
        "anchors": [{"kf": int(k), "rot_deg": round(_rot_deg(anchors[k][0]), 3),
                     "t_m": round(float(np.linalg.norm(anchors[k][1])), 4)}
                    for k in a_kfs],
        "keyframes_warped": int(n_kf - 1 - a_kfs[0]),
        "max_step_between_keyframes_mm": round(max(step) * 1000, 1),
    }
    log(f"  loop closure spread over keyframes {a_kfs[0]+1}..{n_kf-1}: "
        f"anchors {[(k, round(_rot_deg(anchors[k][0]), 2), round(float(np.linalg.norm(anchors[k][1])), 3)) for k in a_kfs]}, "
        f"max step between neighbouring keyframes "
        f"{max(step)*1000:.1f} mm — no seam")

    # 5) APPLY: one-level undo backup, then bake --------------------------
    _p(55, "applying: backup (one-level undo) + rewrite cloud & poses...")
    undo = output_dir / UNDO_DIR
    if undo.exists():
        shutil.rmtree(undo)
    undo.mkdir()
    shutil.copy2(ply_path, undo / "cleaned_cloud.ply")
    shutil.copy2(output_dir / "camera_poses.txt", undo / "camera_poses.txt")
    if (output_dir / "segmentation_result.json").exists():
        shutil.copy2(output_dir / "segmentation_result.json",
                     undo / "segmentation_result.json")

    xyz_new = xyz
    # depth expansion stays per chunk (along each point's own ray)
    for c, e in per_chunk.items():
        if e["_k"] != 1.0:
            mc = chunk == c
            cams = np.stack([cam_center.get(int(f), np.zeros(3))
                             for f in fg[mc]])
            xyz_new[mc] = cams + (xyz_new[mc] - cams) * e["_k"]
    # rigid warp per KEYFRAME: points grouped by their keyframe index
    n_moved = 0
    order = np.argsort(ks, kind="stable")
    bounds = np.searchsorted(ks[order], np.arange(-1, n_kf + 1))
    for k in range(n_kf):
        if k <= a_kfs[0]:
            continue
        sel = order[bounds[k + 1]:bounds[k + 2]]
        if not len(sel):
            continue
        xyz_new[sel] = xyz_new[sel] @ R_kf[k].T + t_kf[k]
        n_moved += len(sel)
    data["x"] = xyz_new[:, 0].astype(data.dtype["x"])
    data["y"] = xyz_new[:, 1].astype(data.dtype["y"])
    data["z"] = xyz_new[:, 2].astype(data.dtype["z"])
    _write_ply(ply_path, header, data)
    log(f"  cleaned_cloud.ply rewritten: {n_moved:,} pts warped "
        f"(keyframes {a_kfs[0]+1}..{n_kf-1})")
    _update_result_obbs(output_dir, xyz_new, log)

    # cameras follow their keyframe's transform
    n_cam = 0
    for k in range(n_kf):
        if k <= a_kfs[0]:
            continue
        R, t = R_kf[k], t_kf[k]
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
                    if k not in ("idx", "seg_idx", "centroid", "frames")} | {
                        "n_points": len(cp["idx"]),
                        "n_object_points": len(cp["seg_idx"]),
                        "n_frames": len(cp["frames"]),
                        "is_reference": cp is ref.get(cp["iid"])}
                   for cp in copies],
        "chunks": [{k: v for k, v in e.items()
                    if not k.startswith("_")}
                   for e in per_chunk.values()],
        "distribution": distribution,
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
        "keyframes_warped": distribution["keyframes_warped"],
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
    if (undo / "floor_transform.npz").exists():
        shutil.copy2(undo / "floor_transform.npz",
                     output_dir / "floor_transform.npz")
    if (undo / "segmentation_result.json").exists():
        shutil.copy2(undo / "segmentation_result.json",
                     output_dir / "segmentation_result.json")
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


# ── chunk boxes + manual gizmo correction (USER 2026-09-06) ──────────────
def compute_chunk_boxes(output_dir: Path) -> dict:
    """Floor-aligned OBB per chunk (center, size, yaw) from provenance —
    the viewer draws them (all deselected by default) so the user can SEE
    where each chunk lies and probe errors with a gizmo. Cached by cloud
    mtime in chunk_boxes.json."""
    output_dir = Path(output_dir)
    ply_path = output_dir / "cleaned_cloud.ply"
    cache_p = output_dir / "chunk_boxes.json"
    mtime = ply_path.stat().st_mtime
    if cache_p.exists():
        try:
            cached = json.loads(cache_p.read_text())
            if cached.get("cloud_mtime") == mtime:
                return cached
        except Exception:  # noqa: BLE001
            pass
    _, data = _read_ply(ply_path)
    xyz = np.stack([data["x"], data["y"], data["z"]],
                   axis=1).astype(np.float64)
    fg = data["frame_global"].astype(np.int64)
    frames = [int(x) for x in
              (output_dir / "camera_frames.txt").read_text().split()]
    fmax = int(fg.max())
    kf_arr = np.full(fmax + 1, -1, dtype=np.int64)
    for k, f in enumerate(frames):
        if f <= fmax:
            kf_arr[f] = k
    ks = kf_arr[np.clip(fg, 0, fmax)]
    n_chunks = (len(frames) + 29) // 30
    chunk = np.where(ks >= 0, np.minimum(ks // 30, n_chunks - 1), -1)
    boxes = []
    rng = np.random.default_rng(0)
    for c in range(n_chunks):
        idx = np.where(chunk == c)[0]
        if len(idx) < 100:
            continue
        if len(idx) > 200000:
            idx = idx[rng.choice(len(idx), 200000, replace=False)]
        P = xyz[idx]
        cen = P.mean(0)
        # yaw from XZ covariance (floor-aligned box)
        Q = P[:, [0, 2]] - cen[[0, 2]]
        w, V = np.linalg.eigh(Q.T @ Q)
        ax = V[:, np.argmax(w)]
        yaw = float(np.arctan2(ax[1], ax[0]))
        ca, sa = np.cos(-yaw), np.sin(-yaw)
        u = ca * Q[:, 0] - sa * Q[:, 1]
        v = sa * Q[:, 0] + ca * Q[:, 1]
        y = P[:, 1] - cen[1]
        lo = np.percentile(np.stack([u, y, v], 1), 1, axis=0)
        hi = np.percentile(np.stack([u, y, v], 1), 99, axis=0)
        boxes.append({
            "chunk": int(c),
            "center": [round(float(x), 4) for x in
                       (cen + [(lo[0] + hi[0]) / 2 * np.cos(yaw)
                               - (lo[2] + hi[2]) / 2 * np.sin(yaw),
                               (lo[1] + hi[1]) / 2,
                               (lo[0] + hi[0]) / 2 * np.sin(yaw)
                               + (lo[2] + hi[2]) / 2 * np.cos(yaw)])],
            "size": [round(float(hi[i] - lo[i]), 4) for i in range(3)],
            "yaw": round(yaw, 5),
            "n_points": int(len(np.where(chunk == c)[0])),
        })
    out = {"cloud_mtime": mtime, "n_chunks": n_chunks, "boxes": boxes,
           "provenance": "tool_measured"}
    cache_p.write_text(json.dumps(out))
    return out


def apply_manual_chunk(output_dir: Path, chunk_id: int, matrix16: list,
                       log=_log_default, progress=None) -> dict:
    """Bake the user's gizmo adjustment of ONE chunk (USER 2026-09-06:
    'el gizmo permite mover y si pone guardar queda ajustado y se debe
    recalcular cleaned cloud y potree'). p' = R p + t for every point of
    the chunk, cameras move with it; one-level undo backup; Potree
    deleted and rebuilt. Same Approve/Undo flow as the automatic
    correction."""
    output_dir = Path(output_dir)
    state = load_state(output_dir)
    if state.get("status") == "pending":
        raise RuntimeError("a correction is pending approval — approve or "
                           "undo it before saving a manual one")
    M = np.array(matrix16, dtype=np.float64).reshape(4, 4)
    R, t = M[:3, :3], M[:3, 3]
    # guard: rotation must be orthonormal (no scale — the gizmo has none)
    if abs(abs(np.linalg.det(R)) - 1.0) > 1e-3:
        raise RuntimeError("transform carries scale/shear — only rotation "
                           "+ translation are allowed")

    def _p(pct, msg):
        log(msg)
        if progress:
            progress(pct, msg)

    _p(5, f"manual correction of chunk {chunk_id}: rot "
          f"{_rot_deg(R):.2f}°, |t| {float(np.linalg.norm(t)):.3f} m")
    ply_path = output_dir / "cleaned_cloud.ply"
    header, data = _read_ply(ply_path)
    xyz = np.stack([data["x"], data["y"], data["z"]],
                   axis=1).astype(np.float64)
    fg = data["frame_global"].astype(np.int64)
    frames = [int(x) for x in
              (output_dir / "camera_frames.txt").read_text().split()]
    kf_index = {f: k for k, f in enumerate(frames)}
    fmax = int(fg.max())
    kf_arr = np.full(fmax + 1, -1, dtype=np.int64)
    for f, k in kf_index.items():
        if f <= fmax:
            kf_arr[f] = k
    ks = kf_arr[np.clip(fg, 0, fmax)]
    n_chunks = (len(frames) + 29) // 30
    chunk = np.where(ks >= 0, np.minimum(ks // 30, n_chunks - 1), -1)
    mc = chunk == int(chunk_id)
    if not mc.any():
        raise RuntimeError(f"chunk {chunk_id} has no points")

    _p(30, "backup (one-level undo) + rewriting cloud & poses...")
    undo = output_dir / UNDO_DIR
    if undo.exists():
        shutil.rmtree(undo)
    undo.mkdir()
    shutil.copy2(ply_path, undo / "cleaned_cloud.ply")
    shutil.copy2(output_dir / "camera_poses.txt",
                 undo / "camera_poses.txt")
    if (output_dir / "segmentation_result.json").exists():
        shutil.copy2(output_dir / "segmentation_result.json",
                     undo / "segmentation_result.json")
    xyz[mc] = xyz[mc] @ R.T + t
    data["x"] = xyz[:, 0].astype(data.dtype["x"])
    data["y"] = xyz[:, 1].astype(data.dtype["y"])
    data["z"] = xyz[:, 2].astype(data.dtype["z"])
    _write_ply(ply_path, header, data)
    poses = []
    for ln in (output_dir / "camera_poses.txt").read_text().splitlines():
        vals = [float(x) for x in ln.split()]
        if len(vals) == 16:
            poses.append(np.array(vals).reshape(4, 4))
    n_cam = 0
    for f, k in kf_index.items():
        if min(k // 30, n_chunks - 1) == int(chunk_id):
            poses[k][:3, :3] = R @ poses[k][:3, :3]
            poses[k][:3, 3] = R @ poses[k][:3, 3] + t
            n_cam += 1
    (output_dir / "camera_poses.txt").write_text("\n".join(
        " ".join(f"{x:.9f}" for x in P.reshape(-1)) for P in poses) + "\n")
    log(f"  {int(mc.sum()):,} pts + {n_cam} cameras moved")
    _update_result_obbs(output_dir, xyz, log)

    _p(60, "deleting Potree and rebuilding from the corrected cloud...")
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
        raise RuntimeError("Potree rebuild FAILED — undo is available "
                           "(correction_undo/)")
    _save_state(output_dir, {
        "status": "pending", "mode": "manual",
        "applied_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "chunks": [int(chunk_id)],
        "points_moved": int(mc.sum()),
        "rigid": {"rot_deg": round(_rot_deg(R), 3),
                  "t_m": round(float(np.linalg.norm(t)), 4)},
    })
    _p(100, f"✅ manual chunk correction applied — awaiting your verdict "
            f"(Approve or Undo)")
    return load_state(output_dir)


def align_floor_y0(output_dir: Path, chunk_ids: List[int],
                   log=_log_default, progress=None) -> dict:
    """USER DESIGN 2026-09-06: RANSAC each SELECTED chunk's floor plane and
    put it at Y=0 — tilt correction included (the floor normal is
    straightened to vertical, pivoting on the chunk's floor centroid) plus
    the vertical translation. UNSELECTED chunks accompany their neighbours:
    drift is smooth along the walk, so selected chunks act as ANCHORS and
    every other chunk gets the interpolated correction (slerp rotation +
    lerp Δy by chunk index; constant extension past the first/last anchor).
    A real step between two anchors is PRESERVED — its chunk moves like its
    neighbours and keeps the relative relation, losing only the drift.
    A selected chunk whose floor RANSAC fails the guards (near-vertical
    normal, enough inliers) is DEMOTED to interpolated and loudly reported.
    One pending correction → Approve/Undo, like everything else."""
    from scipy.spatial.transform import Rotation, Slerp
    output_dir = Path(output_dir)
    state = load_state(output_dir)
    if state.get("status") == "pending":
        raise RuntimeError("a correction is pending approval — approve or "
                           "undo it before running floor alignment")
    rng = np.random.default_rng(0)

    def _p(pct, msg):
        log(msg)
        if progress:
            progress(pct, msg)

    _p(2, "loading cloud + provenance...")
    ply_path = output_dir / "cleaned_cloud.ply"
    header, data = _read_ply(ply_path)
    xyz = np.stack([data["x"], data["y"], data["z"]],
                   axis=1).astype(np.float64)
    fg = data["frame_global"].astype(np.int64)
    frames = [int(x) for x in
              (output_dir / "camera_frames.txt").read_text().split()]
    kf_index = {f: k for k, f in enumerate(frames)}
    fmax = int(fg.max())
    kf_arr = np.full(fmax + 1, -1, dtype=np.int64)
    for f, k in kf_index.items():
        if f <= fmax:
            kf_arr[f] = k
    ks = kf_arr[np.clip(fg, 0, fmax)]
    n_chunks = (len(frames) + 29) // 30
    chunk = np.where(ks >= 0, np.minimum(ks // 30, n_chunks - 1), -1)

    # 1) floor RANSAC per SELECTED chunk → anchor transforms ---------------
    UP = np.array([0.0, 1.0, 0.0])
    MAX_TILT_DEG = 10.0
    MIN_INLIERS = 5000
    anchors = {}          # chunk → (R 3x3, t 3)
    per_chunk_report = []
    for c in sorted(set(int(x) for x in chunk_ids)):
        idx = np.where(chunk == c)[0]
        if len(idx) < MIN_INLIERS:
            per_chunk_report.append({"chunk": c, "role": "demoted",
                                     "why": f"only {len(idx)} points"})
            log(f"  ch{c:02d}: DEMOTED to interpolated — only "
                f"{len(idx)} points")
            continue
        P = xyz[idx]
        # lowest band: floor candidates
        y0 = np.percentile(P[:, 1], 5)
        band = P[(P[:, 1] < y0 + 0.5)]
        if len(band) < MIN_INLIERS:
            per_chunk_report.append({"chunk": c, "role": "demoted",
                                     "why": "no low band"})
            log(f"  ch{c:02d}: DEMOTED — no low horizontal band")
            continue
        S = band if len(band) <= 120000 else band[
            rng.choice(len(band), 120000, replace=False)]
        best, bn = None, -1
        cos_max = np.cos(np.radians(MAX_TILT_DEG))
        for _ in range(400):
            a, b_, c_ = S[rng.choice(len(S), 3, replace=False)]
            nrm = np.cross(b_ - a, c_ - a)
            ln = np.linalg.norm(nrm)
            if ln < 1e-9:
                continue
            nrm /= ln
            if nrm[1] < 0:
                nrm = -nrm
            if nrm @ UP < cos_max:
                continue                     # not near-horizontal plane
            cnt = int((np.abs((S - a) @ nrm) < 0.02).sum())
            if cnt > bn:
                bn, best = cnt, (nrm, a)
        if best is None or bn < MIN_INLIERS * 0.5:
            per_chunk_report.append({"chunk": c, "role": "demoted",
                                     "why": f"floor RANSAC failed "
                                            f"({bn} inliers)"})
            log(f"  ch{c:02d}: DEMOTED — floor RANSAC failed "
                f"({bn} inliers)")
            continue
        nrm, a = best
        inl = np.abs((S - a) @ nrm) < 0.03
        c_f = S[inl].mean(0)
        nrm = np.linalg.svd(S[inl] - c_f, full_matrices=False)[2][2]
        if nrm[1] < 0:
            nrm = -nrm
        tilt = float(np.degrees(np.arccos(np.clip(nrm @ UP, -1, 1))))
        # rotation that takes the floor normal to vertical (pivot c_f)
        axis = np.cross(nrm, UP)
        s_ = np.linalg.norm(axis)
        if s_ > 1e-9:
            R = Rotation.from_rotvec(
                axis / s_ * np.arctan2(s_, nrm @ UP)).as_matrix()
        else:
            R = np.eye(3)
        t = c_f - R @ c_f
        t[1] -= float(c_f[1])          # floor plane → y = 0
        anchors[c] = (R, t)
        per_chunk_report.append({
            "chunk": c, "role": "anchor",
            "tilt_deg": round(tilt, 3),
            "dy_m": round(-float(c_f[1]), 4),
            "floor_inliers": int(inl.sum())})
        log(f"  ch{c:02d}: ANCHOR — tilt {tilt:.2f}°, floor at "
            f"{c_f[1]*100:+.1f} cm → Δy {-c_f[1]*100:+.1f} cm "
            f"({int(inl.sum()):,} inliers)")

    if not anchors:
        raise RuntimeError("no selected chunk produced a trustworthy floor "
                           "plane — nothing to align")

    # 2) interpolate every chunk between/past the anchors ------------------
    _p(30, f"interpolating {n_chunks - len(anchors)} unselected chunk(s) "
           f"between {len(anchors)} anchor(s)...")
    a_ids = sorted(anchors.keys())
    rots = Rotation.from_matrix(np.stack([anchors[c][0] for c in a_ids]))
    slerp = Slerp(a_ids, rots) if len(a_ids) > 1 else None
    per_chunk = {}
    for c in range(n_chunks):
        if c in anchors:
            per_chunk[c] = anchors[c]
            continue
        if c <= a_ids[0]:
            per_chunk[c] = anchors[a_ids[0]]
            role = f"extended from ch{a_ids[0]:02d}"
        elif c >= a_ids[-1]:
            per_chunk[c] = anchors[a_ids[-1]]
            role = f"extended from ch{a_ids[-1]:02d}"
        else:
            lo = max(a for a in a_ids if a < c)
            hi = min(a for a in a_ids if a > c)
            w = (c - lo) / (hi - lo)
            R = slerp([c]).as_matrix()[0]
            t = (1 - w) * anchors[lo][1] + w * anchors[hi][1]
            per_chunk[c] = (R, t)
            role = f"interpolated ch{lo:02d}..ch{hi:02d} (w={w:.2f})"
        per_chunk_report.append({"chunk": c, "role": role})

    # 3) apply everything as ONE pending correction ------------------------
    _p(45, "applying: backup (one-level undo) + rewrite cloud & poses...")
    undo = output_dir / UNDO_DIR
    if undo.exists():
        shutil.rmtree(undo)
    undo.mkdir()
    shutil.copy2(ply_path, undo / "cleaned_cloud.ply")
    shutil.copy2(output_dir / "camera_poses.txt",
                 undo / "camera_poses.txt")
    if (output_dir / "segmentation_result.json").exists():
        shutil.copy2(output_dir / "segmentation_result.json",
                     undo / "segmentation_result.json")
    for c, (R, t) in per_chunk.items():
        mc = chunk == c
        if not mc.any():
            continue
        xyz[mc] = xyz[mc] @ R.T + t
    data["x"] = xyz[:, 0].astype(data.dtype["x"])
    data["y"] = xyz[:, 1].astype(data.dtype["y"])
    data["z"] = xyz[:, 2].astype(data.dtype["z"])
    _write_ply(ply_path, header, data)
    poses = []
    for ln in (output_dir / "camera_poses.txt").read_text().splitlines():
        vals = [float(x) for x in ln.split()]
        if len(vals) == 16:
            poses.append(np.array(vals).reshape(4, 4))
    for f, k in kf_index.items():
        c = min(k // 30, n_chunks - 1)
        R, t = per_chunk[c]
        poses[k][:3, :3] = R @ poses[k][:3, :3]
        poses[k][:3, 3] = R @ poses[k][:3, 3] + t
    (output_dir / "camera_poses.txt").write_text("\n".join(
        " ".join(f"{x:.9f}" for x in P.reshape(-1)) for P in poses) + "\n")
    log(f"  cloud + {len(poses)} cameras rewritten (every chunk moved: "
        f"{len(anchors)} anchored, {n_chunks - len(anchors)} accompanying)")
    _update_result_obbs(output_dir, xyz, log)

    _p(60, "deleting Potree and rebuilding from the corrected cloud...")
    potree = output_dir / "potree"
    if potree.exists():
        shutil.rmtree(potree)
    import sys
    server_dir = str(Path(__file__).resolve().parents[1])
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)
    from potree_converter import convert_ply_to_potree
    if not convert_ply_to_potree(output_dir.parent, force=True):
        raise RuntimeError("Potree rebuild FAILED — undo is available "
                           "(correction_undo/)")
    # the cloud's floor now sits at Y=0 BY CONSTRUCTION — a stale display
    # floor transform (computed on the drifted cloud) would shift the whole
    # scene off the grid (USER 2026-09-06: grid floating 52 cm above the
    # floor). Reset it to identity; the previous one rides the undo backup.
    ft = output_dir / "floor_transform.npz"
    if ft.exists():
        shutil.copy2(ft, undo / "floor_transform.npz")
    np.savez(ft, s=np.float64(1.0), R=np.eye(3), t=np.zeros(3))
    log("  floor_transform reset to identity (floor is at Y=0 by "
        "construction)")
    _save_state(output_dir, {
        "status": "pending", "mode": "floor_align",
        "applied_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "chunks": a_ids,
        "points_moved": int((chunk >= 0).sum()),
        "detail": per_chunk_report,
    })
    _p(100, "✅ floor alignment applied — awaiting your verdict "
            "(Approve or Undo)")
    return load_state(output_dir)


def _update_result_obbs(output_dir: Path, xyz: np.ndarray,
                        log=_log_default) -> None:
    """After a correction moved the cloud's COORDINATES: segment membership
    (globalIndices) is untouched, so the cached segmentation_result.json
    stays valid — only its OBBs are stale. Recompute them in place from the
    moved points (in display space via the session's floor transform) so a
    session load NEVER re-runs matching/DBSCAN (USER 2026-09-06: that runs
    when segmenting, and nowhere else)."""
    res_path = output_dir / "segmentation_result.json"
    if not res_path.exists():
        return
    try:
        result = json.loads(res_path.read_text())
        ft = output_dir / "floor_transform.npz"
        if ft.exists():
            d = np.load(ft)
            s_, R_, t_ = float(d["s"]), d["R"], d["t"]
        else:
            s_, R_, t_ = 1.0, np.eye(3), np.zeros(3)
        from segmentation.pipeline import _compute_obb
        n_done = 0
        for inst in result.get("instances", []):
            g = np.asarray(inst.get("globalIndices") or [], dtype=np.int64)
            g = g[(g >= 0) & (g < len(xyz))]
            if len(g) < 10:
                continue
            disp = (s_ * (xyz[g] @ R_.T)) + t_
            inst["obb"] = _compute_obb(disp)
            n_done += 1
        res_path.write_text(json.dumps(result))
        log(f"  segmentation_result.json OBBs updated in place "
            f"({n_done} instance(s)) — no re-matching needed")
    except Exception as e:  # noqa: BLE001
        log(f"  OBB update failed (non-fatal): {e}")
