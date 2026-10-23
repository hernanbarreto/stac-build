"""The classical cross-check — COLMAP 4.x / GLOMAP as an INDEPENDENT photogrammetry of the same
keyframes (audit 2026-10-23, phase 3d; claude_stac.txt plan h2; USER: "lo mismo con colmap …
usa lo mejor que existe").

Two things, two moments:
  1. BEFORE F4 (the chain's step f3_classical): SIFT features, sequential matching with the
     vocabulary-tree loop detection, the global mapper — ``evaluation.colmap_baseline``'s own
     commands (the battery's h2), in its own env, no learned model anywhere. Every VERIFIED
     two-view geometry between keyframes far apart along the walk is a LOOP CANDIDATE
     (``loop_candidates.json``, stamped with the reconstruction id and the keyframe list): F4 opens
     a loop window on each, so the revisits SALAD missed still constrain the drift.
  2. AFTER F5 (``cross_check`` in refine.json): the classical trajectory Sim3-aligned to F5's over
     the registered keyframes (Umeyama with scale — COLMAP's metre is its own — one Huber
     reweighting at the median), the discrepancy per keyframe in metres and degrees, the registered
     share. Two photogrammetries that share nothing but the pixels agreeing within their errors
     CERTIFY the poses; where they part, the keyframes are named. An instrument, never a correction.

Outputs ``output/precision/classical/`` (the COLMAP workspace: database, sparse_global) and
``classical.json``; a report of another reconstruction or keyframe list is refused by name.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

DIRNAME = "classical"
REPORT_NAME = "classical.json"
LOOPS_NAME = "loop_candidates.json"
LOG_TAG = "[classical]"
VERSION = 1
MAX_IMAGE_ID = 2147483647        # COLMAP's pair_id = image_id1 * MAX_IMAGE_ID + image_id2


class ClassicalError(RuntimeError):
    pass


def keyframe_list_sha(kf: Sequence[int]) -> str:
    return hashlib.sha256(" ".join(str(int(f)) for f in kf).encode()).hexdigest()


def pair_ids(pair_id: int) -> Tuple[int, int]:
    return int(pair_id // MAX_IMAGE_ID), int(pair_id % MAX_IMAGE_ID)


def loop_pairs_from_database(db_path: Path, kf: Sequence[int], min_inliers: int, min_gap: int
                             ) -> List[Tuple[int, int, int]]:
    """(i, j, inliers) over the verified two-view geometries of ``database.db`` between keyframes
    at least ``min_gap`` apart in the walk, with at least ``min_inliers`` inliers."""
    from evaluation.colmap_io import frame_of_name
    kidx = {int(f): i for i, f in enumerate(kf)}
    con = sqlite3.connect(str(db_path))
    try:
        names = {int(r[0]): str(r[1]) for r in con.execute("SELECT image_id, name FROM images")}
        rows = con.execute("SELECT pair_id, rows FROM two_view_geometries").fetchall()
    finally:
        con.close()
    out: List[Tuple[int, int, int]] = []
    for pid, n in rows:
        a, b = pair_ids(int(pid))
        fa, fb = frame_of_name(names.get(a, "")), frame_of_name(names.get(b, ""))
        if fa is None or fb is None or fa not in kidx or fb not in kidx:
            continue
        i, j = kidx[fa], kidx[fb]
        if abs(i - j) < int(min_gap) or int(n or 0) < int(min_inliers):
            continue
        out.append((min(i, j), max(i, j), int(n)))
    out = sorted(set(out))
    return out


def run_classical(session_dir: Path, pcfg, log: Callable = print, runner: Optional[Callable] = None) -> Dict[str, Any]:
    """The step: features + matches + global mapper on the keyframes; the loop candidates and the
    report, stamped. ``runner``: replaces ``evaluation.colmap_baseline``'s two stages (tests)."""
    from correction.epoch import RECONSTRUCTION_ID_KEY, reconstruction_id_or_none
    from intake.quality import read_session_epochs
    from precision.config import raw_of
    session_dir = Path(session_dir)
    out = session_dir / "output"
    cc = pcfg.classical
    raw = raw_of(pcfg)
    try:
        ccfg = dict(raw["evaluation"]["colmap"])
    except (KeyError, TypeError) as e:
        raise ClassicalError("config.yaml: evaluation.colmap (the classical baseline's configuration) is missing") from e
    sel = session_dir / "frames" / "selected_frames.json"
    if not sel.exists():
        raise ClassicalError(f"{sel} is missing — the keyframes are not selected")
    files = sorted(json.loads(sel.read_text()).get("selected_files", []))
    from evaluation.colmap_io import frame_of_name
    kf = [frame_of_name(n) for n in files]
    if any(f is None for f in kf) or len(kf) < 3:
        raise ClassicalError(f"{sel}: {len(kf)} keyframe file(s), need 3 with a frame number each")
    ws = out / "precision" / DIRNAME
    ws.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if runner is None:
        from evaluation import colmap_baseline as CB
        CB.prepare_images(session_dir / "frames", files, ws)
        feat = CB.ensure_features_and_matches(ws, "SIFT", "SIFT_BRUTEFORCE", ccfg, log)
        mapped = CB.map_sparse(ws, "global", ccfg, log)
    else:
        feat, mapped = runner(ws, files, ccfg, log)
    models = sorted(mapped.get("models", []), key=lambda m: -int(m["n_images"]))
    if not models:
        raise ClassicalError("the global mapper registered no model")
    pairs = loop_pairs_from_database(ws / "database.db", kf, int(cc.loop_min_inliers), int(cc.loop_min_gap_keyframes))
    epochs = read_session_epochs(session_dir)
    rid = reconstruction_id_or_none(out)
    stamp = {RECONSTRUCTION_ID_KEY: rid, "keyframes_sha256": keyframe_list_sha(kf), "n_keyframes": len(kf), **epochs}
    loops = {"version": VERSION, "stage": "classical", "provenance": "tool_measured", **stamp,
             "rule": f"verified two-view geometries with >= {int(cc.loop_min_inliers)} inliers between keyframes "
                     f">= {int(cc.loop_min_gap_keyframes)} apart (COLMAP sequential + vocabulary-tree matching)",
             "pairs": [[int(kf[i]), int(kf[j]), int(n)] for i, j, n in pairs]}
    (out / "precision" / LOOPS_NAME).write_text(json.dumps(loops, indent=1))
    rep = {"version": VERSION, "stage": "classical", "provenance": "tool_measured", **stamp,
           "features": feat, "mapping": mapped, "largest_model": models[0],
           "registered_share": float(models[0]["n_images"]) / float(len(kf)),
           "loop_candidates": len(pairs), "seconds": round(time.time() - t0, 1)}
    (ws / REPORT_NAME).write_text(json.dumps(rep, indent=1, default=float))
    log(f"{LOG_TAG} global mapper: {len(models)} model(s), the largest {models[0]['n_images']}/{len(kf)} keyframes; "
        f"{len(pairs)} loop candidate(s) >= {cc.loop_min_gap_keyframes} keyframes apart → {out / 'precision' / LOOPS_NAME}")
    return rep


def classical_loops(out_dir: Path, kf: Sequence[int], rid: Optional[str]) -> Tuple[List[Tuple[int, int]], dict]:
    """F4's reader: the loop candidates as keyframe-INDEX pairs when the file belongs to THIS
    reconstruction and THIS keyframe list; otherwise none, with the reason."""
    from correction.epoch import same_reconstruction
    p = Path(out_dir) / "precision" / LOOPS_NAME
    if not p.exists():
        return [], {"present": False, "taken": False, "reason": "no loop_candidates.json (f3_classical did not run)"}
    doc = json.loads(p.read_text())
    taken, why = same_reconstruction(doc, rid)
    if not taken:
        return [], {"present": True, "taken": False, "reason": why}
    if doc.get("keyframes_sha256") != keyframe_list_sha(kf):
        return [], {"present": True, "taken": False, "reason": "measured on another keyframe list"}
    kidx = {int(f): i for i, f in enumerate(kf)}
    pairs = [(kidx[int(a)], kidx[int(b)]) for a, b, _n in doc.get("pairs", []) if int(a) in kidx and int(b) in kidx]
    return pairs, {"present": True, "taken": True, "n_pairs": len(pairs), "rule": doc.get("rule")}


def umeyama_sim3(src: np.ndarray, dst: np.ndarray, w: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray, np.ndarray]:
    """(s, R, t) with dst ≈ s R src + t, weighted."""
    src = np.asarray(src, np.float64); dst = np.asarray(dst, np.float64)
    w = np.ones(len(src)) if w is None else np.asarray(w, np.float64)
    w = w / w.sum()
    ms = (w[:, None] * src).sum(0); md = (w[:, None] * dst).sum(0)
    S = src - ms; D = dst - md
    H = (S * w[:, None]).T @ D
    U, sig, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Dm = np.diag([1.0, 1.0, d if d != 0 else 1.0])
    R = Vt.T @ Dm @ U.T
    var_s = float(np.sum(w[:, None] * S ** 2))
    s = float(np.sum(sig * np.diag(Dm)) / var_s) if var_s > 0 else 1.0
    t = md - s * (R @ ms)
    return s, R, t


def cross_check(out_dir: Path, kf: Sequence[int], w2c_f5: np.ndarray, rid: Optional[str]) -> Dict[str, Any]:
    """refine.json["classical_cross_check"]: the classical model's poses against F5's."""
    from correction.epoch import same_reconstruction
    from evaluation.colmap_io import frame_of_name, list_models, read_model
    from precision.scale_lock import centres_of
    ws = Path(out_dir) / "precision" / DIRNAME
    rp = ws / REPORT_NAME
    if not rp.exists():
        return {"measured": False, "reason": "no classical model (f3_classical did not run)"}
    rep = json.loads(rp.read_text())
    taken, why = same_reconstruction(rep, rid)
    if not taken:
        return {"measured": False, "reason": why}
    if rep.get("keyframes_sha256") != keyframe_list_sha(kf):
        return {"measured": False, "reason": "the classical model was built on another keyframe list"}
    sparse = ws / "sparse_global"
    dirs = list_models(sparse) if sparse.exists() else []
    if not dirs:
        return {"measured": False, "reason": f"{sparse}: no model"}
    model = max((read_model(d) for d in dirs), key=lambda m: m.n_images)
    kidx = {int(f): i for i, f in enumerate(kf)}
    rows = []
    for im in model.images.values():
        f = frame_of_name(im.name)
        if f is None or f not in kidx:
            continue
        W = np.asarray(im.w2c(), np.float64)
        rows.append((kidx[f], W[:3, :3], W[:3, 3]))
    if len(rows) < 3:
        return {"measured": False, "reason": f"{len(rows)} registered keyframe(s), need 3"}
    idx = np.array([r[0] for r in rows])
    Rc = np.stack([r[1] for r in rows]); tc = np.stack([r[2] for r in rows])
    C_cl = -np.einsum("nji,nj->ni", Rc, tc)                     # classical centres (its own metre)
    W5 = np.asarray(w2c_f5, np.float64)[idx]
    C_f5 = centres_of(W5)
    s, R, t = umeyama_sim3(C_cl, C_f5)
    res = np.linalg.norm((s * (C_cl @ R.T) + t) - C_f5, axis=1)
    med = float(np.median(res))
    if med > 0.0:
        w = np.minimum(1.0, med / np.maximum(res, np.finfo(float).tiny))
        s, R, t = umeyama_sim3(C_cl, C_f5, w)
        res = np.linalg.norm((s * (C_cl @ R.T) + t) - C_f5, axis=1)
    # rotation: the frame rotation between the two worlds from the ROTATIONS themselves — the
    # chordal mean of classical_kᵀ · f5_k over the keyframes (one Huber reweighting) — then the
    # per-keyframe discrepancy angle(classical_k · A vs f5_k), independent of the centres' fit
    def chordal_mean(ws):
        M = sum(w * (Rc[k].T @ W5[k, :3, :3]) for k, w in enumerate(ws))
        U, _s, Vt = np.linalg.svd(M)
        D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
        return U @ D @ Vt

    def angles(A):
        out = []
        for k in range(len(rows)):
            Rd = (Rc[k] @ A) @ W5[k, :3, :3].T
            out.append(float(np.arccos(np.clip((np.trace(Rd) - 1.0) / 2.0, -1.0, 1.0))))
        return np.asarray(out)
    A = chordal_mean(np.ones(len(rows)))
    a0 = angles(A)
    ma = float(np.median(a0))
    if ma > 0.0:
        A = chordal_mean(np.minimum(1.0, ma / np.maximum(a0, np.finfo(float).tiny)))
    ang = np.degrees(angles(A))
    worst = np.argsort(-res)[:5]
    return {"measured": True, "n_registered": int(len(rows)), "n_keyframes": int(len(kf)),
            "registered_share": float(len(rows)) / float(len(kf)),
            "alignment": {"scale_f5_over_classical": float(s), "method": "Umeyama Sim3 on the registered keyframe "
                                                                       "centres, one Huber reweighting at the median"},
            "position_m": {"median": float(np.median(res)), "p90": float(np.percentile(res, 90)),
                           "max": float(res.max())},
            "rotation_deg": {"median": float(np.median(ang)), "p90": float(np.percentile(ang, 90)),
                             "max": float(ang.max())},
            "worst_keyframes": [{"frame": int(kf[idx[k]]), "position_m": float(res[k]), "rotation_deg": float(ang[k])}
                                for k in worst],
            "reading": "two independent photogrammetries of the same pixels: where they agree within their errors "
                       "the poses are certified by both; the worst keyframes name where they part"}


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    import sys
    from precision.config import load_precision_config
    ap = argparse.ArgumentParser(prog="python -m precision.classical_check",
                                 description="COLMAP / GLOMAP on the keyframes: loop candidates for F4, the pose cross-check for F5.")
    ap.add_argument("--session", required=True)
    args = ap.parse_args(argv)
    from repro import require_exclusive_gpu
    require_exclusive_gpu(log=print)
    run_classical(Path(args.session), load_precision_config(session_dir=Path(args.session)))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
