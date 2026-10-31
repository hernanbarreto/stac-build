"""The numbers the precision audit (claude_stac.txt "ACTUALIZACIÓN 2026-10-23", step 4) asks to read
after a session ran the audited chain — printed, and written as JSON beside them. Numbers for
Hernán's eye; never a verdict.

    python -m tools.precision_audit_summary --session <scan dir> [--out <json>]

Reads (each one optional — an absent file is said, never assumed): output/precision/refine.json
(odometry, prior_sigma_ladder, loops, scale_lock, rolling_shutter, classical_cross_check, rungs),
output/precision/tracks.json (classical_loops, loop_bridge), output/precision/depth_on_f5.json
(fusion: owners / consumed / free_space / unverified / representation, bend model, tier 0),
output/rejected_points.npz (observations not published, by reason and by keyframe),
output/precision/classical/loop_candidates.json, output/certify_acta.json (the correction's epochs)."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def _load(p: Path) -> Optional[Dict[str, Any]]:
    try:
        d = json.loads(p.read_text())
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


def _pick(d: Optional[Dict[str, Any]], *keys, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def summarise(session_dir: Path) -> Dict[str, Any]:
    out_dir = Path(session_dir) / "output"
    prec = out_dir / "precision"
    s: Dict[str, Any] = {"session": str(session_dir), "files": {}}
    rf = _load(prec / "refine.json")
    s["files"]["refine.json"] = rf is not None
    if rf:
        rungs = rf.get("rungs") or {}
        s["refine"] = {
            "applied_rung": rf.get("applied_rung"), "n_landmarks": rf.get("n_landmarks"),
            "rungs": {k: {kk: v.get(kk) for kk in ("taken", "fit_rms_px", "heldout_median_px", "n_heldout_tracks",
                                                   "n_points", "termination", "applied", "would_improve")
                          if isinstance(v, dict) and kk in v}
                      for k, v in rungs.items() if isinstance(v, dict)},
            "odometry": rf.get("odometry"), "prior_sigma_ladder": rf.get("prior_sigma_ladder"),
            "loops": rf.get("loops"), "scale_lock": rf.get("scale_lock"),
            "rolling_shutter": {k: v for k, v in (rf.get("rolling_shutter") or {}).items()
                                if k in ("tau_frames", "taken", "search", "verdict", "reason", "heldout_before_px",
                                         "heldout_after_px", "model")} if isinstance(rf.get("rolling_shutter"), dict) else rf.get("rolling_shutter"),
            "classical_cross_check": rf.get("classical_cross_check"),
            "camera": rf.get("camera"), "gauge": rf.get("gauge"), "judge": rf.get("judge"),
        }
    tr = _load(prec / "tracks.json")
    s["files"]["tracks.json"] = tr is not None
    if tr:
        s["tracks"] = {k: tr.get(k) for k in ("n_tracks", "n_obs", "n_keyframes", "n_witnesses", "windows",
                                              "classical_loops", "loop_bridge") if k in tr}
        lb = tr.get("loop_bridge") or {}
        if isinstance(lb, dict) and isinstance(lb.get("pairs"), dict):
            pairs = lb["pairs"]
            s["tracks"]["loop_bridge_summary"] = {
                "pairs": len(pairs), "bridged": sum(1 for p in pairs.values() if p.get("bridged_tracks")),
                "not_bridged": [k for k, p in pairs.items() if p.get("reason")],
                "bridged_tracks_total": lb.get("bridged_tracks"), "links": lb.get("links")}
    d5 = _load(prec / "depth_on_f5.json")
    s["files"]["depth_on_f5.json"] = d5 is not None
    if d5:
        s["depth_on_f5"] = {k: d5.get(k) for k in ("fusion", "bend", "tier0", "colmap_tier", "n_keyframes", "n_points",
                                                   "representation", "free_space", "unverified") if k in d5}
    rp = out_dir / "rejected_points.npz"
    s["files"]["rejected_points.npz"] = rp.is_file()
    if rp.is_file():
        try:
            with np.load(rp, allow_pickle=True) as z:
                keys = list(z.keys())
                rec: Dict[str, Any] = {"keys": keys, "n": int(len(z[keys[0]])) if keys else 0}
                if "reason" in z:
                    reasons = z["reason"]
                    rec["by_reason"] = dict(Counter(str(r) for r in reasons.tolist()))
                for fk in ("frame_global", "frame", "keyframe"):
                    if fk in z:
                        fr = z[fk]
                        c = Counter(int(f) for f in fr.tolist())
                        rec["by_keyframe_top10"] = c.most_common(10)
                        rec["keyframes_with_rejections"] = len(c)
                        break
                s["rejected_points"] = rec
        except Exception as e:  # noqa: BLE001
            s["rejected_points"] = {"error": f"{type(e).__name__}: {e}"}
    lc = _load(prec / "loop_candidates.json")
    s["files"]["loop_candidates.json"] = lc is not None
    if lc:
        s["classical_loop_candidates"] = {k: lc.get(k) for k in ("n_pairs", "pairs", "reason", "reconstruction_id",
                                                                 "taken", "n_candidates") if k in lc}
        if isinstance(lc.get("pairs"), list):
            s["classical_loop_candidates"]["n_pairs"] = len(lc["pairs"])
            s["classical_loop_candidates"].pop("pairs", None)
    acta = _load(out_dir / "certify_acta.json")
    s["files"]["certify_acta.json"] = acta is not None
    if acta:
        s["certification"] = {k: acta.get(k) for k in ("epochs", "epoch", "verdict", "gate_warnings", "stages", "repeatability",
                                                        "closures", "n_closures") if k in acta}
    ge = _load(out_dir / "geometry_epoch.json")
    s["geometry_epoch"] = ge.get("epoch") if ge else None
    cam = _load(out_dir / "camera.json")
    if cam:
        s["camera"] = {"params": cam.get("params"), "source": cam.get("source"), "camera_epoch": cam.get("camera_epoch"),
                       "report": cam.get("report")}
    return s


def classical_series(session_dir: Path, out_dir: Path) -> Optional[Dict[str, Any]]:
    """One series per classical mapper found (``sparse_<mapper>`` under output/precision/classical):
    {mapper: record}; a declaration when none exists."""
    cl = Path(session_dir) / "output" / "precision" / "classical"
    dirs = sorted(d for d in cl.glob("sparse_*") if d.is_dir()) if cl.exists() else []
    if not dirs:
        return {"measured": False, "reason": f"{cl}: no sparse_<mapper> model directory"}
    return {d.name[len("sparse_"):]: _classical_series_one(session_dir, out_dir, d, "_" + d.name[len("sparse_"):])
            for d in dirs}


def _classical_series_one(session_dir: Path, out_dir: Path, sparse: Path, tag: str) -> Optional[Dict[str, Any]]:
    """WHERE the classical (COLMAP / GLOMAP) trajectory and F5's part, keyframe by keyframe: the
    rotation residual after the best common world rotation and the position residual after the
    Sim3 alignment (the cross-check's own recipe), plus the consecutive-keyframe rotation STEPS of
    both trajectories — a step of 90° or 180° in one of them, against ~8° in the other, is a
    broken segment of that model, not drift. CSV + PNG in ``out_dir``; None when the model or the
    poses are missing (said)."""
    from evaluation.colmap_io import frame_of_name, list_models, read_model
    from precision.classical_check import umeyama_sim3
    out = Path(session_dir) / "output"
    poses_p, frames_p = out / "camera_poses.txt", out / "camera_frames.txt"
    if not sparse.exists() or not poses_p.is_file() or not frames_p.is_file():
        return {"measured": False, "reason": f"missing {sparse if not sparse.exists() else poses_p}"}
    dirs = list_models(sparse)
    if not dirs:
        return {"measured": False, "reason": f"{sparse}: no model"}
    model = max((read_model(d) for d in dirs), key=lambda m: m.n_images)
    frames = [int(x) for x in frames_p.read_text().split()]
    kidx = {f: i for i, f in enumerate(frames)}
    P = np.loadtxt(str(poses_p)).reshape(-1, 4, 4)
    rows = sorted((kidx[frame_of_name(im.name)], im.w2c()) for im in model.images.values()
                  if frame_of_name(im.name) in kidx)
    if len(rows) < 3:
        return {"measured": False, "reason": f"{len(rows)} registered keyframe(s)"}
    idx = np.array([r[0] for r in rows])
    Wc = np.stack([r[1] for r in rows])
    Rc, tc = Wc[:, :3, :3], Wc[:, :3, 3]
    Cc = -np.einsum("nji,nj->ni", Rc, tc)
    W5 = np.linalg.inv(P[idx])
    R5 = W5[:, :3, :3]
    C5 = P[idx][:, :3, 3]
    s_, Rw, t_ = umeyama_sim3(Cc, C5)
    pos = np.linalg.norm((s_ * (Cc @ Rw.T) + t_) - C5, axis=1)
    M = sum(Rc[k].T @ R5[k] for k in range(len(idx)))
    U, _s, Vt = np.linalg.svd(M)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    A = U @ D @ Vt
    def _ang(X):
        return float(np.degrees(np.arccos(np.clip((np.trace(X) - 1.0) / 2.0, -1.0, 1.0))))
    rot = np.array([_ang((Rc[k] @ A) @ R5[k].T) for k in range(len(idx))])
    step_c = np.array([_ang(Rc[k + 1] @ Rc[k].T) for k in range(len(idx) - 1)])
    step_5 = np.array([_ang(R5[k + 1] @ R5[k].T) for k in range(len(idx) - 1)])
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_p = out_dir / f"classical_vs_f5_by_keyframe{tag}.csv"
    with open(csv_p, "w") as f:
        f.write("keyframe_index,frame,rotation_residual_deg,position_residual_m,step_classical_deg,step_f5_deg\n")
        for k in range(len(idx)):
            sc = step_c[k] if k < len(step_c) else float("nan")
            s5 = step_5[k] if k < len(step_5) else float("nan")
            f.write(f"{int(idx[k])},{frames[idx[k]]},{rot[k]:.4f},{pos[k]:.5f},{sc:.4f},{s5:.4f}\n")
    rec = {"measured": True, "n_registered": int(len(idx)), "scale_f5_over_classical": float(s_),
           "rotation_residual_deg": {"median": float(np.median(rot)), "p90": float(np.percentile(rot, 90))},
           "position_residual_m": {"median": float(np.median(pos)), "p90": float(np.percentile(pos, 90))},
           "largest_step_disagreements": [{"from_kf": int(idx[k]), "to_kf": int(idx[k + 1]),
                                           "classical_step_deg": float(step_c[k]), "f5_step_deg": float(step_5[k])}
                                          for k in np.argsort(-np.abs(step_c - step_5))[:8]],
           "csv": str(csv_p),
           "reading": "a consecutive-keyframe rotation step of 90-180 deg in ONE trajectory where the other steps "
                      "a few degrees is a broken segment of that model, not drift"}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
        axes[0].plot(idx, rot, ".", label="rotation residual classical vs F5 (deg)")
        axes[0].plot(idx, 100 * pos, ".", label="position residual (cm, after Sim3)")
        axes[0].set_title(f"classical ({tag.strip(chr(95))} mapper) against F5, keyframe by keyframe"); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
        axes[1].plot(idx[:-1], step_c, ".", label="classical consecutive step (deg)")
        axes[1].plot(idx[:-1], step_5, ".", label="F5 consecutive step (deg)")
        axes[1].set_xlabel("keyframe index"); axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        png_p = out_dir / f"classical_vs_f5_by_keyframe{tag}.png"
        fig.savefig(png_p, dpi=120)
        plt.close(fig)
        rec["png"] = str(png_p)
    except Exception as e:  # noqa: BLE001
        rec["png_error"] = str(e)
    return rec


def _fmt(v, nd=4):
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return str(v)


def print_summary(s: Dict[str, Any]) -> None:
    print(f"== precision audit summary — {s['session']}")
    print("files:", ", ".join(f"{k}={'yes' if v else 'NO'}" for k, v in s["files"].items()))
    print("geometry epoch:", s.get("geometry_epoch"))
    if "camera" in s:
        p = s["camera"]["params"] or []
        print(f"camera ({s['camera'].get('source')}, epoch {s['camera'].get('camera_epoch')}): "
              + (f"fx {p[0]:.2f} fy {p[1]:.2f} cx {p[2]:.2f} cy {p[3]:.2f} k1 {p[4]:+.4f} k2 {p[5]:+.4f}" if len(p) >= 6 else str(p)))
    rf = s.get("refine")
    if rf:
        print(f"F5: applied rung {rf.get('applied_rung')}, landmarks {rf.get('n_landmarks')}")
        for k, v in (rf.get("rungs") or {}).items():
            print(f"  rung {k}: taken {v.get('taken')} fit rms {_fmt(v.get('fit_rms_px'))} px, held-out median "
                  f"{_fmt(v.get('heldout_median_px'))} px over {v.get('n_heldout_tracks')} tracks, {v.get('termination')}")
        for key in ("odometry", "prior_sigma_ladder", "loops", "scale_lock", "rolling_shutter", "classical_cross_check", "judge"):
            if rf.get(key) is not None:
                print(f"  {key}: {json.dumps(rf[key], default=str)[:600]}")
    tr = s.get("tracks")
    if tr:
        print("F4:", json.dumps({k: v for k, v in tr.items() if k in ("n_tracks", "n_obs", "classical_loops", "loop_bridge_summary")},
                                default=str)[:900])
    d5 = s.get("depth_on_f5")
    if d5:
        print("F6:", json.dumps(d5, default=str)[:1200])
    rp = s.get("rejected_points")
    if rp:
        print("rejected observations:", json.dumps({k: v for k, v in rp.items() if k != "keys"}, default=str)[:600])
    if s.get("classical_loop_candidates"):
        print("classical loop candidates:", json.dumps(s["classical_loop_candidates"], default=str)[:300])
    if s.get("certification"):
        print("certification:", json.dumps(s["certification"], default=str)[:800])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.precision_audit_summary")
    ap.add_argument("--session", required=True)
    ap.add_argument("--out", default=None, help="JSON path (default <session>/output/precision/audit_summary.json)")
    args = ap.parse_args(argv)
    s = summarise(Path(args.session))
    s["classical_by_keyframe"] = classical_series(Path(args.session), Path(args.session) / "output" / "precision")
    print_summary(s)
    cs = s.get("classical_by_keyframe") or {}
    if cs.get("measured"):
        print(f"classical vs F5 by keyframe: rotation median {cs['rotation_residual_deg']['median']:.1f} deg, position median "
              f"{cs['position_residual_m']['median'] * 100:.0f} cm; largest step disagreements "
              f"{[(d['from_kf'], d['to_kf'], round(d['classical_step_deg'], 1), round(d['f5_step_deg'], 1)) for d in cs['largest_step_disagreements'][:4]]} "
              f"→ {cs['csv']}")
    out = Path(args.out) if args.out else Path(args.session) / "output" / "precision" / "audit_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(s, indent=1, default=str))
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
