"""§10.12 adversarial suite — synthetic scenarios the pipeline is known to
struggle with, each run end to end (instance loops → certification loop →
witnesses) and each reported with a verdict AND a cause. Criterion: no
silent failure. Scenarios that need real imagery (glass and reflections,
people in motion) are declared ``not_runnable`` with the reason instead of
being faked.

    python -m reconstruction.quality.adversarial --out <dir>
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

SCENARIOS = ("corridor_out_and_back", "symmetric_hall", "rotation_only_stretch", "static_stretch",
             "glass_and_reflections", "moving_people")
NOT_RUNNABLE = {"glass_and_reflections": "needs real imagery (specular surfaces are a photometric "
                                         "failure the analytic renderer cannot produce)",
                "moving_people": "needs real imagery and the Qwen dynamic classification "
                                 "(§4.4); the synthetic scene has no moving instance"}


def _instances_of(sess, labels=("wall", "column", "box")):
    inst, iid = {}, 1
    for p in sess.scene.prims:
        if getattr(p, "label", "") in labels:
            inst[iid] = {"label": p.label, "oids": [p.oid]}
            iid += 1
    return inst


def build_scenario(name: str, root: Path, n_kf: int = 72, H: int = 40, W: int = 56, seed: int = 0):
    """Write the synthetic session of one scenario; returns (session, meta)."""
    import sys
    from tests.synth_metric import (make_session, write_session_dir, write_aligned_chunks, write_images,
                                    synthetic_tracks, corridor_scene, out_and_back_trajectory, hall_scene,
                                    hall_trajectory, corridor_loop_scene, loop_trajectory,
                                    rotation_only_trajectory, static_trajectory, with_stretch, chain_drift)
    if name == "corridor_out_and_back":
        sess = make_session(H=H, W=W, scene=corridor_scene(), poses=out_and_back_trajectory(n_kf))
        steps = np.zeros((n_kf, 6)); steps[:, 5] = 0.002; steps[n_kf // 4:n_kf // 2, 1] = np.radians(0.03)
        D = chain_drift(sess.poses, steps)
    elif name == "symmetric_hall":
        sess = make_session(H=H, W=W, scene=hall_scene(), poses=hall_trajectory(n_kf))
        steps = np.zeros((n_kf, 6)); steps[:, 3] = 0.002; steps[:, 1] = np.radians(0.02)
        D = chain_drift(sess.poses, steps)
    elif name in ("rotation_only_stretch", "static_stretch"):
        base = loop_trajectory(n_kf, extra_laps=0.12)
        extra = (rotation_only_trajectory(12, pos=tuple(base[n_kf // 3][:3, 3]))
                 if name == "rotation_only_stretch"
                 else static_trajectory(12, pos=tuple(base[n_kf // 3][:3, 3]),
                                        forward=tuple(base[n_kf // 3][:3, 2])))
        poses = with_stretch(base, extra, n_kf // 3)
        sess = make_session(H=H, W=W, scene=corridor_loop_scene(), poses=poses)
        steps = np.zeros((len(poses), 6)); steps[:, 3] = 0.002
        D = chain_drift(sess.poses, steps)
    else:
        raise ValueError(name)
    inst = _instances_of(sess)
    write_session_dir(root, sess, inst, point_stride=2, drift_by_kf=D)
    write_aligned_chunks(root, sess, chunk_size=24, overlap=12, drift_by_kf=D)
    write_images(root, sess)
    synthetic_tracks(root, sess, win=12, stride=6, per_frame=30, seed=seed)
    return sess, {"n_kf": int(sess.n_kf), "drift_max_m": float(max(np.linalg.norm((D[g] @ sess.poses[g])[:3, 3]
                                                                                   - sess.poses[g][:3, 3])
                                                                    for g in range(sess.n_kf)))}


def run_scenario(name: str, cfg, correction_cfg, work: Path, log: Callable = print, **certify_kw) -> dict:
    from reconstruction.certify.run import certify_session
    from reconstruction.witness.frames import frames_from_arrays
    from reconstruction.witness.depth_tracks import load_tracks
    rec = {"scenario": name, "status": None, "cause": None, "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if name in NOT_RUNNABLE:
        rec.update({"status": "not_runnable", "cause": NOT_RUNNABLE[name]})
        return rec
    t0 = time.time()
    root = work / name / "s"
    try:
        sess, meta = build_scenario(name, root)
        rec["scene"] = meta
        base = frames_from_arrays(sess.depth, sess.K, sess.poses, sess.frame_numbers)
        acta = certify_session(root, cfg, operator="adversarial", log=log, correction_cfg=correction_cfg,
                               base_frames=base, tracks=load_tracks(root / "output"), use_fork_edges=False,
                               **certify_kw)
        from correction.session import read_poses
        P = read_poses(root / "output" / "camera_poses.txt")
        err = [float(np.linalg.norm(P[g][:3, 3] - sess.poses[g][:3, 3])) for g in range(sess.n_kf)]
        rec["pose_error_after_m"] = {"median": float(np.median(err)), "max": float(np.max(err))}
        rec["loop"] = {"stopped_at": acta["stopped_at"], "stop_reason": acta["stop_reason"],
                       "epoch_final": acta["epoch_final"],
                       "iterations": [{"verdict": it.get("verdict"), "reason": it.get("reason"),
                                       "gates": [g for g in it.get("gates", []) if not g["passed"]]}
                                      for it in acta["iterations"]]}
        w = acta["metrics_final"].get("witnesses") or {}
        rec["witnesses"] = w.get("status_fraction")
        rec["duplicates_after"] = acta["metrics_final"]["duplicates"]["n"]
        rejected = [it for it in acta["iterations"] if it.get("verdict") == "rejected"]
        if rejected:
            rec["status"] = "failed"
            rec["cause"] = f"iteration rejected: {rejected[0].get('reason')}"
        elif rec["pose_error_after_m"]["max"] > rec["scene"]["drift_max_m"]:
            rec["status"] = "failed"
            rec["cause"] = (f"pose error after ({rec['pose_error_after_m']['max']:.3f} m) exceeds the "
                            f"injected drift ({rec['scene']['drift_max_m']:.3f} m)")
        else:
            rec["status"] = "passed"
    except Exception as e:      # noqa: BLE001 — declared in the report, never swallowed
        rec["status"] = "failed"
        rec["cause"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()[-3000:]
    rec["elapsed_s"] = round(time.time() - t0, 1)
    return rec


def run_suite(cfg=None, correction_cfg=None, scenarios=SCENARIOS, work: Optional[Path] = None,
              out_json: Optional[Path] = None, log: Callable = print, **certify_kw) -> dict:
    from reconstruction.loops.config import load_loops_config
    cfg = cfg or load_loops_config()
    work = Path(work) if work else Path(tempfile.mkdtemp(prefix="stac_adv_"))
    t0 = time.time()
    rows = [run_scenario(s, cfg, correction_cfg, work, log, **certify_kw) for s in scenarios]
    for r in rows:
        log(f"[adversarial] {r['scenario']}: {r['status']}" + (f" — {r['cause']}" if r.get("cause") else ""))
    silent = [r["scenario"] for r in rows if r["status"] is None or (r["status"] != "passed" and not r.get("cause"))]
    rep = {"version": 1, "scenarios": rows, "n_passed": sum(1 for r in rows if r["status"] == "passed"),
           "n_failed": sum(1 for r in rows if r["status"] == "failed"),
           "n_not_runnable": sum(1 for r in rows if r["status"] == "not_runnable"),
           "silent_failures": silent, "elapsed_s": round(time.time() - t0, 1), "provenance": "tool_measured"}
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(rep, indent=1, default=float))
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="adversarial suite (§10.12)")
    ap.add_argument("--out", required=True, help="report json path")
    ap.add_argument("--work", default=None)
    args = ap.parse_args(argv)
    rep = run_suite(work=args.work, out_json=Path(args.out))
    return 0 if not rep["silent_failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
