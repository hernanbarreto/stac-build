"""Data behind the visual validation kit (§11): everything the viewer draws
comes from the session's own records — no new measurement here.

  * trajectory + edges: keyframe positions (camera_poses.txt), the odometry
    chain, the loop edges the graphs used (keyframe_graph.json, the
    certification acta, the reconstruction's own loop_edges.json) with their
    verdict — accepted (green), vetoed / rejected (red, with the reason),
    scale_break (orange), ambiguous candidates (amber);
  * duplicates: instances still written twice (duplicates.json) and the
    revisited places' offsets before / after the last iteration;
  * epochs: the chain of pending epochs, which of them have a Potree octree
    for the before/after toggle.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def _load(p: Path) -> Optional[dict]:
    return json.loads(p.read_text()) if p.exists() else None


def _poses(output_dir: Path):
    from correction.session import read_poses
    p = output_dir / "camera_poses.txt"
    if not p.exists():
        return None, []
    poses = read_poses(p)
    frames = []
    cf = output_dir / "camera_frames.txt"
    if cf.exists():
        frames = [int(float(x)) for x in cf.read_text().split()]
    return poses, frames


def kit_edges(output_dir) -> dict:
    output_dir = Path(output_dir)
    poses, frames = _poses(output_dir)
    if poses is None:
        return {"n_keyframes": 0, "positions": [], "frames": [], "odometry": [], "loops": [], "duplicates": []}
    pos = poses[:, :3, 3]
    n = len(pos)
    loops: List[dict] = []
    seen = set()

    def _add(i, j, kind, source, reason=None, residual_m=None, extra=None):
        if i is None or j is None or not (0 <= int(i) < n and 0 <= int(j) < n):
            return
        key = (int(i), int(j), kind, source)
        if key in seen:
            return
        seen.add(key)
        loops.append({"i": int(i), "j": int(j), "kind": kind, "source": source, "reason": reason,
                      "residual_m": residual_m, **(extra or {})})

    # the post-hoc keyframe graph (last solve): used vs vetoed
    kg = _load(output_dir / "keyframe_graph.json") or {}
    for v in kg.get("vetoed", []):
        _add(v.get("i"), v.get("j"), "vetoed", "keyframe_graph",
             reason=f"demanded {v.get('correction_m', 0):.2f} m > budget {v.get('budget_m', 0):.2f} m",
             residual_m=v.get("correction_m"))
    # the certification acta: every iteration's measured loops with their verdict
    acta = _load(output_dir / "certify_acta.json") or {}
    for it in acta.get("iterations", []):
        for m in it.get("loops", []):
            kind = "accepted" if m.get("accepted") else "rejected"
            _add(m.get("i"), m.get("j"), kind, f"revisit@iter{it.get('iteration')}",
                 reason=m.get("reason"), residual_m=m.get("icp_rms_m") or m.get("offset_before_m"),
                 extra={"offset_before_m": m.get("offset_before_m"), "offset_after_m": m.get("offset_after_m"),
                        "observability": m.get("observability"), "iteration": it.get("iteration")})
    # the reconstruction's own bridges (fork): accepted / scale_break / rejected
    le = _load(output_dir / "maplong_run" / "loop_edges.json") or {}
    for e in le.get("edges", []):
        ke = e.get("keyframe_edge") or {}
        st = e.get("status")
        kind = {"accepted": "accepted", "scale_break": "scale_break"}.get(st, "rejected")
        _add(ke.get("i"), ke.get("j"), kind, "bridge", reason=e.get("reason"),
             residual_m=e.get("residual_m") or ke.get("sigma_m"), extra={"s_ab": e.get("s_ab")})
    # instance candidates the spatial gate judged (ambiguous / split / reject)
    lc = _load(output_dir / "loop_candidates.json") or {}
    for c in lc.get("candidates", []):
        verdict = c.get("verdict")
        if verdict == "loop":
            continue
        kind = "ambiguous" if verdict == "ambiguous" else "rejected"
        _add(c.get("i"), c.get("j"), kind, "instance",
             reason=f"{c.get('label')}#{c.get('instance_id')}: {(c.get('gate') or {}).get('reason') or verdict}",
             extra={"instance_id": c.get("instance_id"), "label": c.get("label")})
    # duplicates: instances still in two copies + the revisited places' offsets
    dups: List[dict] = []
    dj = _load(output_dir / "duplicates.json") or {}
    for d in dj.get("duplicates", []):
        kfs = d.get("keyframes") or [None, None]
        dups.append({"instance_id": d.get("instance_id"), "label": d.get("label"), "i": kfs[0], "j": kfs[1],
                     "separation_m": d.get("separation_m"), "verdict": d.get("verdict"), "source": "instance"})
    last = acta.get("iterations", [])[-1] if acta.get("iterations") else None
    if last:
        for m in last.get("loops", []):
            if m.get("duplicated") or (m.get("offset_before_m") or 0) > 0:
                dups.append({"instance_id": None, "label": "revisit", "i": m.get("i"), "j": m.get("j"),
                             "separation_m": m.get("offset_before_m"), "closure_after_m": m.get("offset_after_m"),
                             "verdict": "accepted" if m.get("accepted") else "rejected", "source": "revisit"})
    for d in dups:
        for k in ("i", "j"):
            kk = d.get(k)
            d[k + "_pos"] = pos[int(kk)].tolist() if kk is not None and 0 <= int(kk) < n else None
    return {"n_keyframes": int(n), "positions": pos.tolist(), "frames": frames,
            "odometry": [[k, k + 1] for k in range(n - 1)], "loops": loops, "duplicates": dups,
            "legend": {"accepted": "green", "scale_break": "orange", "vetoed": "red", "rejected": "red",
                       "ambiguous": "amber", "odometry": "grey"}}


def epoch_layers(output_dir) -> dict:
    """The current epoch and the pending previous epochs that carry a
    Potree octree (the before/after toggle needs one per side)."""
    from correction.epoch import current_epoch
    from correction.apply import pending_prev_dirs
    output_dir = Path(output_dir)
    cur = current_epoch(output_dir)
    prev = []
    for d in pending_prev_dirs(output_dir):
        try:
            ep = int(d.name.split("_")[-1])
        except ValueError:
            continue
        prev.append({"epoch": ep, "potree": (d / "potree" / "metadata.json").exists()})
    return {"epoch": cur, "current_potree": (output_dir / "potree" / "metadata.json").exists(),
            "previous": prev}
