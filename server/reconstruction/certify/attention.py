"""§11 attention list: the N places where the system is LEAST sure — the
operator looks there first, not where the cloud looks nice. Built from the
session's own records: loops judged ambiguous, closures beyond the drift
budget, scale_break edges, advisory gate warnings of the applied epochs
(the certification applies and declares — USER 2026-09-13), saturated
stages, keyframes with the lowest mv_votes, stretches without loop
coverage, split instances, remaining duplicates. Every entry carries a "fly
to" anchor (a point or a keyframe) and its evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np


def _load(p: Path) -> Optional[dict]:
    return json.loads(p.read_text()) if p.exists() else None


def attention_list(output_dir, n_low_votes_keyframes: int = 5) -> dict:
    output_dir = Path(output_dir)
    items: List[dict] = []
    from correction.session import read_poses
    poses_p = output_dir / "camera_poses.txt"
    poses = read_poses(poses_p) if poses_p.exists() else None
    frames = []
    cf = output_dir / "camera_frames.txt"
    if cf.exists():
        frames = [int(float(x)) for x in cf.read_text().split()]

    def _kf_anchor(k):
        if poses is None or k is None or not (0 <= int(k) < len(poses)):
            return None
        return {"keyframe": int(k), "frame": (frames[int(k)] if int(k) < len(frames) else None),
                "position": poses[int(k)][:3, 3].tolist()}

    cands = _load(output_dir / "loop_candidates.json") or {}
    for c in cands.get("candidates", []):
        if c.get("verdict") == "ambiguous":
            items.append({"kind": "loop_ambiguous", "severity": 2,
                          "text": f"{c.get('label')}#{c.get('instance_id')}: identity ambiguous between "
                                  f"keyframes {c.get('i')} and {c.get('j')} ({c.get('gate', {}).get('reason')})",
                          "anchor": _kf_anchor(c.get("i")), "anchor_b": _kf_anchor(c.get("j")),
                          "evidence": c.get("gate")})
        if c.get("split") is not None:
            items.append({"kind": "instance_split", "severity": 2,
                          "text": f"{c.get('label')}#{c.get('instance_id')} was split (implausible identity) "
                                  f"→ new instance {c.get('split')}",
                          "anchor": _kf_anchor(c.get("i")), "anchor_b": _kf_anchor(c.get("j")),
                          "evidence": c.get("gate")})
    dups = _load(output_dir / "duplicates.json") or {}
    for d in dups.get("duplicates", []):
        if d.get("verdict") in ("loop", "ambiguous"):
            items.append({"kind": "duplicate", "severity": 3,
                          "text": f"{d.get('label')}#{d.get('instance_id')} exists twice "
                                  f"({d.get('separation_m', 0):.2f} m apart) — target 0",
                          "anchor": _kf_anchor((d.get("keyframes") or [None])[0]),
                          "anchor_b": _kf_anchor((d.get("keyframes") or [None, None])[1]), "evidence": d})
    edges = _load(output_dir / "maplong_run" / "loop_edges.json") or {}
    for e in edges.get("edges", []):
        if e.get("status") == "scale_break":
            ke = e.get("keyframe_edge") or {}
            items.append({"kind": "scale_break", "severity": 3,
                          "text": f"loop {ke.get('i')}↔{ke.get('j')}: scale break "
                                  f"(s_ab {e.get('s_ab')}) — kept, not applied",
                          "anchor": _kf_anchor(ke.get("i")), "anchor_b": _kf_anchor(ke.get("j")),
                          "evidence": {k: v for k, v in e.items() if k != "candidate"}})
        elif e.get("status") == "rejected":
            ke = e.get("keyframe_edge") or {}
            items.append({"kind": "loop_rejected", "severity": 1,
                          "text": f"loop candidate {ke.get('i')}↔{ke.get('j')} rejected: {e.get('reason')}",
                          "anchor": _kf_anchor(ke.get("i")), "evidence": {"reason": e.get("reason")}})
    acta = _load(output_dir / "certify_acta.json") or {}
    for it in acta.get("iterations", []):
        if it.get("verdict") != "applied":
            continue
        if it.get("regressed"):
            items.append({"kind": "objective_regressed", "severity": 4,
                          "text": f"epoch {it.get('epoch_to')} left the session WORSE than it found it: "
                                  f"objective {it.get('objective_prev', 0):.4f} → {it.get('objective', 0):.4f} "
                                  f"({it.get('improvement', 0) * 100:+.1f}%) — applied (advisory); "
                                  f"look at it before approving",
                          "anchor": None, "evidence": {"iteration": it.get("iteration"),
                                                       "objective_prev": it.get("objective_prev"),
                                                       "objective": it.get("objective"),
                                                       "improvement": it.get("improvement")}})
        for w in it.get("gate_warnings", []):
            items.append({"kind": "gate_warning", "severity": 3,
                          "text": f"epoch {it.get('epoch_to')}: gate ⚠ {w} — applied (advisory), "
                                  f"judge it: Approve or Undo",
                          "anchor": None, "evidence": {"iteration": it.get("iteration"),
                                                       "gates": [g for g in it.get("gates", []) if not g.get("passed")]}})
        for w in ((it.get("stages") or {}).get("poses") or {}).get("gate_warnings", []) or []:
            items.append({"kind": "gate_warning", "severity": 2,
                          "text": f"epoch {it.get('epoch_to')}: pose graph ⚠ {w} — applied (advisory)",
                          "anchor": None, "evidence": {"iteration": it.get("iteration")}})
    kg = _load(output_dir / "keyframe_graph.json") or {}
    for v in kg.get("over_budget", []):
        items.append({"kind": "loop_over_budget", "severity": 2,
                      "text": f"loop {v.get('i')}↔{v.get('j')}: closure {v.get('correction_m', 0):.2f} m "
                              f"beyond the drift budget {v.get('budget_m', 0):.2f} m over a "
                              f"{v.get('walk_m', 0):.1f} m walk — applied; look here",
                      "anchor": _kf_anchor(v.get("i")), "anchor_b": _kf_anchor(v.get("j")), "evidence": v})
    cov = kg.get("loop_coverage") or {}
    for u in cov.get("uncovered", []):
        items.append({"kind": "no_loop_coverage", "severity": 2,
                      "text": f"keyframes {u.get('kf_from')}–{u.get('kf_to')} ({u.get('length_m', 0):.1f} m) "
                              f"have no loop within {cov.get('radius_m')} m",
                      "anchor": _kf_anchor(u.get("kf_from")), "anchor_b": _kf_anchor(u.get("kf_to")),
                      "evidence": u})
    auth = _load(output_dir / "maplong_run" / "authority.json") or {}
    for stage, rec in auth.items():
        if isinstance(rec, dict) and (rec.get("saturated") or rec.get("exceeded")):
            items.append({"kind": "authority_saturated", "severity": 2,
                          "text": f"stage {stage} used {rec.get('fraction_used', 0) * 100:.0f}% of its authority"
                                  + (" — EXCEEDED (identity)" if rec.get("exceeded") else ""),
                          "anchor": None, "evidence": rec})
    pa = (kg.get("gates") or {}).get("authority") or {}
    if pa.get("saturated") or pa.get("exceeded"):
        items.append({"kind": "authority_saturated", "severity": 2,
                      "text": f"post-hoc pose graph used {pa.get('fraction_used', 0) * 100:.0f}% of its authority",
                      "anchor": None, "evidence": pa})
    # keyframes with the lowest multi-view support
    ply = output_dir / "cleaned_cloud.ply"
    if ply.exists():
        from correction.session import read_ply
        _, data = read_ply(ply)
        names = data.dtype.names or ()
        if "mv_votes" in names and "frame_global" in names and frames:
            fg = np.asarray(data["frame_global"], np.int64)
            mv = np.asarray(data["mv_votes"], np.float64)
            kf_of = {f: k for k, f in enumerate(frames)}
            order = np.argsort(fg, kind="stable")
            sfg = fg[order]
            starts = np.concatenate([[0], np.flatnonzero(np.diff(sfg)) + 1])
            ends = np.concatenate([starts[1:], [len(sfg)]])
            per = []
            for s, e in zip(starts, ends):
                f = int(sfg[s]); k = kf_of.get(f)
                if k is None or e - s < 50:
                    continue
                per.append((float(mv[order[s:e]].mean()), k, int(e - s)))
            per.sort()
            for mean_votes, k, n in per[:int(n_low_votes_keyframes)]:
                items.append({"kind": "low_mv_votes", "severity": 1,
                              "text": f"keyframe {k}: mean mv_votes {mean_votes:.2f} over {n:,} points",
                              "anchor": _kf_anchor(k), "evidence": {"mv_votes_mean": mean_votes, "n": n}})
    items.sort(key=lambda it: -it["severity"])
    return {"version": 1, "n": len(items), "items": items}
