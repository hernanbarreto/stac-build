"""SAM3 instances as loop detectors (claude_stac.txt §4.4) — geometry decides.

Two signals per instance, both proposals until the spatial gate (§4.5) rules:
  * TEMPORAL: an instance re-identified by tracking continuity that appears in
    two keyframe windows farther apart than ``loops.min_gap_keyframes`` is a
    candidate (i, j) = the central keyframes of the two windows;
  * SPATIAL: an instance whose points fall in two disjoint 3-D clusters
    (DBSCAN; centroids farther than ``duplicate_min_sep_m``, no bridging
    points) is a candidate AND the duplicate metric of §10.4.

Every candidate is gated: ``loop`` → written to ``maplong_run/loop_closures.txt``
(source ``instance``, or ``instance:<class>`` when the proposer is not a
structural instance — σ inflated by ``loops.semantic.nonstructural_sigma_factor``
wherever the edge is measured, never dropped: USER 2026-09-13 "nunca debe
descartarse un duplicado detectado por SAM3") for the exact-bridge machinery
of the fork AND measured post-hoc as a pose edge by
:func:`reconstruction.certify.loops_posthoc.instance_edges` inside the
certification stage of the pipeline; ``ambiguous`` → same, flagged (σ
inflated downstream); ``split`` → the fused instance is divided
(:mod:`reconstruction.loops.split`) — no loop; ``reject`` → recorded. The
full evidence lands in ``output/loop_candidates.json`` and the duplicates in
``output/duplicates.json``; instance classes (Qwen, structural | movable |
dynamic) in ``output/loop_semantics.json`` for the verifier's semantic check
(§4.2.4) — only ``dynamic`` (people, vehicles) never proposes. A manual
correction (marked objects) is the SAME candidate kind, source ``manual``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from reconstruction.loops import spatial_gate as sg
from reconstruction.loops.config import MetricGraphConfig, load_loops_config

CANDIDATES_JSON = "loop_candidates.json"
DUPLICATES_JSON = "duplicates.json"
SEMANTICS_JSON = "loop_semantics.json"
LOOP_CLOSURES_TXT = "loop_closures.txt"


# ── trajectory view over a session (poses + K + raw points per keyframe) ────

class SessionView:
    """TrajectoryView for the spatial gate over a reconstructed session:
    keyframe index = frame index (camera_frames.txt order = the fork's
    img_list order), poses from camera_poses.txt, K from session_io on the
    trace grid, points from the cleaned cloud's provenance."""

    def __init__(self, session, session_dir: Path):
        from segmentation.session_io import _load_camera_source
        from reconstruction.surface_fit.hole_audit import _k_grid
        self.session = session
        self.n_frames = session.n_kf
        cam = _load_camera_source(Path(session_dir), Path(session.output_dir))
        if cam is None:
            raise RuntimeError("no camera source (poses + intrinsics) for this session")
        grid = _k_grid(Path(session.output_dir))
        if grid is None:
            raise RuntimeError("cannot derive the intrinsics grid from cleaned_cloud.ply")
        kw, kh, _ = grid
        self.hw = (int(kh), int(kw))
        self._K = {}
        for k, f in enumerate(session.frames):
            K = cam.K_for(int(f))
            if K is not None:
                self._K[k] = np.asarray(K, np.float64)
        self._centres = session.poses[:, :3, 3].copy()
        order = np.argsort(session.ks, kind="stable")
        self._ks_sorted = session.ks[order]
        self._order = order

    def pose(self, g):
        g = int(g)
        return self.session.poses[g] if 0 <= g < self.n_frames else None

    def K(self, g):
        return self._K.get(int(g))

    def centres(self):
        return self._centres

    def _own_idx(self, g):
        lo = np.searchsorted(self._ks_sorted, int(g), side="left")
        hi = np.searchsorted(self._ks_sorted, int(g), side="right")
        return self._order[lo:hi]

    def depth(self, g):
        """Z-buffer of the frame's OWN cloud points on the trace grid (per-point
        provenance pixel_row/pixel_col) — the raw-cloud occlusion witness."""
        g = int(g)
        idx = self._own_idx(g)
        if len(idx) == 0 or self.pose(g) is None:
            return None
        M = np.linalg.inv(self.pose(g))
        z = (self.session.xyz[idx] @ M[:3, :3].T + M[:3, 3])[:, 2]
        H, W = self.hw
        pr = np.clip(self.session.data["pixel_row"][idx].astype(np.int64), 0, H - 1)
        pc = np.clip(self.session.data["pixel_col"][idx].astype(np.int64), 0, W - 1)
        zb = np.full((H, W), np.inf)
        np.minimum.at(zb, (pr, pc), z)
        zb[~np.isfinite(zb)] = 0.0
        return zb

    def points(self, g, n, seed=0):
        idx = self._own_idx(g)
        if len(idx) > n:
            idx = np.random.default_rng(seed).choice(idx, int(n), replace=False)
        return self.session.xyz[idx]


# ── clusters ────────────────────────────────────────────────────────────────

def disjoint_clusters(pts: np.ndarray, cfg) -> List[np.ndarray]:
    """Spatially separated groups of an instance's points (indices into pts),
    groups below cluster_min_points dropped, largest first.

    The question is whether the instance's points form two blobs with empty
    space between them — a copy of one object left in two places. Answering it
    with DBSCAN over every point does not survive contact with a real instance:
    at eps 15 cm on a surface sampled every 7.6 mm each point has thousands of
    neighbours, and sklearn materialises that neighbourhood graph. The floor of
    pccr 2026-09-14, 5,696,387 points, took the process past the container's
    109 GB and it was killed with no traceback. It never showed before because
    the instances had no points at all to cluster.

    Connected components over an occupied-voxel grid at the same eps answer the
    same question in one pass and a few MB: two points fall in the same group
    exactly when a chain of occupied voxels joins them. ``dbscan_min_samples``
    no longer marks stragglers as noise — ``cluster_min_points`` already drops
    anything too small to be a copy.
    """
    if len(pts) < cfg.cluster_min_points:
        return []
    eps = float(cfg.dbscan_eps_m)
    key = np.floor(np.asarray(pts, np.float64) / eps).astype(np.int64)
    vox, inv = np.unique(key, axis=0, return_inverse=True)
    n_vox = len(vox)
    if n_vox == 1:
        return [np.arange(len(pts))] if len(pts) >= cfg.cluster_min_points else []

    base = vox.min(axis=0)
    span = (vox.max(axis=0) - base + 3).astype(np.int64)
    if float(span[0]) * float(span[1]) * float(span[2]) > 9.0e18:
        return [np.arange(len(pts))]

    def _pack(v):
        d = v - base + 1
        return (d[:, 0] * span[1] + d[:, 1]) * span[2] + d[:, 2]

    packed = _pack(vox)
    order = np.argsort(packed, kind="stable")
    packed_sorted = packed[order]

    parent = np.arange(n_vox)

    def _find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return int(a)

    # 26-neighbourhood, each offset once (the mirrored half is redundant)
    offsets = [(dx, dy, dz)
               for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
               if (dx, dy, dz) > (0, 0, 0)]
    for off in offsets:
        nb = _pack(vox + np.asarray(off, np.int64))
        pos = np.searchsorted(packed_sorted, nb)
        ok = pos < n_vox
        pos = np.where(ok, pos, 0)
        hit = ok & (packed_sorted[pos] == nb)
        if not hit.any():
            continue
        for a, b in zip(np.flatnonzero(hit), order[pos[hit]]):
            ra, rb = _find(int(a)), _find(int(b))
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

    roots = np.array([_find(i) for i in range(n_vox)], dtype=np.int64)
    labels = roots[inv]
    out = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        if len(idx) >= cfg.cluster_min_points:
            out.append(idx)
    out.sort(key=len, reverse=True)
    return out


def _bridging(pts: np.ndarray, a: np.ndarray, b: np.ndarray, eps: float) -> bool:
    """True when some point of the instance lies between the two clusters
    (within eps of the segment joining their centroids) — then they are one
    connected object, not two copies."""
    ca, cb = pts[a].mean(0), pts[b].mean(0)
    d = cb - ca
    L = float(np.linalg.norm(d))
    if L < 1e-9:
        return True
    u = d / L
    rest = np.setdiff1d(np.arange(len(pts)), np.concatenate([a, b]))
    if len(rest) == 0:
        return False
    v = pts[rest] - ca
    t = v @ u
    mid = (t > 0.0) & (t < L)
    if not mid.any():
        return False
    perp = np.linalg.norm(v[mid] - np.outer(t[mid], u), axis=1)
    return bool((perp <= eps).any())


# ── candidates ──────────────────────────────────────────────────────────────

@dataclass
class Candidate:
    kind: str            # temporal | duplicate | manual
    instance_id: int
    label: str
    i: int
    j: int
    idx_a: np.ndarray    # global point indices of copy A (around i)
    idx_b: np.ndarray    # global point indices of copy B (around j)


def _median_kf(ks: np.ndarray) -> int:
    return int(np.median(ks))


def instance_candidates(session, inst: dict, cfg: MetricGraphConfig) -> List[Candidate]:
    from correction.units import visits_from_keyframes
    iid = int(inst.get("instance_id", inst.get("id")))
    label = str(inst.get("label", "segment"))
    gi = np.asarray(inst.get("globalIndices") or [], dtype=np.int64)
    gi = gi[(gi >= 0) & (gi < session.n_points)]
    if len(gi) < cfg.loops.cluster_min_points:
        return []
    ks = session.ks[gi]
    ok = ks >= 0
    gi, ks = gi[ok], ks[ok]
    cands: List[Candidate] = []
    # temporal windows
    visits = visits_from_keyframes(ks, cfg.loops.min_gap_keyframes // 2)
    vis_idx = []
    for v in visits:
        sel = gi[np.isin(ks, v)]
        if len(sel) >= cfg.loops.cluster_min_points:
            vis_idx.append((v, sel))
    for a in range(len(vis_idx)):
        for b in range(a + 1, len(vis_idx)):
            va, sa = vis_idx[a]
            vb, sb = vis_idx[b]
            if vb[0] - va[-1] <= cfg.loops.min_gap_keyframes:
                continue
            cands.append(Candidate("temporal", iid, label,
                                   i=_median_kf(session.ks[sb]), j=_median_kf(session.ks[sa]),
                                   idx_a=sb, idx_b=sa))
    # spatial clusters
    pts = session.xyz[gi]
    clusters = disjoint_clusters(pts, cfg.loops)
    for a in range(len(clusters)):
        for b in range(a + 1, len(clusters)):
            ca, cb = clusters[a], clusters[b]
            # the OBSERVABLE separation: plane offset / lateral axis offset for
            # two pieces of one surface (a wall past a doorway is not a
            # duplicate), centroid distance otherwise
            sep = float(sg.same_surface_rule(pts[ca], pts[cb], cfg.loops.spatial)["distance_m"])
            if sep < cfg.loops.duplicate_min_sep_m:
                continue
            if _bridging(pts, ca, cb, cfg.loops.dbscan_eps_m):
                continue
            ka, kb = _median_kf(ks[ca]), _median_kf(ks[cb])
            # a DUPLICATE is a revisit: the two clusters must come from
            # keyframes at least min_gap_keyframes apart. Two clusters written
            # by the SAME visit (a wall in two pieces past a doorway, a fused
            # segmentation of two neighbours) are never a loop — an edge
            # between adjacent keyframes closes nothing and, measured on
            # partial pieces, demands a correction that is pure noise
            # (certify smoke: 82↔81, 76↔76 vetoed at 1 m); they are pieces
            # (same surface) or a split (the spatial gate says which)
            if abs(int(ka) - int(kb)) < cfg.loops.min_gap_keyframes:
                continue
            if ka >= kb:
                cands.append(Candidate("duplicate", iid, label, i=ka, j=kb,
                                       idx_a=gi[ca], idx_b=gi[cb]))
            else:
                cands.append(Candidate("duplicate", iid, label, i=kb, j=ka,
                                       idx_a=gi[cb], idx_b=gi[ca]))
    return cands


# ── persistence helpers ─────────────────────────────────────────────────────

def merge_loop_closures(output_dir: Path, new: List[dict], log=print) -> Path:
    """Append candidates to maplong_run/loop_closures.txt (the file the fork's
    get_loop_pairs consumes), keeping the vendor's own SALAD lines; pairs
    already present are not duplicated."""
    import sys
    vendor = Path(__file__).resolve().parents[3] / "vendor" / "VGGT-Long"
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    from loop_utils.loop_bridges import load_loop_candidates, write_loop_candidates
    run_dir = Path(output_dir) / "maplong_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / LOOP_CLOSURES_TXT
    existing = load_loop_candidates(str(path))
    have = {(c["i"], c["j"]) for c in existing}
    merged = list(existing)
    n_add = 0
    for c in new:
        key = (int(c["i"]), int(c["j"]))
        if key in have:
            continue
        have.add(key)
        merged.append({"i": key[0], "j": key[1], "sim": c.get("sim"), "source": c["source"]})
        n_add += 1
    write_loop_candidates(str(path), merged, header="merged by reconstruction.loops")
    log(f"[instance-loops] {n_add} candidate(s) added to {path} ({len(merged)} total)")
    return path


def add_manual_candidate(output_dir, i: int, j: int, instance_ids: List[int],
                         log=print) -> Path:
    """§4.4: a marked-object correction is the SAME edge, source ``manual``."""
    return merge_loop_closures(Path(output_dir),
                               [{"i": int(max(i, j)), "j": int(min(i, j)), "sim": None,
                                 "source": "manual"}], log=log)


# ── the detector ────────────────────────────────────────────────────────────

def detect_instance_loops(output_dir, session_dir, cfg: Optional[MetricGraphConfig] = None,
                          log: Callable[[str], None] = print, apply_splits: bool = True,
                          budget_override: Optional[Dict[str, float]] = None) -> dict:
    """Run the detector over every instance of the session. Returns the
    report written to output/loop_candidates.json."""
    from correction.session import load_session
    from segmentation.erase import _mask_obj_by_iid
    from reconstruction.loops.semantic_classes import classify_instances
    from reconstruction.loops.split import split_instance

    cfg = cfg or load_loops_config()
    output_dir, session_dir = Path(output_dir), Path(session_dir)
    t0 = time.time()
    session = load_session(output_dir)
    res_path = output_dir / "segmentation_result.json"
    if not res_path.exists():
        raise RuntimeError(f"{res_path} missing — the instance detector needs a segmentation")
    instances = json.loads(res_path.read_text()).get("instances") or []
    view = SessionView(session, session_dir)

    # classes (Qwen) → recorded with every candidate; dynamic instances excluded,
    # every other class proposes (non-structural ones with an inflated σ)
    oid_of = _mask_obj_by_iid(output_dir)
    frames_of = {}
    for inst in instances:
        iid = int(inst.get("instance_id", inst.get("id")))
        gi = np.asarray(inst.get("globalIndices") or [], dtype=np.int64)
        gi = gi[(gi >= 0) & (gi < session.n_points)]
        frames_of[iid] = sorted({int(f) for f in np.unique(session.fg[gi])})
    # The crops are looked up in seg_masks.npz, which is keyed by KEYFRAME
    # POSITION, while frames_of above carries the cloud's REAL video frame
    # numbers. Built ONCE here, over the whole keyframe set, because the
    # translation cannot be derived from one instance's handful of frames.
    try:
        import numpy as _np
        from segmentation.pipeline import _mask_frame_lookup
        _mask_frames = _np.load(output_dir / "seg_masks.npz",
                                allow_pickle=True)["frames"].tolist()
        cloud_to_mask = _mask_frame_lookup(
            output_dir, _mask_frames, sorted({int(f) for f in _np.unique(session.fg)}))
    except Exception as e:  # noqa: BLE001 — classification degrades, never fails the run
        log(f"[loop-class] mask frame lookup unavailable ({e}) — keys used as-is")
        cloud_to_mask = {}
    classes = classify_instances(output_dir, session_dir, instances, cfg.loops.semantic,
                                 oid_of, frames_of, cloud_to_mask, log=log)
    # per-frame label lists for the verifier (§4.2.4) — keyed by REAL frame number
    per_frame: Dict[str, Dict[str, List[str]]] = {}
    for inst in instances:
        iid = int(inst.get("instance_id", inst.get("id")))
        cls = classes.get(iid, {}).get("class", cfg.loops.semantic.default_class)
        for f in frames_of.get(iid, []):
            rec = per_frame.setdefault(str(int(f)), {"structural": [], "movable": [], "dynamic": []})
            rec[cls].append(str(inst.get("label", "segment")).lower())
    (output_dir / SEMANTICS_JSON).write_text(json.dumps(
        {"version": 1, "classes": {str(k): v for k, v in classes.items()},
         "frames": per_frame, "provenance": "vlm_proposed"}, indent=1))

    # structural context per candidate (rule 4): other structural instances
    # whose points sit near copy A — evaluated lazily
    structural_ids = [int(i.get("instance_id", i.get("id"))) for i in instances
                      if classes.get(int(i.get("instance_id", i.get("id"))), {}).get("class")
                      == "structural"]

    report = {"version": 1, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
              "budget_source": (budget_override or {}).get("source", "drift_rate_model"),
              "candidates": [], "splits": [], "duplicates": [],
              "n_instances": len(instances), "n_structural": len(structural_ids)}
    to_write: List[dict] = []

    def _gate_candidate(inst, cand, cls):
        pts_a, pts_b = session.xyz[cand.idx_a], session.xyz[cand.idx_b]
        gate = sg.gate_instance_pair(cand.i, cand.j, view, cfg.loops.spatial, pts_a, pts_b,
                                     cand.label, context_pass=None,
                                     budget_override=budget_override)
        if gate.get("rules", {}).get("context") is not None and gate["verdict"] == "ambiguous":
            # rule 4: count neighbouring structural instances that pass 1–3
            n_ctx = 0
            ca = pts_a.mean(0)
            for oid_ in structural_ids:
                if oid_ == int(inst.get("instance_id", inst.get("id"))):
                    continue
                other = next((x for x in instances
                              if int(x.get("instance_id", x.get("id"))) == oid_), None)
                if other is None:
                    continue
                og = np.asarray(other.get("globalIndices") or [], dtype=np.int64)
                og = og[(og >= 0) & (og < session.n_points)]
                if len(og) < cfg.loops.cluster_min_points:
                    continue
                oc = session.xyz[og].mean(0)
                if np.linalg.norm(oc - ca) > cfg.loops.spatial.corridor_width_m:
                    continue
                for oc_ in instance_candidates(session, other, cfg):
                    g2 = sg.gate_instance_pair(oc_.i, oc_.j, view, cfg.loops.spatial,
                                               session.xyz[oc_.idx_a], session.xyz[oc_.idx_b],
                                               oc_.label, context_pass=None,
                                               budget_override=budget_override)
                    if g2["verdict"] in ("loop", "ambiguous") and \
                            g2["rules"]["separation"]["verdict"] == "loop":
                        n_ctx += 1
                        break
            gate = sg.gate_instance_pair(cand.i, cand.j, view, cfg.loops.spatial, pts_a,
                                         pts_b, cand.label, context_pass=n_ctx,
                                         budget_override=budget_override)
        return gate

    queue = list(instances)
    n_rounds = 0
    while queue:
        inst = queue.pop(0)
        iid = int(inst.get("instance_id", inst.get("id")))
        cls = classes.get(iid, {}).get("class", cfg.loops.semantic.default_class)
        if cls == "dynamic":
            continue
        split_done = False
        for cand in instance_candidates(session, inst, cfg):
            gate = _gate_candidate(inst, cand, cls)
            rec = {"kind": cand.kind, "instance_id": iid, "label": cand.label,
                   "class": cls, "i": int(cand.i), "j": int(cand.j),
                   "n_points": [int(len(cand.idx_a)), int(len(cand.idx_b))],
                   "gate": gate, "verdict": gate["verdict"], "provenance": "tool_measured"}
            if cand.kind == "duplicate":
                report["duplicates"].append({
                    "instance_id": iid, "label": cand.label,
                    "separation_m": gate["rules"]["separation"]["distance_m"],
                    "keyframes": [int(cand.i), int(cand.j)], "verdict": gate["verdict"]})
            if gate["verdict"] in ("loop", "ambiguous"):
                # USER 2026-09-13: "nunca debe descartarse un duplicado detectado
                # por SAM3" — geometry decided; the class only tags the source
                # (the bridge verifier and the post-hoc instance edge inflate σ
                # for a non-structural proposer: loops.semantic.nonstructural_
                # sigma_factor). dynamic instances left the queue above.
                to_write.append({"i": int(cand.i), "j": int(cand.j), "sim": None,
                                 "source": "instance" if cls == "structural" else f"instance:{cls}"})
                rec["written"] = True
            elif gate["verdict"] == "split" and apply_splits:
                smaller = cand.idx_a if len(cand.idx_a) <= len(cand.idx_b) else cand.idx_b
                try:
                    sp = split_instance(output_dir, iid, smaller,
                                        {"reason": gate["reason"], "candidate": rec}, log=log)
                    report["splits"].append({"instance_id": iid, **sp["ledger"]})
                    rec["split"] = sp["new_instance_id"]
                    split_done = True
                except ValueError as e:
                    rec["split_error"] = str(e)
            report["candidates"].append(rec)
            if split_done:
                break
        if split_done:
            # the instance changed on disk: re-read it (and its new sibling) and
            # judge the remaining candidates on the CURRENT point sets, never
            # on stale indices — bounded by the number of instances
            n_rounds += 1
            if n_rounds > 4 * max(len(instances), 1):
                raise RuntimeError("instance split loop did not converge — "
                                   "the segmentation keeps producing split verdicts")
            fresh = json.loads(res_path.read_text()).get("instances") or []
            by_id = {int(x.get("instance_id", x.get("id"))): x for x in fresh}
            new_iid = rec["split"]
            if new_iid in by_id:
                classes[new_iid] = dict(classes.get(iid, {}))
                instances.append(by_id[new_iid])
                if cls == "structural":
                    structural_ids.append(new_iid)
                queue.append(by_id[new_iid])
            if iid in by_id:
                for k_, x in enumerate(instances):
                    if int(x.get("instance_id", x.get("id"))) == iid:
                        instances[k_] = by_id[iid]
                queue.insert(0, by_id[iid])
    if to_write:
        merge_loop_closures(output_dir, to_write, log=log)
    report["n_written"] = len(to_write)
    # §4.6 regulated dimensions of structural instances → ABSOLUTE scale rows
    # (§5.2) for the next scale-graph solve (scale_absolute_rows.json)
    if cfg.structural.regulated_dims:
        from reconstruction.loops import structural as st
        from correction.units import load_chunk_plan
        plan = load_chunk_plan(output_dir)
        ranges = plan["chunk_ranges"] if plan else None
        downs = session.poses[:, :3, 1]
        g_down = downs.mean(0); g_down = g_down / (np.linalg.norm(g_down) + 1e-12)
        cls_map = {int(k): v.get("class", cfg.loops.semantic.default_class) for k, v in classes.items()}
        rows = st.regulated_rows(session, instances, cls_map, cfg.structural.regulated_dims,
                                 cfg.scale.sigma_regulated, -g_down, chunk_ranges=ranges,
                                 dims_pct=(cfg.loops.spatial.dims_pct_lo, cfg.loops.spatial.dims_pct_hi))
        if rows:
            p = st.write_absolute_rows(output_dir, rows, ranges)
            log(f"[instance-loops] {len(rows)} regulated-dimension scale row(s) → {p.name}")
        report["regulated_rows"] = rows
    report["elapsed_s"] = round(time.time() - t0, 1)
    (output_dir / CANDIDATES_JSON).write_text(json.dumps(report, indent=1, default=float))
    (output_dir / DUPLICATES_JSON).write_text(json.dumps(
        {"version": 1, "n_duplicates": len(report["duplicates"]),
         "duplicates": report["duplicates"], "target": 0}, indent=1, default=float))
    log(f"[instance-loops] {len(report['candidates'])} candidate(s): "
        f"{len(to_write)} written as loops, {len(report['splits'])} split(s), "
        f"{len(report['duplicates'])} duplicate(s) — {output_dir / CANDIDATES_JSON}")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SAM3 instance loop detector (claude_stac §4.4)")
    ap.add_argument("--session", required=True, help="session dir (holds frames/ and output/)")
    ap.add_argument("--no-split", action="store_true")
    args = ap.parse_args(argv)
    sd = Path(args.session)
    detect_instance_loops(sd / "output", sd, apply_splits=not args.no_split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
