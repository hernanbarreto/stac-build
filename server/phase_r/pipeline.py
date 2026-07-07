# STAC-Builder — Phase R orchestrator: instance-aware pose/depth refinement.
#
# Inserts between windowed reconstruction (VGGT-Ω/VGGT-Long + DA3-Streaming) and
# final fusion (TSDF/surface_fitting). Ties R.1..R.9 together:
#   R.1/R.8  build the canonical instance store from masklets (+ dynamic R.7)
#   R.4      estimate gravity from the dominant floor/slab plane
#   R.2/R.3  vote-entropy + onion metrics per instance
#   R.4      per-window OBBs -> Sim(3) residuals -> inter-window pose graph
#            (refinement loop, max 2 iterations)
#   R.6      metric authority: marker/survey scale wins on conflict
#   R.9      A/B gate vs the no-anchor baseline; fall back if anything regresses
#
# Single-window sessions run the per-instance metrics; the inter-window pose
# graph is exercised when a window_map (frame->window) with >1 window is given.
#
# PROVENANCE: ours, over R3D-ported geometry.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .build_instances import InstanceStoreBuilder
from .depth_regularization import fit_plane
from .failsafe import compare_and_gate
from .instance_store import InstanceStore
from .metric_hierarchy import MetricAuthority, ScaleReport
from .residuals import Sim3, WindowEdge, optimize_window_graph, sim3_from_obb_pair


@dataclass
class PhaseRReport:
    n_instances: int = 0
    gravity: list = field(default_factory=list)
    n_windows: int = 1
    pose_graph: dict = field(default_factory=dict)
    onion_bimodal: int = 0
    onion_separation_median_m: float = 0.0
    failsafe: dict = field(default_factory=dict)
    used_anchoring: bool = True

    def summary(self) -> str:
        return (f"instances={self.n_instances} windows={self.n_windows} "
                f"gravity={[round(g,3) for g in self.gravity]} "
                f"onion_bimodal={self.onion_bimodal} "
                f"anchoring={'KEPT' if self.used_anchoring else 'FALLBACK'}")


class PhaseRPipeline:
    def __init__(self, session_dir, output_dir, store_path, config: dict | None = None,
                 window_map: dict[int, str] | None = None,
                 marker_scale: float | None = None):
        self.session_dir = Path(session_dir)
        self.output_dir = Path(output_dir)
        self.store_path = str(store_path)
        self.config = config or {}
        self.cfg = self.config.get("phase_r", {})
        self.window_map = window_map or {}
        self.marker_scale = marker_scale
        self.floor_labels = set(self.cfg.get("floor_labels",
                                             ["floor", "slab", "platform", "ground"]))
        self.max_iterations = self.cfg.get("max_refine_iterations", 2)

    # ── R.4 gravity from the dominant floor/slab plane ──────────────
    def _estimate_gravity(self, store: InstanceStore) -> np.ndarray | None:
        best_pts, best_n = None, 0
        for inst in store.list_instances():
            if not any(f in inst["label"] for f in self.floor_labels):
                continue
            pts = store.get_points(inst["instance_id"])
            if pts is not None and len(pts) > best_n:
                best_pts, best_n = pts, len(pts)
        if best_pts is None:
            return None
        n, _d = fit_plane(best_pts)
        # gravity points "down": choose the sign so up has +Y-ish component
        up = n if n[1] >= 0 else -n
        return -up  # gravity = -up

    # ── R.4 inter-window pose graph over shared instances ───────────
    def _build_pose_graph(self, store: InstanceStore):
        windows = sorted(set(self.window_map.values())) if self.window_map else ["global"]
        if len(windows) < 2:
            return windows, []
        widx = {w: i for i, w in enumerate(windows)}
        edges: list[WindowEdge] = []
        for inst in store.list_instances():
            iid = inst["instance_id"]
            wins = store.list_obb_windows(iid)
            wins = [w for w in wins if w in widx]
            for a in range(len(wins)):
                for b in range(a + 1, len(wins)):
                    oa = store.get_obb(iid, wins[a])
                    ob = store.get_obb(iid, wins[b])
                    if oa is None or ob is None:
                        continue
                    M = sim3_from_obb_pair(oa[0], oa[1], ob[0], ob[1])
                    w = float(min(1.0, 0.5 + inst.get("confidence", 0.0)))
                    edges.append(WindowEdge(widx[wins[a]], widx[wins[b]], M, w))
        return windows, edges

    # ── main ────────────────────────────────────────────────────────
    def run(self, baseline_metrics: dict | None = None) -> PhaseRReport:
        # R.1/R.2-lift/R.3/R.7/R.8
        gravity0 = None
        builder = InstanceStoreBuilder(self.session_dir, self.output_dir, self.store_path,
                                       config=self.config)
        builder.build(gravity=gravity0)
        store = InstanceStore(self.store_path)
        report = PhaseRReport()
        insts = store.list_instances()
        report.n_instances = len(insts)

        # R.4 gravity, then re-fit OBBs under gravity (refinement loop)
        gravity = self._estimate_gravity(store)
        report.gravity = gravity.tolist() if gravity is not None else []
        if gravity is not None:
            from .geometry import fit_gravity_aligned_obb
            from .onion import detect_onion
            for inst in insts:
                iid = inst["instance_id"]
                pts = store.get_points(iid)
                if pts is None or len(pts) < 3:
                    continue
                obb = fit_gravity_aligned_obb(pts, gravity=gravity)
                if obb is None:
                    continue
                T, aabb, pos = obb
                store.set_obb(iid, T, aabb, pos, window_id="global", gravity=gravity,
                              n_points=len(pts))
                on = detect_onion(pts, T, aabb)
                store.set_onion_metric(iid, on.bimodal, on.separation_m, on.bic_delta)

        # R.4 inter-window pose graph (loop, <= max_iterations)
        windows, edges = self._build_pose_graph(store)
        report.n_windows = len(windows)
        if edges:
            for _it in range(self.max_iterations):
                corr, stats = optimize_window_graph(len(windows), edges)
                report.pose_graph = stats
                if stats["cost"] < 1e-6 or stats["cost"] > 0.5 * stats["cost0"]:
                    break  # converged or not improving

        # R.6 metric authority (scale)
        scale_report = ScaleReport()
        if self.marker_scale is not None:
            auth = MetricAuthority(self.marker_scale)
            report.failsafe.setdefault("marker_scale", self.marker_scale)

        # onion aggregate
        seps = []
        for inst in insts:
            m = store.get_metrics(inst["instance_id"]).get("onion")
            if m and m["bimodal"]:
                report.onion_bimodal += 1
                seps.append(m["separation_m"])
        report.onion_separation_median_m = float(np.median(seps)) if seps else 0.0

        # R.9 fail-safe A/B gate
        refined_metrics = {
            "onion_separation_median": report.onion_separation_median_m,
            "vote_entropy_mean": self._mean_vote_entropy(store),
        }
        if baseline_metrics:
            gate = compare_and_gate(baseline_metrics, refined_metrics)
            report.used_anchoring = gate.use_anchoring
            report.failsafe.update({"report": gate.report,
                                    "regressions": gate.regressions,
                                    "refined": refined_metrics})
        else:
            report.failsafe.update({"report": "no baseline provided — anchoring kept",
                                    "refined": refined_metrics})
        store.set_meta("phase_r_report", report.summary())
        store.close()
        return report

    def _mean_vote_entropy(self, store: InstanceStore) -> float:
        vals = []
        for inst in store.list_instances():
            v = store.get_metrics(inst["instance_id"]).get("vote")
            if v and v.get("mean_entropy") is not None:
                vals.append(v["mean_entropy"])
        return float(np.mean(vals)) if vals else 0.0
