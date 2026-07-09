# STAC-Builder — Phase R orchestrator: instance-aware pose/depth refinement.
#
# Inserts between windowed reconstruction (VGGT-Ω/VGGT-Long + DA3-Streaming) and
# final fusion (TSDF/surface_fitting). Ties R.1..R.9 together:
#   R.1/R.8  build the canonical instance store from masklets (+ dynamic R.7)
#   R.2      multi-view plurality vote + entropy (per point/instance/region)
#   R.3      onion metric per instance, PER SEAM between windows, + heatmaps
#   R.4      gravity + per-window OBBs -> Sim(3) residuals -> inter-window pose
#            graph, with a REAL refinement loop (max 2 iterations): each pass
#            re-applies the corrections to the OBBs (== re-lifting the points
#            with corrected poses, since the lift is linear through c2w),
#            rebuilds the edges and re-solves until converged.
#   R.6      metric authority: the marker/survey scale ALWAYS wins on conflict
#            (conflict logged with magnitude); DA3 top-25% S kept as a prior in
#            the solve; S per window before/after reported.
#   R.9      A/B gate vs the no-anchor baseline (computed automatically from
#            the pre-refinement store); fall back if ANY metric regresses.
#
# R.1 RULE (inviolable, spec): cross-window re-identification is EXCLUSIVELY by
# tracking continuity through the overlap frames. Appearance matching between
# instances not connected by tracking is PROHIBITED — in tunnels/stations the
# columns repeat every N metres and an appearance match creates false loop
# closures that destroy the reconstruction (perceptual aliasing). Nothing in
# this package matches instances by appearance; keep it that way.
#
# Single-window sessions run the per-instance metrics; the inter-window pose
# graph is exercised when a window_map (frame->window) with >1 window is given.
#
# PROVENANCE: ours, over R3D-ported geometry.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .build_instances import (InstanceStoreBuilder, build_vote_views,
                              run_scene_vote)
from .depth_regularization import fit_plane
from .failsafe import compare_and_gate
from .geometry import fit_gravity_aligned_obb
from .instance_store import InstanceStore
from .metric_hierarchy import MetricAuthority, ScaleReport
from .onion import detect_onion, onion_heatmap
from .residuals import (PrimitiveEdge, Sim3, WindowEdge, fit_cylinder_primitive,
                        fit_plane_primitive, optimize_window_graph,
                        transform_cylinder, transform_plane)
from .vote import View

# Every step logs here; the phase_r worker forwards this logger to the pipe →
# server.log + UI, so a full run is debuggable from the log alone.
logger = logging.getLogger("PhaseR")


@dataclass
class PhaseRReport:
    n_instances: int = 0
    gravity: list = field(default_factory=list)
    n_windows: int = 1
    pose_graph: dict = field(default_factory=dict)
    iterations: int = 0
    converged: bool = False
    onion_bimodal: int = 0
    onion_separation_median_m: float = 0.0
    seams: dict = field(default_factory=dict)
    scale_report: dict = field(default_factory=dict)
    scale_conflicts: list = field(default_factory=list)
    failsafe: dict = field(default_factory=dict)
    used_anchoring: bool = True
    writeback: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (f"instances={self.n_instances} windows={self.n_windows} "
                f"gravity={[round(g,3) for g in self.gravity]} "
                f"iters={self.iterations} onion_bimodal={self.onion_bimodal} "
                f"anchoring={'KEPT' if self.used_anchoring else 'FALLBACK'}")


def _sim3_matrix(M: Sim3) -> np.ndarray:
    """Sim(3) as a 4x4 similarity [[s·R, t], [0, 1]]."""
    T = np.eye(4)
    T[:3, :3] = M.s * M.R
    T[:3, 3] = M.t
    return T


def _apply_sim3_to_obb(M: Sim3, obb):
    """Transform an OBB (T, aabb, pos) by a window Sim(3) correction: the frame
    rotates/translates rigidly, the extents scale by s."""
    T, aabb, pos = obb
    T2 = np.asarray(T, float).copy()
    T2[:3, :3] = M.R @ T2[:3, :3]
    T2[:3, 3] = M.apply(T2[:3, 3])
    return T2, np.asarray(aabb, float) * M.s, T2[:3, 3]


class PhaseRPipeline:
    def __init__(self, session_dir, output_dir, store_path, config: dict | None = None,
                 window_map: dict[int, str] | None = None,
                 marker_scale: float | None = None,
                 window_scale_priors: dict[str, float] | None = None):
        self.session_dir = Path(session_dir)
        self.output_dir = Path(output_dir)
        self.store_path = str(store_path)
        self.config = config or {}
        self.cfg = self.config.get("phase_r", {})
        self.window_map = {int(k): v for k, v in (window_map or {}).items()}
        # R.6 marker/survey authority: explicit argument wins; else the config
        # (phase_r.marker_scale — e.g. a surveyed distance or the 1435 mm track
        # gauge measured on this scene). None → no marker, DA3 prior only.
        if marker_scale is None:
            mk = self.cfg.get("marker_scale")
            marker_scale = float(mk) if mk else None
        self.marker_scale = marker_scale
        # DA3 top-confidence scale S per window (prior, R.6) — wired by the
        # phase_r worker from the raw DA3/omega depth ratios per window.
        self.window_scale_priors = window_scale_priors or {}
        # R.4 fine primitives for structural classes (plane/cylinder residuals)
        self.primitive_residuals = bool(self.cfg.get("primitive_residuals", True))
        self.primitive_weight = float(self.cfg.get("primitive_weight", 1.0))
        self.plane_classes = set(self.cfg.get("plane_classes",
                                              ["wall", "slab", "vault", "platform", "floor"]))
        self.cylinder_classes = set(self.cfg.get("cylinder_classes", ["column"]))
        self.floor_labels = set(self.cfg.get("floor_labels",
                                             ["floor", "slab", "platform", "ground"]))
        self.max_iterations = int(self.cfg.get("max_refine_iterations", 2))
        self.converge_tol = float(self.cfg.get("refine_convergence_tol", 1e-3))
        self.min_points = int(self.cfg.get("min_points", 30))
        self.vote_max_views = int(self.cfg.get("vote_max_views", 20))

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

    # ── R.4 per-window OBBs (from per-point source frames) ──────────
    def _windows(self) -> list[str]:
        return sorted(set(self.window_map.values())) if self.window_map else ["global"]

    def _residual_eligible(self, inst: dict) -> bool:
        """Spec (Phase 1 item 5): dubious instances may vote but generate NO
        pose residuals until validated or above threshold."""
        return inst["status"] != "dubious"

    def _fit_window_obbs(self, store: InstanceStore, gravity) -> dict:
        """Group each instance's points by window and fit a gravity-aligned OBB
        per window (R.4). Returns {(iid, window): [T, aabb, pos, weight]}."""
        obb_map: dict = {}
        windows = self._windows()
        if len(windows) < 2:
            return obb_map
        for inst in store.list_instances():
            if inst["status"] == "dynamic_excluded" or not self._residual_eligible(inst):
                continue
            iid = inst["instance_id"]
            pts = store.get_points(iid)
            fids = store.get_point_frames(iid)
            if pts is None or fids is None or len(pts) < self.min_points:
                continue
            wins = np.array([self.window_map.get(int(f), windows[0]) for f in fids])
            weight = float(min(1.0, 0.5 + inst.get("confidence", 0.0)))
            for w in np.unique(wins):
                sel = wins == w
                if int(sel.sum()) < self.min_points:
                    continue
                obb = fit_gravity_aligned_obb(
                    pts[sel], gravity=gravity,
                    trim_percent=float(self.cfg.get("obb_trim_percent", 1.0)))
                if obb is None:
                    continue
                T, aabb, pos = obb
                store.set_obb(iid, T, aabb, pos, window_id=str(w), gravity=gravity,
                              n_points=int(sel.sum()))
                obb_map[(iid, str(w))] = [T, aabb, pos, weight]
        return obb_map

    def _fit_window_primitives(self, store: InstanceStore, gravity) -> dict:
        """R.4 fine primitives: per STRUCTURAL instance and window, fit the
        surface primitive on that window's points — plane for plane-like
        classes, cylinder axis for column-like. Returns
        {(iid, window): (prim, kind, weight)}. Empty when disabled or <2 windows."""
        prim_map: dict = {}
        if not self.primitive_residuals:
            return prim_map
        windows = self._windows()
        if len(windows) < 2:
            return prim_map
        for inst in store.list_instances():
            if inst["status"] == "dynamic_excluded" or not self._residual_eligible(inst):
                continue
            label = str(inst["label"])
            kind = ("plane" if any(c in label for c in self.plane_classes) else
                    "cylinder" if any(c in label for c in self.cylinder_classes) else None)
            if kind is None:
                continue
            iid = inst["instance_id"]
            pts = store.get_points(iid)
            fids = store.get_point_frames(iid)
            if pts is None or fids is None or len(pts) < self.min_points:
                continue
            wins = np.array([self.window_map.get(int(f), windows[0]) for f in fids])
            weight = float(min(1.0, 0.5 + inst.get("confidence", 0.0)))
            for w in np.unique(wins):
                sel = wins == w
                if int(sel.sum()) < self.min_points:
                    continue
                prim = (fit_plane_primitive(pts[sel]) if kind == "plane"
                        else fit_cylinder_primitive(pts[sel]))
                if prim is not None:
                    prim_map[(iid, str(w))] = (prim, kind, weight)
        logger.info(f"[R.4] fine primitives fitted: {len(prim_map)} "
                    f"(instance, window) pairs "
                    f"(plane classes {sorted(self.plane_classes)}, "
                    f"cylinder classes {sorted(self.cylinder_classes)})")
        return prim_map

    @staticmethod
    def _edges_from_primitives(prim_map: dict, windows: list[str]) -> list[PrimitiveEdge]:
        widx = {w: i for i, w in enumerate(windows)}
        by_iid: dict[int, list[str]] = {}
        for (iid, w) in prim_map:
            by_iid.setdefault(iid, []).append(w)
        edges: list[PrimitiveEdge] = []
        for iid, wins in by_iid.items():
            wins = sorted(w for w in wins if w in widx)
            for a in range(len(wins)):
                for b in range(a + 1, len(wins)):
                    pa, kind_a, wt = prim_map[(iid, wins[a])]
                    pb, kind_b, _ = prim_map[(iid, wins[b])]
                    if kind_a != kind_b:
                        continue
                    edges.append(PrimitiveEdge(widx[wins[a]], widx[wins[b]],
                                               pa, pb, kind_a, wt))
        return edges

    @staticmethod
    def _apply_sim3_to_primitives(prim_map: dict, corr, widx: dict) -> dict:
        """Re-lift the primitives under the iteration's corrections (same role
        as _apply_sim3_to_obb for the OBBs)."""
        out = {}
        for (iid, w), (prim, kind, wt) in prim_map.items():
            M = corr[widx[w]]
            prim2 = transform_plane(prim, M) if kind == "plane" else transform_cylinder(prim, M)
            out[(iid, w)] = (prim2, kind, wt)
        return out

    @staticmethod
    def _edges_from_obbs(obb_map: dict, windows: list[str]) -> list[WindowEdge]:
        from .residuals import sim3_from_obb_pair
        widx = {w: i for i, w in enumerate(windows)}
        by_iid: dict[int, list[str]] = {}
        for (iid, w) in obb_map:
            by_iid.setdefault(iid, []).append(w)
        edges: list[WindowEdge] = []
        for iid, wins in by_iid.items():
            wins = sorted(w for w in wins if w in widx)
            for a in range(len(wins)):
                for b in range(a + 1, len(wins)):
                    Ta, aa, _pa, wt = obb_map[(iid, wins[a])]
                    Tb, ab, _pb, _ = obb_map[(iid, wins[b])]
                    M = sim3_from_obb_pair(Ta, aa, Tb, ab)
                    edges.append(WindowEdge(widx[wins[a]], widx[wins[b]], M, wt))
        return edges

    # ── R.3 onion per seam (adjacent windows) ────────────────────────
    def _seam_onion(self, store: InstanceStore, windows: list[str],
                    corrections: dict[str, np.ndarray] | None = None,
                    persist: bool = False) -> dict[str, list[float]]:
        """Per-seam onion metric: for each adjacent window pair, pool each
        shared instance's points from BOTH windows and run the bimodality test.
        Returns {seam: [separations_m per shared instance]}."""
        seams: dict[str, list[float]] = {}
        if len(windows) < 2:
            return seams
        for inst in store.list_instances():
            if inst["status"] == "dynamic_excluded":
                continue
            iid = inst["instance_id"]
            pts = store.get_points(iid)
            fids = store.get_point_frames(iid)
            if pts is None or fids is None or len(pts) < self.min_points:
                continue
            if corrections:
                from .build_instances import _apply_window_corrections
                pts = _apply_window_corrections(store, iid, pts, corrections)
            wins = np.array([self.window_map.get(int(f), windows[0]) for f in fids])
            for a in range(len(windows) - 1):
                wa, wb = windows[a], windows[a + 1]
                sel = (wins == wa) | (wins == wb)
                if (int((wins == wa).sum()) < self.min_points
                        or int((wins == wb).sum()) < self.min_points):
                    continue
                sub = pts[sel]
                obb = fit_gravity_aligned_obb(sub)
                if obb is None:
                    continue
                on = detect_onion(sub, obb[0], obb[1], min_points=self.min_points)
                seam = f"{wa}|{wb}"
                seams.setdefault(seam, []).append(on.separation_m)
                if persist:
                    store.set_onion_metric(iid, on.bimodal, on.separation_m,
                                           on.bic_delta, seam=seam)
        return seams

    @staticmethod
    def _seam_median(seams: dict[str, list[float]]) -> float:
        vals = [v for lst in seams.values() for v in lst]
        return float(np.median(vals)) if vals else 0.0

    # ── global OBB + onion + heatmap refresh ─────────────────────────
    def _refresh_global_geometry(self, store: InstanceStore, gravity,
                                 corrections: dict | None = None,
                                 persist: bool = True) -> tuple[int, float]:
        """(Re)fit the global gravity-aligned OBB + onion metric + heatmap for
        every instance, optionally on window-corrected points. The segmentation
        OBB (cleaned cloud — exactly what the viewer draws) stays canonical
        when present; the trimmed fit on lifted points is the fallback. Returns
        (n_bimodal, median separation over bimodal instances)."""
        from .build_instances import load_seg_display_obbs
        seg_obbs = load_seg_display_obbs(self.output_dir)
        trim = float(self.cfg.get("obb_trim_percent", 1.0))
        n_bimodal, seps = 0, []
        for inst in store.list_instances():
            if inst["status"] == "dynamic_excluded":
                continue
            iid = inst["instance_id"]
            pts = store.get_points(iid)
            if pts is None or len(pts) < 3:
                continue
            if corrections:
                from .build_instances import _apply_window_corrections
                pts = _apply_window_corrections(store, iid, pts, corrections)
            if iid in seg_obbs and not corrections:
                T, aabb = seg_obbs[iid]
                obb = (T, aabb, T[:3, 3])
            else:
                obb = fit_gravity_aligned_obb(pts, gravity=gravity, trim_percent=trim)
            if obb is None:
                continue
            T, aabb, pos = obb
            on = detect_onion(pts, T, aabb, min_points=self.min_points)
            if persist:
                store.set_obb(iid, T, aabb, pos, window_id="global", gravity=gravity,
                              n_points=len(pts))
                hm = (onion_heatmap(pts, T, aabb)
                      if len(pts) >= self.min_points else None)
                store.set_onion_metric(iid, on.bimodal, on.separation_m, on.bic_delta,
                                       heatmap_json=json.dumps(hm) if hm else None)
            if on.bimodal:
                n_bimodal += 1
                seps.append(on.separation_m)
        return n_bimodal, (float(np.median(seps)) if seps else 0.0)

    def _corrected_views(self, views: list[View],
                         corrections: dict) -> list[View]:
        """Re-vote support: apply each window's 4x4 similarity to its views'
        c2w so views and corrected points stay consistent (a within-window
        projection is invariant; only cross-window ones move)."""
        default = self._windows()[0]
        out = []
        for v in views:
            w = self.window_map.get(int(v.fid), default) if v.fid is not None else default
            M = corrections.get(w)
            if M is None:
                out.append(v)
            else:
                out.append(View(np.asarray(M, float) @ np.asarray(v.c2w, float),
                                v.K, v.masks, v.wh, fid=v.fid))
        return out

    @staticmethod
    def _edge_scale_std(edges: list[WindowEdge]) -> float:
        """Consistency of the inter-window scales implied by the shared
        instances (R.6 diagnostic): std of log s over all edges."""
        if not edges:
            return 0.0
        return float(np.std([np.log(max(e.measured.s, 1e-9)) for e in edges]))

    # ── main ────────────────────────────────────────────────────────
    def run(self, baseline_metrics: dict | None = None) -> PhaseRReport:
        # R.1/R.2/R.3/R.7/R.8 — build the baseline store (points + frame ids,
        # global OBBs, onion, baseline vote)
        builder = InstanceStoreBuilder(self.session_dir, self.output_dir, self.store_path,
                                       config=self.config)
        builder.build(gravity=None)
        store = InstanceStore(self.store_path)
        report = PhaseRReport()
        insts = store.list_instances()
        report.n_instances = len(insts)
        windows = self._windows()
        report.n_windows = len(windows)

        # R.4 gravity, then refit baseline global OBBs/onion/heatmaps under it
        gravity = self._estimate_gravity(store)
        report.gravity = gravity.tolist() if gravity is not None else []
        n_bi, sep_med = self._refresh_global_geometry(store, gravity, persist=True)

        # R.9 baseline metrics (auto A/B) — computed BEFORE any refinement;
        # caller-provided values (e.g. reprojection RMSE) override/extend.
        base_seams = self._seam_onion(store, windows, persist=False)
        obb_map = self._fit_window_obbs(store, gravity)
        edges = self._edges_from_obbs(obb_map, windows)
        prim_map = self._fit_window_primitives(store, gravity)
        prim_edges = self._edges_from_primitives(prim_map, windows)
        logger.info(f"[R.4] pose graph inputs: {len(windows)} windows, "
                    f"{len(edges)} OBB edges, {len(prim_edges)} primitive edges, "
                    f"scale priors for {len(self.window_scale_priors)} windows, "
                    f"marker_scale={self.marker_scale}")
        auto_baseline = {
            "vote_entropy_mean": self._mean_vote_entropy(store),
            "onion_separation_median": sep_med,
            "seam_error_median": self._seam_median(base_seams),
            "scale_S_std": self._edge_scale_std(edges),
        }
        if baseline_metrics:
            auto_baseline.update(baseline_metrics)

        # R.4 inter-window pose graph — REAL refinement loop (<= max_iterations):
        # solve -> clamp scale (R.6 authority) -> apply to OBBs (== re-lift) ->
        # rebuild edges (re-residuals) -> repeat until converged.
        widx = {w: i for i, w in enumerate(windows)}
        # R.6 — S prior per window: DA3 top-25% scale when provided; otherwise
        # 1.0 (= "the reconstruction is already metrically consistent, do not
        # rescale windows unless the instance evidence insists").
        priors = {widx[w]: float(self.window_scale_priors.get(w, 1.0))
                  for w in windows}
        auth = MetricAuthority(self.marker_scale) if self.marker_scale else None
        totals = [Sim3.identity() for _ in windows]
        converged = False
        if edges or prim_edges:
            for it in range(self.max_iterations):
                corr, stats = optimize_window_graph(
                    len(windows), edges,
                    scale_priors=priors or None,
                    primitive_edges=prim_edges or None,
                    primitive_weight=self.primitive_weight,
                    min_window_edges=int(self.cfg.get("min_window_edges", 2)))
                report.pose_graph = stats
                report.iterations = it + 1
                # R.6 — marker authority clamps any correction that would move
                # the metric scale beyond tolerance (marker wins, logged)
                if auth is not None:
                    for i, M in enumerate(corr):
                        anchor_scale = self.marker_scale * M.s
                        resolved = auth.resolve_scale(anchor_scale, window_id=windows[i])
                        if resolved != anchor_scale:
                            corr[i] = Sim3(1.0, M.R, M.t)  # keep pose, drop rescale
                step = max(M.magnitude() for M in corr)
                _locked = stats.get("locked_windows", [])
                logger.info(f"[R.4] iteration {it + 1}: cost {stats['cost0']:.6f} → "
                            f"{stats['cost']:.6f} ({stats['n_edges']} OBB + "
                            f"{stats.get('n_primitive_edges', 0)} primitive edges), "
                            f"max window step {step:.4f}"
                            + (f"; {len(_locked)} window(s) LOCKED at identity "
                               f"(<min_window_edges evidence): "
                               f"{[windows[i] for i in _locked]}" if _locked else ""))
                totals = [c.compose(t) for c, t in zip(corr, totals)]
                # apply == re-lift: transform the per-window OBBs + primitives,
                # rebuild both edge sets
                for (iid, w), entry in obb_map.items():
                    M = corr[widx[w]]
                    T2, aabb2, pos2 = _apply_sim3_to_obb(M, (entry[0], entry[1], entry[2]))
                    obb_map[(iid, w)] = [T2, aabb2, pos2, entry[3]]
                edges = self._edges_from_obbs(obb_map, windows)
                if prim_map:
                    prim_map = self._apply_sim3_to_primitives(prim_map, corr, widx)
                    prim_edges = self._edges_from_primitives(prim_map, windows)
                if step < self.converge_tol or stats["cost"] < 1e-6:
                    converged = True
                    break
        report.converged = converged
        logger.info(f"[R.4] pose graph {'converged' if converged else 'iteration-capped'} "
                    f"after {report.iterations} iteration(s); per-window corrections: "
                    + ", ".join(f"{w}: |t|={np.linalg.norm(totals[widx[w]].t):.4f}m "
                                f"s={totals[widx[w]].s:.5f}" for w in windows))

        # corrections as 4x4 (+ embedded frame->window map for point re-lift)
        corrections = {w: _sim3_matrix(totals[widx[w]]) for w in windows}
        corrections["__window_map__"] = dict(self.window_map)
        any_correction = any(t.magnitude() > 1e-9 for t in totals)

        # R.6 — S per window before/after (DA3 prior × correction scale)
        scale_report = ScaleReport()
        for w in windows:
            s_before = float(self.window_scale_priors.get(w, 1.0))
            scale_report.set(w, s_before, s_before * totals[widx[w]].s)
        report.scale_report = {"before": scale_report.before,
                               "after": scale_report.after,
                               "stability": scale_report.stability()}
        if auth is not None:
            report.scale_conflicts = auth.log.conflicts
        store.set_meta("scale_report", json.dumps(report.scale_report))
        if report.scale_conflicts:
            store.set_meta("scale_conflicts", json.dumps(report.scale_conflicts))

        # R.9 refined metrics — re-vote + re-onion on CORRECTED geometry
        views = []
        try:
            views = build_vote_views(self.session_dir, self.output_dir,
                                     dynamic_labels=builder.dynamic_labels,
                                     max_views=self.vote_max_views)
        except Exception:
            pass
        if any_correction:
            trial_views = self._corrected_views(views, corrections) if views else []
            trial_vote = run_scene_vote(store, trial_views, corrections=corrections,
                                        persist=False) if views else {"mean_entropy": 0.0}
            ref_seams = self._seam_onion(store, windows, corrections=corrections,
                                         persist=False)
            _nb, ref_sep = self._refresh_global_geometry(store, gravity,
                                                         corrections=corrections,
                                                         persist=False)
            refined_metrics = {
                "vote_entropy_mean": trial_vote["mean_entropy"] or auto_baseline["vote_entropy_mean"],
                "onion_separation_median": ref_sep,
                "seam_error_median": self._seam_median(ref_seams),
                "scale_S_std": self._edge_scale_std(edges),
            }
        else:
            ref_seams = base_seams
            refined_metrics = dict(auto_baseline)

        gate = compare_and_gate(auto_baseline, refined_metrics)
        report.used_anchoring = gate.use_anchoring
        logger.info(f"[R.9] A/B gate: anchoring {'KEPT' if gate.use_anchoring else 'FALLBACK'}"
                    + (f" — regressions: {gate.regressions}" if gate.regressions else "")
                    + f"; baseline={auto_baseline} refined={refined_metrics}")
        report.failsafe.update({"report": gate.report, "regressions": gate.regressions,
                                "baseline": auto_baseline, "refined": refined_metrics})
        if self.marker_scale is not None:
            report.failsafe.setdefault("marker_scale", self.marker_scale)

        # persist the chosen geometry
        if report.used_anchoring and any_correction:
            for inst in insts:
                iid = inst["instance_id"]
                pts = store.get_points(iid)
                fids = store.get_point_frames(iid)
                if pts is None:
                    continue
                from .build_instances import _apply_window_corrections
                store.set_points(iid, _apply_window_corrections(store, iid, pts, corrections),
                                 frame_ids=fids)
            # refined per-window OBBs + residual history
            for (iid, w), (T, aabb, pos, _wt) in obb_map.items():
                store.set_obb(iid, T, aabb, pos, window_id=str(w), gravity=gravity)
                store.set_window_history(iid, str(w), residual_applied=True)
            # global OBB/onion/heatmaps + seams + votes on the final geometry
            self._refresh_global_geometry(store, gravity, persist=True)
            self._seam_onion(store, windows, persist=True)
            if views:  # final vote: corrected points (persisted) + corrected views
                run_scene_vote(store, self._corrected_views(views, corrections),
                               persist=True)
        else:
            # baseline kept — still persist the seam diagnostics of the baseline
            self._seam_onion(store, windows, persist=True)

        # onion aggregate for the report (post-decision store state)
        report.onion_bimodal, report.onion_separation_median_m = 0, 0.0
        seps = []
        for inst in insts:
            m = store.get_metrics(inst["instance_id"]).get("onion")
            if m and m["bimodal"]:
                report.onion_bimodal += 1
                seps.append(m["separation_m"])
        report.onion_separation_median_m = float(np.median(seps)) if seps else 0.0
        report.seams = {k: {"median_separation_m": float(np.median(v)), "n": len(v)}
                        for k, v in ref_seams.items()}

        # R.4/R.5 WRITEBACK — close the loop into the fusion, gated by the A/B
        # fail-safe: only write refined artifacts when anchoring is kept.
        report.writeback = self._writeback(store, windows, totals, report.used_anchoring)

        store.set_meta("phase_r_report", report.summary())
        store.close()
        # persist the report next to the store — every caller (pipeline stage,
        # endpoint, CLI, demo) leaves the same artifact, and the pipeline's
        # resume probe uses it to know this session is already anchored
        try:
            (self.output_dir / "phase_r_report.json").write_text(json.dumps({
                "summary": report.summary(),
                "n_instances": report.n_instances,
                "n_windows": report.n_windows,
                "iterations": report.iterations,
                "converged": report.converged,
                "onion_bimodal": report.onion_bimodal,
                "onion_separation_median_m": report.onion_separation_median_m,
                "seams": report.seams,
                "scale_report": report.scale_report,
                "scale_conflicts": report.scale_conflicts,
                "failsafe": report.failsafe,
                "used_anchoring": report.used_anchoring,
                "writeback": report.writeback,
            }, indent=2, default=str))
        except Exception:
            pass
        return report

    # ── R.4/R.5 writeback into the fusion inputs (gated) ────────────
    def _writeback(self, store, windows, corr, used_anchoring: bool) -> dict:
        wb: dict = {"poses": [], "depth": None, "gated_out": False}
        if not used_anchoring:
            wb["gated_out"] = True   # A/B regressed → keep baseline fusion inputs
            return wb
        if not self.cfg.get("writeback_poses", True):
            return wb
        try:
            from .writeback import compose_refined_poses, write_refined_camera_poses
            from segmentation.session_io import _load_camera_source
            cam = _load_camera_source(self.session_dir, self.output_dir)
            if cam is not None and corr is not None and len(windows) > 1:
                refined = compose_refined_poses(cam.pose_map, self.window_map, windows, corr)
                wb["poses"] = write_refined_camera_poses(self.output_dir, refined)
            else:
                wb["poses_note"] = ("single window / no inter-window correction — "
                                    "poses unchanged (identity writeback)")
        except Exception as e:  # never let writeback break the pipeline
            wb["poses_error"] = str(e)
        # R.4 — correct the CLOUD geometry too (chunks pre-merge in the
        # anchored order; merged cloud via frame_global on resumed sessions),
        # so the fusion inputs AND the viewer see the anchored result.
        if self.cfg.get("writeback_clouds", True) and corr is not None and len(windows) > 1:
            try:
                from .writeback import apply_corrections_to_clouds
                wb["clouds"] = apply_corrections_to_clouds(
                    self.output_dir, windows, corr, self.window_map)
            except Exception as e:
                wb["clouds_error"] = str(e)
            # a window correction with s ≠ 1 rescales the world → the camera-frame
            # depth of that window's frames must scale by s too, or the TSDF
            # integrates depth inconsistent with the corrected poses/cloud
            try:
                from .writeback import apply_corrections_to_aligned_depth
                wb["aligned_depth"] = apply_corrections_to_aligned_depth(
                    self.output_dir, windows, corr, self.window_map)
            except Exception as e:
                wb["aligned_depth_error"] = str(e)
        if self.cfg.get("writeback_depth", False):
            try:
                from .writeback import apply_depth_regularization
                wb["depth"] = apply_depth_regularization(
                    self.output_dir, store, self.session_dir, self.config)
            except Exception as e:
                wb["depth_error"] = str(e)
        logger.info(f"[R.4/R.5] writeback: {wb}")
        return wb

    def _mean_vote_entropy(self, store: InstanceStore) -> float:
        vals = []
        for inst in store.list_instances():
            v = store.get_metrics(inst["instance_id"]).get("vote")
            if v and v.get("mean_entropy") is not None:
                vals.append(v["mean_entropy"])
        return float(np.mean(vals)) if vals else 0.0
