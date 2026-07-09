# STAC-Builder — Phase R tests: primitive residuals, single-window pseudo-split,
# per-window scale priors, gated refined-S apply.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from phase_r.residuals import (CylinderPrim, PlanePrim, PrimitiveEdge, Sim3,  # noqa: E402
                               WindowEdge, fit_cylinder_primitive,
                               fit_plane_primitive, optimize_window_graph,
                               primitive_residual, transform_cylinder,
                               transform_plane)


def _rand_rot(rng, max_deg=10.0):
    from scipy.spatial.transform import Rotation
    v = rng.normal(size=3)
    v = v / np.linalg.norm(v) * np.deg2rad(max_deg) * rng.uniform(0.2, 1.0)
    return Rotation.from_rotvec(v).as_matrix()


# ── primitive fitting ────────────────────────────────────────────────
def test_fit_plane_primitive_recovers_plane():
    rng = np.random.default_rng(0)
    n = np.array([0.0, 1.0, 0.0])
    pts = rng.uniform(-5, 5, (500, 3))
    pts[:, 1] = 2.0 + rng.normal(0, 0.005, 500)      # y ≈ 2 plane
    p = fit_plane_primitive(pts)
    assert p is not None
    assert abs(abs(float(p.n @ n)) - 1.0) < 1e-3      # normal aligned (either sign)
    assert abs(abs(p.d) - 2.0) < 0.01                 # |offset| = 2


def test_fit_cylinder_primitive_recovers_axis_and_radius():
    rng = np.random.default_rng(1)
    theta = rng.uniform(0, 2 * np.pi, 800)
    z = rng.uniform(0, 4, 800)
    r_true = 0.30
    pts = np.column_stack([r_true * np.cos(theta), r_true * np.sin(theta), z])
    pts += rng.normal(0, 0.003, pts.shape)
    cy = fit_cylinder_primitive(pts)
    assert cy is not None
    assert abs(abs(float(cy.a @ np.array([0, 0, 1.0]))) - 1.0) < 1e-2
    assert abs(cy.r - r_true) < 0.02


# ── primitive transforms + residuals ─────────────────────────────────
def test_plane_transform_consistency():
    """Points on a plane, transformed by a Sim3, must lie on the transformed plane."""
    rng = np.random.default_rng(2)
    p = PlanePrim(np.array([0.0, 0.0, 1.0]), -5.0)    # z = 5
    M = Sim3(1.3, _rand_rot(rng, 25), np.array([0.4, -0.2, 1.1]))
    pts = np.column_stack([rng.uniform(-3, 3, 50), rng.uniform(-3, 3, 50), np.full(50, 5.0)])
    moved = np.array([M.apply(x) for x in pts])
    p2 = transform_plane(p, M)
    res = moved @ p2.n + p2.d
    assert np.abs(res).max() < 1e-9


def test_cylinder_transform_consistency():
    rng = np.random.default_rng(3)
    cy = CylinderPrim(np.array([0.0, 0.0, 1.0]), np.zeros(3), 0.3)
    M = Sim3(2.0, _rand_rot(rng, 30), np.array([1.0, 2.0, 3.0]))
    cy2 = transform_cylinder(cy, M)
    assert abs(cy2.r - 0.6) < 1e-12                    # radius scales
    theta = rng.uniform(0, 2 * np.pi, 30)
    pts = np.column_stack([0.3 * np.cos(theta), 0.3 * np.sin(theta),
                           rng.uniform(0, 2, 30)])
    moved = np.array([M.apply(x) for x in pts])
    d = moved - cy2.c
    radial = d - np.outer(d @ cy2.a, cy2.a)
    assert np.abs(np.linalg.norm(radial, axis=1) - cy2.r).max() < 1e-9


def test_primitive_residual_zero_when_agreeing():
    p = PlanePrim(np.array([0.0, 1.0, 0.0]), -2.0)
    assert np.allclose(primitive_residual(p, p, "plane"), 0)
    # sign-flipped normal must also read as agreement
    q = PlanePrim(-p.n, -p.d)
    assert np.allclose(primitive_residual(p, q, "plane"), 0)
    cy = CylinderPrim(np.array([0.0, 0.0, 1.0]), np.array([1.0, 2.0, 0.0]), 0.3)
    cy_shifted_along_axis = CylinderPrim(cy.a, cy.c + np.array([0, 0, 3.0]), cy.r)
    # a shift ALONG the axis is not a disagreement (infinite cylinder)
    assert np.allclose(primitive_residual(cy, cy_shifted_along_axis, "cylinder"), 0)


# ── pose graph with primitive edges ──────────────────────────────────
def test_graph_recovers_window_offset_from_plane_primitives_only():
    """Two windows disagree by a known translation; the SAME wall/floor planes
    seen from both should let the graph recover it WITHOUT OBB edges.

    NOTE: plane-only constraints leave a one-parameter scale↔translation gauge
    (s·d − n·t = d admits t = (s−1)·d·n for any s) — production always passes
    per-window scale priors (default 1.0), which is exactly what fixes it, so
    the test does the same."""
    t_err = np.array([0.15, -0.08, 0.10])
    M_err = Sim3(1.0, np.eye(3), t_err)               # window 1 is offset
    walls = [PlanePrim(np.array([1.0, 0.0, 0.0]), -2.0),
             PlanePrim(np.array([0.0, 1.0, 0.0]), -1.5),
             PlanePrim(np.array([0.0, 0.0, 1.0]), -4.0)]
    edges = []
    for w in walls:
        edges.append(PrimitiveEdge(0, 1, w, transform_plane(w, M_err), "plane", 1.0))
    corr, stats = optimize_window_graph(2, [], primitive_edges=edges,
                                        scale_priors={1: 1.0})
    # X_1 must undo the offset: X_1 ≈ M_err^{-1}
    got = corr[1].compose(M_err)
    assert got.magnitude() < 1e-3, (got.t, got.s)
    assert stats["n_primitive_edges"] == 3


def test_graph_obb_plus_primitives_beats_biased_obb():
    """Partial-view OBB bias: OBB edges alone pull windows to a WRONG offset;
    adding unbiased plane primitives must drag the solution toward truth."""
    # truth: windows agree perfectly (identity correction expected)
    biased = Sim3(1.0, np.eye(3), np.array([0.30, 0.0, 0.0]))   # biased OBB measurement
    obb_edges = [WindowEdge(0, 1, biased, 1.0)]
    walls = [PlanePrim(np.array([1.0, 0.0, 0.0]), -2.0),
             PlanePrim(np.array([0.0, 1.0, 0.0]), -1.5),
             PlanePrim(np.array([0.0, 0.0, 1.0]), -4.0)]
    prim_edges = [PrimitiveEdge(0, 1, w, w, "plane", 1.0) for w in walls]
    # min_window_edges=1 on the OBB-only solve: it demonstrates the bias the
    # guard exists to prevent (production default 2 would just lock the window)
    corr_obb, _ = optimize_window_graph(2, obb_edges, min_window_edges=1)
    corr_mix, _ = optimize_window_graph(2, obb_edges, primitive_edges=prim_edges,
                                        primitive_weight=4.0)
    err_obb = np.linalg.norm(corr_obb[1].t)
    err_mix = np.linalg.norm(corr_mix[1].t)
    assert err_mix < err_obb * 0.5, (err_obb, err_mix)


def test_scale_priors_pull_window_scale():
    """A scale prior ≠ 1 (from the per-window DA3 S) must move the solved
    window scale toward it when the edges don't resist."""
    edges = [WindowEdge(0, 1, Sim3.identity(), 0.01)]  # near-zero-information edge
    corr, _ = optimize_window_graph(2, edges, scale_priors={1: 1.05},
                                    scale_prior_weight=50.0, min_window_edges=1)
    assert abs(corr[1].s - 1.05) < 5e-3


# ── single-window pseudo-split ───────────────────────────────────────
def test_single_window_pseudo_split(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "workers"))
    from workers.phase_r_worker import _window_map
    out = tmp_path / "output"
    (out / "omega_run").mkdir(parents=True)
    n = 150                                            # fits ONE 120/60 chunk? 150 frames
    (out / "omega_run" / "camera_frames.txt").write_text(" ".join(str(i * 4) for i in range(n)))
    cfg = {"reconstruction": {"backend": "vggtomega",
                              "vggtomega": {"chunk_size": 200, "chunk_overlap": 60}},
           "phase_r": {"single_window_split_frames": 60}}
    wmap = _window_map(tmp_path, out, cfg)
    wins = sorted(set(wmap.values()))
    assert len(wins) >= 2 and all(w.startswith("s") for w in wins), wins
    # contiguous split: frame order maps to non-decreasing window ids
    ids = [wmap[i * 4] for i in range(n)]
    assert ids == sorted(ids)
    # disabled → single window preserved
    cfg["phase_r"]["single_window_split_frames"] = 0
    wmap2 = _window_map(tmp_path, out, cfg)
    assert len(set(wmap2.values())) == 1


# ── gated refined-S apply ────────────────────────────────────────────
def _mk_sr(delta_pct, n_frames=20, s_marker=10.0):
    return {"s_marker": s_marker, "s_refined": s_marker * (1 + delta_pct / 100.0),
            "delta_pct": delta_pct, "n_frames": n_frames}


def test_apply_refined_scale_gates(tmp_path):
    from phase_r.scale_check import apply_refined_scale
    cfg = {"phase_r": {"scale_recheck_apply": True, "scale_recheck_min_frames": 10,
                       "scale_recheck_min_delta_pct": 0.2,
                       "scale_recheck_max_delta_pct": 10.0}}
    db = tmp_path / "scene_r.db"
    # too few frames
    r = apply_refined_scale(tmp_path, db, _mk_sr(-2.7, n_frames=5), cfg)
    assert not r["applied"] and "structural frames" in r["reason"]
    # delta below noise floor
    r = apply_refined_scale(tmp_path, db, _mk_sr(0.05), cfg)
    assert not r["applied"] and "below min" in r["reason"]
    # delta absurdly large → refuse
    r = apply_refined_scale(tmp_path, db, _mk_sr(25.0), cfg)
    assert not r["applied"] and "exceeds max" in r["reason"]
    # master switch off
    r = apply_refined_scale(tmp_path, db, _mk_sr(-2.7),
                            {"phase_r": {"scale_recheck_apply": False}})
    assert not r["applied"]


def test_apply_refined_scale_applies_and_rescales_store(tmp_path):
    from phase_r.instance_store import InstanceStore
    from phase_r.scale_check import apply_refined_scale
    out = tmp_path
    (out / ".metric_scale_applied").write_text("s=10.000000\n")
    # minimal store with one instance
    db = out / "scene_r.db"
    st = InstanceStore(db)
    st.upsert_instance(1, "wall")
    pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], np.float32)
    st.set_points(1, pts, frame_ids=np.array([0, 4], np.int32))
    T = np.eye(4); T[:3, 3] = [1, 1, 1]
    st.set_obb(1, T, np.array([0, 1, 0, 1, 0, 1], float), np.array([1.0, 1, 1]))
    st.close()
    cfg = {"phase_r": {"scale_recheck_apply": True, "scale_recheck_min_frames": 10,
                       "scale_recheck_min_delta_pct": 0.2,
                       "scale_recheck_max_delta_pct": 10.0}}
    sr = _mk_sr(-2.0, n_frames=20, s_marker=10.0)      # δ = 0.98
    r = apply_refined_scale(out, db, sr, cfg)
    assert r["applied"] and abs(r["delta"] - 0.98) < 1e-9
    assert "s=9.8" in (out / ".metric_scale_applied").read_text()
    st = InstanceStore(db)
    assert np.allclose(st.get_points(1), pts * 0.98, atol=1e-5)
    T2, aabb2, pos2 = st.get_obb(1)
    assert np.allclose(T2[:3, 3], [0.98] * 3) and np.allclose(aabb2[1], 0.98)
    st.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


def test_underdetermination_guard_locks_sparse_windows():
    """A window supported by a single edge must stay at identity (the edge
    would otherwise be satisfied exactly, copying its noise into the pose)."""
    bad = Sim3(0.05, np.eye(3), np.array([3.0, 0.0, 0.0]))   # absurd single edge
    edges = [WindowEdge(0, 1, bad, 1.0)]
    corr, stats = optimize_window_graph(2, edges, min_window_edges=2)
    assert stats["locked_windows"] == [1]
    assert corr[1].magnitude() < 1e-12                        # untouched
    # with 2 edges the window unlocks and IS corrected
    edges2 = [WindowEdge(0, 1, Sim3(1.0, np.eye(3), np.array([0.1, 0, 0])), 1.0),
              WindowEdge(0, 1, Sim3(1.0, np.eye(3), np.array([0.1, 0, 0])), 1.0)]
    corr2, stats2 = optimize_window_graph(2, edges2, min_window_edges=2,
                                          scale_priors={1: 1.0})
    assert stats2["locked_windows"] == []
    assert abs(np.linalg.norm(corr2[1].t) - 0.1) < 1e-3
