"""The chunk-level Sim(3) graph (USER 2026-11-04, vendor/VGGT-Long/loop_utils/chunk_graph.py):
the algebra, the chain the seams alone give and its round trip through the fork's
accumulate_sim3_transforms, a consistent ring left alone, a scale drift chained along the seams
closed by loops measured on the truth, the DA3 priors holding a loopless chain, and the
leave-one-out judge. CPU only, synthetic."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

FORK = Path(__file__).resolve().parents[2] / "vendor" / "VGGT-Long"
sys.path.insert(0, str(FORK))

from loop_utils import chunk_graph as G  # noqa: E402


def _rand_sim3(rng, s_sigma=0.05, t_scale=2.0, rot=0.3):
    return (float(np.exp(rng.normal(0, s_sigma))), G._rodrigues(rng.normal(0, rot, 3)),
            rng.normal(0, t_scale, 3))


def _close(A, B, tol=1e-9):
    return (abs(A[0] - B[0]) < tol and np.allclose(A[1], B[1], atol=tol)
            and np.allclose(A[2], B[2], atol=tol))


def test_the_algebra_and_the_rotation_vector_round_trip():
    rng = np.random.default_rng(0)
    A, B = _rand_sim3(rng), _rand_sim3(rng)
    x = rng.normal(0, 1, (5, 3))
    assert np.allclose(G.apply(G.compose(A, B), x), G.apply(A, G.apply(B, x)))
    assert _close(G.compose(A, G.inverse(A)), G.identity())
    for w in (np.zeros(3), np.array([1e-9, 0, 0]), rng.normal(0, 1, 3), np.array([0, 0, np.pi - 1e-8])):
        assert np.allclose(G._rodrigues(G._rotvec(G._rodrigues(w))), G._rodrigues(w), atol=1e-7)


def _truth(n, rng):
    """World poses of n chunks along a walk."""
    poses = [G.identity()]
    for k in range(1, n):
        step = (1.0, G._rodrigues(np.array([0, 0.25, 0]) + rng.normal(0, 0.02, 3)),
                np.array([1.5, 0.0, 0.3]) + rng.normal(0, 0.05, 3))
        poses.append(G.compose(poses[-1], step))
    return poses


def _edge(a, b, C, centroid=(0.0, 0.0, 2.0), sigma=0.05, lever=2.0, bias=None, bridge=None):
    """The edge a→b the poses C imply (chunk-a onto chunk-b coordinates), optionally biased."""
    M = G.compose(G.inverse(C[b]), C[a])
    if bias is not None:
        M = G.compose(bias, M)
    return {"a": a, "b": b, "s": M[0], "R": M[1], "t": M[2], "centroid": list(centroid),
            "sigma_m": sigma, "lever_m": lever, "bridge": bridge}


def test_the_chain_round_trips_through_the_forks_accumulation():
    from loop_utils.sim3utils import accumulate_sim3_transforms
    rng = np.random.default_rng(1)
    C = _truth(5, rng)
    seams = [_edge(k + 1, k, C) for k in range(4)]
    chain = G.chain_poses(5, seams)
    for k in range(5):
        assert _close(chain[k], C[k], 1e-9)
    acc = accumulate_sim3_transforms(G.sequential(chain))
    for k in range(1, 5):
        s, R, t = acc[k - 1]
        assert _close((s, R, t), chain[k], 1e-9)


def test_a_consistent_ring_is_left_as_the_chain():
    rng = np.random.default_rng(2)
    C = _truth(6, rng)
    seams = [_edge(k + 1, k, C) for k in range(5)]
    loops = [_edge(5, 0, C, sigma=0.3, lever=5.0, bridge=0)]
    priors = {k: {"log_s": 0.0, "sigma": 0.05} for k in range(6)}
    sol = G.solve(6, seams, loops, priors)
    assert sol["converged"]
    for k in range(6):
        assert _close(sol["poses"][k], C[k], 1e-6)
    assert max(sol["residuals"]) < 1e-6


def test_a_scale_drift_along_the_seams_is_closed_by_the_loops():
    """Every seam carries +3 % of scale (the chain grows 1.03^7 ≈ 23 % by the end, pccr's
    start↔end disagreement); loops measured on the truth close it."""
    rng = np.random.default_rng(3)
    n = 8
    C = _truth(n, rng)
    seams = [_edge(k + 1, k, C, sigma=0.08, lever=2.5,
                   bias=(1.03, np.eye(3), np.zeros(3))) for k in range(n - 1)]
    loops = [_edge(n - 1, 0, C, sigma=0.3, lever=4.0, bridge=0),
             _edge(n - 2, 1, C, sigma=0.3, lever=4.0, bridge=1),
             _edge(n - 1, 2, C, sigma=0.3, lever=4.0, bridge=2)]
    priors = {k: {"log_s": 0.0, "sigma": 0.08 / np.sqrt(3)} for k in range(n)}
    chain = G.chain_poses(n, seams)
    sol = G.solve(n, seams, loops, priors)
    assert sol["converged"]
    s_chain = chain[n - 1][0] / C[n - 1][0]
    s_sol = sol["poses"][n - 1][0] / C[n - 1][0]
    assert s_chain > 1.2                          # the drift the seams chained
    assert abs(np.log(s_sol)) < 0.5 * abs(np.log(s_chain))
    for e in loops:
        assert (G.edge_error_m(sol["poses"][e["a"]], sol["poses"][e["b"]], e)
                < 0.5 * G.edge_error_m(chain[e["a"]], chain[e["b"]], e))


def test_the_da3_priors_hold_a_loopless_chain_toward_their_scales():
    rng = np.random.default_rng(4)
    n = 6
    C = _truth(n, rng)
    seams = [_edge(k + 1, k, C, sigma=0.08, lever=2.5,
                   bias=(1.03, np.eye(3), np.zeros(3))) for k in range(n - 1)]
    strong = {k: {"log_s": 0.0, "sigma": 0.01} for k in range(n)}
    none = G.solve(n, seams, [], {})
    held = G.solve(n, seams, [], strong)
    assert abs(np.log(none["poses"][n - 1][0]) - 5 * np.log(1.03)) < 1e-6   # the chain itself
    assert abs(np.log(held["poses"][n - 1][0])) < abs(np.log(none["poses"][n - 1][0]))


def test_the_judge_applies_closures_that_predict_each_other():
    rng = np.random.default_rng(5)
    n = 9
    C = _truth(n, rng)
    seams = [_edge(k + 1, k, C, sigma=0.08, lever=2.5,
                   bias=(1.03, np.eye(3), np.zeros(3))) for k in range(n - 1)]
    loops = [_edge(a, b, C, sigma=0.2, lever=4.0, bridge=i)
             for i, (a, b) in enumerate([(8, 0), (7, 0), (8, 1), (6, 1), (7, 2), (8, 2)])]
    priors = {k: {"log_s": 0.0, "sigma": 0.08 / np.sqrt(3)} for k in range(n)}
    out = G.judge_leave_one_out(n, seams, loops, priors, error_factor=1.1, confidence=0.95,
                                log=lambda m: None)
    assert out["improves"], out.get("reason")
    assert len(out["judges"]) == 6 and out["discarded"] == 0
    few = G.judge_leave_one_out(n, seams, loops[:3], priors, error_factor=1.1, confidence=0.95,
                                log=lambda m: None)
    assert not few["improves"] and "needed to judge" in few["reason"]


def test_closures_that_disagree_with_each_other_are_not_applied():
    rng = np.random.default_rng(6)
    n = 9
    C = _truth(n, rng)
    seams = [_edge(k + 1, k, C, sigma=0.08, lever=2.5) for k in range(n - 1)]    # a clean chain
    junk = [_edge(a, b, C, sigma=0.2, lever=4.0, bridge=i,
                  bias=_rand_sim3(np.random.default_rng(100 + i), s_sigma=0.1, t_scale=0.8, rot=0.1))
            for i, (a, b) in enumerate([(8, 0), (7, 0), (8, 1), (6, 1), (7, 2), (8, 2)])]
    priors = {k: {"log_s": 0.0, "sigma": 0.08 / np.sqrt(3)} for k in range(n)}
    out = G.judge_leave_one_out(n, seams, junk, priors, error_factor=1.1, confidence=0.95,
                                log=lambda m: None)
    assert not out["improves"]


# ── the fork's stage, run from its source (vggt_long.VGGT_Long) ──

import ast  # noqa: E402
import json  # noqa: E402
import textwrap  # noqa: E402

SRC = (FORK / "vggt_long.py").read_text()


def _method(name):
    tree = ast.parse(SRC)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "VGGT_Long")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return textwrap.dedent(ast.get_source_segment(SRC, fn))


def _bind(obj, *names):
    ns = {"np": np, "os": os}
    for nm in names:
        exec(_method(nm), ns)
        setattr(obj, nm, (lambda f: (lambda *a, **k: f(obj, *a, **k)))(ns[nm]))
    return obj


class _Self:
    pass


def test_a_seam_edge_records_where_and_how_far_its_copies_were_compared():
    s = _Self()
    s._stac_seam_residuals = {2: 0.07}
    ov, H, W = 3, 4, 5
    pm = np.zeros((ov, H, W, 3))
    pm[..., 2] = 4.0                                     # every point 4 m in front of z = 0
    pm[..., 0] = np.arange(W)[None, None, :] * 0.1
    ext = np.tile(np.eye(4), (10, 1, 1))                  # chunk k's 10 frames, cameras at origin
    conf = np.ones((ov, H, W))
    conf[0, 0, 0] = 0.0                                  # one pixel left out
    _bind(s, "_stac_record_seam_edge")
    s._stac_record_seam_edge(2, ov, {"extrinsic": ext}, pm, conf, np.ones((ov, H, W)), None,
                             (1.02, np.eye(3), np.array([0.1, 0.0, 0.0])))
    e = s._stac_seam_edges[2]
    assert e["a"] == 3 and e["b"] == 2 and e["s"] == 1.02 and e["sigma_m"] == 0.07 and e["exact"]
    assert np.allclose(e["centroid"], [0.2, 0.0, 4.0])
    assert 4.0 <= e["lever_m"] <= 4.03


def test_the_stage_judges_applies_and_records_the_graph(tmp_path):
    from loop_utils.sim3utils import accumulate_sim3_transforms
    rng = np.random.default_rng(7)
    n = 9
    C = _truth(n, rng)
    s = _Self()
    s.chunk_indices = [(k * 10, k * 10 + 20) for k in range(n)]
    s._stac_seam_edges = {k: dict(_edge(k + 1, k, C, sigma=0.08, lever=2.5,
                                        bias=(1.03, np.eye(3), np.zeros(3))), exact=True)
                          for k in range(n - 1)}
    s._stac_loop_edges = []
    for i, (a, b) in enumerate([(8, 0), (7, 0), (8, 1), (6, 1), (7, 2), (8, 2)]):
        e = _edge(a, b, C, sigma=0.2, lever=4.0, bridge=i)
        s._stac_loop_edges.append({"a": a, "b": b, "R_ab": e["R"], "t_ab": e["t"], "s_ab": e["s"],
                                   "centroid_b": e["centroid"], "range_m": 4.0, "sigma_m": 0.2,
                                   "status": "accepted", "bridge": i})
    s._stac_scale_inputs = {"s_da3": {k: 1.0 for k in range(n)}, "n_anchors": {k: 3 for k in range(n)},
                            "sigma_anchor": 0.08}
    s._stac_scales_applied = {k: 1.0 for k in range(n)}
    s._stac_graph_cfg = lambda: {"heldout_confidence": 0.95, "improvement_error_factor": 1.1}
    s._stac_stamp_digest = lambda: None
    s.output_dir = str(tmp_path)
    s.sim3_list = []
    _bind(s, "_stac_chunk_graph")
    s._stac_chunk_graph()
    doc = json.loads((tmp_path / "chunk_graph.json").read_text())
    assert doc["applied"] == "seams + closures + DA3 priors" and doc["judge"]["improves"]
    assert len(doc["closures"]) == 6 and len(doc["seams"]) == n - 1 and len(doc["priors"]) == n
    for c in doc["closure_errors"]:
        assert c["final_m"] < 0.5 * c["chain_m"]
    acc = accumulate_sim3_transforms(s.sim3_list)
    poses = [G.identity()] + [tuple(x) for x in acc]
    assert abs(np.log(poses[n - 1][0])) < 0.5 * abs(np.log(1.03 ** (n - 1)))


def test_without_enough_closures_the_seams_and_the_priors_decide(tmp_path):
    rng = np.random.default_rng(8)
    n = 5
    C = _truth(n, rng)
    s = _Self()
    s.chunk_indices = [(k * 10, k * 10 + 20) for k in range(n)]
    s._stac_seam_edges = {k: dict(_edge(k + 1, k, C, sigma=0.08, lever=2.5), exact=True)
                          for k in range(n - 1)}
    s._stac_seam_edges[1]["sigma_m"] = None              # a starved seam
    s._stac_loop_edges = []
    s._stac_scale_inputs, s._stac_scales_applied = {}, {}
    s._stac_graph_cfg = lambda: {"heldout_confidence": 0.95, "improvement_error_factor": 1.1}
    s._stac_stamp_digest = lambda: None
    s.output_dir = str(tmp_path)
    _bind(s, "_stac_chunk_graph")
    s._stac_chunk_graph()
    doc = json.loads((tmp_path / "chunk_graph.json").read_text())
    assert doc["applied"] == "seams + DA3 priors (no closure)"
    assert doc["seams"][1]["sigma_source"].startswith("starved seam")
    assert doc["seams"][1]["sigma_m"] == 0.08
    for k, p in enumerate(doc["poses"]):
        assert abs(p["s"] - C[k][0]) < 1e-6 and np.allclose(p["t"], C[k][2], atol=1e-6)


def test_every_verified_closure_enters_even_when_the_judge_cannot_tell(tmp_path):
    """USER 2026-11-04 ("sí"): closures enter like a SLAM's — 3 true closures with local offsets
    (fewer than the judge's 5) and one metres off: the true ones close, the far one barely moves
    the graph (the Cauchy loss)."""
    rng = np.random.default_rng(9)
    n = 8
    C = _truth(n, rng)
    s = _Self()
    s.chunk_indices = [(k * 10, k * 10 + 20) for k in range(n)]
    s._stac_seam_edges = {k: dict(_edge(k + 1, k, C, sigma=0.03, lever=2.0,
                                        bias=(1.02, np.eye(3), np.zeros(3))), exact=True)
                          for k in range(n - 1)}
    loops = [(7, 0, 0.08, None), (5, 2, 0.08, None), (6, 1, 0.08, None),
             (7, 3, 0.17, (1.0, G._rodrigues(np.array([0, 0.5, 0])), np.array([1.5, 0, 0.5])))]
    s._stac_loop_edges = []
    for i, (a, b, sg, bias) in enumerate(loops):
        e = _edge(a, b, C, sigma=sg, lever=4.0, bridge=i, bias=bias)
        s._stac_loop_edges.append({"a": a, "b": b, "R_ab": e["R"], "t_ab": e["t"], "s_ab": e["s"],
                                   "centroid_b": e["centroid"], "range_m": 4.0, "sigma_m": sg,
                                   "status": "accepted", "bridge": i})
    s._stac_scale_inputs = {"s_da3": {k: 1.0 for k in range(n)}, "n_anchors": {k: 3 for k in range(n)},
                            "sigma_anchor": 0.08}
    s._stac_scales_applied = {k: 1.0 for k in range(n)}
    s._stac_graph_cfg = lambda: {"heldout_confidence": 0.95, "improvement_error_factor": 1.1}
    s._stac_stamp_digest = lambda: None
    s.output_dir = str(tmp_path)
    _bind(s, "_stac_chunk_graph")
    s._stac_chunk_graph()
    doc = json.loads((tmp_path / "chunk_graph.json").read_text())
    assert doc["applied"] == "seams + closures + DA3 priors"
    assert doc["judge"]["improves"] is False                # measured, recorded, not deciding
    errs = {c["bridge"]: c for c in doc["closure_errors"]}
    for i in (0, 1, 2):
        assert errs[i]["final_m"] < 0.6 * errs[i]["chain_m"]
    assert errs[3]["final_m"] > 0.8 * errs[3]["chain_m"]    # the false one is weighed down


def test_the_mode_is_read_strictly():
    class _M:
        def __init__(self, m):
            self.config = {"Model": m}
    ns = {"np": np, "os": os}
    exec(_method("_stac_chunk_graph_mode"), ns)
    f = ns["_stac_chunk_graph_mode"]
    assert f(_M({})) is None and f(_M({"chunk_graph": {"mode": "sim3"}})) == "sim3"
    with pytest.raises(ValueError, match="sim3"):
        f(_M({"chunk_graph": {"mode": "sl4"}}))


def test_a_sim3_closure_carries_no_scale_term_in_its_sigma():
    """pccr 2026-11-04: verify_loop given no scale disagreement reads log s_ab of the measurement
    and adds it to σ (the start↔end closure: held-out 7.9 cm → σ 28 cm); in the chunk Sim(3)
    graph the scale is the graph's variable — the stage passes 0."""
    body = _method("_stac_verify_loops")
    assert "_sres = (0.0 if sim3 else" in body
    from loop_utils.loop_bridges import verify_loop
    meas = {"ok": True, "s_ab": 1.10, "R_ab": np.eye(3).tolist(), "t_ab": [0, 0, 0],
            "range_m": 2.7, "residual_m": 0.02, "holdout_residual_m": 0.079,
            "holdout_ratio": 1.0, "n_corr": 50000}
    import yaml
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from reconstruction.loops.config import fork_model_loops, load_loops_config
    live = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text())
    cfg = fork_model_loops(load_loops_config(live), str(Path(__file__).resolve().parents[1]))
    with_term = verify_loop(meas, cfg)
    without = verify_loop(meas, cfg, scale_disagreement_log=0.0)
    assert without["checks"]["scale"]["sigma_scale_m"] == 0.0
    assert with_term["checks"]["scale"]["sigma_scale_m"] > 0.2


# ── 2026-11-04: closures by geometry and the chain-consistency test ──

def _corridor_out_and_back(n_half=40, step=0.25):
    """A corridor walked along +x, then walked back along −x a metre to the side: keyframes
    0..n_half−1 look +x, n_half..2n_half−1 look −x."""
    from loop_utils.revisits import propose, quick_visibility  # noqa: F401
    poses, pts = [], []
    rng = np.random.default_rng(0)
    for k in range(2 * n_half):
        back = k >= n_half
        x = (k if not back else 2 * n_half - 1 - k) * step
        y = 1.0 if back else 0.0
        fwd = np.array([-1.0, 0, 0]) if back else np.array([1.0, 0, 0])
        down = np.array([0, 0, -1.0])
        right = np.cross(down, fwd)
        M = np.eye(4)
        M[:3, :3] = np.stack([right, down, fwd], 1)
        M[:3, 3] = [x, y, 1.5]
        poses.append(M)
        # the corridor's walls 2 m to either side, ahead of the camera
        d = rng.uniform(1.0, 6.0, 200)
        lat = rng.choice([-1.5, 2.5], 200)
        P = np.stack([x + fwd[0] * d, np.full(200, 0.0) + lat, rng.uniform(0.2, 2.8, 200)], 1)
        pts.append(P)
    K = np.array([[300.0, 0, 232], [0, 300.0, 416], [0, 0, 1]])
    return np.stack(poses), np.stack([K] * len(poses)), pts


def test_the_geometry_proposes_the_revisit_between_chunks_that_share_no_frame():
    from loop_utils.revisits import propose, quick_visibility
    poses, Ks, pts = _corridor_out_and_back()
    n = len(poses)
    owner = np.minimum(np.arange(n) // 16, 4)               # chunks of 16 keyframes
    cent = poses[:, :3, 3]
    walk = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(cent, axis=0), axis=1))])
    cfg = {"min_depth_m": 0.1, "max_depth_m": 15.0, "frustum_margin_px": 0.0}
    F = quick_visibility(pts, poses, Ks, (832, 464), cfg, walk, lambda L: (0.05, 1.0))
    assert F.shape == (n, n) and F[0, 1] > 0.5
    props = propose(F, owner, walk, min_walk_m=1.0, min_visible_frac=0.3)
    assert props, "the walk back sees the walk out"
    for (a, b), lst in props.items():
        assert abs(a - b) >= 2 and 1 <= len(lst) <= 3
        for s, i, j in lst:
            assert s >= 0.3 and abs(walk[j] - walk[i]) >= 1.0 and owner[i] != owner[j]
        assert [x[0] for x in lst] == sorted([x[0] for x in lst], reverse=True)
    assert propose(F, owner, walk, min_walk_m=1e9, min_visible_frac=0.3) == {}


def test_the_bridge_planner_adds_the_geometric_revisits_in_sim3_mode():
    body = _method("_stac_plan_bridges")
    assert "self._stac_geometric_revisits(view, gate_mod, lcfg," in body
    assert 'if self._stac_chunk_graph_mode() == "sim3":' in body
    geo = _method("_stac_geometric_revisits")
    assert "gate_mod.gate_frame_pair(int(i), int(j), view, sp)" in geo
    assert '"source": "geometry"' in geo
    assert "covered(ki, ra, kj, rb, existing)" in geo
    stage = _method("_stac_chunk_graph")
    # no pre-filter of the closures (USER 2026-09-16: no separation bar rejects anything) — the
    # judge decides on every verified closure, the robust loss weighs the far ones down
    assert "CG.judge_leave_one_out(n, seams, loops, priors" in stage


def test_a_closure_covers_a_candidate_only_between_the_same_two_chunks():
    from loop_utils.revisits import covered
    existing = [(5, (120, 140), 4, (100, 120))]             # a SALAD pair inside one pass
    assert not covered(3, (81, 101), 5, (115, 135), existing)          # pccr's 3↔5 revisit
    assert covered(4, (105, 118), 5, (125, 139), existing)             # the same chunks, swapped
    assert covered(5, (125, 139), 4, (105, 118), existing)
    assert not covered(5, (141, 150), 4, (105, 118), existing)         # window a apart
    assert not covered(3, (81, 101), 5, (115, 135), [])
