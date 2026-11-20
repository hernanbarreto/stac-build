"""F5 carries the cube graph (precision/graph_prior.py): the edges it takes from what the cube run left, and their
residuals and derivatives in the solver's unknowns. CPU, synthetic."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision import graph_prior as GP                     # noqa: E402
from precision.ba_torch import so3_exp                       # noqa: E402

D = dict(dtype=torch.float64)
EYE = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def _run(tmp_path, closures=(), n=10, sets=((0, 1, 2, 3, 4, 5), (3, 4, 5, 6, 7, 8, 9)), seams=None):
    d = tmp_path / "maplong_run"
    d.mkdir(parents=True, exist_ok=True)
    owner = [0] * 5 + [1] * (n - 5)
    (d / GP.LAYOUT_NAME).write_text(json.dumps({"chunk_sets": [list(s) for s in sets], "owner": owner[:n]}))
    seams = [{"a": 1, "b": 0, "sigma_m": 0.02, "lever_m": 2.0}] if seams is None else seams
    (d / GP.GRAPH_NAME).write_text(json.dumps({
        "seams": seams, "closures": list(closures),
        "poses": [{"s": 1.0, "R": EYE, "t": [0.0, 0.0, 0.0]} for _ in sets]}))
    return d


def _closure(i, j, t=(0.0, 0.0, 0.0), sigma=0.05):
    return {"a": 1, "b": 0, "s": 1.0, "R": EYE, "t": list(t), "centroid": [0.5, 0.0, 2.0], "lever_m": 1.5,
            "sigma_m": sigma, "revisit": [i, j]}


def test_every_keyframe_of_a_cube_is_tied_to_its_middle_keyframe_with_the_seams_own_residual(tmp_path):
    ed = GP.graph_edges(_run(tmp_path), 10)
    cube = ed["kind"] == GP.KIND_CUBE
    pairs = sorted(zip(ed["k"][cube].tolist(), ed["l"][cube].tolist()))
    assert pairs == sorted([(3, m) for m in (0, 1, 2, 4, 5)] + [(6, m) for m in (3, 4, 5, 7, 8, 9)])
    assert np.all(ed["sigma_m"][cube] == 0.02) and np.all(ed["lever_m"][cube] == 2.0)
    assert ed["held"].tolist() == [3] and not (ed["kind"] == GP.KIND_CLOSURE).any()
    # the shared keyframes 3, 4, 5 are tied to BOTH cubes: the seam is carried by the stars
    assert {(3, 4), (6, 4)} <= set(pairs) and (6, 3) in pairs


def test_a_closure_ties_its_two_keyframes_as_far_as_the_solved_graph_honours_it(tmp_path):
    ed = GP.graph_edges(_run(tmp_path, [_closure(1, 8), _closure(0, 9, t=(1.0, 0.0, 0.0))]), 10)
    cl = ed["kind"] == GP.KIND_CLOSURE
    assert list(zip(ed["k"][cl].tolist(), ed["l"][cl].tolist())) == [(1, 8), (0, 9)]
    honoured, missed = ed["sigma_m"][cl]
    assert abs(honoured - 0.05) < 1e-12                      # the solved graph agrees with it: its own fit's sigma
    assert abs(missed - np.hypot(0.05, 1.0)) < 1e-9          # off by 1 m at its surfaces: it ties nothing
    assert GP.summary(ed)["closure"]["edges"] == 2 and GP.summary(ed)["edges"] == 13


def test_what_cannot_be_weighed_fails_naming_it(tmp_path):
    with pytest.raises(GP.GraphPriorError, match="is missing"):
        GP.graph_edges(tmp_path / "nothing", 10)
    with pytest.raises(GP.GraphPriorError, match="places 10 keyframes, the session has 12"):
        GP.graph_edges(_run(tmp_path / "a"), 12)
    with pytest.raises(GP.GraphPriorError, match="ONE cube has no seam"):
        GP.graph_edges(_run(tmp_path / "b", sets=(tuple(range(10)),), seams=[]), 10)
    with pytest.raises(GP.GraphPriorError, match="outside 0"):
        GP.graph_edges(_run(tmp_path / "c", [_closure(1, 30)]), 10)


def test_the_derivatives_are_those_of_the_solvers_own_update():
    torch.manual_seed(0)
    N, E = 7, 9
    R, t = so3_exp(torch.randn(N, 3, **D) * 0.7), torch.randn(N, 3, **D)
    k = torch.tensor([0, 1, 2, 3, 4, 5, 0, 2, 6])
    l = torch.tensor([1, 2, 3, 4, 5, 6, 4, 6, 1])
    Rk, tk = GP.relative(R, t, k, l)
    e0, _, _ = GP.edge_terms(R, t, k, l, Rk, tk, torch.ones(E, **D), torch.ones(E, **D))
    assert float(e0.abs().max()) == 0.0                      # at the reference there is no residual at all
    # a reference ~10 degrees and 5 cm away: the derivatives must hold far from zero residual
    R0, t0 = so3_exp(torch.randn(E, 3, **D) * 0.17) @ Rk, tk + torch.randn(E, 3, **D) * 0.05
    sig_t = torch.rand(E, **D) * 0.05 + 0.01
    sig_r = sig_t / 1.9
    e, Jk, Jl = GP.edge_terms(R, t, k, l, R0, t0, sig_t, sig_r)
    h = 1e-6
    for cam in range(N):
        for a in range(6):
            d6 = torch.zeros(N, 6, **D)
            d6[cam, a] = h
            ep, _, _ = GP.edge_terms(so3_exp(d6[:, :3]) @ R, t + d6[:, 3:], k, l, R0, t0, sig_t, sig_r, with_jac=False)
            em, _, _ = GP.edge_terms(so3_exp(-d6[:, :3]) @ R, t - d6[:, 3:], k, l, R0, t0, sig_t, sig_r, with_jac=False)
            ana = (torch.where((k == cam)[:, None], Jk[:, :, a], torch.zeros(E, 6, **D))
                   + torch.where((l == cam)[:, None], Jl[:, :, a], torch.zeros(E, 6, **D)))
            assert float(((ep - em) / (2 * h) - ana).abs().max()) < 1e-6 * max(1.0, float(e.abs().max()))


def test_the_logarithm_and_the_exponential_undo_each_other_down_to_the_smallest_angles():
    torch.manual_seed(1)
    axis = torch.randn(6, 3, **D)
    axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True)
    for angle in (1e-12, 1e-7, 1e-3, 0.5, 3.0):             # every angle under half a turn has one rotation vector
        w = axis * angle
        assert float((GP.so3_log(so3_exp(w)) - w).abs().max()) <= 1e-12 * max(1.0, angle / 1e-3)
