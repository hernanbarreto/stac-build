"""The chunk graph's solve with its residuals in one batch on a device (USER 2026-11-19: "Todo debe hacerse en
placa" / "codificar todos los pasos en la placa porque es tiempo perdido") against loop_utils.chunk_graph, the
path of record."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

FORK = Path(os.environ.get("STAC_FORK_DIR") or Path(__file__).resolve().parents[2] / "vendor" / "VGGT-Long")
sys.path.insert(0, str(FORK))
from loop_utils import chunk_graph as CG  # noqa: E402
from loop_utils import chunk_graph_torch as CT  # noqa: E402

DEV = "cpu"


def _rand_pose(rng, big=False):
    w = rng.normal(size=3) * (1.2 if big else 0.2)
    return (float(np.exp(rng.normal() * 0.15)), CG._rodrigues(w), rng.normal(size=3) * 2.0)


def _graph(rng, n=9, n_loops=14, noise=0.01):
    """A walk of n chunks with seams between neighbours (and next-but-one) and loops between far ones: the
    edges measured from true poses plus noise, in chunk_graph's convention (M maps a onto b)."""
    true = [CG.identity()] + [_rand_pose(rng) for _ in range(n - 1)]

    def edge(a, b, sig, loop):
        M = CG.compose(CG.inverse(true[b]), true[a])
        dM = (float(np.exp(rng.normal() * noise)), CG._rodrigues(rng.normal(size=3) * noise), rng.normal(size=3) * noise)
        M = CG.compose(dM, M)
        e = {"a": a, "b": b, "s": M[0], "R": M[1], "t": M[2], "centroid": rng.normal(size=3) * 3.0,
             "lever_m": float(rng.uniform(0.5, 3.0)), "sigma_m": sig}
        if loop:
            e["revisit"] = [a, b]
        return e
    seams = [edge(k + 1, k, float(rng.uniform(0.005, 0.03)), False) for k in range(n - 1)]
    seams += [edge(k + 2, k, float(rng.uniform(0.01, 0.05)), False) for k in range(n - 2)]
    pairs = [(int(a), int(b)) for a, b in rng.integers(0, n, size=(n_loops, 2)) if abs(a - b) > 2]
    loops = [edge(a, b, float(rng.uniform(0.02, 0.08)), True) for a, b in pairs]
    if loops:                                                      # one wrong loop: Cauchy must carry it
        loops[0]["t"] = np.asarray(loops[0]["t"]) + np.array([1.5, -0.7, 0.4])
    return true, seams, loops


@pytest.mark.parametrize("robust_seams", [False, True])
def test_the_batched_residuals_are_the_records(robust_seams):
    rng = np.random.default_rng(3)
    true, seams, loops = _graph(rng)
    n = len(true)
    priors = {2: {"log_s": 0.05, "sigma": 0.02}, 5: {"log_s": -0.03, "sigma": 0.04}}
    prob = CT.Problem(n, seams, loops, priors, robust_seams, DEV)
    edges = seams + loops
    for trial in range(6):
        P = [CG.identity()] + [_rand_pose(rng, big=(trial % 2 == 1)) for _ in range(n - 1)]
        x = CG._pack(P)
        Pu = CG._unpack(x, n)
        want = []
        for i, e in enumerate(edges):
            r = CG.edge_residual(Pu[e["a"]], Pu[e["b"]], e)
            want.append(r if (robust_seams or i < len(seams)) else CG._cauchy_whitened(r))
        for k in sorted(priors):
            want.append(np.array([(np.log(Pu[k][0]) - priors[k]["log_s"]) / priors[k]["sigma"]]))
        want = np.concatenate(want)
        got = prob.fun(x)
        assert got.shape == want.shape
        assert np.abs(got - want).max() <= 1e-12 * max(1.0, np.abs(want).max()), np.abs(got - want).max()


def test_the_rotation_vector_keeps_the_records_three_branches():
    import torch
    rng = np.random.default_rng(5)
    ws = [np.zeros(3), np.array([1e-9, -2e-9, 5e-10]), np.array([0.3, -0.2, 0.9]),
          np.array([np.pi - 3e-7, 0.0, 0.0]), (np.pi - 2e-7) * np.array([0.6, 0.0, 0.8]), rng.normal(size=3)]
    R = np.stack([CG._rodrigues(w) for w in ws])
    got = CT._rotvec(torch.as_tensor(R, dtype=torch.float64)).numpy()
    want = np.stack([CG._rotvec(r) for r in R])
    assert np.abs(got - want).max() <= 1e-12
    back = CT._rodrigues(torch.as_tensor(np.stack(ws), dtype=torch.float64)).numpy()
    assert np.abs(back - R).max() <= 1e-15


def test_the_jacobian_is_scipys_own_finite_differences():
    from scipy.optimize._numdiff import approx_derivative
    rng = np.random.default_rng(11)
    true, seams, loops = _graph(rng, n=7, n_loops=10)
    n = len(true)

    def record(x):
        P = CG._unpack(x, n)
        r = [CG.edge_residual(P[e["a"]], P[e["b"]], e) for e in seams]
        r += [CG._cauchy_whitened(CG.edge_residual(P[e["a"]], P[e["b"]], e)) for e in loops]
        return np.concatenate(r)
    x = CG._pack([CG.identity()] + [_rand_pose(rng) for _ in range(n - 1)])
    want = approx_derivative(record, x, method="2-point", f0=record(x))
    got = CT.Problem(n, seams, loops, {}, False, DEV).jac(x)
    # a finite difference with a step of 1.5e-8 turns a rounding of 1e-16 in f into 1e-8 in the derivative
    assert np.abs(got - want).max() <= 2e-6 * max(1.0, np.abs(want).max()), np.abs(got - want).max()


@pytest.mark.parametrize("robust_seams", [False, True])
def test_the_solve_ends_where_the_record_ends(robust_seams):
    rng = np.random.default_rng(7)
    true, seams, loops = _graph(rng, n=10, n_loops=18)
    n = len(true)
    rec = CG.solve(n, seams, loops, {}, robust_seams=robust_seams)
    got = CT.solve(n, seams, loops, {}, robust_seams=robust_seams, device=DEV)
    assert rec["converged"] and got["converged"]
    assert abs(got["cost"] - rec["cost"]) <= 1e-9 * max(1.0, rec["cost"])
    ds = max(abs(np.log(a[0] / b[0])) for a, b in zip(got["poses"], rec["poses"]))
    dt = max(np.abs(np.asarray(a[2]) - np.asarray(b[2])).max() for a, b in zip(got["poses"], rec["poses"]))
    dR = max(np.abs(np.asarray(a[1]) - np.asarray(b[1])).max() for a, b in zip(got["poses"], rec["poses"]))
    # the record against ITSELF from a start 1e-13 away: how far its own last digits are determined
    start = [(p[0] * (1 + 1e-13), p[1], np.asarray(p[2]) + 1e-13) for p in CG.chain_poses(n, seams)]
    rec2 = CG.solve(n, seams, loops, {}, x0=start, robust_seams=robust_seams)
    own = max(np.abs(np.asarray(a[2]) - np.asarray(b[2])).max() for a, b in zip(rec2["poses"], rec["poses"]))
    assert dt <= max(1e-6, 10.0 * own) and ds <= max(1e-6, 10.0 * own) and dR <= max(1e-6, 10.0 * own), (ds, dt, dR, own)
    assert np.allclose(got["residuals"], rec["residuals"], rtol=0, atol=1e-4)


def test_the_twin_keeps_the_records_signature_and_refuses_what_it_is_not():
    rng = np.random.default_rng(1)
    true, seams, loops = _graph(rng, n=5, n_loops=4)
    one = CT.solve(1, [], [], {}, device=DEV)
    assert one == {"poses": one["poses"], "converged": True, "cost": 0.0, "nfev": 0, "residuals": []}
    out = CT.solve(len(true), seams, loops, {}, robust_seams=False, device=DEV)
    assert set(out) == {"poses", "converged", "cost", "nfev", "status", "residuals"}
    assert len(out["residuals"]) == len(seams) + len(loops)
    with pytest.raises(ValueError, match="unknown device"):
        CT.solve(len(true), seams, loops, {}, device="numpy")
