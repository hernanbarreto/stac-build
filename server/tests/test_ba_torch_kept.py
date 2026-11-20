"""F5's solver in chunks keeps what is the same at every LM trial — each chunk's pair plan, and its Jacobians between
a linearisation and its trials — within two memory bounds (USER 2026-11-20: "todo lo que no cambie el resultado debe
programarse mas rapido"): refine.torch_ba.keep_pair_plans_gb / keep_jacobians_gb. Every step, every solve and every
rung THE SAME BITS as the record (0: everything made again at every trial)."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from precision import ba_torch as BA  # noqa: E402

_spec = importlib.util.spec_from_file_location("ba_scenes", Path(__file__).with_name("test_ba_torch.py"))
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)


def _tb(plans=0.0, jac=0.0):
    return SimpleNamespace(**vars(T.TB), keep_pair_plans_gb=plans, keep_jacobians_gb=jac)


def _problem(tb, cam_free=(0, 1, 2, 3), blocks=1, chunk_obs=97):
    w2c, cam_idx, pt_idx, uv, X0 = T._flat_problem()
    block_of = (np.arange(len(w2c)) * blocks // len(w2c)).astype(int)
    kw = dict(device="cpu", sigmas=np.full(len(w2c), 0.05),
              prior_centres=-np.einsum("nji,nj->ni", w2c[:, :3, :3], w2c[:, :3, 3]) + 0.01, torch_ba=tb)
    return BA._ChunkedProblem(w2c, [list(T.CAM)] * blocks, block_of, cam_idx, pt_idx, uv, X0, 1.0, cam_free,
                              chunk_obs=chunk_obs, **kw)


def _trials(p):
    """A linearisation, trials at several dampings (a repeated one: a rejected trial changes only λ), an accepted
    step, and the trials of the next linearisation."""
    out = []
    for round_ in range(2):
        for lam in (1e-4, 1e-4, 3.0, 1e-1):
            (dc, dp), cost, pred, why = p.step(lam)
            out.append((dc.clone(), dp.clone(), cost, pred, why))
        p.R, p.t, p.cam, p.X = p.apply(dc, dp)
        p.invalidate()
        out.append((p.total_cost(),))
    return out


def _same(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert len(x) == len(y)
        for u, v in zip(x, y):
            assert torch.equal(u, v) if isinstance(u, torch.Tensor) else u == v


GB = float(1 << 30)


@pytest.mark.parametrize("cam_free,blocks", [((), 1), ((0, 1, 2, 3, 4, 5, 6, 7), 1), ((0, 1), 3)])
def test_every_trial_is_the_records_whatever_is_kept(cam_free, blocks):
    record = _problem(_tb(), cam_free, blocks)
    assert len(record.chunks) > 5
    ref = _trials(record)
    assert not record._plans and not record._jac and all(sub._pairs is None for _, _, sub in record.chunks)
    for plans, jac in ((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)):
        p = _problem(_tb(plans, jac), cam_free, blocks)
        _same(ref, _trials(p))
        p.linearize()
        assert (len(p._plans) == len(p.chunks)) == (plans > 0) and (len(p._jac) == len(p.chunks)) == (jac > 0)


def test_what_does_not_fit_its_bound_is_made_again_and_nothing_changes():
    record = _problem(_tb())
    ref = _trials(record)
    full = _problem(_tb(1.0, 1.0))
    _trials(full)
    full.linearize()
    plan_bytes, jac_bytes = full._plan_bytes, full._jac_bytes
    assert plan_bytes > 0 and jac_bytes > 0
    part = _problem(_tb(0.45 * plan_bytes / GB, 0.45 * jac_bytes / GB))
    _same(ref, _trials(part))
    part.linearize()
    part.step(1e-4)
    assert 0 < len(part._plans) < len(part.chunks) and 0 < len(part._jac) < len(part.chunks)
    assert part._plan_bytes <= 0.45 * plan_bytes and part._jac_bytes <= 0.45 * jac_bytes
    kept = [sub._pairs is not None for _, _, sub in part.chunks]
    assert kept == [i in part._plans for i in range(len(part.chunks))]


def test_a_whole_solve_and_a_whole_rung_are_the_records(monkeypatch):
    a, b = _problem(_tb()), _problem(_tb(1.0, 1.0))
    sa, sb = BA.solve(a, 60), BA.solve(b, 60)
    for k in ("initial_cost", "final_cost", "num_successful_steps", "num_unsuccessful_steps", "termination_type"):
        assert sa[k] == sb[k], k
    assert sa["num_successful_steps"] > 2
    for k in ("R", "t", "cam", "X"):
        assert torch.equal(getattr(a, k), getattr(b, k)), k
    # the rung interface (precision.refine's contract), in chunks
    w2c, pts, groups, X = T._scene()
    w0, X0 = T._perturb(w2c, X, np.random.default_rng(7))
    kw = dict(sigmas=np.full(len(w2c), 0.5), prior_w2c=w0)
    monkeypatch.setattr(BA, "CHUNKED_ABOVE_OBS", 0)
    monkeypatch.setattr(BA, "CHUNK_OBS", 211)
    cfg0 = SimpleNamespace(**{**vars(T.CFG), "torch_ba": _tb()})
    cfg1 = SimpleNamespace(**{**vars(T.CFG), "torch_ba": _tb(1.0, 1.0)})
    r0 = BA.run_rung_torch("R1", w0, T.CAM, T.WH, groups, X0, cfg=cfg0, device="cpu", **kw)
    r1 = BA.run_rung_torch("R1", w0, T.CAM, T.WH, groups, X0, cfg=cfg1, device="cpu", **kw)
    assert r0.termination == r1.termination == "CONVERGENCE" and r0.final_cost == r1.final_cost
    assert np.array_equal(r0.w2c, r1.w2c) and all(np.array_equal(r0.X[t], r1.X[t]) for t in r0.X)
    assert np.array_equal(np.asarray(r0.params_by_block), np.asarray(r1.params_by_block)) and r0.fit_rms_px == r1.fit_rms_px


def test_the_two_bounds_are_typed():
    import yaml
    from precision.config import load_precision_config
    raw = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text())
    tb = raw["reconstruction"]["precision"]["refine"]["torch_ba"]
    assert "keep_pair_plans_gb" in tb and "keep_jacobians_gb" in tb, "declared, not left to a default"
    c = load_precision_config(raw).refine.torch_ba
    assert c.keep_pair_plans_gb == 0.0 and c.keep_jacobians_gb == 0.0, "the record's values"
    for key in ("keep_pair_plans_gb", "keep_jacobians_gb"):
        for bad in (-1, True, "8", float("inf")):
            tb[key] = bad
            with pytest.raises(ValueError, match=key):
                load_precision_config(raw)
        tb[key] = 12
    c = load_precision_config(raw).refine.torch_ba
    assert c.keep_pair_plans_gb == 12.0 and c.keep_jacobians_gb == 12.0
