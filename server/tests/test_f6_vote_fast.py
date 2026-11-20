"""F6's multi-view vote with the session's maps resident on the device and the witnesses in batches (USER
2026-11-20: "todo lo que no cambie el resultado debe programarse mas rapido") — f6_torch.two_sided_vote_fast
against f6_torch.two_sided_vote, the record: EVERY entry of the result the same bit for bit, in every mode the
vote has, and the cloud step's own vote (depth_on_f5.edge_keeping_vote) the same maps and totals."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from precision import depth_on_f5 as D5  # noqa: E402
from precision import f6_torch as FT  # noqa: E402

H, W = 40, 56
K = np.array([[48.0, 0, W / 2 - 0.5], [0, 48.0, H / 2 - 0.5], [0, 0, 1]])


def _same(a: torch.Tensor, b: torch.Tensor, what=""):
    assert a.dtype == b.dtype and a.shape == b.shape, (what, a.dtype, b.dtype, a.shape, b.shape)
    if a.dtype.is_floating_point:
        assert torch.equal(torch.isnan(a), torch.isnan(b)), what
        assert torch.equal(torch.nan_to_num(a, nan=-7.0), torch.nan_to_num(b, nan=-7.0)), what
    else:
        assert torch.equal(a, b), what


def _same_vote(r: dict, f: dict):
    assert sorted(r) == sorted(f)
    for k in r:
        _same(r[k], f[k], k)


def _scene(n=11, seed=3, with_holes=True):
    """A room corner (a floor, a back wall, a box in front of it) seen by ``n`` cameras walking along x and
    turning a little: each keyframe's depth is the nearest surface along every pixel's ray (+ noise, a few
    flyers, invalid pixels), so neighbours agree, occlude and see through one another."""
    rng = np.random.default_rng(seed)
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1)
    order, dep, judge, c2w = [], {}, {}, {}
    for i in range(n):
        f = 10 + 7 * i
        th = 0.06 * (i - n / 2)
        R = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
        C = np.array([0.35 * i, 0.02 * np.sin(i), 0.05 * i])
        d = rays @ R.T
        with np.errstate(divide="ignore", invalid="ignore"):
            t_wall = (6.0 - C[2]) / d[..., 2]                                  # the wall z = 6
            t_floor = (1.2 - C[1]) / d[..., 1]                                 # the floor y = 1.2
            t_box = (4.0 - C[2]) / d[..., 2]                                   # a box face z = 4, x in [1, 2.2], y < 1.2
        Xb = C + d * t_box[..., None]
        box = (Xb[..., 0] > 1.0) & (Xb[..., 0] < 2.2) & (Xb[..., 1] > 0.2)
        t = np.where((t_floor > 0) & (t_floor < t_wall), t_floor, t_wall)
        t = np.where(box & (t_box > 0) & (t_box < t), t_box, t)
        z = t * (1 + rng.normal(size=t.shape) * 0.002)                        # (rays have z = 1: t is the depth)
        fly = rng.random(t.shape) < 0.01
        z[fly] *= rng.uniform(0.5, 0.9, int(fly.sum()))
        ok = np.isfinite(z) & (z > 0.2)
        if with_holes:
            ok &= rng.random(t.shape) > 0.05
        c = np.eye(4); c[:3, :3] = R; c[:3, 3] = C
        order.append(f); dep[f] = np.where(ok, z, 0).astype(np.float32); c2w[f] = c
        judge[f] = ok & (rng.random(t.shape) > 0.1)
    w2c = {f: np.linalg.inv(c2w[f]) for f in order}
    return order, dep, judge, c2w, w2c


FUSION = {"max_reproj_px": 1.5, "max_reproj_deg": 1.0, "max_depth_rel": 0.02, "tri_min_deg": 0.5, "tri_max_deg": 60.0,
          "mode": "keep", "witness_max_m": 15.0, "repair": False, "occlusion_veto": True, "majority": True,
          "in_place": True}
MODES = {
    "tau, every view counts": dict(),
    "tau, a vote counts when it tests the depth": dict(min_parallax_deg=2.0),
    "colmap keep + in place (the recipe)": dict(min_parallax_deg=2.0, fusion=FUSION),
    "colmap keep, no range, no repair": dict(min_parallax_deg=2.0, fusion=dict(FUSION, witness_max_m=0.0, in_place=False)),
    "colmap + repair, a short range": dict(min_parallax_deg=2.0, fusion=dict(FUSION, witness_max_m=5.5, repair=True)),
    "adjust": dict(min_parallax_deg=2.0, fusion=dict(FUSION, mode="adjust")),
    "adjust without a range": dict(fusion=dict(FUSION, mode="adjust", witness_max_m=0.0)),
}


@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize("batch", [1, 3, 32])
def test_the_fast_vote_is_the_records_bit_for_bit(mode, batch):
    order, dep, judge, c2w, w2c = _scene()
    res = FT.Resident(dep, judge)
    kw = MODES[mode]
    for i in (0, 4, len(order) - 1):
        cand = (dep[order[i]] > 0) & (np.arange(H * W).reshape(H, W) % 11 != 0)
        extra = [order[j] for j in (len(order) - 1 - i, (i + 5) % len(order)) if abs(j - i) > 2]
        rec = FT.two_sided_vote(i, order, dep, judge, cand, K, c2w, w2c, (-2, -1, 1, 2), 0.02, extra=extra, **kw)
        fast = FT.two_sided_vote_fast(i, order, res, cand, K, c2w, w2c, (-2, -1, 1, 2), 0.02, extra=extra,
                                      batch=batch, **kw)
        _same_vote(rec, fast)
        assert int(rec["agree"].sum()) > 0 and rec["splats"].shape[0] == len(extra) + sum(
            0 <= i + d < len(order) for d in (-2, -1, 1, 2))


def test_a_keyframe_without_witnesses_and_one_without_candidates():
    order, dep, judge, c2w, w2c = _scene(n=3)
    res = FT.Resident(dep, judge)
    for kw in (dict(), dict(min_parallax_deg=2.0, fusion=FUSION)):
        rec = FT.two_sided_vote(1, order, dep, judge, np.ones((H, W), bool), K, c2w, w2c, (-9, 9), 0.02, **kw)
        fast = FT.two_sided_vote_fast(1, order, res, np.ones((H, W), bool), K, c2w, w2c, (-9, 9), 0.02, **kw)
        _same_vote(rec, fast)
        assert rec["splats"].shape[0] == 0
        none = np.zeros((H, W), bool)
        _same_vote(FT.two_sided_vote(1, order, dep, judge, none, K, c2w, w2c, (-1, 1), 0.02, **kw),
                   FT.two_sided_vote_fast(1, order, res, none, K, c2w, w2c, (-1, 1), 0.02, **kw))


def test_the_one_step_z_buffer_is_the_two_sorts():
    rng = np.random.default_rng(5)
    for n_pix, n in ((1, 0), (7, 3), (500, 20000), (4096, 100), (64, 5000)):
        p = torch.as_tensor(rng.integers(0, n_pix, n), dtype=torch.int64)
        zp = torch.as_tensor(rng.integers(1, 40, n) / 4.0 + (rng.random(n) < 0.5) * rng.random(n), dtype=torch.float64)
        a, b = FT.zbuffer_sorted(p, zp, n_pix), FT.zbuffer_min(p, zp, n_pix)
        assert torch.equal(a, b) and a.shape == (n_pix,) and a.dtype == torch.float64
        if n == 0:
            assert bool(torch.isinf(a).all())
    assert FT.zbuffer_method() in ("min", "sorted")


def test_the_z_buffer_is_allowed_under_the_strict_deterministic_mode():
    from repro import deterministic_torch
    with deterministic_torch(0):
        FT._ZBUFFER.clear()
        assert FT.zbuffer_method() == "min", "on this device the reduction is available: the fast vote uses it"
        order, dep, judge, c2w, w2c = _scene(n=5)
        res = FT.Resident(dep, judge)
        cand = dep[order[2]] > 0
        _same_vote(FT.two_sided_vote(2, order, dep, judge, cand, K, c2w, w2c, (-2, -1, 1, 2), 0.02, min_parallax_deg=2.0,
                                     fusion=FUSION),
                   FT.two_sided_vote_fast(2, order, res, cand, K, c2w, w2c, (-2, -1, 1, 2), 0.02, min_parallax_deg=2.0,
                                          fusion=FUSION))


def test_with_the_sorted_z_buffer_the_fast_vote_is_still_the_record(monkeypatch):
    monkeypatch.setitem(FT._ZBUFFER, FT.device(), "sorted")
    order, dep, judge, c2w, w2c = _scene(n=6)
    res = FT.Resident(dep, judge)
    cand = dep[order[3]] > 0
    _same_vote(FT.two_sided_vote(3, order, dep, judge, cand, K, c2w, w2c, (-2, -1, 1, 2), 0.02),
               FT.two_sided_vote_fast(3, order, res, cand, K, c2w, w2c, (-2, -1, 1, 2), 0.02, batch=2))


def test_a_witnesss_surface_is_kept_within_its_budget_and_recomputed_the_same():
    order, dep, judge, c2w, w2c = _scene(n=6)
    big, small = FT.Resident(dep, judge), FT.Resident(dep, judge, surface_bytes=1)
    cand = dep[order[2]] > 0
    for i in (2, 3, 2):
        a = FT.two_sided_vote_fast(i, order, big, cand, K, c2w, w2c, (-2, -1, 1, 2), 0.02)
        b = FT.two_sided_vote_fast(i, order, small, cand, K, c2w, w2c, (-2, -1, 1, 2), 0.02)
        _same_vote(a, b)
    assert len(small._surf) == 1 and len(big._surf) > 3


@pytest.mark.parametrize("mode", ["tau", "colmap", "adjust"])
def test_the_cloud_steps_vote_gives_the_same_maps_and_totals(mode):
    order, dep, judge, c2w, _ = _scene(n=9, seed=11)
    valid = {f: dep[f] > 0 for f in order}
    extra = {order[1]: [order[7]], order[6]: [order[0], order[2]]}

    def run(vote_batch):
        d = {f: dep[f].copy() for f in order}
        kw = dict(log=lambda *a: None, extra=extra, vote_batch=vote_batch)
        if mode == "tau":
            kw.update(min_parallax_deg=2.0, min_support=2)
        else:
            kw.update(min_parallax_deg=2.0, min_support=2,
                      fusion=dict(FUSION, mode="adjust" if mode == "adjust" else "keep", max_depth_rel=0.0))
        return D5.edge_keeping_vote(order, d, {f: valid[f].copy() for f in order}, {f: judge[f].copy() for f in order},
                                    K, dict(c2w), None, (-2, -1, 1, 2), 90.0, 2, mode == "tau", **kw)
    final_r, tau_r, tot_r = run(0)
    for vb in (1, 4):
        final_f, tau_f, tot_f = run(vb)
        assert tau_r == tau_f and tot_r == tot_f
        for f in order:
            assert np.array_equal(final_r[f][0], final_f[f][0]) and final_r[f][0].dtype == final_f[f][0].dtype
            assert np.array_equal(final_r[f][1], final_f[f][1]) and final_r[f][1].dtype == final_f[f][1].dtype
    assert tot_r["out"] > 0


def test_every_scenario_of_the_votes_own_tests_gives_the_records_bits(monkeypatch):
    """tests/test_depth_on_f5.py and tests/test_turn_witnesses.py walk the vote through its rules (the flyer, the
    rescue, the adjust, the repair, the occlusion veto, the range, the majority, in place, the turn witnesses):
    here every call they make of two_sided_vote is ALSO made the fast way and held to it."""
    import importlib.util
    seen = []
    record = FT.two_sided_vote

    def both(i, order, depth, judge, cand, K_, c2w, w2c, neighbors, tau, **kw):
        r = record(i, order, depth, judge, cand, K_, c2w, w2c, neighbors, tau, **kw)
        res = FT.Resident(depth, judge)
        for batch in (1, 5):
            _same_vote(r, FT.two_sided_vote_fast(i, order, res, cand, K_, c2w, w2c, neighbors, tau, batch=batch, **kw))
        seen.append(sorted(k for k in kw if kw[k] is not None))
        return r
    monkeypatch.setattr(FT, "two_sided_vote", both)
    n_tests = 0
    for name in ("test_depth_on_f5.py", "test_turn_witnesses.py"):
        spec = importlib.util.spec_from_file_location("votes_" + name[:-3], Path(__file__).with_name(name))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for fn in sorted(n for n in dir(mod) if n.startswith("test_")):
            f = getattr(mod, fn)
            if f.__code__.co_argcount == 0:
                before = len(seen)
                f()
                n_tests += len(seen) > before
    assert len(seen) > 40 and n_tests >= 8, (len(seen), n_tests)


def test_the_key_is_typed():
    import yaml
    from precision.config import load_precision_config
    raw = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text())
    assert load_precision_config(raw).cloud.vote_batch == 0, "declared, at the record's value"
    for bad in (-1, True, 2.5, "8"):
        raw["reconstruction"]["precision"]["cloud"]["vote_batch"] = bad
        with pytest.raises(ValueError, match="vote_batch"):
            load_precision_config(raw)
    raw["reconstruction"]["precision"]["cloud"]["vote_batch"] = 32
    assert load_precision_config(raw).cloud.vote_batch == 32
