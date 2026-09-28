"""P2 — the continuous metric gauge along the walk (claude_stac.txt §4-F2):
a scale drift known along the walk is recovered within σ; when the windows are
noisier than the mono instrument the held-out chooses mono; relative rows move
only differences; the walk stays continuous under the correction; rows read
from DA3 window files carry the drift; a whole run writes gauge.json and the
v2 block of scale_diagnostics.json."""

import json
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision import gauge as G                                # noqa: E402
from precision.config import load_precision_config              # noqa: E402

GCFG = load_precision_config().gauge
WALK_M = 60.0


def _drift(c):
    """log scale the cloud still needs: +14 % spread linearly over the walk (pccr's
    measured first-half/second-half DA3 ratio)."""
    return math.log(1.14) * c / WALK_M - 0.06


def _rows(inst, n_groups, per_group, noise, seed, sigma=float("nan")):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        for k in range(per_group):
            c = (g + (k + 0.5) / per_group) * WALK_M / n_groups
            rows.append(G.Row(inst, _drift(c) + rng.normal(0, noise), sigma, g, c=c))
    return rows


def test_known_drift_is_recovered_within_sigma():
    rows = _rows("da3_mono", 12, 6, 0.01, 0)
    model = G.solve(rows, WALK_M, GCFG, log=lambda *a: None)
    knots, x = np.array(model["knots_m"]), np.array(model["x"])
    sig = model["sigma_by_instrument"]["da3_mono"]
    assert 0.005 < sig < 0.02                              # the scatter it was given
    for c in np.linspace(2, WALK_M - 2, 15):
        # a fit over ~70 rows is far tighter than one row's σ
        assert abs(float(G.hat(c, knots) @ x) - _drift(c)) < sig


def test_heldout_chooses_the_quieter_instrument():
    wins = _rows("da3_windows", 12, 1, 0.08, 1, sigma=0.01)
    noise = np.array([r.y - _drift(r.c) for r in wins])
    sample_sigma = G.MAD_TO_SIGMA * float(np.median(np.abs(noise - np.median(noise))))
    rows = wins + _rows("da3_mono", 12, 6, 0.01, 2)
    model = G.solve(rows, WALK_M, GCFG, log=lambda *a: None)
    assert model["applied_instrument"] == "da3_mono"
    assert model["heldout_by_instrument"]["da3_mono"] < model["heldout_by_instrument"]["da3_windows"]
    assert model["choice_verdict"]["against"] == "da3_windows"
    assert model["choice_verdict"]["improves"]
    assert model["choice_verdict"]["decision"] == "lowest_heldout_beyond_noise"
    # the windows' own σ (0.01) was optimistic: the scatter measured held-out replaced
    # it, and it is the scatter this sample actually has (not the nominal 0.08)
    sw = model["sigma_by_instrument"]["da3_windows"]
    assert sw > 2 * 0.01
    assert 0.5 * sample_sigma < sw < 2.0 * sample_sigma


def test_a_lower_heldout_within_the_noise_keeps_the_configured_default():
    """Two instruments of the same quality: the mono one happens to hold out 20 % lower
    on this sample, but the paired bootstrap cannot tell them apart — the default (the
    first of gauge.instruments) stays, and the verdict says why."""
    assert GCFG.instruments[0] == "da3_windows"
    rows = _rows("da3_windows", 12, 6, 0.01, 3) + _rows("da3_mono", 12, 6, 0.01, 10)
    model = G.solve(rows, WALK_M, GCFG, log=lambda *a: None)
    held = model["heldout_by_instrument"]
    assert held["da3_mono"] < 0.9 * held["da3_windows"]
    v = model["choice_verdict"]
    assert not v["improves"] and v["ci_low"] <= 0.0 <= v["ci_high"]
    assert v["candidate"] == "da3_mono" and v["against"] == "da3_windows"
    assert v["decision"] == "default_kept_within_noise"
    assert model["applied_instrument"] == "da3_windows"


def test_huber_iterates_to_its_tolerance_and_reports_a_capped_run():
    rng = np.random.default_rng(8)
    om = rng.uniform(1, 8, (40, 60))
    inst = om * 1.25 * np.exp(rng.standard_t(2, om.shape) * 0.05)
    g = G._log_gain(inst, om, None, GCFG)
    assert g.converged and not g.mad_zero and 1 < g.iterations < GCFG.huber_max_iter
    capped = G._log_gain(inst, om, None, replace(GCFG, huber_max_iter=1))
    assert not capped.converged and capped.iterations == 1
    assert G._log_gain(inst, om, None, GCFG) == g            # identical inputs, identical bits


def test_huber_with_zero_mad_returns_the_agreeing_majority_exactly():
    """More than half the pixels agree exactly: the MAD is 0, no scale is invented (the
    old stand-in 1.0 made every pixel an inlier and the outliers pulled the mean) — the
    limit of the estimate, the value the majority agrees on, comes back flagged."""
    om = np.full((20, 30), 2.0)
    inst = np.full((20, 30), 2.5)
    inst[:8] = 40.0                                         # 40 % of garbage
    g = G._log_gain(inst, om, None, GCFG)
    assert g.converged and g.mad_zero
    assert abs(g.log - math.log(1.25)) < 1e-15


def test_non_converged_gains_leave_no_row_and_are_reported(tmp_path):
    chainage = _write_session(tmp_path, _drift, _drift)
    wins, mono, rep = G.da3_rows(tmp_path, chainage, 0.0, replace(GCFG, huber_max_iter=1))
    assert wins == [] and mono == []
    assert rep and all(not r["row"] and r["huber"]["not_converged"]
                       and r["huber"]["mono_not_converged"] for r in rep)


def test_relative_rows_constrain_differences_only():
    knots = G.knots_for(WALK_M, GCFG.knot_walk_m)
    absr = [G.Row("da3_mono", 0.0, 0.01, g, c=c) for g, c in enumerate(np.linspace(0, WALK_M, 8))]
    rel = [G.Row("visit_drift", 0.1, 0.001, 0, c_ab=(10.0, 50.0))]
    x = G.fit(absr + rel, knots, 1.0)
    diff = float((G.hat(50.0, knots) - G.hat(10.0, knots)) @ x)
    assert diff > 0.05                                     # the closure pulled the difference


def test_continuous_transforms_keep_the_walk_continuous():
    c = np.cumsum(np.ones((20, 3)) * [0.1, 0.0, 0.05], axis=0)
    k, t = G.continuous_transforms(c, np.ones(20))
    assert np.allclose(t, 0) and np.allclose(k, 1)
    k, t = G.continuous_transforms(c, np.full(20, 1.1))
    new = c + t
    assert np.allclose(new[0], c[0])                        # the first camera stays
    assert np.allclose(np.diff(new, axis=0), 1.1 * np.diff(c, axis=0))
    s = np.linspace(1.0, 1.2, 20)
    k, t = G.continuous_transforms(c, s)
    steps = np.linalg.norm(np.diff(c + t, axis=0), axis=1)
    assert np.allclose(steps, 0.5 * (s[1:] + s[:-1]) * np.linalg.norm(np.diff(c, axis=0), axis=1))


def test_huber_gain_resists_outliers():
    rng = np.random.default_rng(3)
    om = rng.uniform(1, 8, (40, 60)).astype(np.float32)
    inst = om * 1.25 * np.exp(rng.normal(0, 0.01, om.shape)).astype(np.float32)
    inst[:4] = 50.0                                         # a band of garbage
    g = G._log_gain(inst, om, np.ones_like(om), GCFG)
    assert g.converged and abs(g.log - math.log(1.25)) < 0.01


def _write_session(tmp, drift_windows, drift_mono, n=96, log_s0=0.0):
    out = tmp / "output"
    (out / "omega_run" / "results_output").mkdir(parents=True)
    wdir = out / "da3_windows"
    wdir.mkdir()
    rng = np.random.default_rng(4)
    om = {f: rng.uniform(1, 6, (12, 16)).astype(np.float32) for f in range(n)}
    for f, d in om.items():
        np.savez(out / "omega_run" / "results_output" / f"frame_{f}.npz", depth=d)
    chainage = {f: f * WALK_M / (n - 1) for f in range(n)}
    from intake.walk import plan_windows
    for i, (a, b) in enumerate(plan_windows(n, 32, 0.5)):
        fr = list(range(a, b))
        dw = np.array([om[f] * math.exp(drift_windows(chainage[f]) + log_s0) for f in fr])
        dm = np.array([om[f] * math.exp(drift_mono(chainage[f]) + log_s0) for f in fr])
        np.savez(wdir / f"window_{i:04d}.npz", frames=np.array(fr), depth=dw.astype(np.float32),
                 depth_mono=dm.astype(np.float32), conf=np.ones_like(dw, np.float32),
                 extrinsics=np.tile(np.eye(4), (len(fr), 1, 1)),
                 intrinsics=np.tile(np.eye(3), (len(fr), 1, 1)),
                 scale_factor=np.float64(1), is_metric=np.int64(1))
    return chainage


def test_window_and_mono_rows_carry_the_drift(tmp_path):
    chainage = _write_session(tmp_path, _drift, _drift, log_s0=math.log(1.3))
    wins, mono, rep = G.da3_rows(tmp_path, chainage, math.log(1.3), GCFG)
    assert len(wins) == len(rep) >= 5 and len(mono) == len(chainage)
    assert all(r["row"] and not r["huber"]["not_converged"] for r in rep)
    for r in mono:
        assert abs(r.y - _drift(r.c)) < 1e-4               # the global scale is subtracted
    for r in wins:
        assert abs(r.y - _drift(r.c)) < 0.02               # a window spans some drift
        assert r.sigma >= 0
    assert all(abs(w["log_s_near_band"] - w["log_s"]) < 0.05 for w in rep)


def test_run_measures_and_reports_without_applying(tmp_path, monkeypatch):
    chainage = _write_session(tmp_path, _drift, _drift)
    frames = sorted(chainage)
    walk = {"walk_length_m": WALK_M,
            "chainage": [{"frame": f, "chainage_m": chainage[f]} for f in frames],
            "windows": []}
    from intake.walk import plan_windows
    walk["windows"] = [{"frames": [a, b - 1]} for a, b in plan_windows(len(frames), 32, 0.5)]
    (tmp_path / "intake").mkdir()
    (tmp_path / "intake" / "walk.json").write_text(json.dumps(walk))
    poses = np.tile(np.eye(4), (len(frames), 1, 1))
    poses[:, 0, 3] = [chainage[f] for f in frames]
    import correction.session as CS
    monkeypatch.setattr(CS, "load_session",
                        lambda out: SimpleNamespace(frames=frames, poses=poses))
    cfg = SimpleNamespace(**{**GCFG.__dict__, "instruments": ("da3_windows", "da3_mono")})
    doc = G.run_gauge(tmp_path, cfg, apply=False, log=lambda *a: None)
    assert doc["applied"] is False and doc["applied_instrument"] in ("da3_windows", "da3_mono")
    assert abs(math.log(doc["s_keyframes"]["max"]) - _drift(WALK_M)) < 0.01
    assert abs(math.log(doc["s_keyframes"]["min"]) - _drift(0.0)) < 0.01
    diag = json.loads((tmp_path / "output" / "scale_diagnostics.json").read_text())
    assert diag["v2"]["applied_instrument"] == doc["applied_instrument"]
    assert not G.gauge_applied(tmp_path / "output")


def test_gauge_refuses_a_corrected_epoch(tmp_path, monkeypatch):
    import correction.epoch as E
    monkeypatch.setattr(E, "current_epoch", lambda out: 2)
    with pytest.raises(G.GaugeError, match="epoch 2"):
        G.run_gauge(tmp_path, GCFG, apply=False, log=lambda *a: None)
