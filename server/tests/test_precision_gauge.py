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
    import precision.poses_epoch as PE          # the gauge reads poses only — no cloud
    monkeypatch.setattr(PE, "load_poses", lambda out: (frames, poses))
    cfg = SimpleNamespace(**{**GCFG.__dict__, "instruments": ("da3_windows", "da3_mono")})
    doc = G.run_gauge(tmp_path, cfg, apply=False, log=lambda *a: None)
    assert doc["applied"] is False and doc["applied_instrument"] in ("da3_windows", "da3_mono")
    assert abs(math.log(doc["s_keyframes"]["max"]) - _drift(WALK_M)) < 0.01
    assert abs(math.log(doc["s_keyframes"]["min"]) - _drift(0.0)) < 0.01
    diag = json.loads((tmp_path / "output" / "scale_diagnostics.json").read_text())
    assert diag["v2"]["applied_instrument"] == doc["applied_instrument"]
    # the scale source and its capture files (plan point 73): no VIO / Stray here → null
    assert doc["scale_source"] == doc["applied_instrument"]
    assert doc["scale_inputs"] == {"vio": None, "vio_sha256": None, "stray_depth_dir": None}
    assert diag["v2"]["scale_source"] == doc["scale_source"]
    assert diag["v2"]["scale_inputs"] == doc["scale_inputs"]
    assert not G.gauge_applied(tmp_path / "output")


def test_gauge_refuses_a_corrected_epoch(tmp_path, monkeypatch):
    import correction.epoch as E
    monkeypatch.setattr(E, "current_epoch", lambda out: 2)
    with pytest.raises(G.GaugeError, match="epoch 2"):
        G.run_gauge(tmp_path, GCFG, apply=False, log=lambda *a: None)


def test_calibrated_confidence_sets_the_da3_pixel_weights(tmp_path):
    """claude_stac.txt §4-F6: with the session's confidence calibration on disk, a DA3
    pixel weighs 1/q² (q its calibrated |error| quantile) — low-confidence pixels that
    carry a +30 % bias stop pulling the window's gain. Taken only when it was measured on
    THIS reconstruction at epoch 0 (plan point 33)."""
    from precision import confidence as CAL
    chainage = _write_session(tmp_path, lambda c: 0.0, lambda c: 0.0)
    (tmp_path / "output" / "camera_frames.txt").write_text(
        "\n".join(str(f) for f in sorted(chainage)) + "\n")
    wdir = tmp_path / "output" / "da3_windows"
    for p in sorted(wdir.glob("window_*.npz")):
        with np.load(p) as z:
            d = {k: z[k] for k in z.files}
        conf = np.full(d["depth"].shape, 0.9, np.float32)
        conf[..., ::2] = 0.1                               # every other column: unreliable
        d["depth"] = np.where(conf < 0.5, d["depth"] * 1.3, d["depth"]).astype(np.float32)
        d["conf"] = conf
        np.savez(p, **d)
    raw_wins, _m, raw_rep = G.da3_rows(tmp_path, chainage, 0.0, GCFG)
    rng = np.random.default_rng(0)
    c = rng.uniform(0.0, 1.0, 20000)
    err = np.where(c < 0.5, 0.3, 0.0) + rng.normal(0, 0.005, c.size)
    table = CAL.calibrate(err, c, rng.uniform(1, 6, c.size), conf_bins=2, dist_bins=1,
                          quantile=0.95, min_bin_samples=50)
    CAL.write_calibration(tmp_path / "output", "tier0", {"da3": table},
                          {"geometry_epoch": 0, "camera_epoch": 0}, {})
    cal_wins, _m, cal_rep = G.da3_rows(tmp_path, chainage, 0.0, GCFG)
    assert all("calibrated" in r["pixel_weights"] for r in cal_rep if r["row"])
    assert all(r["pixel_weights"] == "da3 confidence" for r in raw_rep if r["row"])
    raw_bias = np.median([abs(w.y) for w in raw_wins])
    cal_bias = np.median([abs(w.y) for w in cal_wins])
    assert cal_bias < 0.01 < raw_bias



def _calibrated_session(tmp_path):
    from precision import confidence as CAL
    chainage = _write_session(tmp_path, lambda c: 0.0, lambda c: 0.0)
    out = tmp_path / "output"
    (out / "camera_frames.txt").write_text("\n".join(str(f) for f in sorted(chainage)) + "\n")
    rng = np.random.default_rng(0)
    c = rng.uniform(0.0, 1.0, 2000)
    table = CAL.calibrate(np.abs(rng.normal(0, 0.01, c.size)), c, rng.uniform(1, 6, c.size),
                          conf_bins=2, dist_bins=1, quantile=0.95, min_bin_samples=50)
    return out, table


def test_a_calibration_of_a_later_epoch_or_another_reconstruction_is_not_f2s_input(tmp_path):
    """Plan point 33: F2 runs on epoch 0 — a calibration its own chain's F6 wrote later (epoch >= 1)
    is never its input, nor one measured on another reconstruction (epoch numbers restart with
    every reconstruction); the report names why."""
    from precision import confidence as CAL
    out, table = _calibrated_session(tmp_path)
    CAL.write_calibration(out, "tier0", {"da3": table}, {"geometry_epoch": 2, "camera_epoch": 1}, {})
    cal, rep = G.calibration_for_gauge(out)
    assert cal is None and rep["present"] and not rep["taken"] and "epoch" in rep["reason"]
    CAL.write_calibration(out, "tier0", {"da3": table}, {"geometry_epoch": 0, "camera_epoch": 0}, {})
    cal, rep = G.calibration_for_gauge(out)
    assert cal is not None and rep["taken"]
    # the same file on another reconstruction (one Omega record changed) is refused
    rec = sorted((out / "omega_run" / "results_output").glob("frame_*.npz"))[0]
    with np.load(rec) as z:
        d = {k: z[k] for k in z.files}
    d["depth"] = d["depth"] * 1.0001
    np.savez(rec, **d)
    cal, rep = G.calibration_for_gauge(out)
    assert cal is None and "another reconstruction" in rep["reason"]


def test_scale_rows_and_known_dimensions_need_this_reconstructions_id(tmp_path):
    """Plan point 34: an epoch NUMBER does not name a reconstruction — the rows enter only with
    this reconstruction's id, verified on read; unstamped files are refused and reported."""
    from correction.epoch import RECONSTRUCTION_ID_KEY, reconstruction_id
    out, _t = _calibrated_session(tmp_path)
    frames = list(range(96))
    chainage = {f: f * WALK_M / 95 for f in frames}
    rows = {"version": 1, "measured_on_epoch": 0,
            "rows": [{"i": 5, "j": 80, "k_b": 1.1, "residual_m": 0.01, "D_b_m": 3.0}]}
    (out / "scale_loop_rows.json").write_text(json.dumps(rows))
    dims = {"dimensions": [{"role": "gauge", "frames": [10, 11], "measured_m": 1.0, "true_m": 1.05,
                            "sigma_m": 0.001, "id": "door"}]}
    (out / "known_dimensions.json").write_text(json.dumps(dims))
    rep = {}
    assert G.visit_rows(out, frames, chainage, lambda c: 0, 0, report=rep) == []
    assert G.known_dim_rows(out, chainage, lambda c: 0, report=rep) == []
    assert all(not r["taken"] and "reconstruction_id" in r["reason"] for r in rep.values())
    rid = reconstruction_id(out)
    for name, doc in (("scale_loop_rows.json", rows), ("known_dimensions.json", dims)):
        (out / name).write_text(json.dumps({**doc, RECONSTRUCTION_ID_KEY: rid}))
    rep = {}
    assert len(G.visit_rows(out, frames, chainage, lambda c: 0, 0, report=rep)) == 1
    assert len(G.known_dim_rows(out, chainage, lambda c: 0, report=rep)) == 1
    assert all(r["taken"] for r in rep.values())
    (out / "scale_loop_rows.json").write_text(json.dumps({**rows, RECONSTRUCTION_ID_KEY: "0" * 64}))
    rep = {}
    assert G.visit_rows(out, frames, chainage, lambda c: 0, 0, report=rep) == []
    assert "another reconstruction" in rep["scale_loop_rows.json"]["reason"]


def test_knots_sit_at_fixed_positions_and_the_fit_moves_continuously_across_a_multiple():
    """Plan point 39: the knots are multiples of knot_walk_m whatever the walk's length — a walk
    crossing a multiple gains one free knot past its data instead of re-spacing every knot, so
    the fitted curve does not jump."""
    step = GCFG.knot_walk_m
    a, b = G.knots_for(10 * step - 1e-6, step), G.knots_for(10 * step + 1e-6, step)
    assert np.array_equal(a, step * np.arange(11)) and np.array_equal(b, step * np.arange(12))
    assert np.array_equal(b[:11], a)

    def curve(c_max):
        rng = np.random.default_rng(5)
        rows = []
        for g in range(8):
            for k in range(6):
                c = (g + (k + 0.5) / 6) * c_max / 8
                rows.append(G.Row("da3_mono", _drift(c) + rng.normal(0, 0.01), 0.01, g, c=c))
        kn = G.knots_for(c_max, step)
        x = G.fit(rows, kn, 1.0)
        return np.array([float(G.hat(c, kn) @ x) for c in np.linspace(0, 10 * step - 1e-3, 50)])
    below, above = curve(10 * step - 1e-6), curve(10 * step + 1e-6)
    assert np.max(np.abs(below - above)) < 1e-5


def test_lambda_is_the_smoothest_within_the_error_of_the_minimum():
    """Plan point 29: not a strict argmin — the smoothest grid value whose held-out RMS is within
    error_factor x the MEASURED error of the minimum (the bootstrap spread of its mean over the
    held-out windows, fixed seed); the whole curve is returned and lands in gauge.json."""
    rows = _rows("da3_mono", 12, 6, 0.01, 0, sigma=0.01)
    knots = G.knots_for(WALK_M, GCFG.knot_walk_m)
    ch = G.choose_lambda(rows, knots, GCFG.smooth_grid, 2.0)
    assert len(ch.curve) == len(GCFG.smooth_grid)
    best = min(ch.curve, key=lambda c: c["heldout_mean"])
    assert np.isfinite(best["heldout_error"]) and best["heldout_error"] > 0
    bar = best["heldout_mean"] + 2.0 * best["heldout_error"]
    within = [c["lambda"] for c in ch.curve if c["heldout_mean"] <= bar]
    assert ch.lam == max(within) and ch.rule["bar"] == pytest.approx(bar)
    assert ch.rule["bootstrap"] == {"n": G.BOOT_N, "seed": G.BOOT_SEED}
    # the error IS the bootstrap spread of the mean over the windows, from a fixed stream
    k = list(GCFG.smooth_grid).index(best["lambda"])
    per = np.asarray(list(G.loo(rows, knots, best["lambda"]).values()))
    assert best["heldout_error"] == G._bootstrap_error_of_mean(per, [k])
    assert G.choose_lambda(rows, knots, GCFG.smooth_grid, 2.0) == ch            # bit-identical
    # with no tolerance it is the plain argmin; the tolerance only ever moves λ smoother
    strict = G.choose_lambda(rows, knots, GCFG.smooth_grid, 1e-12)
    assert strict.lam == best["lambda"] and ch.lam >= strict.lam
    model = G.solve(rows, WALK_M, GCFG, log=lambda *a: None, error_factor=2.0)
    assert model["lambda_curve_final"]["curve"] and model["lambda_curve_by_instrument"]["da3_mono"]
    assert all("heldout_error" in c for c in model["lambda_curve_final"]["curve"])


def test_the_instrument_switch_also_needs_the_judges_and_an_improvement_beyond_twice_the_error():
    """Plan point 30 (the user's rule, metric_lock.decide_change): a quieter instrument replaces
    the default only when significant, with >= 5 judging windows AND an improvement of at least
    2 x the judges' measured error. Four windows are not enough judges, however clear."""
    wins = _rows("da3_windows", 4, 1, 0.08, 1, sigma=0.01)
    rows = wins + _rows("da3_mono", 4, 6, 0.01, 2)
    model = G.solve(rows, WALK_M, GCFG, log=lambda *a: None, error_factor=2.0)
    v = model["choice_verdict"]
    assert model["applied_instrument"] == "da3_windows" and not v["improves"]
    assert "judges" in v["failed"] and v["n_judges"] == 4 and v["min_judges"] == 5
    # enough windows, but the judges' own measured error is larger than half the improvement
    wins = _rows("da3_windows", 12, 1, 0.08, 1, sigma=0.2)
    model = G.solve(wins + _rows("da3_mono", 12, 6, 0.01, 2), WALK_M, GCFG,
                    log=lambda *a: None, error_factor=2.0)
    v = model["choice_verdict"]
    assert model["applied_instrument"] == "da3_windows"
    assert v["significant"] and v["enough_judges"] and not v["beyond_error"]
    assert v["error"] == pytest.approx(0.2) and v["failed"] == ["error"]


def test_huber_convergence_is_never_finer_than_the_float64_resolution_of_its_sum():
    """Plan point 39: a step below the rounding of a 10^5-sample weighted mean is the sum's own
    rounding, not movement — the gain converges instead of being dropped at the iteration cap."""
    n = 200_000
    g = G._summation_gamma(n)
    assert g == pytest.approx(n * np.finfo(float).eps / 2, rel=1e-6)
    rng = np.random.default_rng(1)
    om = rng.uniform(1, 8, (400, 500))
    inst = om * 1.25 * np.exp(rng.standard_t(2, om.shape) * 0.05)
    tight = G._log_gain(inst, om, None, replace(GCFG, huber_tol=1e-300))
    assert tight.converged and tight.iterations < GCFG.huber_max_iter


def test_each_windows_bootstrap_has_its_own_stream(tmp_path):
    """Plan point 39: a window that drops out leaves every other window's sigma bit-identical."""
    chainage = _write_session(tmp_path, _drift, _drift)
    wins, _m, _r = G.da3_rows(tmp_path, chainage, 0.0, GCFG)
    first = sorted((tmp_path / "output" / "da3_windows").glob("window_*.npz"))[0]
    with np.load(first) as z:
        d = {k: z[k] for k in z.files}
    d["conf"] = np.zeros_like(d["conf"])                    # window 0 gives no gain any more
    np.savez(first, **d)
    wins2, _m, _r = G.da3_rows(tmp_path, chainage, 0.0, GCFG)
    by = {r.group: r.sigma for r in wins}
    assert 0 not in {r.group for r in wins2}
    assert all(r.sigma == by[r.group] for r in wins2)


def test_regeneration_checks_the_card_and_refuses_a_plan_that_is_not_the_walks(tmp_path, monkeypatch):
    """Plan point 26: the window files are regenerated on a card checked free, and a regenerated
    plan that is not walk.json's (a halved window, another card) is refused, not mixed in."""
    import intake.walk as W
    import repro
    chainage = _write_session(tmp_path, _drift, _drift)
    wdir = tmp_path / "output" / "da3_windows"
    from intake.walk import plan_windows
    plan = plan_windows(len(chainage), 32, 0.5)
    spec = {"windows": [[f"/f/{f:06d}.jpg" for f in range(a, b)] for a, b in plan],
            "process_res": 504, "model_id": "m"}
    (wdir / "windows.json").write_text(json.dumps(spec))
    walk = {"windows": [{"frames": [a, b - 1], "n": b - a} for a, b in plan]}
    for p in wdir.glob("window_*.npz"):
        p.unlink()
    assert G.needs_window_regeneration(tmp_path)
    calls = []
    monkeypatch.setattr(repro, "require_exclusive_gpu", lambda log=print: calls.append("gpu") or {})

    def regen(sd, g, py, log=print):
        calls.append("da3")
        (wdir / "windows.json").write_text(json.dumps(spec))
    monkeypatch.setattr(W, "run_da3_windows", regen)
    G.regenerate_windows(tmp_path, GCFG, walk, log=lambda *a: None)
    assert calls == ["gpu", "da3"]                              # the card first, then DA3

    def halved(sd, g, py, log=print):
        p2 = plan_windows(len(chainage), 16, 0.5)
        (wdir / "windows.json").write_text(json.dumps(
            {**spec, "windows": [[f"/f/{f:06d}.jpg" for f in range(a, b)] for a, b in p2]}))
    monkeypatch.setattr(W, "run_da3_windows", halved)
    with pytest.raises(G.GaugeError, match="not the plan walk.json"):
        G.regenerate_windows(tmp_path, GCFG, walk, log=lambda *a: None)

    def busy(log=print):
        raise repro.ReproError("the GPU is NOT free")
    monkeypatch.setattr(repro, "require_exclusive_gpu", busy)
    monkeypatch.setattr(W, "run_da3_windows", regen)
    calls.clear()
    with pytest.raises(repro.ReproError):
        G.regenerate_windows(tmp_path, GCFG, walk, log=lambda *a: None)
    assert calls == []                                          # nothing extracted on a shared card


def test_gauge_json_is_stamped_and_gauge_applied_trusts_only_a_matching_stamp_on_the_live_lineage(tmp_path, monkeypatch):
    """Plan point 146: gauge.json carries the stamp of the F2 run (inputs, code, params,
    reconstruction) and the epoch it produced; gauge_applied answers from that stamp — a missing
    stamp, a stamp that no longer matches, or an applied epoch outside the live lineage FAILS
    naming why; the file's existence decides nothing."""
    chainage = _write_session(tmp_path, _drift, _drift)
    frames = sorted(chainage)
    out = tmp_path / "output"
    walk = {"walk_length_m": WALK_M,
            "chainage": [{"frame": f, "chainage_m": chainage[f]} for f in frames],
            "windows": []}
    from intake.walk import plan_windows
    walk["windows"] = [{"frames": [a, b - 1]} for a, b in plan_windows(len(frames), 32, 0.5)]
    (tmp_path / "intake").mkdir()
    (tmp_path / "intake" / "walk.json").write_text(json.dumps(walk))
    (out / "camera_frames.txt").write_text("\n".join(str(f) for f in frames) + "\n")
    poses = np.tile(np.eye(4), (len(frames), 1, 1))
    poses[:, 0, 3] = [chainage[f] for f in frames]
    from correction.session import write_poses
    write_poses(out / "camera_poses.txt", poses)
    cfg = SimpleNamespace(**{**GCFG.__dict__, "instruments": ("da3_windows", "da3_mono")})
    assert not G.gauge_applied(out)                                # no gauge.json: F2 never ran
    doc = G.run_gauge(tmp_path, cfg, apply=True, log=lambda *a: None)
    assert doc["applied"] and doc["epoch_to"] == 1
    saved = json.loads((out / G.GAUGE_NAME).read_text())
    st = saved[G.GAUGE_STAMP_KEY]
    assert "intake/walk.json" in st["inputs"] and "output/camera_frames.txt" in st["inputs"]
    assert any(k.endswith("precision/gauge.py") for k in st["code"])
    assert "reconstruction.precision.gauge" in st["config"] and saved["reconstruction_id"]
    assert G.gauge_applied(out) is True
    # the walk changes under it (another plan, another intake): the file is not this session's
    (tmp_path / "intake" / "walk.json").write_text(json.dumps({**walk, "walk_length_m": WALK_M + 1}))
    with pytest.raises(G.GaugeError, match="walk.json"):
        G.gauge_applied(out)
    (tmp_path / "intake" / "walk.json").write_text(json.dumps(walk))
    assert G.gauge_applied(out) is True
    # the gauge's epoch undone / another branch selected: epoch 1 is not in the live lineage
    (out / "geometry_epoch.json").write_text(json.dumps({"epoch": 3, "parent_epoch": 0,
                                                         "correction_id": "x", "kind": "transform"}))
    with pytest.raises(G.GaugeError, match="not in the lineage"):
        G.gauge_applied(out)
    (out / "geometry_epoch.json").write_text(json.dumps({"epoch": 3, "parent_epoch": 1,
                                                         "correction_id": "x", "kind": "transform"}))
    assert G.gauge_applied(out) is True                           # 0 → 1 → 3: on the lineage
    # the flag alone, no stamp (a file written by hand or by an older F2): refused, not read
    (out / G.GAUGE_NAME).write_text(json.dumps({"version": 1, "applied": True}))
    with pytest.raises(G.GaugeError, match="no stamp"):
        G.gauge_applied(out)
    # a measure-only run of this session: stamped, not applied → False
    stamped_false = {**saved, "applied": False, "epoch_to": None}
    (out / G.GAUGE_NAME).write_text(json.dumps(stamped_false, default=float))
    assert G.gauge_applied(out) is False
    # … but a measure-only file of ANOTHER reconstruction is refused too
    rec = sorted((out / "omega_run" / "results_output").glob("frame_*.npz"))[0]
    with np.load(rec) as z:
        d = {k: z[k] for k in z.files}
    np.savez(rec, **{**d, "depth": d["depth"] * 1.001})
    with pytest.raises(G.GaugeError, match="reconstruction"):
        G.gauge_applied(out)
