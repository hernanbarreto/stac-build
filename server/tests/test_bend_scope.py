"""F6's depth factor: one per keyframe (the record) or ONE for the whole session (USER 2026-11-20).

``reconstruction.precision.bend.scope`` is read by F6 itself (precision.depth_on_f5.bend_scope) from the job's raw
configuration: declared in precision/config.py it would enter the code stamp of every step and run F2 and F5 again on
a session whose F2 and F5 are being kept."""
import numpy as np
import pytest

from precision import depth_on_f5 as D5
from precision.code_closure import code_closure


def _cfg(scope=None):
    bend = {"enabled": True}
    if scope is not None:
        bend["scope"] = scope
    return {"reconstruction": {"precision": {"bend": bend}}}


def test_absent_is_the_record_and_an_unknown_value_fails_naming_the_key():
    assert D5.bend_scope(_cfg()) == "keyframe"
    assert D5.bend_scope({}) == "keyframe"
    assert D5.bend_scope(_cfg("keyframe")) == "keyframe"
    assert D5.bend_scope(_cfg("session")) == "session"
    with pytest.raises(D5.DepthOnF5Error, match="reconstruction.precision.bend.scope"):
        D5.bend_scope(_cfg("walk"))


def _rows(n, per_kf, rng, factors):
    rows = {}
    for i in range(n):
        u = rng.uniform(0, 464, per_kf); v = rng.uniform(0, 832, per_kf)
        rows[i] = (D5.design(u, v, 464, 832), factors[i] + rng.normal(0, 0.004, per_kf))
    return rows


def test_one_factor_for_every_keyframe_the_robust_fit_of_all_the_rows():
    rng = np.random.default_rng(0)
    n = 40
    factors = np.full(n, 0.95)
    factors[10:14] = 0.86          # a stretch whose own landmarks ask for another factor
    factors[0] = 0.10              # a keyframe whose landmarks are nonsense
    rows = _rows(n, 300, rng, factors)
    kf_coefs, kf_info = D5.bend_coefficients(rows, n, 0, 20, 1.345, 10, 1.1)
    kf_prov = {i: {"status": "verified"} for i in range(n)}
    assert abs(kf_coefs[0][0] - 0.10) < 0.01 and abs(kf_coefs[11][0] - 0.86) < 0.01   # the record follows each one
    coefs, prov, info = D5.session_bend(rows, n, 1.345, 10, kf_coefs, kf_prov)
    c0 = info["c0"]
    pooled = np.concatenate([rows[i][1] for i in range(n)])
    expected, _sig = D5.irls_huber_fit(np.ones((len(pooled), 1)), pooled, 1.345, 10)
    assert c0 == float(expected[0])                       # the bend's own fit, the window = the whole session
    assert abs(c0 - 0.95) < 0.003                         # the odd keyframes do not move it
    for i in range(n):
        assert np.array_equal(coefs[i], np.array([c0, 0.0, 0.0]))
        assert prov[i]["status"] == "session" and prov[i]["fit"] is None and prov[i]["window"] is None
        assert prov[i]["keyframe_c0"] == float(kf_coefs[i][0]) and prov[i]["keyframe_status"] == "verified"
    w = info["per_keyframe_would_apply"]
    assert info["rows"] == n * 300 and info["keyframes_with_rows"] == n
    assert w["min"] == pytest.approx(float(min(kf_coefs[i][0] for i in range(n))))
    assert w["max"] == pytest.approx(float(max(kf_coefs[i][0] for i in range(n))))


def test_a_keyframe_without_landmarks_takes_the_session_factor_too():
    rng = np.random.default_rng(1)
    n = 6
    rows = _rows(n, 200, rng, np.full(n, 0.97))
    rows[3] = (np.zeros((0, 3)), np.zeros(0))             # the record leaves this one at k = 1
    kf_coefs, _ = D5.bend_coefficients(rows, n, 0, 20, 1.345, 10, 1.1)
    assert np.array_equal(kf_coefs[3], D5.IDENTITY)
    coefs, prov, info = D5.session_bend(rows, n, 1.345, 10, kf_coefs, {i: {"status": "x"} for i in range(n)})
    assert coefs[3][0] == info["c0"] and abs(info["c0"] - 0.97) < 0.002 and info["keyframes_with_rows"] == n - 1


def test_no_landmark_in_the_whole_session_fails():
    rows = {i: (np.zeros((0, 3)), np.zeros(0)) for i in range(3)}
    with pytest.raises(D5.DepthOnF5Error, match="no landmark row"):
        D5.session_bend(rows, 3, 1.345, 10, {i: D5.IDENTITY for i in range(3)}, {i: {} for i in range(3)})


def test_the_key_stays_out_of_the_earlier_steps_code_stamps():
    """an F6-only decision must never run F2 – F5 again: F6's module is in no earlier step's closure"""
    for module in ("precision.camera", "precision.gauge", "precision.tracks", "precision.refine"):
        files, _external = code_closure(module)
        assert not any(str(p).endswith("precision/depth_on_f5.py") for p in files), module


def test_the_scope_enters_f6s_own_stamp():
    from precision import runner
    f6 = next(s for s in runner.STEPS if s.key == "f6_bend")
    assert "raw:reconstruction.precision.bend" in f6.config
    assert f6.config.index("raw:reconstruction.precision.bend") > f6.config.index("bend")   # the raw section wins
