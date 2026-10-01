"""The epoch transaction's MLS re-consolidation (correction/apply.py step 9b)
obeys the user's ONE switch, reconstruction.precision.cloud.consolidate.

Edge audit 2026-10-01 #4: the user switched consolidation off on 2026-09-30 and
the certify transaction kept running the MLS (it read only its own
correction.apply.reconsolidate), rounding every crease by 3-4 mm.
correction.apply.reconsolidate stays readable and is subordinate: the MLS runs
only when both say true.

Synthetic (tests/synth_correction.build_scene), no GPU: the MLS and the
semantic-service stop are replaced by recorders — the real stop would kill a
running vLLM.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as server_config                                    # noqa: E402
from correction.apply import (reconsolidation_decision,           # noqa: E402
                              stage_transaction)
from correction.session import load_session, read_ply              # noqa: E402
from precision.config import PrecisionConfigError                 # noqa: E402
from tests.synth_correction import build_scene, make_correction_cfg  # noqa: E402

SERVER = Path(__file__).resolve().parents[1]


def _raw(consolidate):
    with open(SERVER / "config.yaml") as f:
        raw = yaml.safe_load(f)
    raw["reconstruction"]["precision"]["cloud"]["consolidate"] = consolidate
    return raw


@pytest.mark.parametrize("master,sub,runs", [(False, True, False), (False, False, False),
                                             (True, False, False), (True, True, True)])
def test_both_switches_must_say_true(master, sub, runs):
    cfg = make_correction_cfg(**{"apply.reconsolidate": sub})
    ok, why = reconsolidation_decision(cfg, _raw(master))
    assert ok is runs
    if not master:
        assert "reconstruction.precision.cloud.consolidate is false" in why


def test_a_missing_switch_fails_the_load():
    raw = _raw(False)
    del raw["reconstruction"]["precision"]["cloud"]["consolidate"]
    with pytest.raises(PrecisionConfigError, match="consolidate"):
        reconsolidation_decision(make_correction_cfg(**{"apply.reconsolidate": True}), raw)


def _patched(monkeypatch, master):
    """The server-wide config with the user's switch set, and recorders in place
    of the MLS and of the semantic-service stop."""
    cl = server_config.cfg["reconstruction"]["precision"]["cloud"]
    monkeypatch.setitem(cl, "consolidate", master)
    calls = {"mls": [], "stop": []}
    import reconstruction.surface_fit.consolidate as cons
    import workers.base as wb

    def fake_mls(out_dir, **kw):
        calls["mls"].append(Path(out_dir))
        return {"n_points": 0, "mean_move_mm": 0.0, "p95_move_mm": 0.0}
    monkeypatch.setattr(cons, "scene_consolidate", fake_mls)
    monkeypatch.setattr(wb, "stop_semantic_service",
                        lambda *a, **k: calls["stop"].append(k.get("stage")))
    return calls


def _stage(scene, cfg, logs):
    sess = load_session(scene.output_dir)
    n_kf = sess.n_kf
    R = np.tile(np.eye(3), (n_kf, 1, 1))
    t = np.zeros((n_kf, 3))
    t[n_kf // 2:, 1] = 0.05                        # a real warp: half the walk moves 5 cm
    k = np.ones(n_kf)
    info = stage_transaction(sess, cfg, R, t, k, correction_id="test",
                             scale_diag_new=None, log=logs.append)
    return sess, info, R, t, k


def test_switch_off_the_transaction_moves_no_point(tmp_path, monkeypatch):
    scene = build_scene(tmp_path)
    calls = _patched(monkeypatch, master=False)
    logs = []
    sess, info, R, t, k = _stage(scene, make_correction_cfg(**{"apply.reconsolidate": True}),
                                 logs)
    assert calls["mls"] == [] and calls["stop"] == [], calls
    assert info["reconsolidation"] == {
        "enabled": False, "ran": False,
        "why": info["reconsolidation"]["why"]}
    assert any("re-consolidation OFF" in m and "precision.cloud.consolidate" in m
               for m in logs), logs
    # the staged cloud is EXACTLY the warp — nothing moved after it
    from correction.apply import warp_full_cloud
    want, _ = warp_full_cloud(sess, R, t, k, log=lambda m: None)
    _, staged = read_ply(Path(info["tx_dir"]) / "cleaned_cloud.ply")
    got = np.stack([staged["x"], staged["y"], staged["z"]], 1).astype(np.float64)
    want32 = want.astype(np.float32).astype(np.float64)
    assert np.array_equal(got, want32)


def test_switch_on_and_subordinate_on_runs_it_once(tmp_path, monkeypatch):
    scene = build_scene(tmp_path)
    calls = _patched(monkeypatch, master=True)
    logs = []
    _, info, *_ = _stage(scene, make_correction_cfg(**{"apply.reconsolidate": True}), logs)
    assert calls["mls"] == [Path(info["tx_dir"])]
    assert len(calls["stop"]) == 1
    assert info["reconsolidation"]["enabled"] and info["reconsolidation"]["ran"]


def test_subordinate_off_keeps_it_off_under_the_users_on(tmp_path, monkeypatch):
    scene = build_scene(tmp_path)
    calls = _patched(monkeypatch, master=True)
    logs = []
    _, info, *_ = _stage(scene, make_correction_cfg(**{"apply.reconsolidate": False}), logs)
    assert calls["mls"] == [] and calls["stop"] == []
    assert "correction.apply.reconsolidate is false" in info["reconsolidation"]["why"]


def test_an_unreadable_switch_stages_nothing(tmp_path, monkeypatch):
    scene = build_scene(tmp_path)
    cl = server_config.cfg["reconstruction"]["precision"]["cloud"]
    monkeypatch.delitem(cl, "consolidate")
    with pytest.raises(PrecisionConfigError):
        _stage(scene, make_correction_cfg(**{"apply.reconsolidate": True}), [])
    assert not any(scene.output_dir.glob("_tx_epoch_*"))
