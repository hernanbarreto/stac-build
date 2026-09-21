"""claude_stac.txt F4 — smoke test of the visual validation kit's data over a
synthetic session (§11): after a certification run the kit's endpoints'
builders deliver the trajectory with its edges and verdicts, the duplicates
list with fly-to anchors, the attention list, the epoch chain and the
per-epoch quality reports the acta panel shows. (`ui` compiles: build.)"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_certify_f3 import _cfg, _write, _drift, _run, N_KF   # noqa: E402
from tests.synth_metric import make_session, corridor_loop_scene, loop_trajectory   # noqa: E402


@pytest.fixture(scope="module")
def certified(tmp_path_factory):
    sess = make_session(H=40, W=56, scene=corridor_loop_scene(), poses=loop_trajectory(N_KF, extra_laps=0.15))
    root = _write(tmp_path_factory.mktemp("kit") / "s", sess, _drift(sess))
    acta = _run(root, sess, _cfg(), max_iters=1)
    # the correction is the visit-drift loop, which publishes its epochs
    # before the iterations; an iteration that applies nothing is the normal
    # ending, so what the kit needs is that an epoch reached the session
    assert acta["epoch_final"] >= 1, acta["stop_reason"]
    return root / "output", acta


def test_kit_edges_duplicates_and_attention(certified):
    out, acta = certified
    from reconstruction.certify.kit import kit_edges, epoch_layers
    from reconstruction.certify.attention import attention_list
    k = kit_edges(out)
    assert k["n_keyframes"] == N_KF and len(k["positions"]) == N_KF and len(k["odometry"]) == N_KF - 1
    kinds = {e["kind"] for e in k["loops"]}
    assert "accepted" in kinds, kinds
    for e in k["loops"]:
        assert 0 <= e["i"] < N_KF and 0 <= e["j"] < N_KF
        assert e["kind"] in ("accepted", "rejected", "scale_break", "ambiguous")
        if e["kind"] == "rejected":
            assert e["reason"], e                      # every red edge says why
    revisit = [e for e in k["loops"] if e["source"].startswith("revisit")]
    assert revisit and all(e.get("offset_before_m") is not None for e in revisit if e["kind"] == "accepted")
    # duplicates carry fly-to anchors
    for d in k["duplicates"]:
        if d["i"] is not None:
            assert len(d["i_pos"]) == 3
    assert set(k["legend"]) >= {"accepted", "rejected", "scale_break", "ambiguous", "odometry"}
    # attention list: sorted by severity, every item with text (+ anchor when it has a place)
    att = attention_list(out)
    assert att["n"] == len(att["items"])
    sev = [it["severity"] for it in att["items"]]
    assert sev == sorted(sev, reverse=True)
    for it in att["items"]:
        assert it["kind"] and it["text"]
        if it["anchor"] is not None:
            assert len(it["anchor"]["position"]) == 3
    assert any(it["kind"] == "low_mv_votes" for it in att["items"])     # the cloud has witnesses
    # epoch chain for the before/after toggle
    ep = epoch_layers(out)
    # one epoch per correction the visit-drift loop published, all of them
    # still on disk for the before/after toggle
    assert ep["epoch"] == acta["epoch_final"] >= 1
    assert sorted(p["epoch"] for p in ep["previous"]) == list(range(acta["epoch_final"]))
    # every epoch the correction published carries its own octree — that is
    # what the epoch selector loads; epoch 0 is the synthetic fixture's own
    # reconstruction and never had one
    by_epoch = {p["epoch"]: p for p in ep["previous"]}
    assert by_epoch[0]["potree"] is False
    assert all(by_epoch[e]["potree"] is True for e in by_epoch if e > 0), by_epoch


def test_kit_acta_and_reports(certified):
    out, acta = certified
    from reconstruction.quality.report import QUALITY_DIR
    rep = json.loads((out / QUALITY_DIR / "report_epoch_1.json").read_text())
    m = rep["metrics"]
    for key in ("loop_residual", "seam_residual", "closure", "duplicates", "scale", "witnesses",
                "depth_disagreement", "loop_coverage", "authority", "objective"):
        assert key in m, key
    assert rep["comparison_vs_previous"]["objective"] < 0
    it = acta["iterations"][0]
    assert all({"name", "value", "threshold", "passed"} <= set(g) for g in it["gates"])
    a = json.loads((out / "certify_acta.json").read_text())
    assert a["stop_reason"] and a["iterations"][0]["verdict"] in ("applied", "identity")
    # db52016: ONE composed epoch, so the acta carries `correction` with its
    # stages instead of a `visit_drift.epochs` list
    assert a["correction"]["stages"] and a["epoch_final"] >= 1
    assert a["epoch_after_correction"] <= a["epoch_final"]
    assert [st for st in a["correction"]["stages"]
            if st.get("status") == "applied"], a["correction"]
