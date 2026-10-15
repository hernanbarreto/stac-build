"""Phase 4 of the evaluation battery: the stage monitor sums known sleeps, peaks and prices, and
changes no result."""
import json
import time

import pytest

from evaluation import cost as COST
from evaluation import variants as V
from tests import synth_evaluation as S
from tests.test_evaluation_variants import FakeRunner, _raw, _scene_with_refs


def test_monitor_sums_simulated_stage_times_peaks_and_money(tmp_path):
    cfg = S.eval_config(**{"cost.gpu_price_per_hour": {"default": 3.0, "CARD X": 2.0}})
    t = {"now": 0.0}

    def clock():
        return t["now"]

    util = {"v": 50.0, "mem": 1000.0, "proc": 500.0, "rss": 200.0}

    def sampler(pid):
        return util["v"], util["mem"], util["proc"], util["rss"]

    m = COST.StageMonitor(tmp_path, "v", cfg.cost, card_key="CARD X", prices=cfg.cost.gpu_price_per_hour,
                          video_min=2.0, sampler=sampler, clock=clock, interval_s=1000.0)
    m.start()                                   # sample at t=0 (stage setup, 50 %)
    t["now"] = 10.0; util.update(v=100.0, mem=4000.0, rss=900.0); m.stage("A", "start")   # setup: 10 s at 50 %
    t["now"] = 40.0; util.update(v=0.0, mem=1500.0, rss=300.0); m.stage("B", "start")     # A: 30 s at 100 %
    t["now"] = 100.0
    rep = m.stop()                              # B: 60 s at 0 %
    st = rep["stages"]
    assert st["setup"]["wall_s"] == pytest.approx(10.0) and st["setup"]["gpu_time_s"] == pytest.approx(5.0)
    assert st["A"]["wall_s"] == pytest.approx(30.0) and st["B"]["wall_s"] == pytest.approx(60.0)
    # each interval is charged at the utilization sampled at its START (piecewise constant)
    assert st["A"]["gpu_time_s"] == pytest.approx(30.0) and st["B"]["gpu_time_s"] == pytest.approx(0.0)
    assert st["A"]["gpu_mem_peak_mib"] == 4000.0 and st["A"]["rss_peak_mib"] == 900.0
    tot = rep["totals"]
    assert tot["wall_s"] == pytest.approx(100.0) and tot["gpu_time_s"] == pytest.approx(35.0)
    assert tot["gpu_hours"] == pytest.approx(35 / 3600) and rep["price_per_gpu_hour_usd"] == 2.0
    assert tot["cost_usd"] == pytest.approx(35 / 3600 * 2.0) and tot["gpu_mem_peak_mib"] == 4000.0
    assert rep["per_video_minute"]["wall_s"] == pytest.approx(50.0)
    assert json.loads((tmp_path / "cost.json").read_text())["totals"]["wall_s"] == pytest.approx(100.0)
    # an unlisted card takes the default price
    m2 = COST.StageMonitor(tmp_path, "w", cfg.cost, card_key="other", prices=cfg.cost.gpu_price_per_hour,
                           sampler=sampler, clock=clock, interval_s=1000.0)
    assert m2.price == 3.0 and m2.price_source == "default"


def test_monitor_samples_in_the_background_with_real_sleeps(tmp_path):
    cfg = S.eval_config()
    m = COST.StageMonitor(tmp_path, "v", cfg.cost, card_key=None, prices=cfg.cost.gpu_price_per_hour,
                          sampler=lambda pid: (0.0, 0.0, 0.0, 10.0), interval_s=0.05)
    m.start(); m.stage("A"); time.sleep(0.3); m.stage("B"); time.sleep(0.2)
    rep = m.stop()
    assert rep["stages"]["A"]["wall_s"] == pytest.approx(0.3, abs=0.08)
    assert rep["stages"]["B"]["wall_s"] == pytest.approx(0.2, abs=0.08)
    assert rep["n_samples"] >= 8 and rep["totals"]["rss_peak_mib"] == 10.0


def test_instrumentation_does_not_change_the_results(tmp_path):
    scene, doc = _scene_with_refs(tmp_path)
    cfg = S.eval_config(**{"endpoints.depth_window_px": 1, "curves.min_bin_n": 1, "variants_keep": "all"})
    specs = [V.VariantSpec("reference", "chain"), V.VariantSpec("no_f6_bend", "chain",
                                                                overrides={"reconstruction.precision.bend.enabled": False})]
    plain = V.run_scene(V.SceneSpec("synth", scene.session_dir), specs, tmp_path / "run_plain", cfg, FakeRunner(), _raw(),
                        log=lambda *a: None, plots=False)
    factory = lambda out_dir, name: COST.StageMonitor(out_dir, name, cfg.cost, card_key=None,  # noqa: E731
                                                       prices=cfg.cost.gpu_price_per_hour,
                                                       sampler=lambda pid: (10.0, 100.0, 50.0, 20.0), interval_s=0.05)
    mon = V.run_scene(V.SceneSpec("synth", scene.session_dir), specs, tmp_path / "run_mon", cfg, FakeRunner(), _raw(),
                      log=lambda *a: None, monitor_factory=factory, plots=False)
    for n in ("reference", "no_f6_bend"):
        a = {k: v for k, v in plain["results"][n]["summary"].items()}
        b = {k: v for k, v in mon["results"][n]["summary"].items()}
        assert a == b
        assert plain["results"][n]["coverage"] == mon["results"][n]["coverage"]
        assert "cost" in mon["results"][n] and mon["results"][n]["cost"]["totals"]["wall_s"] > 0
        assert (tmp_path / "run_mon" / "synth" / n / "cost.json").exists()
        assert "reconstruction" in mon["results"][n]["cost"]["stages"] and "evaluation" in mon["results"][n]["cost"]["stages"]
    rows = {r["variant"]: r for r in mon["comparison"]}
    assert rows["reference"]["gpu_hours"] is not None and rows["reference"]["cost_usd"] is not None
