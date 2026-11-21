"""docs/plan_determinismo.md points 50, 54, 56, 63 on the F6 side: the chunk check's floor plane is
a seeded numpy RANSAC (the same plane every run, Open3D untouched), PointDiT runs under the
deterministic torch context and records it, the correction id of a published cloud is derived from
the cloud (not drawn), no report of the chain carries a clock (timing sidecars do) and every report
names the reconstruction it was measured on. CPU, synthetic."""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

QUIET = lambda *a: None  # noqa: E731


# ── point 50: the floor plane ─────────────────────────────────────────────

def test_the_seeded_plane_ransac_never_touches_open3d_and_repeats_bit_for_bit(monkeypatch):
    from reconstruction.geometry import primitives as P
    rng = np.random.default_rng(0)
    pts = np.c_[rng.uniform(-3, 3, 20000), rng.normal(0, 0.005, 20000), rng.uniform(-3, 3, 20000)]
    pts = np.concatenate([pts, rng.uniform(-3, 3, (4000, 3))])

    class Boom:
        class geometry:
            @staticmethod
            def PointCloud(*a, **k):
                raise AssertionError("Open3D must not be used with a seed")

        class utility:
            Vector3dVector = staticmethod(lambda x: x)
    monkeypatch.setattr(P, "o3d", Boom)
    a = P.fit_plane_ransac(pts, dist_thresh=0.02, iters=200, min_inlier_frac=0.2, measure_curvature=False, seed=0)
    b = P.fit_plane_ransac(pts, dist_thresh=0.02, iters=200, min_inlier_frac=0.2, measure_curvature=False, seed=0)
    assert a is not None and np.array_equal(a.normal, b.normal) and a.d == b.d
    assert np.array_equal(a.inliers, b.inliers) and a.inliers.dtype == bool and a.inlier_frac > 0.7
    assert abs(abs(a.normal[1]) - 1.0) < 1e-3


def test_chunk_check_measures_the_same_plane_twice_without_a_clock_in_its_report(tmp_path):
    import test_precision_chunk_check as TC
    from precision import chunk_check as CC
    from precision.config import load_precision_config
    from correction.epoch import RECONSTRUCTION_ID_KEY, reconstruction_id
    p = load_precision_config()
    pcfg = replace(p, chunk_check=replace(p.chunk_check, pixel_stride=1, min_points=50, bootstrap=100))
    _, chain = TC.build_session(tmp_path, "clean")
    out = tmp_path / "output"
    rep1 = CC.run_check(tmp_path, pcfg, log=QUIET, chainage=chain)
    txt1 = (out / "precision" / CC.CHECK_NAME).read_bytes()
    rep2 = CC.run_check(tmp_path, pcfg, log=QUIET, chainage=chain)
    txt2 = (out / "precision" / CC.CHECK_NAME).read_bytes()
    assert txt1 == txt2 and rep1["plane"] == rep2["plane"]
    doc = json.loads(txt1)
    assert "seconds" not in doc and doc["params"]["floor_plane"].startswith("seeded numpy RANSAC")
    assert doc[RECONSTRUCTION_ID_KEY] == reconstruction_id(out)
    timing = json.loads((out / "precision" / "chunk_check.timing.json").read_text())
    assert timing["seconds_check"] >= 0 and timing["seconds_total"] >= timing["seconds_check"]
    # the cloud metrics report it folds in carries no clock either; its sidecar does
    cm = json.loads((out / "precision" / "cloud_metrics.json").read_text())
    assert "measured_at" not in cm and "seconds" not in cm and RECONSTRUCTION_ID_KEY in cm
    cmt = json.loads((out / "precision" / "cloud_metrics.timing.json").read_text())
    assert "measured_at" in cmt and "seconds" in cmt


# ── point 54: PointDiT under the deterministic context ────────────────────

def test_pointdit_inference_runs_under_deterministic_torch_and_records_it():
    import torch
    from precision import pointdit_runner as PR
    from precision.config import load_precision_config
    cfg = load_precision_config().mono_detail
    seen = {}

    def generate(x):
        seen["deterministic"] = torch.are_deterministic_algorithms_enabled()
        seen["warn_only"] = torch.is_deterministic_algorithms_warn_only_enabled()
        seen["benchmark"] = torch.backends.cudnn.benchmark
        seen["tf32"] = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
        g = x[:, 1:2]
        return torch.cat([torch.zeros_like(g), torch.zeros_like(g), g - g.mean()], 1)
    r = PR.PointDiTRunner(cfg, log=QUIET, device="cpu", generate=generate)
    img = (np.random.default_rng(0).random((48, 64, 3)) * 255).astype(np.uint8)
    z1, _ = r.depth(img)
    assert seen == {"deterministic": True, "warn_only": False, "benchmark": False, "tf32": (False, False)}
    fp = r.footprint()
    assert fp["seed"] == cfg.seed and fp["card"] is None and fp["autocast_dtype"] is None
    assert fp["numerics"]["deterministic_algorithms"] is True and fp["numerics"]["seed"] == cfg.seed
    assert fp["numerics"]["cudnn_benchmark"] is False and fp["numerics"]["matmul_allow_tf32"] is False
    assert "vram_peak_gb" not in fp and r.timing() == {}
    z2, _ = r.depth(img)
    assert np.array_equal(z1, z2)
    # the stage's report keeps the footprint and moves the clocks to the timing sidecar
    from precision import depth_on_f5 as D
    from precision.mono_detail import StageReport
    rep = StageReport(params={"footprint": fp, "timing": {"vram_peak_gb": 1.0}, "bars": {}},
                      seconds_pointdit=3.0, seconds_total=5.0)
    doc = D.mono_report(rep, cfg)
    assert "seconds_pointdit" not in doc and "timing" not in doc["params"] and doc["params"]["footprint"] == fp
    assert D.mono_timing(rep) == {"seconds_pointdit": 3.0, "seconds_total": 5.0, "vram_peak_gb": 1.0}


# ── points 56 / 63: the published cloud's id, meta and report ─────────────

def _publish(tmp_path: Path, monkeypatch, n: int = 30) -> tuple:
    from precision import corrected_cloud as CC
    from precision.epoch0_cloud import _write_ply_xyzrgb
    out = tmp_path / "output"
    out.mkdir(parents=True)
    rng = np.random.default_rng(0)
    _write_ply_xyzrgb(out / "cleaned_cloud.ply", rng.standard_normal((20, 3)).astype(np.float32),
                      rng.integers(0, 255, (20, 3)).astype(np.uint8))
    (out / "geometry_epoch.json").write_text(json.dumps({"epoch": 0}))
    tmp = out / CC.TX_TMP
    tmp.mkdir()
    rng = np.random.default_rng(1)
    _write_ply_xyzrgb(tmp / "cleaned_cloud.ply", rng.standard_normal((n, 3)).astype(np.float32),
                      rng.integers(0, 255, (n, 3)).astype(np.uint8))

    def fake_potree(session_dir, force=False, ply_override=None, potree_dir_override=None):
        Path(potree_dir_override).mkdir(parents=True)
        (Path(potree_dir_override) / "metadata.json").write_text(json.dumps({"points": n}))
        return True
    import potree_converter
    monkeypatch.setattr(potree_converter, "convert_ply_to_potree", fake_potree)
    rep = CC.publish(tmp_path, tmp, {"stage": "corrected_cloud", "per_frame": {},
                                     "source_of_depth": "test"}, log=QUIET,
                     columns={"n_consistent": np.arange(n, dtype=np.uint8)})
    return out, rep


def test_the_correction_id_is_derived_from_the_cloud_and_the_report_carries_no_clock(tmp_path, monkeypatch):
    from correction.epoch import RECONSTRUCTION_ID_KEY
    from correction.ledger import read_ledger
    out_a, rep_a = _publish(tmp_path / "a", monkeypatch)
    out_b, rep_b = _publish(tmp_path / "b", monkeypatch)
    assert rep_a["correction_id"] == rep_b["correction_id"] and len(rep_a["correction_id"]) == 8
    assert rep_a["cloud_sha256"] == rep_b["cloud_sha256"]
    assert "seconds" not in rep_a and "created_at" not in rep_a and RECONSTRUCTION_ID_KEY in rep_a
    with np.load(out_a / "origins.npz") as z:
        meta = json.loads(str(z["meta"]))
    assert meta["correction_id"] == rep_a["correction_id"] and meta["cloud_sha256"] == rep_a["cloud_sha256"]
    assert RECONSTRUCTION_ID_KEY in meta
    assert (out_a / "corrected_cloud.json").read_bytes() == (out_b / "corrected_cloud.json").read_bytes()
    assert (out_a / "geometry_epoch.json").read_bytes() == (out_b / "geometry_epoch.json").read_bytes()
    la, lb = read_ledger(out_a), read_ledger(out_b)
    assert la == lb and "created_at" not in la[0]
    # a different cloud gets a different id
    out_c, rep_c = _publish(tmp_path / "c", monkeypatch, n=31)
    assert rep_c["correction_id"] != rep_a["correction_id"]


def test_timing_sidecars_live_next_to_their_reports(tmp_path):
    from precision import corrected_cloud as CC
    p = CC.write_timing(tmp_path / "x" / "depth_on_f5.json", {"seconds": 1.5})
    assert p == tmp_path / "x" / "depth_on_f5.timing.json" and json.loads(p.read_text()) == {"seconds": 1.5}
    assert CC.timing_path(Path("a/b/chunk_check.json")).name == "chunk_check.timing.json"


# ── the config keys the points added fail the load when missing ──────────

@pytest.mark.parametrize("section,key", [("bend", "heldout_confidence"), ("bend", "bootstrap"), ("bend", "seed"),
                                         ("mono_detail", "seed")])
def test_a_missing_judge_key_fails_the_load_naming_it(section, key):
    import copy
    from config import cfg as raw_cfg
    from precision.config import PrecisionConfigError, load_precision_config
    bad = copy.deepcopy(raw_cfg)
    del bad["reconstruction"]["precision"][section][key]
    with pytest.raises(PrecisionConfigError, match=f"{section}.{key}"):
        load_precision_config(bad)
    good = load_precision_config()
    assert good.bend.heldout_confidence == 0.95 and good.bend.bootstrap >= 1 and good.mono_detail.seed == 0
