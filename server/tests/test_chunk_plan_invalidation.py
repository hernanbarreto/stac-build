"""USER 2026-10-05: a new chunk plan wipes every product of the old one and everything
downstream before Omega runs — old and new chunk products never live side by side.

USER 2026-10-06: the plan is the co-visibility plan's EXPLICIT ranges (variable lengths, each
seam its own overlap) and Omega's resolution adapts to the largest chunk; the comparison is on
the ranges, the keyframe count and the resolution — never on a size/overlap, never on the
planner's report. A single pass is the one range [0, n), recorded by its own run config."""

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workers.map_worker import _invalidate_on_new_chunk_plan, chunk_plan_doc  # noqa: E402

PCCR = [(0, 63), (30, 169), (63, 189), (169, 211), (189, 289)]
PCCR_OTHER = [(0, 63), (30, 170), (63, 189), (170, 211), (189, 289)]   # one cut moved by 1 kf


def _res(r=832, mode="max_size", gb=79.99):
    return {"resolution": r, "mode": mode, "grid_wh": [464, 832], "card_total_gb": gb}


def _plan(ranges, n=289, res=None, covis=None):
    return chunk_plan_doc(ranges, n, 17.3, covis or {"D_total": 34.71, "measurement": "measured"},
                          res or _res())


def _legacy_plan(size, ov, n=1998):
    from reconstruction.chunk_plan import chunk_ranges
    return {"version": 1, "phase": "walk-planned", "n_keyframes": n, "chunk_size": size,
            "overlap": ov, "chunk_ranges": [[a, b] for a, b in chunk_ranges(n, size, ov)]}


def _session(tmp_path, old_plan=None, old_config=None):
    out = tmp_path / "output"
    ml = out / "maplong_run"
    for d in ("_tmp_results_unaligned", "_tmp_results_aligned", "_tmp_results_loop", "pcd", "sky_masks"):
        (ml / d).mkdir(parents=True)
    (ml / "_tmp_results_unaligned" / "chunk_0.npy").write_bytes(b"x")
    (ml / "_tmp_results_loop" / "loop_1_2_3_4.npy").write_bytes(b"x")
    for f in ("loop_closures.txt", "salad_calibration.json", "frame_list.json", "metric_lock.json",
              "chunk_health.json", "vggt_omega_config.yaml", "chunk_sim3.json"):
        (ml / f).write_text("{}")
    for d in ("da3_run", "da3_windows", "intake", "omega_run", "precision", "potree"):
        (out / d).mkdir()
    for f in ("cleaned_cloud.ply", "camera.json", "geometry_epoch.json", "segmentation.json",
              "salad_revisit_reference.json"):
        (out / f).write_text("{}")
    if old_plan is not None:
        (out / "chunk_plan.json").write_text(json.dumps(old_plan))
    if old_config is not None:
        (out / "vggt_omega_config.yaml").write_text(yaml.dump({"Model": old_config}))
    return out


def _wiped(out):
    ml = out / "maplong_run"
    for gone in ("_tmp_results_unaligned", "_tmp_results_aligned", "_tmp_results_loop", "pcd",
                 "metric_lock.json", "chunk_health.json", "chunk_sim3.json"):
        assert not (ml / gone).exists(), gone
    for gone in ("omega_run", "precision", "potree", "cleaned_cloud.ply", "camera.json",
                 "geometry_epoch.json", "segmentation.json"):
        assert not (out / gone).exists(), gone
    for kept in ("da3_run", "da3_windows", "intake", "salad_revisit_reference.json"):
        assert (out / kept).exists(), kept
    for kept in ("loop_closures.txt", "salad_calibration.json", "frame_list.json", "sky_masks"):
        assert (ml / kept).exists(), kept
    return True


def test_the_plan_document_carries_the_ranges_and_no_uniform_size():
    p = _plan(PCCR)
    assert p["chunk_ranges"] == [list(r) for r in PCCR]
    assert p["chunk_lengths"] == [63, 139, 126, 42, 100]
    assert p["seam_overlaps"] == [33, 106, 20, 22]
    assert "chunk_size" not in p and "overlap" not in p
    assert p["omega_resolution"]["resolution"] == 832 and p["covis"]["D_total"] == 34.71
    from correction.units import chunks_of_keyframe, load_chunk_plan
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "chunk_plan.json").write_text(json.dumps(p))
        loaded = load_chunk_plan(d)                  # the correction module reads it as is
        assert chunks_of_keyframe(loaded, 50) == [0, 1] and chunks_of_keyframe(loaded, 200) == [3, 4]


def test_same_plan_keeps_everything(tmp_path):
    out = _session(tmp_path, _plan(PCCR))
    assert _invalidate_on_new_chunk_plan(out, _plan(PCCR), log=lambda m: None) is False
    assert (out / "maplong_run" / "_tmp_results_unaligned" / "chunk_0.npy").exists()
    assert (out / "cleaned_cloud.ply").exists()


def test_the_same_ranges_with_another_report_keep_everything(tmp_path):
    """measured vs reused, another card total that resolves to the same resolution: the
    products are the same products."""
    out = _session(tmp_path, _plan(PCCR))
    new = _plan(PCCR, res=_res(gb=47.99), covis={"D_total": 34.71, "measurement": "reused"})
    assert _invalidate_on_new_chunk_plan(out, new, log=lambda m: None) is False
    assert (out / "cleaned_cloud.ply").exists()


def test_other_ranges_over_the_same_keyframes_wipe(tmp_path):
    out = _session(tmp_path, _plan(PCCR))
    logs = []
    assert _invalidate_on_new_chunk_plan(out, _plan(PCCR_OTHER), log=logs.append) is True
    assert _wiped(out)
    assert any("NEW PLAN" in m and "5 chunk(s), lengths 41-140 kf" in m for m in logs), logs


def test_another_omega_resolution_wipes(tmp_path):
    """Same ranges, Omega at another resolution (another card): the chunk predictions on disk
    are on another grid — they must not be resumed next to the new ones."""
    out = _session(tmp_path, _plan(PCCR))
    assert _invalidate_on_new_chunk_plan(out, _plan(PCCR, res=_res(r=672)),
                                         log=lambda m: None) is True
    assert _wiped(out)


def test_a_legacy_uniform_plan_on_disk_is_wiped_and_named(tmp_path):
    out = _session(tmp_path, _legacy_plan(296, 148))
    logs = []
    assert _invalidate_on_new_chunk_plan(out, _plan(PCCR), log=logs.append) is True
    assert _wiped(out)
    assert any("NEW PLAN" in m and "296/148" in m for m in logs), logs


def test_a_single_pass_after_a_chunked_plan_wipes(tmp_path):
    """The single pass passes its one range: the chunked products of the old plan go before
    Omega runs (they used to be left for the fork to trip over)."""
    out = _session(tmp_path, _plan(PCCR))
    single = {"chunk_ranges": [[0, 289]], "n_keyframes": 289, "omega_resolution": _res()}
    assert _invalidate_on_new_chunk_plan(out, single, log=lambda m: None) is True
    assert _wiped(out)


def test_a_single_pass_is_recorded_by_its_own_config(tmp_path):
    """No chunk_plan.json for a single pass: the previous pass's vggt_omega_config.yaml
    (Model.chunk_ranges + omega_resolution/omega_mode) is the record of what is on disk."""
    cfg = {"chunk_ranges": [[0, 183]], "omega_resolution": 1376, "omega_mode": "max_size"}
    single = {"chunk_ranges": [[0, 183]], "n_keyframes": 183,
              "omega_resolution": {"resolution": 1376, "mode": "max_size"}}
    out = _session(tmp_path, None, cfg)
    assert _invalidate_on_new_chunk_plan(out, single, log=lambda m: None) is False
    assert (out / "cleaned_cloud.ply").exists()
    other = dict(single, omega_resolution={"resolution": 1808, "mode": "max_size"})
    assert _invalidate_on_new_chunk_plan(out, other, log=lambda m: None) is True
    assert _wiped(out)


def _frame_list(out, n):
    (out / "maplong_run" / "frame_list.json").write_text(
        json.dumps([f"{i:06d}.jpg" for i in range(n)]))


def test_a_legacy_single_pass_config_with_the_same_layout_keeps_everything(tmp_path):
    """Review 2026-10-06: a session last reconstructed as a single pass BEFORE the explicit
    ranges (chunk_size = n, overlap 0) encodes [(0, n)] — resumed with replace OFF at the same
    resolution it keeps its Omega products, core, segmentation and epochs."""
    legacy = {"chunk_size": 183, "overlap": 0, "omega_resolution": 1920, "omega_mode": "max_size"}
    single = {"chunk_ranges": [[0, 183]], "n_keyframes": 183,
              "omega_resolution": {"resolution": 1920, "mode": "max_size"}}
    out = _session(tmp_path, None, legacy)          # frame_list.json unreadable: n = chunk_size
    (out / "corrections").mkdir()
    (out / "corrections" / "epoch_1.npz").write_bytes(b"x")
    assert _invalidate_on_new_chunk_plan(out, single, log=lambda m: None) is False
    for kept in ("omega_run", "cleaned_cloud.ply", "segmentation.json", "corrections/epoch_1.npz"):
        assert (out / kept).exists(), kept
    assert (out / "maplong_run" / "_tmp_results_unaligned" / "chunk_0.npy").exists()
    _frame_list(out, 183)                            # the fork's own record of the launch
    assert _invalidate_on_new_chunk_plan(out, single, log=lambda m: None) is False
    assert (out / "omega_run").exists()


def test_a_legacy_single_pass_config_with_another_layout_or_resolution_wipes(tmp_path):
    legacy = {"chunk_size": 183, "overlap": 0, "omega_resolution": 1920, "omega_mode": "max_size"}
    out = _session(tmp_path, None, legacy)
    logs = []
    other_res = {"chunk_ranges": [[0, 183]], "n_keyframes": 183,
                 "omega_resolution": {"resolution": 1808, "mode": "max_size"}}
    assert _invalidate_on_new_chunk_plan(out, other_res, log=logs.append) is True
    assert _wiped(out)
    assert any("183/0, Omega at 1920" in m and "Omega at 1808" in m for m in logs), logs
    out2 = _session(tmp_path / "b", None, legacy)
    _frame_list(out2, 183)
    other_n = {"chunk_ranges": [[0, 184]], "n_keyframes": 184,
               "omega_resolution": {"resolution": 1920, "mode": "max_size"}}
    assert _invalidate_on_new_chunk_plan(out2, other_n, log=lambda m: None) is True
    assert _wiped(out2)


def test_a_legacy_chunked_config_is_rebuilt_with_the_vendor_formula(tmp_path):
    """chunk_size/overlap > 0: the vendor's uniform layout over the frames that launch ran
    (frame_list.json) — the same ranges keep, other ranges wipe; with no frame count the layout
    cannot be rebuilt and the products go."""
    from reconstruction.chunk_plan import chunk_ranges
    legacy = {"chunk_size": 60, "overlap": 30, "omega_resolution": 832, "omega_mode": "max_size"}
    same = {"chunk_ranges": [[a, b] for a, b in chunk_ranges(289, 60, 30)], "n_keyframes": 289,
            "omega_resolution": {"resolution": 832, "mode": "max_size"}}
    out = _session(tmp_path, None, legacy)
    _frame_list(out, 289)
    assert _invalidate_on_new_chunk_plan(out, same, log=lambda m: None) is False
    assert (out / "cleaned_cloud.ply").exists()
    assert _invalidate_on_new_chunk_plan(out, _plan(PCCR), log=lambda m: None) is True
    assert _wiped(out)
    out2 = _session(tmp_path / "b", None, legacy)    # no readable frame list, overlap > 0
    logs = []
    assert _invalidate_on_new_chunk_plan(out2, same, log=logs.append) is True
    assert _wiped(out2)
    assert any("pre-2026-10-06 config" in m for m in logs), logs


def test_a_version_1_plan_takes_its_resolution_from_its_run_config(tmp_path):
    from reconstruction.chunk_plan import chunk_ranges
    old = _legacy_plan(60, 30, n=289)
    cfg = {"chunk_size": 60, "overlap": 30, "omega_resolution": 832, "omega_mode": "max_size"}
    same = {"chunk_ranges": [[a, b] for a, b in chunk_ranges(289, 60, 30)], "n_keyframes": 289,
            "omega_resolution": {"resolution": 832, "mode": "max_size"}}
    out = _session(tmp_path, old, cfg)
    assert _invalidate_on_new_chunk_plan(out, same, log=lambda m: None) is False
    assert (out / "cleaned_cloud.ply").exists()
    other = dict(same, omega_resolution={"resolution": 672, "mode": "max_size"})
    assert _invalidate_on_new_chunk_plan(out, other, log=lambda m: None) is True
    assert _wiped(out)


def test_chunks_without_a_recorded_plan_are_wiped_a_first_run_is_untouched(tmp_path):
    out = _session(tmp_path, None)
    assert _invalidate_on_new_chunk_plan(out, _plan(PCCR), log=lambda m: None) is True
    assert not (out / "maplong_run" / "_tmp_results_unaligned").exists()
    fresh = tmp_path / "fresh" / "output"
    (fresh / "da3_run").mkdir(parents=True)
    (fresh / "intake").mkdir()
    assert _invalidate_on_new_chunk_plan(fresh, _plan(PCCR), log=lambda m: None) is False
    assert (fresh / "da3_run").exists()


def test_the_fork_stops_on_a_chunk_of_another_plan_instead_of_reinferring_it_alone():
    src = (Path(__file__).resolve().parents[2] / "vendor" / "VGGT-Long" / "vggt_long.py").read_text()
    assert "class _StacPlanMismatch(RuntimeError)" in src
    assert "except _StacPlanMismatch:\n                raise" in src
    assert "predictions['_stac_range']" in src, "an explicit-layout chunk is stamped by its range"
