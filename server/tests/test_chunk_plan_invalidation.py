"""USER 2026-10-05: a new chunk plan wipes every product of the old one and everything
downstream before Omega runs — old and new chunk products never live side by side."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workers.map_worker import _invalidate_on_new_chunk_plan  # noqa: E402


def _plan(size, ov, n=1998):
    from reconstruction.chunk_plan import chunk_ranges
    return {"version": 1, "phase": "walk-planned", "n_keyframes": n, "chunk_size": size,
            "overlap": ov, "chunk_ranges": [[a, b] for a, b in chunk_ranges(n, size, ov)]}


def _session(tmp_path, old_plan):
    out = tmp_path / "output"
    ml = out / "maplong_run"
    for d in ("_tmp_results_unaligned", "_tmp_results_aligned", "_tmp_results_loop", "pcd", "sky_masks"):
        (ml / d).mkdir(parents=True)
    (ml / "_tmp_results_unaligned" / "chunk_0.npy").write_bytes(b"x")
    (ml / "_tmp_results_loop" / "loop_1_2_3_4.npy").write_bytes(b"x")
    for f in ("loop_closures.txt", "salad_calibration.json", "frame_list.json", "metric_lock.json",
              "chunk_health.json", "vggt_omega_config.yaml"):
        (ml / f).write_text("{}")
    for d in ("da3_run", "da3_windows", "intake", "omega_run", "precision", "potree"):
        (out / d).mkdir()
    for f in ("cleaned_cloud.ply", "camera.json", "geometry_epoch.json", "segmentation.json",
              "salad_revisit_reference.json"):
        (out / f).write_text("{}")
    if old_plan is not None:
        (out / "chunk_plan.json").write_text(json.dumps(old_plan))
    return out


def test_same_plan_keeps_everything(tmp_path):
    out = _session(tmp_path, _plan(296, 148))
    assert _invalidate_on_new_chunk_plan(out, _plan(296, 148), log=lambda m: None) is False
    assert (out / "maplong_run" / "_tmp_results_unaligned" / "chunk_0.npy").exists()
    assert (out / "cleaned_cloud.ply").exists()


def test_a_new_plan_wipes_the_old_products_and_everything_downstream(tmp_path):
    out = _session(tmp_path, _plan(296, 148))
    logs = []
    assert _invalidate_on_new_chunk_plan(out, _plan(293, 146), log=logs.append) is True
    ml = out / "maplong_run"
    for gone in ("_tmp_results_unaligned", "_tmp_results_aligned", "_tmp_results_loop", "pcd",
                 "metric_lock.json", "chunk_health.json"):
        assert not (ml / gone).exists(), gone
    for gone in ("omega_run", "precision", "potree", "cleaned_cloud.ply", "camera.json",
                 "geometry_epoch.json", "segmentation.json"):
        assert not (out / gone).exists(), gone
    for kept in ("da3_run", "da3_windows", "intake", "salad_revisit_reference.json", "chunk_plan.json"):
        assert (out / kept).exists(), kept
    for kept in ("loop_closures.txt", "salad_calibration.json", "frame_list.json", "sky_masks"):
        assert (ml / kept).exists(), kept
    assert any("NEW PLAN" in m and "296/148" in m and "293/146" in m for m in logs), logs


def test_chunks_without_a_recorded_plan_are_wiped_a_first_run_is_untouched(tmp_path):
    out = _session(tmp_path, None)
    assert _invalidate_on_new_chunk_plan(out, _plan(293, 146), log=lambda m: None) is True
    assert not (out / "maplong_run" / "_tmp_results_unaligned").exists()
    fresh = tmp_path / "fresh" / "output"
    (fresh / "da3_run").mkdir(parents=True)
    (fresh / "intake").mkdir()
    assert _invalidate_on_new_chunk_plan(fresh, _plan(293, 146), log=lambda m: None) is False
    assert (fresh / "da3_run").exists()


def test_the_fork_stops_on_a_chunk_of_another_plan_instead_of_reinferring_it_alone():
    src = (Path(__file__).resolve().parents[2] / "vendor" / "VGGT-Long" / "vggt_long.py").read_text()
    assert "class _StacPlanMismatch(RuntimeError)" in src
    assert "except _StacPlanMismatch:\n                raise" in src
