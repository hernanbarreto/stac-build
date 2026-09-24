"""Advisory gates (USER 2026-09-09): gates are measured and reported, the
correction is applied anyway; the user's eye on the epochs is the verdict.

Only the floor alignment is left to exercise it (2026-09-24): the manual
object correction went with the UI "Corrections" button."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from correction.run import run_floor                             # noqa: E402
from tests.synth_correction import build_scene, make_correction_cfg  # noqa: E402


def test_floor_alignment_advisory_applies(tmp_path):
    """A step bar of 1 mm fails the continuity gate on any real alignment; in
    advisory mode the run is APPLIED with the failed gate flagged."""
    scene = build_scene(tmp_path, floor="ramp", floor_slope=0.04)
    rep = run_floor(scene.output_dir, "level", None, "test",
                    cfg=make_correction_cfg(**{"gates.mode": "advisory",
                                               "gates.max_step_mm": 1.0}))
    assert rep["status"] == "applied", rep.get("rejection_reason")
    assert rep["warnings"]
    g = next(g for g in rep["gates"] if g["name"] == "continuity")
    assert g["advisory"] is True and g["passed"] is False
    assert (scene.output_dir / "geometry_epoch.json").exists()
