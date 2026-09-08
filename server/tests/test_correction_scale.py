"""Scale rule vs the DA3 anchors: a k that contradicts the anchors is
rejected without override, and applied WITH override — recorded in the
ledger."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from correction.ledger import ledger_view                    # noqa: E402
from correction.run import run_objects                       # noqa: E402
from tests.synth_correction import build_scene, make_correction_cfg  # noqa: E402


def test_scale_gate_rejects_contradicting_k(tmp_path):
    """DA3 confirms the observed (compressed) depth → the solved k=1/c
    contradicts the anchors → rejected."""
    scene = build_scene(tmp_path, drift_yaw_deg=0.5,
                        drift_t=(0.05, 0.0, 0.02), depth_c=0.90,
                        da3_agrees_with_truth=False)
    rep = run_objects(scene.output_dir, [1, 2], "test",
                      cfg=make_correction_cfg())
    assert rep["status"] == "rejected"
    g = next(g for g in rep["gates"] if g["name"] == "scale_vs_da3")
    assert not g["passed"], g
    assert g["worst_anchor"] is not None


def test_scale_gate_passes_when_da3_agrees(tmp_path):
    """DA3 measured the TRUE depth → the k that re-expands the compressed
    geometry IMPROVES the anchor agreement → passes."""
    scene = build_scene(tmp_path, drift_yaw_deg=0.5,
                        drift_t=(0.05, 0.0, 0.02), depth_c=0.90,
                        da3_agrees_with_truth=True)
    rep = run_objects(scene.output_dir, [1, 2], "test",
                      cfg=make_correction_cfg())
    assert rep["status"] == "pending", rep.get("rejection_reason")
    g = next(g for g in rep["gates"] if g["name"] == "scale_vs_da3")
    assert g["passed"]


def test_override_applies_and_is_recorded(tmp_path):
    scene = build_scene(tmp_path, drift_yaw_deg=0.5,
                        drift_t=(0.05, 0.0, 0.02), depth_c=0.90,
                        da3_agrees_with_truth=False)
    rep = run_objects(scene.output_dir, [1, 2], "operator-x",
                      override_scale_check=True,
                      cfg=make_correction_cfg())
    assert rep["status"] == "pending", rep.get("rejection_reason")
    assert rep["overrides"]["scale_check"]["overridden_by"] == "operator-x"
    rows = ledger_view(scene.output_dir)
    assert rows[-1]["overrides"].get("scale_check"), "ledger must record it"
    # scale_diagnostics regenerated with the per-epoch history
    diag = json.loads(
        (scene.output_dir / "scale_diagnostics.json").read_text())
    assert diag["epochs"][-1]["epoch"] == 1
    assert diag["epochs"][-1]["correction_id"] == rep["correction_id"]
