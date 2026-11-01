import json
import numpy as np
import pytest


def test_the_birth_pixel_grid_is_read_from_the_declaration_first(tmp_path):
    """pccr_audit4 (2026-11-01): depth_on_f5 maps the published birth pixel back through the lens onto the
    ORIGINAL image, while the report's 'lens + carried' used to mean the bend's UNDISTORTED grid — the
    certification projected without the lens and read the lens itself as a 1.35 px camera error. The report
    now declares birth_pixels_through_lens; a report without it keeps the old reading; no report = unknown."""
    import correction.visit_drift as vd
    rep = tmp_path / "corrected_cloud.json"
    assert vd._birth_grid_declared(tmp_path) is None
    rep.write_text(json.dumps({"grid_of_the_bend": {"lens": True, "carried": True}}))
    assert vd._birth_grid_declared(tmp_path) is True and vd._birth_grid_undistorted(tmp_path) is True
    rep.write_text(json.dumps({"grid_of_the_bend": {"lens": True, "carried": True, "birth_pixels_through_lens": True}}))
    assert vd._birth_grid_declared(tmp_path) is False and vd._birth_grid_undistorted(tmp_path) is False
    rep.write_text(json.dumps({"grid_of_the_bend": {"lens": True, "carried": True, "birth_pixels_through_lens": False}}))
    assert vd._birth_grid_declared(tmp_path) is True
    rep.write_text(json.dumps({"grid_of_the_bend": {"lens": False, "carried": False}}))
    assert vd._birth_grid_declared(tmp_path) is False
