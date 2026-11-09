"""USER 2026-11-09 ("los de f6 y f4 hay que pasarlos a ángulo también"): F4's corner spacing, F6's reprojection bound
and PointDiT's detail zone / side reach are ANGLES, read in each stage's px with its camera's focal. On pccr 3108's own
focals (F0's fx 362.93 for F4, F5's fx 380.34 for F6) they give back the validated pixel values exactly; twice the focal
(twice the resolution) gives twice the pixels."""
from precision.camera import angle_px
from precision.config import load_precision_config

F4_FX, F6_FX = 362.9334411621094, 380.3406960008251


def test_pccr_gets_back_its_validated_pixels_and_the_angle_scales_with_the_focal():
    c = load_precision_config()
    assert abs(angle_px(F4_FX, c.tracks.corner_min_distance_deg) - 8.0) < 1e-9
    assert abs(angle_px(F6_FX, c.cloud.fusion_max_reproj_deg) - 2.0) < 1e-9
    assert round(angle_px(F6_FX, c.mono_detail.detail_zone_deg)) == 12
    assert round(angle_px(F6_FX, c.mono_detail.side_reach_deg)) == 3
    assert abs(angle_px(2 * F6_FX, c.cloud.fusion_max_reproj_deg) - 4.0) < 1e-9
