"""
surface_fit stage 2.1 + 4 acceptance tests on synthetic clouds with KNOWN
ground truth: parameter recovery, residuals ≈ injected noise, white-noise vs
structured residual discrimination, findings detection, and the
no-extrapolation support trim.
"""
import numpy as np
import pytest

from reconstruction.surface_fit.plane import fit_plane
from reconstruction.surface_fit.runner import fit_segment
from reconstruction.surface_fit.spatial_test import grid_scalar, morans_i
from reconstruction.surface_fit.support import trimmed_quad_mesh

RNG = np.random.default_rng(42)
SIGMA = 0.003  # 3 mm sensor-ish noise


def make_plane_cloud(n=60_000, size=(4.0, 3.0), normal=(0.2, 0.1, 0.97),
                     d0=1.5, sigma=SIGMA, rng=RNG):
    """Noisy plane patch: n·x + d = 0 with gaussian residuals along n."""
    normal = np.asarray(normal, float)
    normal = normal / np.linalg.norm(normal)
    # build in-plane basis (ref must not be parallel to the normal)
    ref = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, ref); u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    su = rng.uniform(-size[0] / 2, size[0] / 2, n)
    sv = rng.uniform(-size[1] / 2, size[1] / 2, n)
    origin = -d0 * normal
    pts = origin + su[:, None] * u + sv[:, None] * v
    pts += rng.normal(0.0, sigma, n)[:, None] * normal
    return pts, normal, d0, u, v


class TestPlaneRecovery:
    def test_parameters_recovered(self):
        pts, n_true, d_true, *_ = make_plane_cloud()
        m = fit_plane(pts, dist_thresh=0.012)
        assert m is not None
        # orientation may flip (n, d) → (-n, -d); compare canonically
        dot = m.normal @ n_true
        assert abs(dot) > 0.99998          # < ~0.4° error
        d_est = m.d if dot > 0 else -m.d
        assert abs(d_est - d_true) < 0.001  # < 1 mm offset error

    def test_rms_matches_injected_noise(self):
        pts, *_ = make_plane_cloud()
        fs = fit_segment(pts, consolidate_method="none", label="synthetic", models=["plane"])
        assert fs is not None and fs.kind == "plane"
        rms = fs.report.stats.rms_mm
        assert SIGMA * 1000 * 0.9 < rms < SIGMA * 1000 * 1.1

    def test_white_noise_is_unstructured(self):
        pts, *_ = make_plane_cloud()
        fs = fit_segment(pts, consolidate_method="none", label="synthetic", models=["plane"])
        assert fs.report.stats.moran is not None
        assert not fs.report.stats.moran.structured
        assert fs.report.stats.flatness_pass is True


class TestStructuredResiduals:
    def test_double_layer_detected_as_structured(self):
        """Simulated registration bias: half the patch duplicated 8 mm off-plane
        (the 'onion layer' failure mode) → residual field MUST read structured."""
        pts, n_true, *_ = make_plane_cloud(n=40_000)
        ghost = pts[pts[:, 0] > np.median(pts[:, 0])] + 0.008 * n_true
        both = np.vstack([pts, ghost])
        fs = fit_segment(both, consolidate_method="none", label="double_layer", models=["plane"])
        assert fs is not None
        assert fs.report.stats.moran.structured

    def test_bulge_finding_detected(self):
        """A 12 mm gaussian bulge over ~0.5 m radius → one 'bulge' finding with
        position and magnitude, and structured residuals."""
        pts, n_true, d_true, u, v = make_plane_cloud(n=80_000)
        uv = np.column_stack([(pts + d_true * n_true) @ u, (pts + d_true * n_true) @ v])
        center = np.array([0.8, 0.4])
        r2 = ((uv - center) ** 2).sum(1)
        bump = 0.012 * np.exp(-r2 / (2 * 0.35 ** 2))
        pts_b = pts + bump[:, None] * n_true
        fs = fit_segment(pts_b, consolidate_method="none", label="bulge", models=["plane"],
                         finding_dev_mm=5.0, finding_min_area_m2=0.05)
        kinds = [f.kind for f in fs.report.findings]
        assert "bulge" in kinds or "depression" in kinds
        top = fs.report.findings[0]
        assert abs(abs(top.peak_dev_mm) - 12.0) < 5.0
        assert fs.report.stats.flatness_pass is False

    def test_plumb_finding_on_leaning_wall(self):
        """A wall built 6 mm/m out of plumb (desplome). The LS plane absorbs
        the lean into its normal — so plumb is read off the fitted normal vs
        world-up, and must be recovered within ~1 mm/m."""
        lean = 0.006                             # 6 mm horizontal drift per m climbed
        c = lean / np.sqrt(1 + lean * lean)      # normal·up of the leaning wall
        n_wall = np.array([np.sqrt(1 - c * c), 0.0, c])
        pts, *_ = make_plane_cloud(n=60_000, size=(6.0, 3.0),
                                   normal=n_wall, sigma=0.002)
        fs = fit_segment(pts, consolidate_method="none", label="wall", models=["plane"], tilt_mm_per_m=3.0)
        tilt = [f for f in fs.report.findings if f.kind == "tilt"]
        assert tilt, "leaning wall must produce a 'tilt' finding"
        assert abs(tilt[0].gradient_mm_per_m - 6.0) < 1.0
        # lean direction is horizontal and points the way the top drifts (+x here)
        assert abs(tilt[0].gradient_dir @ np.array([0.0, 0.0, 1.0])) < 0.05

    def test_plumb_wall_no_tilt_finding(self):
        """A truly vertical wall must NOT report desplome."""
        pts, *_ = make_plane_cloud(n=40_000, size=(6.0, 3.0),
                                   normal=(1.0, 0.0, 0.0), sigma=0.002)
        fs = fit_segment(pts, consolidate_method="none", label="wall", models=["plane"], tilt_mm_per_m=3.0)
        assert not [f for f in fs.report.findings if f.kind == "tilt"]


class TestMoransI:
    def test_white_noise_grid(self):
        uv = RNG.uniform(0, 5, (50_000, 2))
        vals = RNG.normal(0, 0.003, 50_000)
        res = morans_i(grid_scalar(uv, vals, 0.05))
        assert not res.structured
        assert abs(res.i) < 0.05

    def test_smooth_field_grid(self):
        uv = RNG.uniform(0, 5, (50_000, 2))
        vals = 0.01 * np.sin(uv[:, 0] * 2.0) + RNG.normal(0, 0.001, 50_000)
        res = morans_i(grid_scalar(uv, vals, 0.05))
        assert res.structured
        assert res.i > 0.5


class TestSupportTrim:
    def test_hole_stays_hole(self):
        """Points covering a square minus a 1 m disc → mesh must not cover the disc."""
        uv = RNG.uniform(-2, 2, (80_000, 2))
        hole = np.linalg.norm(uv, axis=1) < 1.0
        uv = uv[~hole]
        tm = trimmed_quad_mesh(uv, resolution=0.05, support_radius=0.08)
        assert len(tm.faces) > 0
        centers = tm.vertices_uv[tm.faces].mean(axis=1)
        # no face center deep inside the hole (allow the support-radius rim)
        assert (np.linalg.norm(centers, axis=1) > 1.0 - 0.08 - 0.05 * 1.5).all()
        # but the surrounding area IS meshed
        assert tm.area_m2 > (16 - np.pi) * 0.75

    def test_full_coverage_fraction(self):
        uv = RNG.uniform(0, 2, (40_000, 2))
        tm = trimmed_quad_mesh(uv, resolution=0.05, support_radius=0.08)
        assert tm.support_fraction > 0.95
