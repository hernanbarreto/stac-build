"""
Stage 2.2-2.4 + escalation acceptance tests on synthetic clouds with known
ground truth. The key property under test: the ladder stops at the LOWEST-DOF
model whose residuals are spatially white — planes stay planes, half-pipes
become cylinders (not B-splines), non-circular vaults become swept profiles,
gently warped sheets fall through to the B-spline.
"""
import numpy as np
import pytest

from reconstruction.surface_fit.runner import fit_segment

RNG = np.random.default_rng(7)
SIGMA = 0.003

LADDER = ["plane", "cylinder", "sphere", "swept_profile", "bspline"]


def make_halfpipe(r=2.5, length=12.0, n=120_000, sigma=SIGMA, rng=None):
    rng = rng or np.random.default_rng(11)
    """Vault: upper half-cylinder, axis = +x, crown up (+z)."""
    s = rng.uniform(-length / 2, length / 2, n)
    phi = rng.uniform(0, np.pi, n)                    # upper half
    rr = r + rng.normal(0, sigma, n)
    pts = np.column_stack([s, rr * np.cos(phi), rr * np.sin(phi)])
    return pts + np.array([10.0, 5.0, 2.0])


def make_dome(r=6.0, cap_deg=40.0, n=80_000, sigma=SIGMA, rng=None):
    rng = rng or np.random.default_rng(12)
    """Spherical cap around +z."""
    cosmin = np.cos(np.deg2rad(cap_deg))
    cosphi = rng.uniform(cosmin, 1.0, n)
    sinphi = np.sqrt(1 - cosphi ** 2)
    theta = rng.uniform(0, 2 * np.pi, n)
    rr = r + rng.normal(0, sigma, n)
    pts = np.column_stack([rr * sinphi * np.cos(theta),
                           rr * sinphi * np.sin(theta),
                           rr * cosphi])
    return pts + np.array([3.0, -2.0, 1.0])


def make_elliptic_vault(a=3.0, b=2.0, length=14.0, n=150_000, sigma=SIGMA, rng=None):
    rng = rng or np.random.default_rng(13)
    """Extruded NON-circular profile (ellipse, upper half) along +x — must
    reject plane/cylinder/sphere and land on swept_profile."""
    s = rng.uniform(-length / 2, length / 2, n)
    phi = rng.uniform(0, np.pi, n)
    rho = a * b / np.sqrt((b * np.cos(phi)) ** 2 + (a * np.sin(phi)) ** 2)
    rho = rho + rng.normal(0, sigma, n)
    return np.column_stack([s, rho * np.cos(phi), rho * np.sin(phi)])


def make_warped_sheet(n=120_000, sigma=SIGMA, rng=None):
    rng = rng or np.random.default_rng(14)
    """Gently free-form: 2 cm amplitude, metre-scale wavelength — quadrics and
    the swept profile can't explain it; the B-spline must."""
    x = rng.uniform(0, 8, n)
    y = rng.uniform(0, 6, n)
    z = 0.02 * np.sin(x / 1.5) * np.cos(y / 1.2) + rng.normal(0, sigma, n)
    return np.column_stack([x, y, z])


class TestCylinder:
    def test_halfpipe_escalates_to_cylinder(self):
        pts = make_halfpipe()
        fs = fit_segment(pts, consolidate_method="none", label="vault", models=LADDER)
        assert fs is not None and fs.kind == "cylinder"
        # a half-pipe offers no 30%-inlier plane, so the path starts at cylinder
        assert fs.escalation_path[-1] == "cylinder"
        assert abs(fs.params["radius"] - 2.5) < 0.01     # < 1 cm radius error
        axis = np.asarray(fs.params["axis_dir"])
        assert abs(axis @ np.array([1.0, 0, 0])) > 0.999
        assert not fs.report.stats.moran.structured
        rms = fs.report.stats.rms_mm
        assert SIGMA * 1000 * 0.85 < rms < SIGMA * 1000 * 1.15

    def test_mesh_is_supported(self):
        pts = make_halfpipe(n=60_000)
        fs = fit_segment(pts, consolidate_method="none", label="vault", models=LADDER)
        assert len(fs.mesh_vertices) > 0
        # every mesh vertex must lie near the cylinder surface AND near data
        assert fs.support_fraction > 0.9


class TestSphere:
    def test_dome_escalates_to_sphere(self):
        pts = make_dome()
        fs = fit_segment(pts, consolidate_method="none", label="dome", models=LADDER)
        assert fs is not None and fs.kind == "sphere"
        assert abs(fs.params["radius"] - 6.0) < 0.02
        center = np.asarray(fs.params["center"])
        assert np.linalg.norm(center - np.array([3.0, -2.0, 1.0])) < 0.02
        assert not fs.report.stats.moran.structured


class TestSweptProfile:
    def test_elliptic_vault_escalates_to_swept(self):
        pts = make_elliptic_vault()
        fs = fit_segment(pts, consolidate_method="none", label="tunnel", models=LADDER)
        assert fs is not None and fs.kind == "swept_profile"
        # sub-mm systematic wiggle may register statistically; what matters is
        # that no RELEVANT structure (vs construction tolerance) remains
        assert not fs.report.stats.moran.relevant
        assert fs.report.stats.rms_mm < SIGMA * 1000 * 1.5
        # profile axis recovered
        axis = np.asarray(fs.params["axis_dir"])
        assert abs(axis @ np.array([1.0, 0, 0])) > 0.995

    def test_profile_shape_recovered(self):
        """The fitted swept surface must contain the TRUE (noise-free) ellipse:
        signed distances of exact profile points ≈ 0."""
        from reconstruction.surface_fit.profile_sweep import fit_swept_profile
        pts = make_elliptic_vault()
        model = fit_swept_profile(pts)
        assert model is not None
        phi = np.linspace(np.deg2rad(5), np.pi - np.deg2rad(5), 200)
        rho = 3.0 * 2.0 / np.sqrt((2.0 * np.cos(phi)) ** 2 + (3.0 * np.sin(phi)) ** 2)
        for s in (-5.0, 0.0, 5.0):
            exact = np.column_stack([np.full_like(phi, s),
                                     rho * np.cos(phi), rho * np.sin(phi)])
            d = np.abs(model.signed_distance(exact))
            assert np.percentile(d, 95) < 0.005          # < 5 mm on the true surface


class TestBSpline:
    def test_warped_sheet_falls_to_bspline(self):
        pts = make_warped_sheet()
        fs = fit_segment(pts, consolidate_method="none", label="slab", models=LADDER)
        assert fs is not None and fs.kind == "bspline"
        assert fs.report.stats.rms_mm < SIGMA * 1000 * 1.3
        assert not fs.report.stats.moran.structured
        assert fs.dof > 50                                # genuinely free-form

    def test_bspline_is_lowpass(self):
        """With 50 cm control spacing the spline must NOT chase 10 cm ripples:
        residuals keep the ripple (structured), proving it can't overfit."""
        n = 120_000
        x = RNG.uniform(0, 8, n)
        y = RNG.uniform(0, 6, n)
        z = 0.008 * np.sin(x / 0.05) + RNG.normal(0, 0.001, n)   # 10 cm-scale ripple
        pts = np.column_stack([x, y, z])
        fs = fit_segment(pts, consolidate_method="none", label="ripple", models=["bspline"],
                         ctrl_point_spacing_m=0.5)
        assert fs is not None
        # the ripple survives in the residual → the spline did not invent it away
        assert fs.report.stats.rms_mm > 4.0


class TestEscalationDiscipline:
    def test_plane_stays_plane_with_full_ladder(self):
        rng = np.random.default_rng(3)
        u = rng.uniform(-2, 2, 50_000)
        v = rng.uniform(-1.5, 1.5, 50_000)
        pts = np.column_stack([u, v, rng.normal(0, SIGMA, 50_000)])
        fs = fit_segment(pts, consolidate_method="none", label="flat", models=LADDER)
        assert fs.kind == "plane"
        assert fs.escalation_path == ["plane"]

    def test_halfpipe_prefers_cylinder_over_bspline(self):
        """DOF discipline: a cylinder (5 DOF) explains a half-pipe; the ladder
        must not run on to the hundreds-of-DOF spline."""
        pts = make_halfpipe(n=80_000)
        fs = fit_segment(pts, consolidate_method="none", label="vault", models=LADDER)
        assert fs.kind == "cylinder"
        assert "bspline" not in fs.escalation_path
