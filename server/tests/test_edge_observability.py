"""USER 2026-09-17: a loop edge may only CARRY the DOF it observes.

`info_t` was already built from the observability spec, so the pose graph was
told the edge observes one direction — and then handed a `Z` built from the
UNPROJECTED Sim3 fit, which says something else entirely.

A Sim3 between two copies of a near-symmetric object lands on rotations drift
cannot produce: pccr measured 165.3 deg on desk#201 and 40.9 deg on the floor,
over a 19.3 m walk. A rotation about a distant centre moves the object 60 cm
and a camera 10 m away by TEN METRES, so the edge told the graph a keyframe had
to move metres to close a 60 cm duplicate. kf_graph reads that as
|(inv(Z) @ Zc).t| / walked — which is where the "drift rates" from 4 to 110
cm/m came from, why the consensus was comparing nonsense, and why the graph
closed 0% of the loop in every run.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from correction.solve import project_solution                     # noqa: E402


def _desk201():
    """The real closure: 165.3 deg, |t| = 10 m, moves its copy ~60 cm."""
    ax = np.array([0.2, 0.95, 0.24]); ax /= np.linalg.norm(ax)
    R = Rotation.from_rotvec(np.deg2rad(165.3) * ax).as_matrix()
    c = np.array([4.0, 1.2, 7.8])
    t = np.array([0.55, 0.05, 0.25]) + c - R @ c
    return R, t, c


def _Z(R, t, Ti, Tj):
    X = np.eye(4); X[:3, :3] = R; X[:3, 3] = t
    return np.linalg.inv(Ti) @ np.linalg.inv(X) @ Tj


def _discrepancy(Z, Ti, Tj):
    """What kf_graph absorbs: the measured relative pose against the chain's."""
    Zc = np.linalg.inv(Ti) @ Tj
    return float(np.linalg.norm((np.linalg.inv(Z) @ Zc)[:3, 3]))


def test_the_raw_fit_asks_a_distant_camera_to_move_metres():
    """The trap, stated once: the object moves 60 cm, the camera 30 m."""
    R, t, c = _desk201()
    pts = c + np.random.default_rng(0).normal(0, 0.35, (2000, 3))
    assert np.median(np.linalg.norm(pts @ R.T + t - pts, axis=1)) < 1.1
    Ti = np.eye(4); Tj = np.eye(4); Tj[:3, 3] = [-5.0, 0.0, -6.0]
    assert _discrepancy(_Z(R, t, Ti, Tj), Ti, Tj) > 10.0


def test_projecting_about_the_copy_brings_the_edge_back_to_its_measurement():
    R, t, c = _desk201()
    Rp, tp = project_solution(R, t, {"mode": "perp_axis",
                                     "axis": np.array([1.0, 0.0, 0.15])}, about=c)
    Ti = np.eye(4); Tj = np.eye(4); Tj[:3, 3] = [-5.0, 0.0, -6.0]
    d = _discrepancy(_Z(Rp, tp, Ti, Tj), Ti, Tj)
    assert d < 0.6, f"{d * 100:.1f} cm"


def test_the_projected_edge_carries_no_rotation():
    """A rotation the object cannot observe must not reach the graph at all."""
    R, t, c = _desk201()
    Rp, _tp = project_solution(R, t, {"mode": "normal",
                                      "normal": np.array([0.0, 1.0, 0.0])}, about=c)
    assert np.allclose(Rp, np.eye(3))


def test_the_drift_rate_becomes_physical():
    """kf_graph divides the discrepancy by the walked distance. Over 19.3 m the
    raw edge reads as metres per metre; the projected one as centimetres."""
    R, t, c = _desk201()
    Ti = np.eye(4); Tj = np.eye(4); Tj[:3, 3] = [-5.0, 0.0, -6.0]
    walked = 19.3
    raw = _discrepancy(_Z(R, t, Ti, Tj), Ti, Tj) / walked
    Rp, tp = project_solution(R, t, {"mode": "perp_axis",
                                     "axis": np.array([1.0, 0.0, 0.15])}, about=c)
    proj = _discrepancy(_Z(Rp, tp, Ti, Tj), Ti, Tj) / walked
    assert raw > 1.0, f"{raw:.2f} m/m"
    assert proj < 0.05, f"{proj * 100:.1f} cm/m"


def test_a_full_observability_edge_is_left_exactly_as_measured():
    """Nothing is projected when the object determines all six DOF."""
    R, t, _c = _desk201()
    R2, t2 = project_solution(R, t, {"mode": "full"})
    assert np.allclose(R2, R) and np.allclose(t2, t)


def test_the_edge_builder_projects_before_it_builds_Z():
    src = (Path(__file__).resolve().parents[1] / "reconstruction" / "certify"
           / "loops_posthoc.py").read_text()
    body = src[src.index("def instance_edges("):src.index("def visit_edges(")]
    i_proj = body.index("solve.project_solution")
    i_Z = body.index("Z = np.linalg.inv(Ti) @ np.linalg.inv(X) @ Tj")
    assert i_proj < i_Z, "the projection has to happen BEFORE Z is built"
    assert 'about=np.asarray(m["centroid_a"]' in body, \
        "projecting without the copy's centroid re-introduces the raw-t trap"
