"""tools/genrecon/presence.py: the keyframes' vote keeps a surface several keyframes measure, drops a point they see
through, and the cubes cover the whole of what stays (USER 2026-11-05)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "genrecon"))
from presence import plan_layout, vote_points  # noqa: E402

W, H, F = 64, 48, 60.0
INTR = [F, F, W / 2, H / 2, 0.0, 0.0, 0.0, 0.0]


def _cam(x):
    """A camera at (x, 0, 0) looking down +z (OpenCV)."""
    c2w = np.eye(4)
    c2w[0, 3] = x
    return c2w


def _wall_points(c2w, k, z=5.0):
    """The pixels of a wall at depth z as keyframe k measured them (each pixel's own point)."""
    rows, cols = np.mgrid[0:H, 0:W]
    x = (cols.ravel() + 0.5 - W / 2) / F * z
    y = (rows.ravel() + 0.5 - H / 2) / F * z
    P = np.stack([x, y, np.full_like(x, z)], 1) + c2w[:3, 3]
    return P, np.full(len(P), k), rows.ravel(), cols.ravel()


def test_a_wall_stays_and_a_floating_point_goes():
    cams = [_cam(x) for x in (-0.3, 0.0, 0.3)]
    Ps, fr, rr, cc = [], [], [], []
    for k, c in enumerate(cams):
        P, f, r, col = _wall_points(c, k)
        Ps.append(P); fr.append(f); rr.append(r); cc.append(col)
    # one point floating 2 m in front of the wall, born in keyframe 1 at its centre pixel
    Ps.append(np.array([[0.0, 0.0, 3.0]])); fr.append(np.array([1])); rr.append(np.array([H // 2])); cc.append(np.array([W // 2]))
    xyz = np.concatenate(Ps); frames = np.concatenate(fr); rows = np.concatenate(rr); cols = np.concatenate(cc)
    # the floater replaces keyframe 1's own wall pixel there (one point per pixel)
    own = np.flatnonzero((frames == 1) & (rows == H // 2) & (cols == W // 2))[0]
    keep_idx = np.ones(len(xyz), bool); keep_idx[own] = False
    xyz, frames, rows, cols = xyz[keep_idx], frames[keep_idx], rows[keep_idx], cols[keep_idx]
    out = vote_points(xyz, frames, rows, cols, [0, 1, 2], np.stack(cams), INTR, W, H, voxel=0.05,
                      min_views=2, block=1, device="cpu", log=lambda m: None)
    floater = len(xyz) - 1
    assert not out["keep"][floater]
    assert out["through"][floater] >= 2
    centre_wall = np.flatnonzero((np.abs(xyz[:, 0]) < 0.3) & (np.abs(xyz[:, 1]) < 0.3) & (xyz[:, 2] > 4.9))
    assert out["keep"][centre_wall].all()


def test_the_cubes_cover_every_point_on_one_lattice():
    rng = np.random.default_rng(0)
    P = rng.uniform([-1.0, -2.0, -0.9], [7.5, 3.0, 4.2], size=(20000, 3))
    L = plan_layout(P, 2.85, log=lambda m: None)
    C = np.asarray(L["centers"]); s = L["chunk_size"]
    inside = np.zeros(len(P), bool)
    for c in C:
        inside |= np.all(np.abs(P - c) <= s / 2, axis=1)
    assert inside.all()
    corners = (C - s / 2 - np.asarray(L["origin"])) / L["voxel"]
    assert np.allclose(corners, np.round(corners), atol=1e-6)
    assert (L["cells"] >= 0).all()
