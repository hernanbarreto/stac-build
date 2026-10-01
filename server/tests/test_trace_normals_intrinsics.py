"""reconstruction/trace_normals: the intrinsics are read per keyframe in both
layouts, the poses can come from the cloud's own (warped) directory, and the
finite difference at a crease / occluding edge stays on the pixel's own face.

Edge audit 2026-10-01 #10: ``_intrinsics_for`` read the first 9 numbers of the
4-column ``fx fy cx cy`` intrinsic.txt as one 3x3 (cy came out as the second
keyframe's fy — pccr 363.6 for 416, an 8 degree shear) and used keyframe 0's K
for every keyframe; the forward difference gave the last pixel before a crease
or a depth step the tangent of the other surface.

Synthetic, no GPU: a tilted plane seen by three keyframes with DIFFERENT
intrinsics and rotations, plus single-frame crease / step depth maps.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstruction import trace_normals as TN   # noqa: E402

H, W = 60, 80
# per-keyframe intrinsics on the depth grid: fx, fy, cx, cy all differ, and no
# keyframe's fy equals its cy (the old reader took K[1,2] from row 1's fy)
INTR = np.array([[70.0, 72.0, 41.0, 28.0],
                 [64.0, 66.0, 38.5, 31.0],
                 [78.0, 75.0, 40.0, 29.5]])
FRAMES = [10, 20, 30]
N_CAM = np.array([0.35, 0.55, -1.0]) / np.linalg.norm([0.35, 0.55, -1.0])   # faces the camera
D_PLANE = float(N_CAM @ np.array([0.0, 0.0, 3.0]))


def _rot(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


ROTS = [np.eye(3), _rot("y", 25.0), _rot("x", -15.0) @ _rot("y", -10.0)]


def _K(row):
    fx, fy, cx, cy = row
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


def _plane_depth(K, n=N_CAM, d=D_PLANE, h=H, w=W):
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    ray = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    return d / (ray @ n)


def _session(tmp: Path, intrinsic_text: str, poses_rot=ROTS) -> Path:
    """A session with maplong_run aligned depth (one chunk), poses, frames and
    the given intrinsic.txt."""
    out = tmp / "output"
    run = out / "maplong_run"
    (run / "_tmp_results_aligned").mkdir(parents=True)
    (run / "frame_list.json").write_text(json.dumps([f"frame_{f:06d}.jpg" for f in FRAMES]))
    depth = np.stack([_plane_depth(_K(r)) for r in INTR]).astype(np.float32)
    np.save(run / "_tmp_results_aligned" / "chunk_0.npy", {"depth": depth}, allow_pickle=True)
    rows = []
    for R in poses_rot:
        M = np.eye(4)
        M[:3, :3] = R
        rows.append(" ".join(f"{x:.12f}" for x in M.reshape(-1)))
    (out / "camera_poses.txt").write_text("\n".join(rows) + "\n")
    (out / "camera_frames.txt").write_text("\n".join(str(f) for f in FRAMES) + "\n")
    (out / "intrinsic.txt").write_text(intrinsic_text)
    return out


def _probe():
    """Interior pixels of every keyframe (away from the image border)."""
    rr, cc = np.mgrid[5:H - 5:7, 5:W - 5:7]
    rr, cc = rr.ravel(), cc.ravel()
    fg = np.concatenate([np.full(len(rr), f) for f in FRAMES]).astype(np.int64)
    return fg, np.tile(rr, len(FRAMES)).astype(np.int64), np.tile(cc, len(FRAMES)).astype(np.int64)


def _angle_deg(a, b):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.degrees(np.arccos(np.clip(np.abs((a * b).sum(-1)), -1, 1)))


def _expected(fg, rots=ROTS):
    return np.stack([rots[FRAMES.index(int(f))] @ N_CAM for f in fg])


def _four_col():
    return "\n".join(" ".join(f"{x}" for x in r) for r in INTR) + "\n"


def test_four_column_rows_give_each_keyframe_its_own_camera(tmp_path):
    """FAILS on the old reader: 12 numbers reshaped as one 3x3 (cy = row 1's
    fy, keyframe 0's fx for every keyframe) shear the plane by degrees."""
    out = _session(tmp_path, _four_col())
    fg, pr, pc = _probe()
    n = TN.normals_from_trace(np.zeros((len(fg), 3)), fg, pr, pc, out)
    assert n is not None
    err = _angle_deg(n.astype(np.float64), _expected(fg))
    assert err.max() < 0.2, f"max normal error {err.max():.3f} deg"


def test_nine_value_rows_and_the_legacy_matrix_parse(tmp_path):
    nine = "\n".join(" ".join(f"{x}" for x in _K(r).reshape(-1)) for r in INTR) + "\n"
    out = _session(tmp_path / "a", nine)
    fg, pr, pc = _probe()
    n = TN.normals_from_trace(np.zeros((len(fg), 3)), fg, pr, pc, out)
    assert _angle_deg(n.astype(np.float64), _expected(fg)).max() < 0.2

    # the legacy single 3x3 (three rows of three) serves every keyframe
    Ks = TN._read_intrinsic_rows(out / "intrinsic.txt")
    assert Ks.shape == (3, 3, 3) and np.allclose(Ks[1], _K(INTR[1]))
    legacy = tmp_path / "legacy.txt"
    legacy.write_text("\n".join(" ".join(f"{x}" for x in row) for row in _K(INTR[0])) + "\n")
    one = TN._read_intrinsic_rows(legacy)
    assert one.shape == (1, 3, 3) and np.allclose(one[0], _K(INTR[0]))
    four = TN._read_intrinsic_rows(_session(tmp_path / "b", _four_col()) / "intrinsic.txt")
    assert np.allclose(four[2], _K(INTR[2]))


def test_unreadable_or_unattributable_intrinsics(tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_text("1 2 3 4 5\n6 7 8 9 10\n")
    with pytest.raises(ValueError, match="fx fy cx cy"):
        TN._read_intrinsic_rows(bad)
    # two rows for three keyframes: no K is borrowed from another keyframe
    out = _session(tmp_path / "s", "\n".join(" ".join(map(str, r)) for r in INTR[:2]) + "\n")
    assert TN._intrinsics_for(out, FRAMES) is None
    fg, pr, pc = _probe()
    assert TN.normals_from_trace(np.zeros((len(fg), 3)), fg, pr, pc, out) is None


def test_poses_dir_rotates_by_the_staged_poses(tmp_path):
    """An epoch transaction stages warped poses next to the warped cloud; the
    normals must follow them, not the session's pre-warp rotation."""
    out = _session(tmp_path, _four_col())
    warp = _rot("y", 12.0)
    staged = tmp_path / "tx"
    staged.mkdir()
    rows = []
    for R in ROTS:
        M = np.eye(4)
        M[:3, :3] = warp @ R
        rows.append(" ".join(f"{x:.12f}" for x in M.reshape(-1)))
    (staged / "camera_poses.txt").write_text("\n".join(rows) + "\n")
    fg, pr, pc = _probe()
    n = TN.normals_from_trace(np.zeros((len(fg), 3)), fg, pr, pc, out, poses_dir=staged)
    want = _expected(fg, [warp @ R for R in ROTS])
    assert _angle_deg(n.astype(np.float64), want).max() < 0.2


# ── one-sided differences at a crease and at an occluding edge ──────────

def _depth_two_planes(K, n_a, d_a, n_b, d_b, c0):
    za = _plane_depth(K, n_a, d_a)
    zb = _plane_depth(K, n_b, d_b)
    u = np.arange(W)[None, :].repeat(H, 0)
    return np.where(u < c0, za, zb), u < c0


@pytest.mark.parametrize("kind", ["crease", "step"])
def test_the_pixels_beside_an_edge_keep_their_own_face(kind):
    K = _K(INTR[0])
    n_a = np.array([0.6, 0.1, -1.0]); n_a /= np.linalg.norm(n_a)
    n_b = np.array([-0.6, 0.1, -1.0]); n_b /= np.linalg.norm(n_b)
    c0 = 40
    if kind == "crease":
        # the two planes meet on the ray through column c0 - 0.5: a roof ridge
        v, u = np.mgrid[0:H, 0:W].astype(np.float64)
        ray = np.array([(c0 - 0.5 - K[0, 2]) / K[0, 0], (H / 2 - K[1, 2]) / K[1, 1], 1.0])
        P = 3.0 * ray
        z, on_a = _depth_two_planes(K, n_a, float(n_a @ P), n_b, float(n_b @ P), c0)
    else:
        # plane A in front, plane B a metre behind: an occluding edge
        z, on_a = _depth_two_planes(K, n_a, float(n_a @ [0, 0, 2.0]), n_b,
                                    float(n_b @ [0, 0, 3.5]), c0)
    nmap = TN._normal_map_from_depth(z.astype(np.float32), K)
    want = np.where(on_a[..., None], n_a, n_b)
    err = _angle_deg(nmap.astype(np.float64), want)
    # every pixel, the two beside the edge and the image border included
    assert err.max() < 0.2, (kind, float(err.max()),
                             np.unravel_index(int(np.argmax(err)), err.shape))


def test_invalid_depth_does_not_poison_its_neighbours():
    """A pixel with no depth (z = 0) is never the side chosen for a valid one
    when the other side lies on its plane."""
    K = _K(INTR[0])
    z = _plane_depth(K).astype(np.float32)
    z[:, 30] = 0.0
    nmap = TN._normal_map_from_depth(z, K)
    err = _angle_deg(nmap[:, [29, 31]].astype(np.float64), N_CAM[None, None, :])
    assert err.max() < 0.2
