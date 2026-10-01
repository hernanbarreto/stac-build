"""The depth on F5 (USER 2026-10-01, pccr epoch 7): Omega's depth bent to F5's landmarks, then a vote.
Synthetic, no GPU: a plane seen by three cameras."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision.depth_on_f5 import bend_coefficients, camera_travels, design, vote  # noqa: E402

H, W = 60, 80
K = np.array([[70.0, 0, W / 2], [0, 70.0, H / 2], [0, 0, 1]])


def _cams():
    out = {}
    for f, x in zip((0, 1, 2), (-0.2, 0.0, 0.2)):
        c = np.eye(4); c[0, 3] = x; out[f] = c
    return out


def test_bend_recovers_a_scale_error_and_borrows_for_a_keyframe_without_landmarks():
    rng = np.random.default_rng(0)
    rows = {}
    for i, true in enumerate((0.9, 0.9, 1.1)):
        u = rng.uniform(0, W, 200); v = rng.uniform(0, H, 200)
        rows[i] = (design(u, v, W, H), np.full(200, true) + rng.normal(0, 1e-3, 200))
    rows[3] = (design(np.zeros(0), np.zeros(0), W, H), np.zeros(0))      # no landmark at all
    c = bend_coefficients(rows, 4, 0, 12, 1.345, 1e-12, 100)
    assert abs(c[0][0] - 0.9) < 1e-2 and abs(c[2][0] - 1.1) < 1e-2
    assert np.allclose(c[3], c[2]), "a keyframe without landmarks borrows its NEAREST fitted neighbour"


def test_vote_removes_a_flyer_and_keeps_the_surface():
    cams = _cams(); order = [0, 1, 2]
    dep = {f: np.full((H, W), 3.0, np.float32) for f in order}
    ent = {f: np.ones((H, W), bool) for f in order}
    dep[1][10:20, 30:40] = 1.5                       # a flyer in front of the wall the others see
    out = vote(dep, ent, K, cams, order, (-1, 1), 0.02)
    z1, a1 = out[1][0], out[1][1]
    assert (z1[10:20, 30:40] == 0).all(), "a point the neighbours see through leaves"
    assert np.allclose(z1[40:50, 30:40], 3.0, atol=1e-3) and (a1[40:50, 30:40] >= 1).all()


def test_the_camera_travels_with_the_cloud(tmp_path):
    out = tmp_path
    (out / "intrinsic.txt").write_text("363 364 232 416\n" * 3)
    prev = out / "_epoch_0"; prev.mkdir()
    (prev / "_manifest.json").write_text(json.dumps({"artifacts": [{"rel": "cleaned_cloud.ply", "existed_before": True}]}))
    camera_travels(out, 0, [391.9, 388.7, 234.8, 414.1, 0, 0, 0, 0], 3, log=lambda *a: None)
    assert (prev / "intrinsic.txt").read_text().startswith("363")
    assert "intrinsic.txt" in {a["rel"] for a in json.loads((prev / "_manifest.json").read_text())["artifacts"]}
    assert (out / "intrinsic.txt").read_text().splitlines()[0].startswith("391.9")


def test_chunk_check_applies_the_published_bend():
    """f6_check measures the cloud that was PUBLISHED: Omega's depth × k(u, v) of the bend report."""
    from precision.chunk_check import scale_map
    c = np.array([0.95, 0.04, -0.03])
    k = scale_map(c[0], (c[1], c[2]), H, W)
    rng = np.random.default_rng(1)
    u = rng.integers(0, W, 50); v = rng.integers(0, H, 50)
    assert np.allclose(k[v, u], design(u.astype(float), v.astype(float), W, H) @ c)
    assert scale_map(1.03, (0.0, 0.0), H, W) == 1.03
