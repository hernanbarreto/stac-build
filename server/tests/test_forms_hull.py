"""FORMS: the measured envelope — the visual hull of the masks carved by the TSDF of the object's
points (USER 2026-10-16: nothing sticks out of the silhouettes, the points put the depth where
they are, the hull is the surface where they are not). Synthetic scene on the CPU."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstruction.forms.hull import hull_mesh  # noqa: E402
from reconstruction.forms.silhouette import Silhouettes  # noqa: E402

DEV = "cpu"


def test_the_hull_spans_the_silhouettes_and_the_points_carve_the_depth():
    from test_forms_silhouette import _scene, _cylinder, SIGMA
    truth, cloud, views = _scene()          # true cylinder r 0.10, L 2.0; cloud r 0.095, L 1.6, noisy
    # the scene's cloud: the same deformed cylinder, dense enough for a depth per pixel
    with torch.no_grad():
        dense = _cylinder(0.095, 0.8).samples(1200000).numpy()      # ~a point per pixel, like the scene cloud
    dense = dense + np.random.default_rng(1).normal(0, SIGMA, dense.shape)
    sil = Silhouettes(views, SIGMA, 1.345, DEV)
    V, F, rec = hull_mesh(cloud, sil.batches, voxel=0.015, trunc=0.04, fill_px=2, min_votes=2,
                          min_inside_frac=0.5, sigma=SIGMA, dev=DEV,
                          scene=torch.as_tensor(dense, dtype=torch.float32), session_key="t")
    assert len(V) > 500 and len(F) > 500 and not rec.get("empty")
    # the length comes back from the silhouettes (the cloud only had 1.6 m)
    y = V[:, 1]
    assert np.percentile(y, 99.5) - np.percentile(y, 0.5) == pytest.approx(2.0, abs=0.06)
    # nothing sticks out of the six silhouettes' hull: the radius stays within the truth + a voxel
    r = np.hypot(V[:, 0], V[:, 2])
    assert np.percentile(r, 99) < 0.10 / np.cos(np.pi / 6) + 0.02       # a hexagonal prism of 6 views
    # where the points measured the surface the radius is theirs (0.095), inside the truth
    mid = np.abs(y - 1.0) < 0.6
    assert np.median(r[mid]) == pytest.approx(0.095, abs=0.012)
    assert rec["hull_voxels"] > 1000 and rec["voxel_m"] == pytest.approx(0.015)


def test_without_a_view_the_hull_is_empty():
    rng = np.random.default_rng(0)
    P = rng.normal(0, 0.1, (500, 3))
    sil = Silhouettes([], 0.01, 1.345, DEV)
    V, F, rec = hull_mesh(P, sil.batches, voxel=0.02, trunc=0.04, fill_px=1, min_votes=2,
                          min_inside_frac=0.5, sigma=0.01, dev=DEV)
    assert len(V) == 0 and rec.get("empty")
