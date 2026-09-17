"""The per-frame correction field is SHARED — no stage may overwrite it.

The bug this guards (pccr 2026-09-16, the first run where a loop actually
closed): the pose graph composes its closure into `_stac_elastic_corr` and
moves the POINTS in the npy directly; the elastic stage then rebuilt that field
from scratch, so the closure stayed in the points and vanished from the
cameras. Points and cameras ended up the size of the closure apart (29 cm
median, 38 cm max), `intra_chunk` measured a disagreement that did not exist
(held-out 2-5 cm historically → 8-53 cm), "repaired" it with corrections up to
129 cm, and the warped depth failed the global scale check by 18%.

It stayed invisible for months because every run rejected its loop edges:
the closure was identity, and overwriting identity loses nothing.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pytest

_VENDOR = Path("/workspace/stac-build/vendor/VGGT-Long")
sys.path.insert(0, str(_VENDOR))

from loop_utils.metric_lock import compose_frame_fields  # noqa: E402


def _field(n, t):
    """n frames all translated by t."""
    M = np.tile(np.eye(4), (n, 1, 1))
    M[:, :3, 3] = np.asarray(t, float)
    return M


class TestCompose:
    def test_the_earlier_stage_survives(self):
        prev = {0: _field(3, [0.30, 0, 0])}          # the closure
        new = {0: _field(3, [0.05, 0, 0])}           # the elastic move
        out = compose_frame_fields(prev, new)
        assert np.allclose(out[0][:, :3, 3], [0.35, 0, 0]), out[0][0]

    def test_order_is_new_on_top_of_old(self):
        """new @ prev — the same convention the pose graph uses when it
        composes into the field (X[g] @ ecorr[k][local])."""
        R = np.eye(4)
        R[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)  # +90° yaw
        prev = {0: np.tile(_field(1, [1.0, 0, 0])[0], (1, 1, 1))}
        new = {0: np.tile(R, (1, 1, 1))}
        out = compose_frame_fields(prev, new)
        expected = R @ prev[0][0]
        assert np.allclose(out[0][0], expected)
        assert not np.allclose(out[0][0], prev[0][0] @ R), "composed backwards"

    def test_no_earlier_field_is_just_the_new_one(self):
        new = {0: _field(2, [0.1, 0, 0])}
        assert np.allclose(compose_frame_fields(None, new)[0], new[0])
        assert np.allclose(compose_frame_fields({}, new)[0], new[0])

    def test_a_chunk_only_the_earlier_field_mentions_is_kept(self):
        prev = {0: _field(2, [0.2, 0, 0]), 1: _field(2, [0.4, 0, 0])}
        out = compose_frame_fields(prev, {0: _field(2, [0.1, 0, 0])})
        assert np.allclose(out[1][:, :3, 3], [0.4, 0, 0])

    def test_identity_composes_to_identity(self):
        """Why it hid for so long: with a rejected closure prev IS identity."""
        prev = {0: np.tile(np.eye(4), (4, 1, 1))}
        new = {0: _field(4, [0.07, 0, 0])}
        assert np.allclose(compose_frame_fields(prev, new)[0], new[0])


class TestTheStageDoesNotOverwrite:
    """Source guard: the elastic stage must compose, never reset the field."""

    SRC = (_VENDOR / "vggt_long.py").read_text()

    def test_the_destructive_assignment_is_gone(self):
        block = self.SRC[self.SRC.index("def _stac_elastic_seams"):
                         self.SRC.index("def _stac_aligned_pose")]
        assert not re.search(r"^\s*self\._stac_elastic_corr = \{\}\s*$", block, re.M), \
            "the elastic stage resets the shared per-frame field again"
        assert "compose_frame_fields" in block

    def test_the_pose_graph_still_composes_into_the_shared_field(self):
        assert "ecorr[k][local] = X[g] @ ecorr[k][local]" in self.SRC

    def test_the_camera_helper_reads_the_shared_field(self):
        block = self.SRC[self.SRC.index("def _stac_aligned_pose"):
                         self.SRC.index("def _stac_intra_chunk")]
        assert "_stac_elastic_corr" in block and "ecorr[k][local] @ M" in block
