"""The per-chunk floor motion leaves no step at the chunk seams (USER 2026-10-01: "hay un escalón que antes no
existía"). pccr kf 62→63 jumped 15.0 cm (F5: 6.9 cm): each keyframe took its OWNER chunk's rigid motion.
Inside the overlap of two chunks the motion now goes gradually from one to the other."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from correction.floor import _blend_across_overlaps  # noqa: E402


def _setup(tmp_path):
    (tmp_path / "chunk_plan.json").write_text(json.dumps({"chunk_ranges": [[0, 40], [20, 60]]}))
    owner = np.array([0] * 30 + [1] * 30)
    R0 = Rotation.from_euler("x", 2.0, degrees=True).as_matrix(); t0 = np.array([0.0, -0.18, 0.0])
    R1 = Rotation.from_euler("x", -1.5, degrees=True).as_matrix(); t1 = np.array([0.02, -0.47, 0.01])
    R = np.stack([R0 if o == 0 else R1 for o in owner]); t = np.stack([t0 if o == 0 else t1 for o in owner])
    return SimpleNamespace(output_dir=tmp_path), owner, R, t, (R0, t0, R1, t1)


def test_no_step_at_the_seam(tmp_path):
    sess, owner, R, t, _ = _setup(tmp_path)
    before = np.abs(np.diff(t[:, 1])).max()
    R2, t2, n = _blend_across_overlaps(sess, owner, R, t)
    assert n == 20
    after = np.abs(np.diff(t2[:, 1])).max()
    assert before > 0.25 and after < before / 10, (before, after)


def test_outside_the_overlap_each_chunk_keeps_its_motion(tmp_path):
    sess, owner, R, t, (R0, t0, R1, t1) = _setup(tmp_path)
    R2, t2, _ = _blend_across_overlaps(sess, owner, R, t)
    assert np.allclose(t2[:20], t0) and np.allclose(R2[:20], R0)
    assert np.allclose(t2[40:], t1) and np.allclose(R2[40:], R1)
    assert np.allclose(t2[20], t0) and np.allclose(t2[39], t1)        # the ends of the overlap


def test_without_a_chunk_plan_nothing_changes(tmp_path):
    sess, owner, R, t, _ = _setup(tmp_path)
    (tmp_path / "chunk_plan.json").unlink()
    R2, t2, n = _blend_across_overlaps(sess, owner, R, t)
    assert n == 0 and np.array_equal(t2, t)
