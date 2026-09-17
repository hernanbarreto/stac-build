"""USER 2026-09-17, in front of the kit's Votes view on pccr: the red band is
not a colour, it is a decision — "la eliminacion quiero que sea real porque no
quiero que esos voladores se usen para computar nada, ni para comparar ni para
siquiera calcular el OBB".

Two things are pinned here: the COUNT the panel shows must be the cloud's own
mv_votes column (never an estimate), and what `witness.drop_statuses` removes
must be named by a measured status — never by a number invented at the call
site, and never `unobserved`, whose zero is the absence of a witness and not a
vote against the point (§6).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstruction.loops.config import LoopsConfigError, load_loops_config   # noqa: E402
from reconstruction.witness.histogram import (CACHE_NAME, below, cached,      # noqa: E402
                                              vote_histogram)
from reconstruction.witness.status import STATUS_CODES                        # noqa: E402
from tests.synth_metric import raw_server_cfg                                 # noqa: E402


def _write_ply(path: Path, mv: np.ndarray, st: np.ndarray) -> None:
    """A minimal cloud carrying the two columns the histogram reads, with the
    same property order and types the pipeline writes."""
    n = len(mv)
    rec = np.zeros(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                             ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                             ("confidence", "<f4"), ("frame_global", "<i4"),
                             ("pixel_row", "<i2"), ("pixel_col", "<i2"),
                             ("mv_votes", "u1"), ("mask_votes", "u1"),
                             ("mask_conflicts", "u1"), ("status", "u1")])
    rec["x"] = np.arange(n, dtype=np.float32)
    rec["mv_votes"] = mv
    rec["status"] = st
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "property float confidence\nproperty int frame_global\n"
              "property short pixel_row\nproperty short pixel_col\n"
              "property uchar mv_votes\nproperty uchar mask_votes\n"
              "property uchar mask_conflicts\nproperty uchar status\n"
              "end_header\n").encode("ascii")
    path.write_bytes(header + rec.tobytes())


def _cloud(tmp_path: Path, counts, status_of=None) -> Path:
    """counts[v] points with mv_votes == v; status from the session's own rule
    (observed and below the rule's minimum -> single_witness)."""
    mv = np.concatenate([np.full(c, v, np.uint8) for v, c in enumerate(counts)])
    if status_of is None:
        st = np.where(mv < 2, STATUS_CODES["single_witness"],
                      STATUS_CODES["verified"]).astype(np.uint8)
    else:
        st = status_of(mv)
    p = tmp_path / "cleaned_cloud.ply"
    _write_ply(p, mv, st)
    return p


# ── the count is the cloud's own column ──────────────────────────────────

def test_the_histogram_counts_every_point_exactly(tmp_path):
    h = vote_histogram(_cloud(tmp_path, [11, 23, 47, 5, 90]))
    assert h["n_points"] == 176
    assert h["counts"] == [11, 23, 47, 5, 90]
    assert sum(h["counts"]) == h["n_points"]          # nothing sampled, nothing lost


def test_a_threshold_takes_everything_strictly_below_it(tmp_path):
    """The shader paints red when votes < threshold; the count must use the
    SAME comparison or the panel would promise a different cloud."""
    h = vote_histogram(_cloud(tmp_path, [11, 23, 47, 5, 90]))
    assert below(h, 0)["points"] == 0                 # nothing is below zero
    assert below(h, 1)["points"] == 11
    assert below(h, 2)["points"] == 11 + 23
    assert below(h, 5)["points"] == 176
    b = below(h, 2)
    assert b["remaining"] == 176 - 34
    assert b["fraction"] == pytest.approx(34 / 176)


def test_the_split_by_status_travels_with_the_count(tmp_path):
    """Without it the panel cannot tell a point measured AGAINST from a point
    never measured at all — and only the first is removed."""
    def mixed(mv):
        st = np.where(mv < 2, STATUS_CODES["single_witness"],
                      STATUS_CODES["verified"]).astype(np.uint8)
        st[:4] = STATUS_CODES["unobserved"]           # four zero-vote unobserved
        return st
    h = vote_histogram(_cloud(tmp_path, [11, 23, 47, 5, 90], status_of=mixed))
    row0 = next(r for r in h["table"] if r["votes"] == 0)
    assert row0["status"]["unobserved"] == 4
    assert row0["status"]["single_witness"] == 7
    assert h["status_counts"]["unobserved"] == 4


def test_a_cloud_without_witnesses_says_so(tmp_path):
    p = tmp_path / "cleaned_cloud.ply"
    rec = np.zeros(3, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
    p.write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 3\n"
                  b"property float x\nproperty float y\nproperty float z\nend_header\n"
                  + rec.tobytes())
    with pytest.raises(RuntimeError, match="no mv_votes"):
        vote_histogram(p)
    assert cached(tmp_path) is None


def test_the_cache_follows_the_cloud_and_not_the_clock(tmp_path):
    """Every stage that rewrites the cloud changes its size or mtime; a cache
    that survived one of those would report a cloud that no longer exists."""
    p = _cloud(tmp_path, [4, 6, 10])
    first = cached(tmp_path)
    assert first["n_points"] == 20
    assert (tmp_path / CACHE_NAME).exists()
    again = cached(tmp_path)
    assert again["source"] == first["source"]          # unchanged cloud, cache reused
    _write_ply(p, np.zeros(7, np.uint8), np.full(7, STATUS_CODES["single_witness"], np.uint8))
    assert cached(tmp_path)["n_points"] == 7           # rewritten cloud, recounted


# ── what is removed is a status, never a number ──────────────────────────

def test_drop_statuses_defaults_to_the_single_witness_band():
    w = load_loops_config().witness
    assert "single_witness" in w.drop_statuses
    assert "unobserved" not in w.drop_statuses, \
        "a point with no witness at all is not evidence against itself (§6)"


def test_an_unknown_status_fails_at_load_naming_it():
    with pytest.raises(LoopsConfigError, match="floaters"):
        load_loops_config(raw_server_cfg(**{"witness.drop_statuses": ["floaters"]}))


def test_a_status_cannot_be_both_removed_and_thinned():
    """clean_statuses is what the voxel/SOR net may thin. Listing a status in
    both is a contradiction the config must refuse instead of resolving by
    order of evaluation."""
    with pytest.raises(LoopsConfigError, match="drop_statuses"):
        load_loops_config(raw_server_cfg(**{"witness.drop_statuses": ["verified"]}))


def test_an_empty_drop_list_removes_nothing():
    cfg = load_loops_config(raw_server_cfg(**{"witness.drop_statuses": []}))
    assert cfg.witness.drop_statuses == ()
