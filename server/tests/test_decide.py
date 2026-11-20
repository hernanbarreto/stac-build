"""The decision by zone (precision/decide.py) on a session written to disk: the scene of tests/test_coherence.py as
keyframe records (in units of their own, 1.25 x the state's), frames, the first pass's map and a camera. CPU."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_coherence as TC                                  # noqa: E402
from precision import coherence as CO                        # noqa: E402
from precision import decide as DZ                           # noqa: E402
from precision.camera import CameraModel, GridMap            # noqa: E402

N, H, W = TC.N, TC.H, TC.W
REC_UNITS = 1.25
FRAMES = list(range(100, 100 + N))
QUIET = lambda *a: None  # noqa: E731


def _camera():
    g = GridMap("omega", W, H, W, H, 0, 0, 0, 0, W, H, W, H)
    return CameraModel(width=W, height=H, params=tuple(TC.K) + tuple(TC.LENS), source="synthetic", camera_epoch=0,
                       omega_grid=g)


@pytest.fixture(scope="module")
def judged(tmp_path_factory):
    from PIL import Image
    d = tmp_path_factory.mktemp("session")
    rdir = d / "output" / CO.RECORDS_REL
    rdir.mkdir(parents=True)
    (d / "frames").mkdir()
    (d / "intake").mkdir()
    depth, rgb, c2w = TC._arrays()
    for k, f in enumerate(FRAMES):
        pose = c2w[k].copy()
        pose[:3, 3] *= REC_UNITS
        np.savez(rdir / f"frame_{f}.npz", depth=(depth[k] * REC_UNITS).astype(np.float32),
                 conf=np.full((H, W), 5.0, np.float32), chunk=np.int64(k // 6), pose_c2w=pose)
        Image.fromarray(rgb[k]).save(d / "frames" / f"{f:06d}.jpg", quality=98)
    (d / "output" / "camera_frames.txt").write_text(" ".join(str(f) for f in FRAMES) + "\n")
    (d / "intake" / "cubes.json").write_text(json.dumps(
        {"kf_covisibility": {"pairs": [[100 + k, 100 + k + 6, 0.8, 0.7] for k in range(6)]}}))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(CO, "params", lambda pcfg, raw, device=None: TC.CFG)
        judge = DZ.Judge(d, None, {}, _camera(), log=QUIET)
    return judge, np.linalg.inv(c2w), d


def _moved(w2c, cams, dx):
    c2w = np.linalg.inv(w2c)
    c2w[list(cams), 0, 3] += dx
    return np.linalg.inv(c2w)


def test_the_session_is_read_with_its_zones_its_units_and_its_pairs(judged):
    judge, w2c, _d = judged
    assert judge.zone_of.tolist() == [0] * 6 + [1] * 6 and judge.maps.frames == tuple(FRAMES)
    assert (0, 6) in judge.pairs and (0, 1) in judge.pairs
    s = judge.state("truth", w2c, _camera())
    assert abs(float(s.factor[0]) - 1.0 / REC_UNITS) < 1e-12        # the records' depth brought to the cameras' units
    rep = CO.summary(judge.measure(s))
    for kind in rep.values():
        assert kind["along_cm"]["median"] < 0.05 and kind["across_cm"]["median"] < 0.3, rep


def test_a_candidate_that_is_not_worse_anywhere_is_accepted(judged):
    judge, w2c, _d = judged
    called = []
    res = DZ.decide_stage(judge, "stage", ("before", w2c, _camera()), ("candidate", w2c.copy(), _camera()),
                          lambda held: called.append(held), log=QUIET)
    assert res["outcome"] == DZ.ACCEPTED and res["attempt"] == 1 and res["held"] is None and not called
    assert res["records"][0]["zones"]["worse"] == [] and res["records"][0]["outcome"] == DZ.ACCEPTED


def test_a_zone_that_got_worse_is_held_and_the_stage_solved_again(judged):
    judge, w2c, _d = judged
    bad = _moved(w2c, range(6, 12), 0.04)                 # the second pass displaced 4 cm sideways: every pair of
    #                                                       different passes says so (six judges, his rule needs five)
    asked = []

    def again(held):
        asked.append(held.copy())
        return "candidate, zones held", w2c.copy(), _camera()
    res = DZ.decide_stage(judge, "stage", ("before", w2c, _camera()), ("candidate", bad, _camera()), again, log=QUIET)
    assert len(asked) == 1 and asked[0].dtype == bool and asked[0][6:].all()      # zone 1's keyframes are held
    assert res["outcome"] == DZ.ACCEPTED_HELD and res["attempt"] == 2 and np.array_equal(res["held"], asked[0])
    first, second = res["records"]
    assert 1 in first["zones"]["worse"] and first["outcome"] is None
    assert second["zones"]["worse"] == [] and second["zones_held"] == first["zones"]["worse"]
    assert judge.maps.depth.device.type == "cpu"          # the maps are back after the solve


def test_still_worse_with_the_zones_held_discards_the_stage(judged):
    judge, w2c, _d = judged
    bad = _moved(w2c, range(6, 12), 0.04)
    res = DZ.decide_stage(judge, "stage", ("before", w2c, _camera()), ("candidate", bad, _camera()),
                          lambda held: ("the same again", bad, _camera()), log=QUIET)
    assert res["outcome"] == DZ.DISCARDED and res["attempt"] is None and res["held"] is None
    assert [r["outcome"] for r in res["records"]] == [None, DZ.DISCARDED]
    res = DZ.decide_stage(judge, "stage", ("before", w2c, _camera()), ("candidate", bad, _camera()), None, log=QUIET)
    assert res["outcome"] == DZ.DISCARDED and len(res["records"]) == 1 and res["records"][0]["outcome"] == DZ.DISCARDED


def test_the_decisions_are_written_per_stage_and_a_stage_replaces_its_own(judged):
    judge, w2c, d = judged
    res = DZ.decide_stage(judge, "stage", ("before", w2c, _camera()), ("candidate", w2c.copy(), _camera()), None,
                          log=QUIET)
    p = DZ.write_decisions(d / "output", "f5_refine", res["records"])
    DZ.write_decisions(d / "output", "another", [])
    DZ.write_decisions(d / "output", "f5_refine", res["records"] * 2)
    doc = json.loads(p.read_text())
    assert doc["version"] == DZ.VERSION and doc["provenance"] == "tool_measured"
    assert sorted(doc["stages"]) == ["another", "f5_refine"] and len(doc["stages"]["f5_refine"]) == 2
    assert doc["stages"]["f5_refine"][0]["depth_scale"]["candidate"] == pytest.approx(1.0 / REC_UNITS)


def test_the_scale_of_the_cameras_against_the_records_is_the_ratio_of_their_steps():
    rng = np.random.default_rng(0)
    rec = np.tile(np.eye(4), (9, 1, 1))
    rec[:, :3, 3] = np.cumsum(rng.uniform(0.01, 0.2, (9, 3)), 0)
    st = rec.copy()
    st[:, :3, 3] = rec[:, :3, 3] * 0.9546 + np.array([3.0, -1.0, 2.0])
    assert abs(DZ.camera_scale(st, rec) - 0.9546) < 1e-12
    bent = st.copy()
    bent[5:, :3, 3] += np.array([0.02, 0.0, 0.0])           # one camera step changed: not one body any more
    with pytest.raises(DZ.DecideError, match="not the records' cameras moved as one body"):
        DZ.camera_scale(bent, rec)
    rec2, st2 = rec.copy(), st.copy()
    rec2[4, :3, 3] = rec2[3, :3, 3]                         # two record cameras at the same place: that step is
    st2[4, :3, 3] = st2[3, :3, 3]                           # left out
    assert abs(DZ.camera_scale(st2, rec2) - 0.9546) < 1e-12
    with pytest.raises(DZ.DecideError, match="do not move"):
        DZ.camera_scale(st, np.tile(np.eye(4), (9, 1, 1)))
