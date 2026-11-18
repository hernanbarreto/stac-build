"""Common points: the same physical point marked on two scans, and the similarity they give.

USER 2026-11-17: "deberían marcarse dos puntos y decir estos son los mismos en ambas … y que se
ajusten a uno de ellos, o a una medida externa … para que se puedan posar uno sobre otro y comparar".
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fusion import common_points as CP  # noqa: E402


def moved(points, scale, yaw_deg, shift):
    th = math.radians(yaw_deg)
    c, n = math.cos(th), math.sin(th)
    R = np.array([[c, 0.0, -n], [0.0, 1.0, 0.0], [n, 0.0, c]])
    return scale * (np.asarray(points) @ R.T) + np.asarray(shift)


def test_the_similarity_of_two_levelled_scans_is_recovered():
    rng = np.random.default_rng(3)
    scan = rng.uniform(-10, 10, (6, 3)) * [1.0, 0.15, 1.0]                   # points of a room: metres across, little in height
    ref = moved(scan, 0.9725, 37.5, [4.0, -0.3, 12.0])
    fit = CP.similarity(scan, ref)
    assert abs(fit["scale"] - 0.9725) < 1e-12 and abs(fit["yaw_deg"] - 37.5) < 1e-10
    assert np.allclose(fit["translation"], [4.0, -0.3, 12.0], atol=1e-10) and fit["rms_m"] < 1e-10 and fit["pairs"] == 6
    M = np.array(fit["matrix"]).reshape(4, 4)
    assert np.allclose((np.c_[scan, np.ones(6)] @ M.T)[:, :3], ref, atol=1e-10)   # the matrix is scan → reference
    assert fit["scale_source"] == "points" and fit["baseline_m"] == pytest.approx(max(np.linalg.norm(a - b) for a in ref for b in ref))
    # the other way round: the inverse factor and turn
    back = CP.similarity(ref, scan)
    assert abs(back["scale"] - 1 / 0.9725) < 1e-12 and abs(back["yaw_deg"] + 37.5) < 1e-10


def test_two_points_fix_the_plan_exactly_and_leave_the_height_they_disagree_by():
    scan = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    ref = moved(scan, 1.04, -90.0, [1.0, 2.0, 3.0]) + [[0, 0.02, 0], [0, -0.02, 0]]   # 2 cm of height each way
    fit = CP.similarity(scan, ref)
    assert abs(fit["scale"] - 1.04) < 1e-12 and abs(fit["yaw_deg"] + 90.0) < 1e-10
    res = np.array(fit["residuals"])
    assert np.allclose(res[:, [0, 2]], 0.0, atol=1e-12)                       # nothing left in plan
    assert np.allclose(res[:, 1], [0.02, -0.02], atol=1e-12) and abs(fit["max_m"] - 0.02) < 1e-12


def test_three_points_with_a_marking_error_leave_residuals():
    scan = np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0], [0.0, 0.0, 4.0], [6.0, 1.0, 4.0]])
    ref = moved(scan, 1.0, 10.0, [0, 0, 0])
    ref[2] += [0.05, 0.0, 0.0]                                                # one point marked 5 cm off
    fit = CP.similarity(scan, ref)
    assert 0.01 < fit["rms_m"] < 0.05 and 0.02 < fit["max_m"] < 0.05 and int(np.argmax(fit["residual_m"])) == 2
    assert abs(fit["scale"] - 1.0) < 0.01 and abs(fit["yaw_deg"] - 10.0) < 0.5


def test_a_scale_given_from_outside_is_kept():
    """"o a una medida externa": the factor comes from a measurement; the points give the turn and the shift."""
    scan = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
    ref = moved(scan, 1.03, 20.0, [1, 0, 1])
    fit = CP.similarity(scan, ref, scale=1.0)
    assert fit["scale"] == 1.0 and fit["scale_source"] == "given" and abs(fit["yaw_deg"] - 20.0) < 1e-10
    assert 0.05 < fit["max_m"] < 0.15                                         # the 3 % it does not absorb, on 5 m
    for bad in (0, -1.0, float("nan"), True):
        with pytest.raises(CP.CommonPointsError, match="positive number"):
            CP.similarity(scan, ref, scale=bad)


def test_points_that_fix_nothing_are_said():
    p = np.array([[1.0, 0.0, 2.0], [1.0, 0.0, 2.0]])
    with pytest.raises(CP.CommonPointsError, match="coincide"):
        CP.similarity(p, p + 1.0)
    vertical = np.array([[1.0, 0.0, 2.0], [1.0, 3.0, 2.0]])
    with pytest.raises(CP.CommonPointsError, match="one vertical"):
        CP.similarity(vertical, vertical)
    with pytest.raises(CP.CommonPointsError, match="at least 2 points"):
        CP.similarity(p[:1], p[:1])


def test_points_are_kept_marked_per_scan_and_solved_between_two(tmp_path):
    A, B, C3 = "2026-08-24/default", "2026-08-31/default", "2026-09-07/default"
    true = dict(scale=0.98, yaw_deg=-12.0, shift=[3.0, 0.1, -7.0])
    in_b = np.array([[0.0, 0.0, 0.0], [8.0, 0.2, 1.0], [2.0, 1.5, 9.0]])       # marked on scan B …
    in_a = moved(in_b, true["scale"], true["yaw_deg"], true["shift"])          # … the same points as scan A has them
    ids = []
    for k in range(3):
        p = CP.add(tmp_path)
        ids.append(p["id"])
        assert p["name"] == f"P{k + 1}" and p["marks"] == {}
        CP.mark(tmp_path, p["id"], B, in_b[k].tolist(), geometry_epoch=4, author="ana")
    with pytest.raises(CP.CommonPointsError, match="0 point.s. marked on both"):
        CP.solve(tmp_path, B, A)
    CP.mark(tmp_path, ids[0], A, in_a[0].tolist(), geometry_epoch=2)
    CP.mark(tmp_path, "P2", A, in_a[1].tolist(), geometry_epoch=2)            # by its name too
    two = CP.solve(tmp_path, B, A)
    assert two["pairs"] == 2 and two["determined"].startswith("exactly") and two["marked_on_one_only"] == ["P3"] and two["applied"] is False
    CP.mark(tmp_path, ids[2], A, in_a[2].tolist(), geometry_epoch=2)
    fit = CP.solve(tmp_path, B, A)                                            # scan B brought onto scan A
    assert fit["pairs"] == 3 and fit["determined"] == "least squares" and [p["name"] for p in fit["points"]] == ["P1", "P2", "P3"]
    assert abs(fit["scale"] - 0.98) < 1e-12 and abs(fit["yaw_deg"] + 12.0) < 1e-10 and fit["rms_m"] < 1e-10
    # a mark again replaces it; a third scan's marks do not enter
    CP.mark(tmp_path, ids[0], C3, [9, 9, 9])
    assert CP.solve(tmp_path, B, A)["pairs"] == 3
    # the geometry of scan A moved since: its marks are of the earlier one — listed, not used
    stale = CP.listing(tmp_path, {A: 3, B: 4})
    assert all(p["marks"][A]["stale"] for p in stale) and not any(p["marks"][B]["stale"] for p in stale)
    with pytest.raises(CP.CommonPointsError, match="3 more were marked on an earlier geometry"):
        CP.solve(tmp_path, B, A, {A: 3, B: 4})
    # unmark, delete, names
    CP.unmark(tmp_path, ids[2], A)
    assert CP.solve(tmp_path, B, A)["pairs"] == 2
    with pytest.raises(KeyError):
        CP.unmark(tmp_path, ids[2], A)
    assert CP.delete(tmp_path, "P1")["id"] == ids[0]
    assert CP.add(tmp_path, "the door's corner")["name"] == "the door's corner" and CP.add(tmp_path)["name"] == "P5"
    with pytest.raises(CP.CommonPointsError, match="exists already"):
        CP.add(tmp_path, "P2")
    with pytest.raises(CP.CommonPointsError, match="ANOTHER scan"):
        CP.solve(tmp_path, A, A)
    with pytest.raises(CP.CommonPointsError, match="three finite numbers"):
        CP.mark(tmp_path, "P2", A, [0, 1])
    assert json.loads((tmp_path / CP.FILE).read_text())["version"] == 1 and not list(tmp_path.glob("*.part"))


# ── the routes ─────────────────────────────────────────────────────────────────────────────────
@pytest.fixture()
def client(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import common_points_api as R
    from correction.epoch import EPOCH_FILE
    project = tmp_path / "projects" / "site"
    outs = {}
    for scan, epoch in (("2026-08-24/default", 2), ("2026-08-31/default", 4)):
        day, source = scan.split("/")
        outs[scan] = project / "scans" / day / f"src_{source}" / "output"
        outs[scan].mkdir(parents=True)
        (outs[scan] / EPOCH_FILE).write_text(json.dumps({"epoch": epoch}))

    class Ctx:
        def __init__(self, scan):
            self.project_dir, self.output_dir, self.session_dir = project, outs[scan], outs[scan].parent

    def resolve(session_id, scan=None):
        if session_id == "old":                                               # a session of the old layout: no project
            d = tmp_path / "old"
            d.mkdir(exist_ok=True)
            return type("Legacy", (), {"session_dir": d, "output_dir": d / "output"})()
        if session_id != "site":
            return type("Missing", (), {"session_dir": tmp_path / "nope", "output_dir": tmp_path / "nope" / "output"})()
        if scan is not None and scan not in outs:
            raise FileNotFoundError(f"no scan {scan}")
        return Ctx(scan or "2026-08-24/default")

    R.configure(resolve)
    app = FastAPI()
    app.include_router(R.router)
    return TestClient(app), outs, project


def test_common_points_through_the_routes(client):
    c, outs, project = client
    A, B = "2026-08-24/default", "2026-08-31/default"
    base = "/api/project/site/common_points"
    assert c.get(base).json() == {"ok": True, "points": []}
    assert c.get("/api/project/nowhere/common_points").status_code == 404
    assert c.get("/api/project/old/common_points").status_code == 409        # one scan: nothing to lay on another
    in_b = np.array([[0.0, 0.0, 0.0], [8.0, 0.2, 1.0], [2.0, 1.5, 9.0]])
    in_a = moved(in_b, 0.98, -12.0, [3.0, 0.1, -7.0])
    for k in range(3):
        r = c.post(base, json={})
        pid = r.json()["point"]
        assert [p["name"] for p in r.json()["points"]][-1] == f"P{k + 1}"
        assert c.put(f"{base}/{pid}/marks", json={"scan": B, "position": in_b[k].tolist()}).status_code == 200
        r = c.put(f"{base}/P{k + 1}/marks", json={"scan": A, "position": in_a[k].tolist()})
        assert r.status_code == 200 and r.json()["points"][k]["marks"][A]["geometry_epoch"] == 2      # the scan's own epoch
    assert c.put(f"{base}/P1/marks", json={"scan": "2030-01-01/x", "position": [0, 0, 0]}).status_code == 404
    assert c.put(f"{base}/P1/marks", json={"scan": A, "position": [0, 0]}).status_code == 400
    assert c.put(f"{base}/P9/marks", json={"scan": A, "position": [0, 0, 0]}).status_code == 404
    fit = c.get(f"{base}/solve", params={"scan": B, "reference": A}).json()
    assert fit["ok"] and fit["applied"] is False and fit["pairs"] == 3
    assert abs(fit["scale"] - 0.98) < 1e-12 and abs(fit["yaw_deg"] + 12.0) < 1e-10 and fit["rms_m"] < 1e-10
    given = c.get(f"{base}/solve", params={"scan": B, "reference": A, "scale": 1.0}).json()
    assert given["scale"] == 1.0 and given["scale_source"] == "given" and given["max_m"] > 0.05
    assert not (project / "project.json").exists()                            # reported: nothing of the project is written
    # the reference scan's geometry moves on: its marks are of the earlier one
    from correction.epoch import EPOCH_FILE
    (outs[A] / EPOCH_FILE).write_text(json.dumps({"epoch": 3}))
    assert all(p["marks"][A]["stale"] for p in c.get(base).json()["points"])
    r = c.get(f"{base}/solve", params={"scan": B, "reference": A})
    assert r.status_code == 409 and "earlier geometry" in r.json()["detail"]
    # unmark, delete
    assert c.delete(f"{base}/P3/marks", params={"scan": A}).status_code == 200
    assert c.delete(f"{base}/P3/marks", params={"scan": A}).status_code == 404
    assert [p["name"] for p in c.delete(f"{base}/P1").json()["points"]] == ["P2", "P3"]
    assert c.delete(f"{base}/P1").status_code == 404
