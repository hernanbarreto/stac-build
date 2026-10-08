"""The scale / orientation markers KNOW what they transformed (reconstruction/applied_marker.py,
scale_align.run, orient.run — docs/plan_determinismo.md point 10) and the poses they write are
float64 round-trip exact with their backups refreshed on every fresh pass (points 45, 10)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import repro                                                    # noqa: E402
from reconstruction import applied_marker as AM                 # noqa: E402
from reconstruction import orient, scale_align                  # noqa: E402


def _poses(n=3, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        T = np.eye(4)
        q = rng.normal(size=3)
        a = np.linalg.norm(q)
        k = q / a
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        T[:3, :3] = np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K
        T[:3, 3] = rng.normal(size=3) * 1.37
        out.append(T)
    return np.array(out)


def _read(p):
    return np.array([[float(x) for x in ln.split()] for ln in p.read_text().splitlines()
                     if len(ln.split()) == 16]).reshape(-1, 4, 4)


def _out(tmp_path, poses):
    out = tmp_path / "output"
    (out / "maplong_run").mkdir(parents=True)
    repro.write_poses_exact(out / "camera_poses.txt", poses)
    repro.write_poses_exact(out / "maplong_run" / "camera_poses.txt", poses)
    return out


def test_record_and_state(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    a, b = out / "a.txt", out / "b.txt"
    a.write_text("A"); b.write_text("B")
    m = out / ".m"
    assert AM.state(m, base=out) == ("absent", {})
    side = AM.record(m, "s=1.5", [a, b, out / "missing.txt"], base=out, extra={"s": 1.5})
    assert side == out / ".m.sha256.json" and m.read_text() == "s=1.5\n"
    doc = json.loads(side.read_text())
    assert set(doc["files"]) == {"a.txt", "b.txt"} and doc["extra"] == {"s": 1.5}
    st, info = AM.state(m, base=out)
    assert st == "current" and info["changed"] == [] and info["gone"] == [] and info["text"] == "s=1.5"
    b.unlink()                                                  # deleted by a cleanup: not a difference
    st, info = AM.state(m, base=out)
    assert st == "current" and info["gone"] == ["b.txt"]
    a.write_text("A2")                                          # rewritten: STALE, named
    st, info = AM.state(m, base=out)
    assert st == "stale" and info["changed"] == ["a.txt"]
    assert "a.txt" in AM.describe(st, info)
    side.unlink()                                               # a legacy marker: unstamped, declared
    st, info = AM.state(m, base=out)
    assert st == "unstamped" and "cannot be verified" in AM.describe(st, info)
    AM.clear(m)
    assert AM.state(m, base=out)[0] == "absent"
    side.write_text("not json")
    m.write_text("s=1\n")
    assert AM.state(m, base=out)[0] == "stale"


def test_apply_scale_writes_exact_poses_and_refreshes_the_backup(tmp_path):
    P = _poses()
    out = _out(tmp_path, P)
    scale_align.apply_scale(out, 1.37)
    want = P.copy(); want[:, :3, 3] *= 1.37
    for base in (out, out / "maplong_run"):
        got = _read(base / "camera_poses.txt")
        assert np.array_equal(got, want), "float64 round-trip exact, bit for bit"
        assert np.array_equal(_read(base / "camera_poses.txt.prescale"), P)
    # a second FRESH pass over other poses refreshes the backup (it used to keep the first)
    P2 = _poses(seed=1)
    repro.write_poses_exact(out / "camera_poses.txt", P2)
    scale_align.apply_scale(out, 2.0)
    assert np.array_equal(_read(out / "camera_poses.txt.prescale"), P2)
    # the lossy pose writer is gone (the ASCII-PLY branch is another writer, not a pose one)
    assert ':.8g}" for x in m.reshape(-1)' not in Path(scale_align.__file__).read_text()
    assert "repro.exact_row(m.reshape(-1))" in Path(scale_align.__file__).read_text()


def test_scale_run_honours_a_current_marker_and_redoes_a_stale_one(tmp_path, monkeypatch):
    P = _poses()
    out = _out(tmp_path, P)
    calls = []
    monkeypatch.setattr(scale_align, "estimate_v2",
                        lambda od, **k: (calls.append("est") or (1.5, {"scale_source": "da3"})))
    logs = []
    s = scale_align.run(out, log=logs.append)
    assert s == 1.5 and calls == ["est"]
    marker = out / ".metric_scale_applied"
    assert marker.read_text() == "s=1.5\n"                      # exact text, parseable by its readers
    assert float(marker.read_text().strip().split("=")[-1]) == 1.5
    doc = json.loads(AM.sidecar_path(marker).read_text())
    assert set(doc["files"]) == {"camera_poses.txt", "maplong_run/camera_poses.txt"}
    scaled = _read(out / "camera_poses.txt")
    # a resume: the products are the transformed ones → reused, nothing estimated or re-scaled
    assert scale_align.run(out, log=logs.append) == 1.5 and calls == ["est"]
    assert np.array_equal(_read(out / "camera_poses.txt"), scaled)
    assert any("marker current" in m for m in logs)
    # the cleanup deleted a recorded file: still current
    (out / "maplong_run" / "camera_poses.txt").unlink()
    assert scale_align.run(out, log=logs.append) == 1.5 and calls == ["est"]
    # a fresh fork pass rewrote the poses in Omega's raw frame: STALE → redone from them
    repro.write_poses_exact(out / "camera_poses.txt", P)
    repro.write_poses_exact(out / "maplong_run" / "camera_poses.txt", P)
    assert scale_align.run(out, log=logs.append) == 1.5 and calls == ["est", "est"]
    assert any("STALE" in m and "camera_poses.txt" in m for m in logs)
    assert np.array_equal(_read(out / "camera_poses.txt"), scaled)
    # a legacy marker without the sidecar is reused and DECLARED
    AM.sidecar_path(marker).unlink()
    assert scale_align.run(out, log=logs.append) == 1.5 and calls == ["est", "est"]
    assert any("predates the sha record" in m for m in logs)


def test_orient_run_honours_a_current_marker_and_redoes_a_stale_one(tmp_path, monkeypatch):
    P = _poses()
    out = _out(tmp_path, P)
    down = np.array([0.0, 0.0, 1.0])
    calls = []
    monkeypatch.setattr(orient, "estimate_gravity",
                        lambda od, log=None: (calls.append("g") or (down, 0.99)))
    logs = []
    T = orient.run(out, log=logs.append)
    assert calls == ["g"] and T.shape == (4, 4) and not np.allclose(T, np.eye(4))
    marker = out / orient.MARKER_NAME
    assert marker.exists() and AM.sidecar_path(marker).exists()
    oriented = _read(out / "camera_poses.txt")
    assert np.array_equal(oriented, np.stack([T @ p for p in P])), "exact (the same matmul)"
    assert np.array_equal(_read(out / "camera_poses.txt.preorient"), P)
    assert np.allclose(orient.run(out, log=logs.append), np.eye(4)) and calls == ["g"]
    assert np.array_equal(_read(out / "camera_poses.txt"), oriented)
    repro.write_poses_exact(out / "camera_poses.txt", P)
    repro.write_poses_exact(out / "maplong_run" / "camera_poses.txt", P)
    T2 = orient.run(out, log=logs.append)
    assert calls == ["g", "g"] and np.array_equal(T2, T)
    assert any("STALE" in m for m in logs)
    assert np.array_equal(_read(out / "camera_poses.txt"), oriented)
    src = Path(orient.__file__).read_text()
    assert ':.8g}" for x in m.reshape(-1)' not in src and "repro.exact_row(m.reshape(-1))" in src


def test_the_markers_readers_still_parse_the_text():
    """The sidecar keeps the marker's text as it was: every reader of `.metric_scale_applied`
    parses float(text.split('=')[-1])."""
    server = Path(__file__).resolve().parents[1]
    for name in ("precision/gauge.py", "reconstruction/loops/kf_graph.py"):
        assert 'split("=")[-1]' in (server / name).read_text(), name
    assert repro.exact_float(0.984577) == "0.984577" and float("s=0.984577".split("=")[-1]) == 0.984577
