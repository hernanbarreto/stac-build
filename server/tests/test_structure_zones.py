"""The structure probe's analysis (reconstruction/walk_structure.py) and the revisited zones made the owners of their
keyframes (reconstruction/zones.py) — USER 2026-11-05, the plan he approved: "1. Pasada de estructura … cada zona
revisitada es un chunk, con todos sus pedazos del recorrido". Synthetic, on the CPU, no Omega."""

import json

import numpy as np
import pytest

from reconstruction import walk_structure as WS
from reconstruction import zones as ZN

H, W = 24, 32
KMAT = np.array([[60.0, 0, 16.0], [0, 60.0, 12.0], [0, 0, 1.0]])


def _walk_out_and_back(n_out=50, n_back=10, step=0.1):
    """A camera sliding sideways along x in front of a wall 3 m away (footprint 1.6 m wide), out for n_out
    keyframes, then back at the start for n_back: the walk LEFT and CAME BACK."""
    xs = [step * i for i in range(n_out)] + [step * i for i in range(n_back)]
    c2w = np.repeat(np.eye(4)[None], len(xs), 0)
    c2w[:, 0, 3] = xs
    vv, uu = np.mgrid[0:H, 0:W]
    depth = np.stack([3.0 * (1.0 + 0.004 * np.sin(0.7 * uu + 1.3 * vv + 0.9 * k)) for k in range(len(xs))])
    return c2w, depth.astype(np.float32)


def test_a_walk_that_left_and_came_back_is_one_zone_of_two_runs(tmp_path):
    c2w, depth = _walk_out_and_back()
    S = len(c2w)
    npz = tmp_path / "probe.npz"
    np.savez(npz, c2w=c2w.astype(np.float32), K=np.repeat(KMAT[None], S, 0).astype(np.float32), depth=depth,
             conf=np.ones_like(depth))
    res = WS.analyse(npz, device="cpu", log=lambda m: None)
    assert res["revisits"]["geometric"] and not res["revisits"]["salad"]
    # the way out up to where its footprint (1.6 m) still reaches the way back (x ≤ 0.9 m), and the way back
    assert all(i < 26 and j >= 50 for i, j, _ in res["revisits"]["geometric"])
    assert len(res["zones"]) == 1
    runs = res["zones"][0]["runs"]
    assert len(runs) == 2 and runs[0][0] == 0 and runs[1][1] == S - 1
    assert not set(range(28, 48)) & set(res["zones"][0]["keyframes"])     # the corridor between is no zone
    assert np.all(np.asarray(res["local_run"]) >= 1)


def test_salad_is_the_second_witness_and_its_bar_comes_with_the_probe(tmp_path):
    K = 60
    c = np.zeros((K, K))
    for i in range(K):
        c[i, max(0, i - 3):i + 4] = 0.8
    sal = np.full((K, K), 0.1)
    for i in range(K):
        sal[i, max(0, i - 3):i + 4] = 0.9
    sal[5, 55] = sal[55, 5] = 0.8
    ell = WS.local_runs(c)
    rv = WS.revisits(c, ell, sal, 0.7)
    assert rv["geometric"] == [] and rv["salad"] == [(5, 55, pytest.approx(0.8))]
    assert WS.revisits(c, ell, sal, 0.85)["salad"] == []                  # under the bar: no witness


def test_zones_that_share_keyframes_are_one_zone():
    z = WS.zones_of([(0, 100, 0.5), (103, 200, 0.5)], 295)          # two revisits whose runs touch at 98–105
    assert z == [{"runs": [[0, 2], [98, 105], [198, 202]],
                  "keyframes": [0, 1, 2] + list(range(98, 106)) + list(range(198, 203))}]
    two = WS.zones_of([(10, 50, 0.5), (60, 120, 0.5)], 295)
    assert [x["runs"] for x in two] == [[[8, 12], [48, 52]], [[58, 62], [118, 122]]]
    keys = [set(x["keyframes"]) for x in two]
    assert not keys[0] & keys[1]                                        # ONE owner per keyframe


def _rot_y(deg):
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def _session(root, c2w_chain, depth, n_chunks_of):
    out = root / "output"
    rec = out / "omega_run" / "results_output"
    rec.mkdir(parents=True)
    (out / "maplong_run").mkdir()
    files = [f"{10 * i:06d}.jpg" for i in range(len(c2w_chain))]
    for g, f in enumerate(files):
        np.savez_compressed(rec / f"frame_{10 * g}.npz", depth=depth[g], pose_c2w=c2w_chain[g],
                            chunk=np.int64(n_chunks_of(g)), frame_global=np.int64(g),
                            conf=np.ones_like(depth[g]), K_omega=KMAT)
    lines = "\n".join(" ".join(repr(float(v)) for v in T.reshape(-1)) for T in c2w_chain) + "\n"
    for base in (out, out / "maplong_run"):
        (base / "camera_poses.txt").write_text(lines)
        (base / "camera_frames.txt").write_text("\n".join(str(10 * g) for g in range(len(files))) + "\n")
    return files


def test_the_zone_owns_its_keyframes_and_carries_the_revisit_as_one_reconstruction(tmp_path, monkeypatch):
    """The chain left the second visit 30 cm off (the duplicate); the zone's joint pass sees both visits in one
    geometry. After the zone the two visits keep the ZONE's relative geometry (the truth), its records carry its own
    chunk id, the camera poses of record follow them, the rest of the walk is untouched."""
    truth, depth = _walk_out_and_back()
    S = len(truth)
    chain = truth.copy()
    chain[50:, 0, 3] += 0.3                                              # the duplicate: the second visit 30 cm off
    files = _session(tmp_path, chain, depth, lambda g: 0 if g < 25 else (1 if g < 45 else 2))
    turn = ["000015.jpg", "000305.jpg", "000555.jpg"]                   # inside run 1, between, inside run 2
    zone = {"runs": [[0, 9], [50, 59]], "keyframes": list(range(10)) + list(range(50, 60))}
    # the zone's own frame: X_truth = s·R·X_zone + t
    s_g, R_g, t_g = 0.5, _rot_y(10.0), np.array([1.0, 2.0, 3.0])
    seen = {}

    def fake_pass(paths, resolution, out_npz, cfg_v, *, sky_cache, salad, log):
        seen["views"] = [p.split("/")[-1] for p in paths]
        c2w, dep = [], []
        for p in paths:
            n = int(p.split("/")[-1].split(".")[0])
            T = truth[n // 10].copy() if n % 10 == 0 else truth[n // 10].copy()
            Tz = np.eye(4)
            Tz[:3, :3] = R_g.T @ T[:3, :3]
            Tz[:3, 3] = R_g.T @ (T[:3, 3] - t_g) / s_g
            c2w.append(Tz)
            dep.append(depth[n // 10] / s_g)
        np.savez(out_npz, c2w=np.array(c2w, np.float32), K=np.repeat(KMAT[None], len(paths), 0).astype(np.float32),
                 depth=np.array(dep, np.float32), conf=np.ones((len(paths), H, W), np.float32))
        return out_npz
    monkeypatch.setattr(ZN, "run_frames_pass", fake_pass)
    rep = ZN.apply_zones(tmp_path, tmp_path / "frames", files, turn, [zone], {}, resolution=832, capacity=100,
                         n_chunks=3, sky_cache=None, log=lambda m: None)
    assert seen["views"] == sorted(files[0:10] + ["000015.jpg"] + files[50:60] + ["000555.jpg"],
                                   key=lambda f: int(f.split(".")[0]))      # the turn frame between runs stays out
    assert rep[0]["inferred"] and rep[0]["chunk"] == 3 and rep[0]["views"] == 22
    rec = {g: dict(np.load(tmp_path / "output/omega_run/results_output" / f"frame_{10 * g}.npz")) for g in range(S)}
    for g in range(S):
        assert int(rec[g]["chunk"]) == (3 if g in zone["keyframes"] else (0 if g < 25 else (1 if g < 45 else 2)))
    for g in range(10, 50):                                             # the rest of the walk: untouched
        np.testing.assert_array_equal(rec[g]["pose_c2w"], chain[g])
    # the two visits in the ZONE's relative geometry (the truth), not the chain's duplicate
    for a, b in [(0, 50), (5, 55), (9, 59)]:
        rel = np.linalg.inv(rec[a]["pose_c2w"]) @ rec[b]["pose_c2w"]
        np.testing.assert_allclose(rel, np.linalg.inv(truth[a]) @ truth[b], atol=2e-3)
        chain_rel = np.linalg.inv(chain[a]) @ chain[b]
        assert abs(chain_rel[0, 3] - rel[0, 3]) > 0.25
    # depth in the walk's units (the fitted scale undoes the zone's 0.5)
    np.testing.assert_allclose(rec[5]["depth"], depth[5], rtol=2e-3)
    # the camera poses of record follow the records (both copies)
    for base in ("output", "output/maplong_run"):
        lines = (tmp_path / base / "camera_poses.txt").read_text().split("\n")
        for g in (3, 20, 57):
            np.testing.assert_allclose(np.array(lines[g].split(), float).reshape(4, 4), rec[g]["pose_c2w"])
    doc = json.loads((tmp_path / "output" / "zones.json").read_text())
    assert doc["chunks_before_zones"] == 3 and doc["zones"][0]["correspondences"] >= 1000


def test_a_zone_the_card_cannot_hold_is_declared_and_not_inferred(tmp_path, monkeypatch):
    truth, depth = _walk_out_and_back()
    files = _session(tmp_path, truth, depth, lambda g: 0)
    before = (tmp_path / "output/camera_poses.txt").read_text()
    monkeypatch.setattr(ZN, "run_frames_pass", lambda *a, **k: pytest.fail("a zone over the card was inferred"))
    logs = []
    rep = ZN.apply_zones(tmp_path, tmp_path / "frames", files, [], [{"runs": [[0, 9], [50, 59]],
                         "keyframes": list(range(10)) + list(range(50, 60))}], {}, resolution=832, capacity=12,
                         n_chunks=1, sky_cache=None, log=logs.append)
    assert rep == [{"zone": 0, "runs": [[0, 9], [50, 59]], "keyframes": 20, "views": 20, "inferred": False,
                    "why": "20 views > 12 one pass holds at 832"}]
    assert any("NOT inferred together" in m and "declared" in m for m in logs)
    assert (tmp_path / "output/camera_poses.txt").read_text() == before
    for g in (0, 55):
        r = np.load(tmp_path / "output/omega_run/results_output" / f"frame_{10 * g}.npz")
        assert int(r["chunk"]) == 0


def test_the_zone_pass_holds_the_zone_keyframes_and_the_turn_frames_inside_its_runs():
    files = [f"{10 * i:06d}.jpg" for i in range(60)]
    v = ZN.zone_pass_frames({"runs": [[0, 2], [50, 52]], "keyframes": [0, 1, 2, 50, 51, 52]}, files,
                            ["000005.jpg", "000025.jpg", "000505.jpg", "000525.jpg", "000300.jpg"])
    assert v == ["000000.jpg", "000005.jpg", "000010.jpg", "000020.jpg", "000500.jpg", "000505.jpg",
                 "000510.jpg", "000520.jpg"]


def test_a_zone_is_only_what_the_chain_of_chunks_does_not_join_already(tmp_path):
    """Two pieces inside one chunk, or across one seam, are reconstructed together already; the zone is the revisit
    whose chunks share no frame (the chunk graph's definition, loop_utils/revisits.py) — pccr 2026-11-05: 6 zones from
    the probe alone, 1 (the start ↔ the end) against its plan."""
    ranges = [(0, 80), (41, 174), (80, 199), (174, 218), (199, 295)]
    assert WS.joined_by_chain(30, 70, ranges)                # one chunk
    assert WS.joined_by_chain(190, 219, ranges)              # across one seam (174–218 ∩ 199–295)
    assert WS.joined_by_chain(50, 150, ranges)               # chunk 41–174 holds both
    assert not WS.joined_by_chain(5, 290, ranges)            # the start and the end: no chunk between shares a frame
    c2w, depth = _walk_out_and_back()
    S = len(c2w)
    npz = tmp_path / "probe.npz"
    np.savez(npz, c2w=c2w.astype(np.float32), K=np.repeat(KMAT[None], S, 0).astype(np.float32), depth=depth,
             conf=np.ones_like(depth))
    apart = WS.analyse(npz, ranges=[(0, 30), (15, 45), (30, 60)], device="cpu", log=lambda m: None)
    assert len(apart["zones"]) == 1 and apart["zone_revisits"]
    together = WS.analyse(npz, ranges=[(0, 60)], device="cpu", log=lambda m: None)   # one pass holds the revisit
    assert together["zones"] == [] and together["revisits"]["geometric"]


def test_the_probe_analysis_never_touches_the_card_in_the_map_worker():
    """pccr 2026-11-05: the co-visibility on cuda inside the map worker's process made Omega's exclusive-card check
    (repro.require_exclusive_gpu) fail the run — the analysis runs on the CPU there."""
    import inspect
    src = inspect.getsource(ZN.structure_probe)
    assert 'WS.analyse(npz, ranges=ranges, device="cpu"' in src
