"""Everything after the cube graph in ONE read of every cube (USER 2026-11-19: "No entiendo por qué siguen habiendo
tantas partes que no usan la placa … es tiempo perdido" / the design: "Dije que si al diseño, dale").

cube_run.CubeRun.close() with Model.cube_closing = single_pass against the step-by-step record (apply → uncertainty →
depth graph → outputs → poses, then the SERVER's per-keyframe records from the placed cubes): every product the
same — the point files and uncertainty maps byte for byte, the origins and records array by array — with each cube
read once and no placed cube written."""
import ast
import functools
import json
import os
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

FORK = Path(os.environ.get("STAC_FORK_DIR") or Path(__file__).resolve().parents[2] / "vendor" / "VGGT-Long")
SERVER = Path(os.environ.get("STAC_SERVER_DIR") or (Path(__file__).resolve().parents[1]
                                                     if (Path(__file__).resolve().parents[1] / "repro.py").exists()
                                                     else "/workspace/stac-build/server"))
for q in (str(FORK), str(SERVER)):
    if q not in sys.path:
        sys.path.insert(0, q)
from loop_utils import cube_run as CR  # noqa: E402
from loop_utils import metric_lock as ML  # noqa: E402
from loop_utils import stac_repro as SR  # noqa: E402

if not os.path.isfile(os.path.join(SR.SERVER_DIR, "repro.py")):       # a fork checked out beside the repo
    SR.SERVER_DIR = str(SERVER)

H, W, S, K_CUBES, STEP = 48, 64, 8, 4, 2
N = STEP * (K_CUBES - 1) + S                                            # 14 keyframes, a frame in up to 4 cubes
STAMP = "c" * 64
SRC = (FORK / "vggt_long.py").read_text()


def _method(name):
    cls = next(n for n in ast.parse(SRC).body if isinstance(n, ast.ClassDef) and n.name == "VGGT_Long")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return textwrap.dedent(ast.get_source_segment(SRC, fn))


class _VL:
    """What the cube run takes from the fork's VGGT_Long: its thresholds run from their source, the rest plain."""

    def __init__(self, out, cfg):
        self.output_dir, self.pcd_dir = str(out), str(out / "pcd")
        self.result_unaligned_dir, self.result_aligned_dir = str(out / "un"), str(out / "al")
        for d in (self.pcd_dir, self.result_unaligned_dir, self.result_aligned_dir):
            os.makedirs(d, exist_ok=True)
        self.config = {"Model": cfg}
        self.img_list = [f"/frames/{g * 5:06d}.jpg" for g in range(N)]
        self._stac_cube_sets = [np.arange(k * STEP, k * STEP + S, dtype=np.int64) for k in range(K_CUBES)]
        holders = [[k for k in range(K_CUBES) if g in self._stac_cube_sets[k]] for g in range(N)]
        self._stac_cube_owner = np.asarray([h[len(h) // 2] for h in holders], np.int64)
        self.chunk_context = [[] for _ in range(K_CUBES)]
        self.reads, self.authority = [], []
        ns = {"np": np, "os": os}
        for nm in ("_stac_conf_threshold", "_stac_pose_conf_floor"):
            exec(_method(nm), ns)
            setattr(self, nm, (lambda f: (lambda *a, **k: f(self, *a, **k)))(ns[nm]))

    def _stac_iter_chunks(self, directory, ks):
        for k in ks:
            self.reads.append((os.path.basename(directory), int(k)))
            yield k, np.load(os.path.join(directory, f"chunk_{k}.npy"), allow_pickle=True).item()

    def _stac_graph_cfg(self):
        return {"heldout_confidence": 0.95, "improvement_error_factor": 1.1}

    def _stac_authority_cfg(self):
        return {"depth_graph_max_log_a": 0.5, "depth_graph_max_b_m": 1.0, "saturation_warn": 0.8}

    def _stac_authority_record(self, stage, fraction, saturated, extra=None):
        self.authority.append((stage, float(fraction), bool(saturated), json.dumps(extra, sort_keys=True)))

    def _stac_stamp_digest(self):
        return STAMP

    def _stac_with_stamp(self, d):
        d["fork_stamp"] = STAMP
        return d

    def _stac_check_fork_stamp(self, d, what):
        if d.get("fork_stamp", d.get("_stac_fork_stamp")) != STAMP:
            raise RuntimeError(f"{what}: another stamp")


def _apply_sim3_direct(point_maps, s, R, t):                              # (loop_utils.sim3utils', which needs numba)
    return s * np.matmul(R, point_maps[..., np.newaxis]).squeeze(-1) + t


def _save_ply(points, colors, confs, output_path, conf_threshold, sample_ratio=1.0):
    c = np.asarray(confs).reshape(-1).astype(np.float32)
    m = (c >= conf_threshold) & (c > 1e-5)
    with open(output_path, "wb") as f:
        f.write(f"ply {int(m.sum())}\n".encode())
        f.write(np.asarray(points, np.float32).reshape(-1, 3)[m].tobytes())
        f.write(np.asarray(colors, np.uint8).reshape(-1, 3)[m].tobytes())


def _rot(rng, mag):
    w = rng.normal(size=3) * mag
    th = np.linalg.norm(w)
    Kx = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]]) / th
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * (Kx @ Kx)


def _scene(tmp_path, tag, cfg_extra=None):
    """Four cubes of eight keyframes along a walk past a slanted wall, each in its own frame and unit (the pose that
    places it is known), each copy of a frame with its own small noise; written as inferred cubes."""
    rng = np.random.default_rng(31)
    out = tmp_path / tag
    cfg = {"frame_ownership": True, "ownership_backfill": True, "depth_graph": True, "cube_range_m": 15.0,
           "Pointcloud_Save": {"conf_percentile": 20.0, "conf_threshold_coef": 0.75, "sample_ratio": 1.0,
                               "use_conf_filter": True}, "pose_fit_conf_min_norm": 0.1}
    cfg.update(cfg_extra or {})
    vl = _VL(out, cfg)
    Kin = np.array([[60.0, 0, W / 2], [0, 60.0, H / 2], [0, 0, 1]])
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    rx, ry = (u - Kin[0, 2]) / Kin[0, 0], (v - Kin[1, 2]) / Kin[1, 1]
    poses = []
    for k, F in enumerate(vl._stac_cube_sets):
        s, R, t = float(np.exp(rng.normal() * 0.2)), _rot(rng, 0.3), rng.normal(size=3)
        poses.append((s, R, t))
        wp = np.empty((S, H, W, 3)); dep = np.empty((S, H, W)); ext = np.tile(np.eye(4), (S, 1, 1))
        for i, g in enumerate(F):
            cx = 0.12 * int(g)
            z = (5.0 + 0.4 * cx) / (1.0 - 0.4 * rx) + rng.normal(size=(H, W)) * 0.004      # the wall z = 5 + 0.4 x
            world = np.stack([cx + rx * z, ry * z, z], -1)
            wp[i] = (world - t) @ R / s
            dep[i] = z / s
            ext[i, :3, :3] = R.T
            ext[i, :3, 3] = (np.array([cx, 0.0, 0.0]) - t) @ R / s
        conf = (rng.random((S, H, W)) * 1.5 + 0.5).astype(np.float32)
        conf[rng.random((S, H, W)) < 0.1] = 0.0
        np.save(Path(vl.result_unaligned_dir) / f"chunk_{k}.npy",
                {"world_points": wp.astype(np.float32), "world_points_conf": conf, "depth": dep.astype(np.float32),
                 "extrinsic": ext.astype(np.float32), "intrinsic": np.tile(Kin, (S, 1, 1)).astype(np.float32),
                 "images": rng.integers(0, 256, size=(S, 3, H, W), dtype=np.uint8), "_stac_frames": F.tolist(),
                 "_stac_context": [], "_stac_fork_stamp": STAMP})
    run = CR.CubeRun(vl, lambda p: None, lambda im: np.asarray(im), _save_ply, _apply_sim3_direct)
    run.poses = poses
    return run, vl, out


def _server_records(out, vl):
    """The step-by-step path's records: the server's own writer on the placed cubes."""
    from workers.map_worker import _emit_omega_depth

    class Pipe:
        def send_log(self, m, level=None):
            pass
    (out / "camera_frames.txt").write_text("\n".join(str(g * 5) for g in range(N)) + "\n")
    (out / "selected_frames.json").write_text(json.dumps({"selected_files": [os.path.basename(p) for p in vl.img_list]}))
    (out / "al").rename(out / "_tmp_results_aligned")
    _emit_omega_depth(out, out / "srv", None, str(out / "selected_frames.json"), Pipe(),
                      cubes=([F.tolist() for F in vl._stac_cube_sets], vl._stac_cube_owner.tolist()))
    return out / "srv" / "omega_run" / "results_output"


def _npz(p):
    with np.load(p) as z:
        return {k: (z[k].dtype.str, z[k].shape, z[k].tobytes()) for k in z.files}


def _products(out):
    got = {}
    for base, _dirs, files in os.walk(out):
        rel = os.path.relpath(base, out)
        if rel.split(os.sep)[0] in ("un", "al", "_tmp_results_aligned", "srv", CR.RECORDS_DIR):
            continue
        for f in files:
            if f in (CR.RECORDS_DOC, "camera_frames.txt", "selected_frames.json"):
                continue
            p = Path(base) / f
            got[os.path.normpath(os.path.join(rel, f))] = _npz(p) if p.suffix == ".npz" else p.read_bytes()
    return got


def _steps(tmp_path, tag="steps"):
    run, vl, out = _scene(tmp_path, tag)
    run.close()
    recs = {p.name: _npz(p) for p in sorted(_server_records(out, vl).glob("frame_*.npz"))}
    return _products(out), recs, vl


def _single(tmp_path, tag="single", again=False, device=None):
    run, vl, out = _scene(tmp_path, tag, {"cube_closing": "single_pass",
                                          **({"cube_closing_device": device} if device else {})})
    run.close()
    if again:                                    # a resume: the same pass over its own products
        vl.reads.clear()
        run2 = CR.CubeRun(vl, lambda p: None, lambda im: np.asarray(im), _save_ply, _apply_sim3_direct)
        run2.poses = run.poses
        run2.close()
    doc = json.load(open(out / CR.RECORDS_DOC))
    recs = {p.name: _npz(p) for p in sorted((out / CR.RECORDS_DIR).glob("frame_*.npz"))}
    return _products(out), recs, vl, doc, out


def _same(a, b):
    assert sorted(a) == sorted(b)
    for k in a:
        assert a[k] == b[k], k


def _force_apply(monkeypatch):
    real = ML.depth_graph_verdict

    @functools.wraps(real)
    def forced(*a, **k):
        v = real(*a, **k)
        v["bounded"] = v["improves"] = True
        return v
    monkeypatch.setattr(ML, "depth_graph_verdict", forced)


def test_the_single_pass_writes_what_the_steps_write_reading_each_cube_once(tmp_path):
    ps, rs, vls = _steps(tmp_path)
    pp, rp, vlp, doc, out = _single(tmp_path)
    assert {"chunk_sim3.json", "uncertainty.json", "depth_graph.json", "camera_poses.txt", "intrinsic.txt",
            os.path.join("pcd", "0_pcd.ply"), os.path.join("pcd", "3_origins.npz")} <= set(ps)
    assert any(k.startswith("uncert" + os.sep) for k in ps), "the two-copy uncertainty was measured"
    _same(ps, pp)
    assert len(rs) == N
    _same(rs, rp)
    assert vlp.reads == [("un", k) for k in range(K_CUBES)], "each cube read once, as inferred"
    assert len(vls.reads) > 5 * K_CUBES and not os.listdir(out / "al"), "no placed cube was written"
    assert all((out / "un" / f"chunk_{k}.npy").exists() for k in range(K_CUBES)), "the inferred cubes are left"
    assert doc["n_frames"] == N and doc["frames"] == [g * 5 for g in range(N)] and doc["fork_stamp"] == STAMP
    assert vls.authority == vlp.authority


def test_the_same_when_the_depth_graph_moves_the_geometry(tmp_path, monkeypatch):
    _force_apply(monkeypatch)
    ps, rs, vls = _steps(tmp_path)
    pp, rp, vlp, _doc, _out = _single(tmp_path)
    a = np.asarray(json.loads(ps["depth_graph.json"])["a"])
    assert json.loads(ps["depth_graph.json"])["verdict"] == "APPLY" and np.abs(a - 1.0).max() > 1e-6
    _same(ps, pp)
    _same(rs, rp)
    # the cubes holding a moved frame were read a second time, and only they
    assert len(vlp.reads) == 2 * K_CUBES and vlp.reads[:K_CUBES] == [("un", k) for k in range(K_CUBES)]
    assert vls.authority == vlp.authority and vls.authority


def test_a_resume_over_its_own_products_reads_once_and_writes_the_same(tmp_path, monkeypatch):
    _force_apply(monkeypatch)
    ps, rs, _vls = _steps(tmp_path)
    pp, rp, vlp, _doc, _out = _single(tmp_path, again=True)
    _same(ps, pp)
    _same(rs, rp)
    assert vlp.reads == [("un", k) for k in range(K_CUBES)], "the solution on disk is applied at the first read"


@pytest.mark.parametrize("forced", [False, True])
def test_with_no_cube_allowed_to_wait_in_memory_the_waiting_ones_are_read_twice(tmp_path, monkeypatch, forced):
    if forced:
        _force_apply(monkeypatch)
    ps, rs, _vls = _steps(tmp_path)
    monkeypatch.setattr(CR, "MAX_PENDING", 0)
    pp, rp, vlp, _doc, _out = _single(tmp_path)
    _same(ps, pp)
    _same(rs, rp)
    assert len(vlp.reads) > (2 if forced else 1) * K_CUBES


@pytest.mark.parametrize("forced,max_pending", [(False, None), (True, None), (True, 0)])
def test_on_a_device_the_single_pass_writes_the_same_bits(tmp_path, monkeypatch, forced, max_pending):
    """loop_utils.closing_torch.Engine in place of the record's numpy pieces (Model.cube_closing_device)."""
    if forced:
        _force_apply(monkeypatch)
    ps, rs, vls = _steps(tmp_path)
    if max_pending is not None:
        monkeypatch.setattr(CR, "MAX_PENDING", max_pending)
    pp, rp, vlp, _doc, _out = _single(tmp_path, device="cpu")
    _same(ps, pp)
    _same(rs, rp)
    assert vls.authority == vlp.authority
    if not forced:
        assert vlp.reads == [("un", k) for k in range(K_CUBES)]


def test_the_server_takes_the_single_pass_records_as_they_are(tmp_path):
    """End to end: the records the server ends up with are the same whichever closing ran — its own writer on
    the placed cubes (steps), or the cube run's records taken by a rename (single pass)."""
    from workers import map_worker as MW
    if not hasattr(MW, "_adopt_omega_records"):
        pytest.skip("this server does not take the cube run's records yet")

    class Pipe:
        def send_log(self, m, level=None):
            pass
    _ps, rs, vls = _steps(tmp_path)
    _pp, rp, vlp, _doc, out = _single(tmp_path)
    (out / "selected_frames.json").write_text(json.dumps({"selected_files": [os.path.basename(p) for p in vlp.img_list]}))
    assert not list((out / "al").iterdir()), "no placed cube for the server to read"
    MW._emit_omega_depth(out, out / "srv", None, str(out / "selected_frames.json"), Pipe(),
                         cubes=([F.tolist() for F in vlp._stac_cube_sets], vlp._stac_cube_owner.tolist()))
    got = {p.name: _npz(p) for p in sorted((out / "srv" / "omega_run" / "results_output").glob("frame_*.npz"))}
    _same(rs, got)
    assert not (out / CR.RECORDS_DIR).exists()


def test_the_mode_is_typed_and_a_placed_cube_refuses_the_single_pass(tmp_path):
    run, vl, out = _scene(tmp_path, "typed", {"cube_closing": "one_pass"})
    with pytest.raises(ValueError, match="cube_closing"):
        run.close()
    run, vl, out = _scene(tmp_path, "placed", {"cube_closing": "single_pass"})
    (out / "al" / "chunk_2.npy").write_bytes(b"x")
    with pytest.raises(RuntimeError, match="PLACED on"):
        run.close()
    run, vl, out = _scene(tmp_path, "default")
    assert run.closing_mode() == "steps", "the record is the default"
