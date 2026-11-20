"""A cube's DA3 lock is measured and applied at its inference, before its one write (USER 2026-11-19: "Todo debe
hacerse en placa … es tiempo perdido" / "Dije que si al diseño, dale") — the same cube file, bit for bit, as the
pass of its own it replaces, which read every cube back from the volume, scaled it and wrote it again."""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

FORK = Path(os.environ.get("STAC_FORK_DIR") or Path(__file__).resolve().parents[2] / "vendor" / "VGGT-Long")
sys.path.insert(0, str(FORK))
from loop_utils import cube_run as CR  # noqa: E402

H, W, S = 40, 36, 5
STAMP = "f" * 64


class _VL:
    def __init__(self, out, n):
        self.output_dir = str(out)
        self.img_list = [f"/frames/{i * 6:06d}.jpg" for i in range(n)]
        self.reads = []

    def _stac_stamp_digest(self):
        return STAMP

    def _stac_with_stamp(self, d):
        d["fork_stamp"] = STAMP
        return d

    def _stac_check_fork_stamp(self, d, what):
        if d.get("fork_stamp", d.get("_stac_fork_stamp")) != STAMP:
            raise RuntimeError(f"{what}: another stamp")

    def _stac_iter_chunks(self, directory, ks):
        for k in ks:
            self.reads.append(int(k))
            yield k, np.load(os.path.join(directory, f"chunk_{k}.npy"), allow_pickle=True).item()


def _pred(rng, F, scale):
    """What an inference leaves for a cube: depth in the cube's own unit (the anchors hold ``scale`` times it)."""
    depth = (rng.uniform(1.0, 6.0, size=(S, H, W))).astype(np.float32)
    K = np.tile(np.array([[30.0, 0, W / 2], [0, 30.0, H / 2], [0, 0, 1]]), (S, 1, 1))
    ext = np.tile(np.eye(4, dtype=np.float32), (S, 1, 1))
    ext[:, :3, 3] = rng.normal(size=(S, 3)).astype(np.float32)
    return {"world_points": rng.normal(size=(S, H, W, 3)).astype(np.float32), "world_points_conf": (rng.random((S, H, W)) + 0.5).astype(np.float32),
            "extrinsic": ext, "intrinsic": K.astype(np.float32), "depth": depth, "_stac_frames": [int(g) for g in F],
            "_stac_context": [], "_stac_fork_stamp": STAMP}, depth.astype(np.float64) * scale


def _run(tmp_path, tag, anchors_for=(0, 1, 2)):
    rng = np.random.default_rng(21)
    out = tmp_path / tag
    (out / "un").mkdir(parents=True)
    anchors = out / "anchors"
    anchors.mkdir()
    # (three cubes with their own frames: an anchor is a frame's, and each cube here has its own unit)
    sets = [np.arange(0, S), np.arange(S, 2 * S), np.arange(2 * S, 3 * S)]
    r = CR.CubeRun.__new__(CR.CubeRun)
    r.vl = _VL(out, 3 * S)
    r.sets, r.K, r.un, r.digest, r.range_m = sets, 3, str(out / "un"), "d" * 64, 15.0
    r.cfg = {"metric_lock": {"enable": True, "anchor_dir": str(anchors), "near_frac": 0.25, "sigma_anchor": 0.08}}
    preds = []
    for k, (F, sc) in enumerate(zip(sets, (2.0, 2.6, 3.1))):
        pred, metric = _pred(rng, F, sc)
        preds.append(pred)
        if k in anchors_for:
            for j, g in enumerate(F):
                np.savez(anchors / f"frame_{int(g) * 6}.npz", depth=(metric[j] * (1 + 0.01 * rng.normal(size=(H, W)))).astype(np.float32))
    return r, preds


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_the_lock_at_the_inference_writes_the_same_cube_as_the_pass_it_replaces(tmp_path):
    # the pass of its own: the cubes written as the inference left them, the lock reads and rewrites each
    a, preds = _run(tmp_path, "pass")
    for k, pred in enumerate(preds):
        np.save(Path(a.un) / f"chunk_{k}.npy", pred)
    a.metric_lock()
    assert a.vl.reads == [0, 1, 2]
    # at the inference: the lock in memory, the cube written once, its record beside it
    b, preds = _run(tmp_path, "inference")
    for k, pred in enumerate(preds):
        rec = b._lock_in_memory(k, pred)
        np.save(Path(b.un) / f"chunk_{k}.npy", pred)
        b._write_lock(k, rec)
    b.metric_lock()
    assert b.vl.reads == [], "a cube that left its record is not read again"
    for k in range(3):
        assert _sha(Path(a.un) / f"chunk_{k}.npy") == _sha(Path(b.un) / f"chunk_{k}.npy"), k
    assert json.load(open(tmp_path / "pass" / "metric_lock.json")) == json.load(open(tmp_path / "inference" / "metric_lock.json"))
    assert a.scales_applied == b.scales_applied and a.scale_inputs == b.scale_inputs
    assert all(abs(a.scales_applied[k] - s) < 0.1 for k, s in enumerate((2.0, 2.6, 3.1)))


def test_a_cube_with_no_anchor_still_takes_the_anchored_cubes_median(tmp_path):
    a, preds = _run(tmp_path, "pass", anchors_for=(0, 2))
    for k, pred in enumerate(preds):
        np.save(Path(a.un) / f"chunk_{k}.npy", pred)
    a.metric_lock()
    b, preds = _run(tmp_path, "inference", anchors_for=(0, 2))
    for k, pred in enumerate(preds):
        rec = b._lock_in_memory(k, pred)
        assert (rec["s"] is None) == (k == 1)
        np.save(Path(b.un) / f"chunk_{k}.npy", pred)
        b._write_lock(k, rec)
    b.metric_lock()
    assert b.vl.reads == [1], "only the cube without an anchor is read (and given the others' median)"
    assert a.scales_applied == b.scales_applied
    assert abs(b.scales_applied[1] - float(np.median([b.scales_applied[0], b.scales_applied[2]]))) < 1e-12
    for k in range(3):
        assert _sha(Path(a.un) / f"chunk_{k}.npy") == _sha(Path(b.un) / f"chunk_{k}.npy"), k


def test_a_record_of_another_layout_or_stamp_is_not_taken(tmp_path):
    b, preds = _run(tmp_path, "inference")
    for k, pred in enumerate(preds):
        rec = b._lock_in_memory(k, pred)
        np.save(Path(b.un) / f"chunk_{k}.npy", pred)
        b._write_lock(k, rec)
    p = Path(b._lock_path(0))
    rec = json.loads(p.read_text())
    p.write_text(json.dumps(dict(rec, layout_sha256="e" * 64)))
    assert b._read_lock(0) is None
    p.write_text(json.dumps(dict(rec, fork_stamp="0" * 64)))
    assert b._read_lock(0) is None
    p.write_text("{not json")
    assert b._read_lock(0) is None
    b.metric_lock()
    assert b.vl.reads == [0], "the cube whose record cannot be taken is read, and found locked already"
    assert abs(b.scales_applied[0] - rec["s"]) < 1e-12


def test_with_the_lock_off_nothing_is_measured(tmp_path):
    b, preds = _run(tmp_path, "inference")
    b.cfg = {"metric_lock": {"enable": False}}
    before = preds[0]["depth"].copy()
    assert b._lock_in_memory(0, preds[0]) is None
    assert np.array_equal(preds[0]["depth"], before) and "_stac_cube_lock" not in preds[0]


def test_a_resume_does_not_read_back_a_cube_its_record_vouches_for(tmp_path, monkeypatch, capsys):
    b, preds = _run(tmp_path, "resume")
    b.al = str(tmp_path / "resume" / "al")
    os.makedirs(b.al)
    b.vl.chunk_context = [[], [], []]
    for k, pred in enumerate(preds):
        rec = b._lock_in_memory(k, pred)
        p = Path(b.un) / f"chunk_{k}.npy"
        np.save(p, pred)
        rec["cube_bytes"] = os.path.getsize(p)
        b._write_lock(k, rec)
    real_load = np.load

    def counted(path, *a, **kw):
        loads.append(os.path.basename(str(path)))
        return real_load(path, *a, **kw)
    loads = []
    monkeypatch.setattr(np, "load", counted)
    for k in range(3):
        b.infer(k)
    assert loads == [] and capsys.readouterr().out.count("vouches for the file") == 3
    # a cube whose size is not the recorded one is not taken on its record's word: it is read (and found whole)
    p0 = Path(b.un) / "chunk_0.npy"
    rec = json.loads(Path(b._lock_path(0)).read_text())
    Path(b._lock_path(0)).write_text(json.dumps(dict(rec, cube_bytes=rec["cube_bytes"] + 1)))
    b.infer(0)
    assert loads == ["chunk_0.npy"] and "on disk — inference skipped" in capsys.readouterr().out
    assert p0.exists()
