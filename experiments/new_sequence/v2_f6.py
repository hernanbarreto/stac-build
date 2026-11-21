"""VERSION 2, steps 10 and 12 — the pipeline's F6 on the session (precision.depth_on_f5), cut in two where the second
tie goes.

    python v2_f6.py depth   <scan dir> <state.npz>
        STEP 10 — "Profundidad de F6: el factor de cada keyframe ajustado a las correspondencias. PointDiT apagado."
        depth_on_f5.compute() as it is, stopped where its depth stage ends (where apply_mono_detail returns — a no-op
        with mono_detail off; what follows in compute() is the vote). Every keyframe's bent depth, its masks, its factor
        and its F5 camera are kept in <state.npz> for the second tie.

    python v2_f6.py publish <scan dir> <second tie's chunk_graph.json> <out report.json>
        STEP 12 — "Voto, limpiador y publicación." compute() again, whole, and publish_cloud(): the same depth stage
        (deterministic), and where it ends the second tie is applied IN MEMORY, before the vote: every keyframe takes
        the Sim(3) of the cube that owns it — its camera moved, its depth multiplied by the cube's scale, its factor
        too (the turn frames take their keyframe's factor and pose inside compute(), so they follow). Then the vote,
        the cleaner and the publication run as they are.

The cameras ON DISK stay F5's: the second tie reaches the published points, not camera_poses.txt (declared in the
checklist)."""
import json
import sys
from pathlib import Path

import numpy as np

SERVER = "/workspace/stac-build/server"
HERE = Path(__file__).resolve().parent
for p in (SERVER, str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)


class _AfterDepth(Exception):
    """Raised where the depth stage ends."""


def _entry(scan, log):
    from precision import f6_torch as FT
    from precision.config import load_precision_config
    pcfg = load_precision_config(session_dir=scan)
    if pcfg.mono_detail.enabled or not pcfg.bend.enabled:
        raise SystemExit("version 2 runs F6 with the bend on and mono_detail off — the session's frozen configuration "
                         "says otherwise")
    rec = FT.require_cuda(int(pcfg.mono_detail.seed))      # the stage entry of the pipeline's F6: the maps on the card
    log(f"[F6] maps on {rec['device']} ({rec['card']}), torch {rec['torch']}, deterministic strict")
    return pcfg


def depth(scan: Path, dst: Path, log) -> int:
    from precision import depth_on_f5 as D
    pcfg = _entry(scan, log)
    cap = {}
    real_mono, real_verify = D.apply_mono_detail, D.verify_keyframes

    def verify(*a, **k):
        r = real_verify(*a, **k)
        cap["coefs"], cap["provenance"] = r
        return r

    def mono(pcfg_, frames, dep, valid, passed, weight, inp, K, c2w, out, log_, _p):
        r = real_mono(pcfg_, frames, dep, valid, passed, weight, inp, K, c2w, out, log_, _p)
        cap.update(frames=[int(f) for f in frames], out=r, K=np.asarray(K, np.float64), c2w=c2w,
                   wh=[int(inp.wh[0]), int(inp.wh[1])], params=[float(v) for v in inp.cam.params])
        raise _AfterDepth()
    D.apply_mono_detail, D.verify_keyframes = mono, verify
    try:
        D.compute(scan, pcfg, log=log)
        raise RuntimeError("compute() returned without reaching the end of its depth stage")
    except _AfterDepth:
        pass
    finally:
        D.apply_mono_detail, D.verify_keyframes = real_mono, real_verify
    dep, valid, passed, _src, _rep = cap["out"]
    frames = cap["frames"]
    coefs = np.asarray([cap["coefs"][i] for i in range(len(frames))], np.float64)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, frames=np.asarray(frames), dep=np.stack([dep[f] for f in frames]).astype(np.float32),
             valid=np.stack([valid[f] for f in frames]), passed=np.stack([passed[f] for f in frames]), coefs=coefs,
             K=cap["K"], c2w=np.stack([cap["c2w"][f] for f in frames]), wh=np.asarray(cap["wh"]),
             params=np.asarray(cap["params"]))
    status = {}
    for p_ in cap["provenance"].values():
        status[p_["status"]] = status.get(p_["status"], 0) + 1
    c0 = coefs[:, 0]
    log(f"[step 10] {len(frames)} keyframes: the factor of each fitted on the correspondences — median {np.median(c0):.4f} "
        f"(min {c0.min():.4f}, max {c0.max():.4f}); how each was decided: {status}; PointDiT off → {dst}")
    dst.with_suffix(".json").write_text(json.dumps({"keyframes": len(frames), "factor_c0": {
        "median": float(np.median(c0)), "min": float(c0.min()), "max": float(c0.max())}, "status": status}, indent=1))
    return 0


def publish(scan: Path, graph_json: Path, report: Path, log) -> int:
    from precision import depth_on_f5 as D
    from v2_common import layout_of
    pcfg = _entry(scan, log)
    out = scan / "output"
    sets, owner, names = layout_of(out)
    num_of = [int(Path(n).stem) for n in names]
    doc = json.loads(graph_json.read_text())
    poses = [(float(q["s"]), np.asarray(q["R"], np.float64), np.asarray(q["t"], np.float64).reshape(3))
             for q in doc["poses"]]
    if len(poses) != len(sets):
        raise SystemExit(f"the second tie holds {len(poses)} cube pose(s), the session {len(sets)} cube(s)")
    cube_of = {num_of[g]: int(owner[g]) for g in range(len(num_of))}
    cap = {}
    real_mono, real_verify = D.apply_mono_detail, D.verify_keyframes

    def verify(*a, **k):
        r = real_verify(*a, **k)
        cap["coefs"] = r[0]
        return r

    def mono(pcfg_, frames, dep, valid, passed, weight, inp, K, c2w, out_, log_, _p):
        dep, valid, passed, src, rep = real_mono(pcfg_, frames, dep, valid, passed, weight, inp, K, c2w, out_, log_, _p)
        moved = []
        for i, f in enumerate(frames):
            s, R, t = poses[cube_of[int(f)]]
            M = np.asarray(c2w[f], np.float64).copy()
            C0 = M[:3, 3].copy()
            M[:3, :3] = R @ M[:3, :3]
            M[:3, 3] = s * (R @ C0) + t
            moved.append(float(np.linalg.norm(M[:3, 3] - C0)))
            c2w[f] = M
            dep[f] = (dep[f].astype(np.float64) * s).astype(np.float32)
            cap["coefs"][i] = np.asarray(cap["coefs"][i], np.float64) * s
        cap["moved"] = moved
        log_(f"[step 12] the second tie applied before the vote: every keyframe by the Sim(3) of its owner cube — "
             f"cube scales {[round(p[0], 5) for p in poses]}; the cameras moved median {np.median(moved) * 100:.2f} cm, "
             f"most {max(moved) * 100:.2f} cm")
        return dep, valid, passed, src, rep
    D.apply_mono_detail, D.verify_keyframes = mono, verify
    try:
        C = D.compute(scan, pcfg, log=log)
    finally:
        D.apply_mono_detail, D.verify_keyframes = real_mono, real_verify
    rep = D.publish_cloud(C, pcfg, log=log)
    v = rep.get("vote", {})
    log(f"[step 12] vote coverage {v.get('coverage', C.cover) * 100:.1f} %; raw points {rep.get('raw_points'):,}; "
        f"published {rep.get('n_points'):,} points as geometry epoch {rep.get('epoch_to')}")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"cameras_moved_m": {"median": float(np.median(cap["moved"])),
                                                      "max": float(max(cap["moved"]))},
                                  "cube_scales": [p[0] for p in poses], "coverage": float(C.cover),
                                  "raw_points": rep.get("raw_points"), "n_points": rep.get("n_points"),
                                  "epoch_to": rep.get("epoch_to")}, indent=1))
    return 0


def main() -> int:
    mode, scan = sys.argv[1], Path(sys.argv[2]).resolve()
    log = lambda m: print(m, flush=True)   # noqa: E731
    if mode == "depth":
        return depth(scan, Path(sys.argv[3]), log)
    if mode == "publish":
        return publish(scan, Path(sys.argv[3]), Path(sys.argv[4]), log)
    raise SystemExit(f"unknown mode {mode!r}: depth | publish")


if __name__ == "__main__":
    sys.exit(main())
