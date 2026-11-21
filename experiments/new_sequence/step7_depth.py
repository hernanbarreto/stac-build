"""THE NEW ORDER, step 7 — "Profundidad, por cubo: el factor de cada keyframe ajustado a las correspondencias, y
después PointDiT en los bordes."

The pipeline's F6 computation (precision.depth_on_f5.compute), unchanged, run on each cube's session: F4's tracks
triangulated with the cube's F5 cameras and the one camera, the factor of every keyframe fitted on those landmarks
(bend_coefficients ... verify_keyframes), then PointDiT at the edges (apply_mono_detail). The run of each cube is
STOPPED where apply_mono_detail returns — what follows in that function is the vote, which is step 10 — and what it
returned is kept: every keyframe's corrected depth, its validity, its confidence-floor mask and its source map, as this
cube holds that keyframe.

Nothing of the depth is computed here: the two functions are the pipeline's, called by the pipeline's own compute();
this script only keeps their results.

    python step7_depth.py <root of the cube sessions> <out dir>"""
import json
import sys
import time
from pathlib import Path

import numpy as np

SERVER = "/workspace/stac-build/server"
if SERVER not in sys.path:
    sys.path.insert(0, SERVER)


class _AfterDepth(Exception):
    """Raised where apply_mono_detail returns: the depth stage of this cube is complete."""


def run_cube(cube_dir: Path, dst: Path, log) -> dict:
    from precision import depth_on_f5 as D
    from precision.config import load_precision_config
    pcfg = load_precision_config(session_dir=cube_dir)
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
    t0 = time.time()
    try:
        D.compute(cube_dir, pcfg, log=log)
        raise RuntimeError("compute() returned without reaching apply_mono_detail")
    except _AfterDepth:
        pass
    finally:
        D.apply_mono_detail, D.verify_keyframes = real_mono, real_verify
    dep, valid, passed, src, rep = cap["out"]
    frames = cap["frames"]
    coefs = np.asarray([cap["coefs"][i] for i in range(len(frames))], np.float64)
    arrays = {"frames": np.asarray(frames), "dep": np.stack([dep[f] for f in frames]).astype(np.float32),
              "valid": np.stack([valid[f] for f in frames]), "passed": np.stack([passed[f] for f in frames]),
              "coefs": coefs, "K": cap["K"], "c2w": np.stack([cap["c2w"][f] for f in frames]),
              "wh": np.asarray(cap["wh"]), "params": np.asarray(cap["params"])}
    if src is not None:
        arrays["src"] = np.stack([np.asarray(src[f]) for f in frames])
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **arrays)
    report = {"keyframes": len(frames), "factor_c0": {"median": float(np.median(coefs[:, 0])),
                                                      "min": float(coefs[:, 0].min()), "max": float(coefs[:, 0].max())},
              "status": {}, "seconds": round(time.time() - t0, 1)}
    for p_ in cap["provenance"].values():
        report["status"][p_["status"]] = report["status"].get(p_["status"], 0) + 1
    if rep is not None:
        m = D.mono_report(rep, pcfg.mono_detail)
        report["pointdit"] = {"tiles": m["tiles"], "pixels": m["pixels"]}
    dst.with_suffix(".json").write_text(json.dumps(report, indent=1, default=float))
    return report


def main() -> int:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    only = [a for a in sys.argv[3:] if a.startswith("cube_")]
    log = lambda m: print(m, flush=True)   # noqa: E731
    dirs = [d for d in sorted(root.glob("cube_*"), key=lambda p: int(p.name.split("_")[1])) if not only or d.name in only]
    # THE STAGE ENTRY OF THE PIPELINE'S F6 (precision.depth_on_f5.main): the maps on the card, strict deterministic
    # torch for the whole process, seeded — "no card, no stage". The first run of this step called compute() without
    # it, so its maps ran on the CPU (slower; the same functions, other kernels).
    from precision import f6_torch as FT
    from precision.config import load_precision_config
    rec = FT.require_cuda(int(load_precision_config(session_dir=dirs[0]).mono_detail.seed))
    log(f"[step 7] maps on {rec['device']} ({rec['card']}), torch {rec['torch']}, deterministic strict")
    for d in dirs:
        dst = out / f"depth_{d.name}.npz"
        if dst.exists() and dst.with_suffix(".json").exists():
            rep = json.loads(dst.with_suffix(".json").read_text())
        else:
            rep = run_cube(d, dst, log)
        f = rep["factor_c0"]
        log(f"[step 7] {d.name}: {rep['keyframes']} keyframes; depth factor per keyframe median {f['median']:.4f} "
            f"(min {f['min']:.4f}, max {f['max']:.4f}); how each factor was decided: {rep['status']}; PointDiT: "
            f"{json.dumps(rep.get('pointdit', {}).get('pixels'))}, tiles {json.dumps(rep.get('pointdit', {}).get('tiles'))}; "
            f"{rep['seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
