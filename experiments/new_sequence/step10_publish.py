"""THE NEW ORDER, step 10 — "Voto, limpiador y publicación."

The pipeline's F6 on the session, unchanged (precision.depth_on_f5.compute, then publish_cloud): the confidence floor
per cube, the vote of every co-visible keyframe and of the turn frames (COLMAP rule: support, angle, reprojection,
range, occlusion veto, majority, in place), the cleaner and the publication with its octree.

The depth is NOT decided here — it was decided cube by cube in step 7, before the tie. So:

  * the job's frozen configuration of the session gets `bend.enabled: false` and `mono_detail.enabled: false` (the
    pipeline's own switches): F6 fits no factor and runs no PointDiT in this step;
  * where F6 would have applied PointDiT (apply_mono_detail, a no-op with the switch off) each keyframe takes the
    depth step 7 gave it in the cube that OWNS it, in the session's final units (x the cube's scale in the graph, x the
    metre factor) — the exact maps of step 7, not a resampling of the records. The masks stay F6's own (validity and
    the confidence floor from the session's records), cut where step 7 left no depth.

The records of the session (written in step 9 from the tied, corrected cubes) hold the same depth; how far the two are
is printed as a check before the vote.

    python step10_publish.py <session scan dir> <step-7 dir>"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cube_sessions as CS  # noqa: E402


def main() -> int:
    scan, d7 = Path(sys.argv[1]), Path(sys.argv[2])
    out = scan / "output"
    from intake import run_config as RC
    from precision import depth_on_f5 as D
    from precision.config import load_precision_config
    from precision.gauge import applied_global_scale
    log = lambda m: print(m, flush=True)   # noqa: E731

    doc = yaml.safe_load(RC.run_config_path(scan).read_text())
    stages = doc.pop(RC.STAGES_KEY, None)
    prec = doc["reconstruction"]["precision"]
    if prec["bend"]["enabled"] or prec["mono_detail"]["enabled"]:
        prec["bend"]["enabled"] = False
        prec["mono_detail"]["enabled"] = False
        RC.freeze_run_config(scan, doc, stages=stages, log=log)
    pcfg = load_precision_config(session_dir=scan)
    # THE STAGE ENTRY OF THE PIPELINE'S F6 (precision.depth_on_f5.main): the maps on the card, strict deterministic
    # torch for the whole process, seeded — "no card, no stage". Without it the vote runs on the CPU.
    from precision import f6_torch as FT
    rec = FT.require_cuda(int(pcfg.mono_detail.seed))
    log(f"[step 10] maps on {rec['device']} ({rec['card']}), torch {rec['torch']}, deterministic strict")

    sets, owner, names = CS.layout(out / "maplong_run")
    cube_scale = [float(p["s"]) for p in json.loads((out / "maplong_run" / "chunk_sim3.json").read_text())["poses"]]
    metre = float(applied_global_scale(out))
    num_of = [int(Path(n).stem) for n in names]
    maps = {}
    for k in range(len(sets)):
        z = np.load(d7 / f"depth_cube_{k}.npz")
        fr = [int(f) for f in z["frames"]]
        for g in sets[k]:
            if owner[g] == k:
                i = fr.index(num_of[g])
                maps[num_of[g]] = (z["dep"][i].astype(np.float64) * (cube_scale[k] * metre),
                                   (z["src"][i] if "src" in z.files else None))
    log(f"[step 10] step 7's depth of the {len(maps)} keyframes, each from the cube that owns it; cube scales in the "
        f"graph {[round(s, 5) for s in cube_scale]}, metre x{metre:.5f}")

    def depth_of_step7(pcfg_, frames, dep, valid, passed, weight, inp, K, c2w, out_, log_, _p):
        ratios, src = [], {}
        for f in frames:
            z7, s7 = maps[int(f)]
            both = (z7 > 0) & (dep[f] > 0)
            if both.any():
                ratios.append(float(np.median(z7[both] / dep[f][both])))
            alive = z7 > 0
            dep[f] = np.where(valid[f] & alive, z7, 0).astype(np.float32)
            valid[f] = valid[f] & alive
            passed[f] = passed[f] & alive
            src[f] = s7
        r = np.asarray(ratios)
        log_(f"[step 10] check — step 7's depth against the session's records, keyframe by keyframe (ratio of the two, "
             f"median over the pixels both hold): median {np.median(r):.6f}, from {r.min():.6f} to {r.max():.6f}")
        return dep, valid, passed, (src if all(v is not None for v in src.values()) else None), None
    D.apply_mono_detail = depth_of_step7

    # THE METRE REACHES THE DEPTH: the session's records are in the units the cubes had BEFORE the metre step (the
    # pipeline never rescales them — its F6 absorbs the metre in the factor it fits). With no factor fitted here, the
    # factor of every keyframe is the KNOWN one: the metre of step 9. It is what the turn frames' depth, which votes
    # too, is multiplied by (a factor of 1 left them 5.5 % short of the cameras in the first run of this step).
    switched_off = D.disabled_bend

    def known_metre(n):
        wb, rule, coefs, prov = switched_off(n)
        for i in coefs:
            coefs[i] = coefs[i].copy()
            coefs[i][0] = metre
        rule = {**rule, "reason": f"no factor is fitted in this step: every keyframe's factor is the metre of step 9, x{metre:.5f}"}
        return wb, rule, coefs, prov
    D.disabled_bend = known_metre
    C = D.compute(scan, pcfg, log=log)
    rep = D.publish_cloud(C, pcfg, log=log)
    v = rep.get("vote", {})
    log(f"[step 10] vote coverage {v.get('coverage', C.cover) * 100:.1f} %; raw points {rep.get('raw_points'):,}; "
        f"published {rep.get('n_points'):,} points as geometry epoch {rep.get('epoch_to')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
