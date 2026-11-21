"""VERSION 3 — Omega's cube stage exactly as the pipeline runs it (vendor/VGGT-Long/vggt_long.py →
loop_utils.cube_run.process_cubes), with ONE substitution: NO DA3 LOCK PER CUBE.

CubeRun.metric_lock (each cube scaled by the median ratio of its DA3 anchors) is replaced: DA3 decides nothing about a
cube (USER 2026-11-21: "da3 metro nada mas" / "sin candado por cubo, sin da3 que decida nada mas que cuanto escalo la
nube para pasarla a metros"). A cube still has to lie in metres for THE RANGE (15 m, the user's rule) to mean 15 m in
the seams' and closures' fits, so each cube takes AMB3R's metre: the scale of the similarity that carries the cube's
own camera centres onto AMB3R's centres of the same keyframes (AMB3R's whole-video pass, already in metres since
step 1). It is applied with the fork's own function (loop_utils.metric_lock.apply_scale) and recorded where the lock
is recorded (metric_lock.json, the cube's `_stac_cube_lock`). The relative scale of the cubes is still decided by the
seams and the graph; the session's metre by the one DA3 factor (scale_align).

Run in place of vggt_long.py with the fork's own arguments (v3_front.py redirects the worker's launch here):
    python v3_omega.py --image_dir <frames> --config <vggt_omega_config.yaml> --save_dir <maplong_run> --selected_frames <json>
AMB3R's keyframe centres: the file named by STAC_V3_AMB3R_CENTRES ({"<video frame>": [x, y, z] in metres})."""
import json
import os
import runpy
import sys

FORK = "/workspace/stac-build/vendor/VGGT-Long"


def _similarity_scale(src, dst):
    """(scale, rms of what is left, in dst's units) of the least-squares similarity src → dst (Umeyama)."""
    import numpy as np
    S, D = src - src.mean(0), dst - dst.mean(0)
    U, sv, Vt = np.linalg.svd(D.T @ S / len(src))
    E = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)))])
    var = float((S ** 2).sum(1).mean())
    if not var > 0:
        raise RuntimeError("the cube's cameras do not move — no scale to AMB3R's metre")
    s = float(np.trace(np.diag(sv) @ E) / var)
    R = U @ E @ Vt
    return s, float(np.sqrt((((s * S @ R.T) - D) ** 2).sum(1).mean()))


def amb3r_lock(self):
    """CubeRun.metric_lock for version 3 (module docstring)."""
    import numpy as np
    from loop_utils import cube_run as CR
    from loop_utils.metric_lock import apply_scale, real_frame_number
    vl = self.vl
    centres = {int(k): np.asarray(v, np.float64) for k, v in json.load(open(os.environ["STAC_V3_AMB3R_CENTRES"])).items()}
    ml = self.cfg.get('metric_lock') or {}
    rep = vl._stac_with_stamp({"layout": "cubes", "layout_sha256": self.digest, "chunks": {},
                               "source": "amb3r: each cube scaled to AMB3R's metre (similarity of its camera "
                                         "centres onto AMB3R's)"})
    applied = {}
    for k, data in vl._stac_iter_chunks(self.un, range(self.K)):
        nums = [real_frame_number(vl.img_list[int(g)]) for g in self.sets[k]]
        C = CR._ext(data)[:, :3, 3]
        if len(C) != len(nums):
            raise RuntimeError(f"{CR.LOG} cube {k}: {len(C)} cameras on disk for {len(nums)} keyframes")
        A = np.stack([centres[int(n)] for n in nums])
        lock = data.get('_stac_cube_lock')
        s_now, rms = _similarity_scale(C, A)
        if lock is not None:                               # a resumed cube: scaled already
            applied[k] = float(lock)
            print(f"{CR.LOG} cube {k}: scaled to AMB3R's metre before (s {applied[k]:.4f})")
        else:
            apply_scale(data, s_now)
            data['_stac_cube_lock'] = float(s_now)
            np.save(CR._path(self.un, k), data)
            applied[k] = float(s_now)
            print(f"{CR.LOG} cube {k}: scaled to AMB3R's metre, s={s_now:.4f} (fit on its {len(nums)} keyframes' "
                  f"cameras, residual {rms * 100:.1f} cm)")
        rep["chunks"][str(k)] = {"s": None, "n_anchors": 0, "ratios": [], "anchor_locals": [],
                                 "s_applied": applied[k], "keyframes": len(nums), "amb3r_fit_rms_m": rms}
        del data
    with open(os.path.join(vl.output_dir, "metric_lock.json"), "w") as f:
        json.dump(rep, f, indent=1, default=float)
    self.scale_inputs = {"s_da3": {}, "n_anchors": {}, "sigma_anchor": float(ml.get('sigma_anchor', 0.08))}
    self.scales_applied = applied


def main() -> None:
    sys.path.insert(0, FORK)
    import loop_utils.cube_run as CR
    CR.CubeRun.metric_lock = amb3r_lock
    sys.argv = [FORK + "/vggt_long.py", *sys.argv[1:]]
    runpy.run_path(FORK + "/vggt_long.py", run_name="__main__")


if __name__ == "__main__":
    main()
