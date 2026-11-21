"""THE ARCHITECTURE OF 2026-11-21, steps 5 and 6 — Omega per cube and the tie, as the pipeline runs them
(vendor/VGGT-Long/vggt_long.py → loop_utils.cube_run.process_cubes), with the cube's size taken from the DA3 depth of
THAT cube.

"Los 15 m se miden con el tamaño de cada cubo, que sale de la profundidad DA3 de ese cubo": CubeRun.metric_lock is
replaced by :func:`cube_size` — each cube is scaled by the median, over its keyframes, of DA3 / Omega (the fork's own
measure, loop_utils.metric_lock.anchor_ratio: the near band, inside the range), where DA3's depth of a keyframe is the
one of the DA3 pass over that same cube (arq_front.py, step 4: one DA3 window per cube, the cube's keyframes). The
seams and the graph decide every cube's scale freely; the session's metre is the one number of step 7.

Run in place of vggt_long.py with the fork's own arguments (arq_front.py redirects the worker's launch here):
    python arq_omega.py --image_dir <frames> --config <vggt_omega_config.yaml> --save_dir <maplong_run> --selected_frames <json>
The DA3 windows: the directory named by STAC_ARQ_DA3_CUBES (window_<cube:04d>.npz: frames, depth)."""
import json
import os
import runpy
import sys

FORK = "/workspace/stac-build/vendor/VGGT-Long"


def cube_size(self):
    """CubeRun.metric_lock for this architecture (module docstring)."""
    import numpy as np
    from loop_utils import cube_run as CR
    from loop_utils.metric_lock import anchor_ratio, apply_scale, real_frame_number
    vl = self.vl
    wdir = os.environ["STAC_ARQ_DA3_CUBES"]
    ml = self.cfg.get('metric_lock') or {}
    near = float(ml.get('near_frac', 0.25))
    rng = float(getattr(self, "range_m", 0.0))
    rep = vl._stac_with_stamp({"layout": "cubes", "layout_sha256": self.digest, "chunks": {},
                               "source": "each cube sized with the DA3 depth of its own DA3 pass"})
    applied = {}
    for k, data in vl._stac_iter_chunks(self.un, range(self.K)):
        nums = [int(real_frame_number(vl.img_list[int(g)])) for g in self.sets[k]]
        with np.load(os.path.join(wdir, f"window_{int(k):04d}.npz")) as z:
            wframes = [int(x) for x in z["frames"]]
            da3 = np.asarray(z["depth"], np.float32)
        if wframes != nums:
            raise RuntimeError(f"{CR.LOG} cube {k}: its DA3 window holds other frames than the cube's keyframes")
        depth = np.asarray(data["depth"])
        conf3 = data.get("world_points_conf")
        kin = None
        if rng > 0:
            kin = data.get("intrinsic")
            if kin is None:
                raise RuntimeError(f"{CR.LOG} cube {k} carries no intrinsic — the range cannot be measured")
            kin = np.asarray(kin, np.float64)
            kin = kin[0] if kin.ndim == 4 else kin
        ratios, locs = [], []
        for local in range(len(nums)):
            conf = None
            if conf3 is not None:
                c = np.asarray(conf3)
                conf = c[local] if local < c.shape[0] else None
            r = anchor_ratio(depth[local], da3[local], conf=conf, near_frac=near,
                             K=(kin[local] if kin is not None else None), range_m=rng)
            if r is not None and np.isfinite(r) and r > 0:
                locs.append(int(local))
                ratios.append(float(r))
        lock = data.get('_stac_cube_lock')
        if lock is not None:                               # a resumed cube: sized already
            applied[k] = float(lock)
            print(f"{CR.LOG} cube {k}: sized before (s {applied[k]:.4f})")
            ratios = [float(r) * float(lock) for r in ratios]
        else:
            if not ratios:
                raise RuntimeError(f"{CR.LOG} cube {k}: no keyframe measures DA3 against Omega — the cube has no size")
            s = float(np.median(ratios))
            apply_scale(data, s)
            data['_stac_cube_lock'] = s
            np.save(CR._path(self.un, k), data)
            applied[k] = s
            print(f"{CR.LOG} cube {k}: sized with its DA3 depth, s={s:.4f} from {len(ratios)} keyframe(s)")
        rep["chunks"][str(k)] = {"s": (float(np.median(ratios)) if ratios else None), "n_anchors": len(ratios),
                                 "ratios": ratios, "anchor_locals": locs, "s_applied": applied[k]}
        del data
    with open(os.path.join(vl.output_dir, "metric_lock.json"), "w") as f:
        json.dump(rep, f, indent=1, default=float)
    self.scale_inputs = {"s_da3": {}, "n_anchors": {}, "sigma_anchor": float(ml.get('sigma_anchor', 0.08))}
    self.scales_applied = applied


def _nothing(self):
    return None


def main() -> None:
    sys.path.insert(0, FORK)
    import loop_utils.cube_run as CR
    CR.CubeRun.metric_lock = cube_size
    CR.CubeRun.uncertainty = _nothing          # the per-frame maps of the cubes' disagreement: no step reads them
    sys.argv = [FORK + "/vggt_long.py", *sys.argv[1:]]
    runpy.run_path(FORK + "/vggt_long.py", run_name="__main__")


if __name__ == "__main__":
    main()
