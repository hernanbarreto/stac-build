"""Step 3 of the architecture (ARQUITECTURA_mapa.md): THE DA3 PROBE on 16 of AMB3R's own keyframes.

The frames: AMB3R's own keyframes (its pass's ``kf``) that are good frames (step 1, arq3_frames), and of them
``focal_probe_frames`` (16) spread uniformly over that list — the focal probe's own rule (intake.focal.probe_files:
np.linspace over the list, rounded). On them, the pipeline's own functions, unchanged:
  * the camera — intake.focal.run_focal_probe: ONE DA3 inference in the committed layout, each frame's K carried to
    native pixels, the per-parameter median; the frames' metric depth kept (focal_depth.npz);
  * AMB3R's metre — reconstruction.amb3r_probe.metric_scale: at each probe frame AMB3R's nearest record projected
    into its camera, DA3's depth over AMB3R's in the near band, the median over the frames — on AMB3R's pass brought
    back to its own units (make_metric's rule).
The probe is handed its frames as the usable list of a quality table that names exactly them (probe_files then
takes all of them). Nothing of the session is written: the probe works in <work>/, `frames` linked to the session's."""
import sys
from pathlib import Path

import numpy as np

SERVER = "/workspace/stac-build/server"
if SERVER not in sys.path:
    sys.path.insert(0, SERVER)


def probe_config():
    import yaml
    from intake.config import load_intake_config
    return load_intake_config(yaml.safe_load((Path(SERVER) / "config.yaml").read_text())).parallax


def load_video(npz: Path) -> dict:
    with np.load(npz) as z:
        return {k: z[k] for k in z.files}


def amb3r_units(video: dict) -> dict:
    """AMB3R's pass in its own units: the stored metre taken out (reconstruction.amb3r_probe.make_metric's rule)."""
    base = dict(video)
    old = float(video["metric_scale"])
    for key in ("pred", "rec_c2w"):
        m = np.asarray(base[key], np.float64).copy()
        m[:, :3, 3] /= old
        base[key] = m
    base["rec_depth"] = (np.asarray(base["rec_depth"], np.float64) / old).astype(np.float32)
    return base


def probe_frames(video: dict, good: set, n: int) -> list:
    """``n`` of AMB3R's own keyframes that are good frames, spread uniformly over that list (in order)."""
    fn = np.asarray(video["frame_no"], np.int64)
    kf = [int(fn[i]) for i in np.asarray(video["kf"], np.int64)]   # kf: positions in AMB3R's frame list
    cand = [f for f in kf if f in good]
    if len(cand) < 2:
        raise RuntimeError(f"{len(cand)} of AMB3R's {len(kf)} keyframes are good frames — the probe needs two")
    idx = np.unique(np.round(np.linspace(0, len(cand) - 1, min(int(n), len(cand)))).astype(int))
    return [cand[i] for i in idx]


def run(scan: Path, work: Path, good: set, python: str, log=print) -> dict:
    """The probe on ``scan``'s AMB3R keyframes: {frames, camera, metre}."""
    from PIL import Image
    from intake.focal import FOCAL_DEPTH_NAME, run_focal_probe
    from intake.quality import QUALITY_VERSION, list_frames
    from reconstruction.amb3r_probe import metric_scale
    scan, work = Path(scan), Path(work)
    pcfg = probe_config()
    video = load_video(scan / "intake" / "amb3r_video.npz")
    chosen = probe_frames(video, good, pcfg.focal_probe_frames)
    paths = list_frames(scan / "frames")
    with Image.open(paths[0]) as im:
        W, H = im.size
    (work / "intake").mkdir(parents=True, exist_ok=True)
    link = work / "frames"
    if not link.is_symlink():
        link.symlink_to(scan / "frames", target_is_directory=True)
    pick = set(chosen)
    quality = {"version": QUALITY_VERSION, "native_w": W, "native_h": H,
               "frames": [{"frame": int(p.stem), "file": p.name, "sharp_rank": 0.0, "usable": int(p.stem) in pick}
                          for p in paths]}
    doc = run_focal_probe(work, quality, pcfg, python=python, log=log)
    with np.load(work / "intake" / FOCAL_DEPTH_NAME) as z:
        focal = {k: z[k] for k in z.files}
    metre = metric_scale(amb3r_units(video), focal)
    return {"frames": chosen, "probe_files": doc["frames"],
            "camera": {k: doc[k] for k in ("fx", "fy", "cx", "cy", "spread_pct")}, "metre": metre}
