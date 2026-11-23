"""Step 5 of the architecture (ARQUITECTURA_mapa.md): THE KEYFRAME SELECTION, with what the steps before it give.

  * the good frames: step 1 (arq3_frames_run.py's result);
  * AMB3R's pass in metres with step 3's metre, and step 3's camera (arq3_probe_run.py's result);
  * the walk from the first good frame, every frame measured against the last keyframe (arq2_select.Measures, on the
    card): the OVERLAP, 92–96 % — v2_select.Overlap, the footprint on AMB3R's points: the map of step 4 is not closed
    (told to him 2026-11-22); the DISPLACEMENT, I1's parallax on AMB3R's points with the turn taken out, its 0.9
    quantile, 1.26° (measured on the validated runs, his OK); the ROTATION, I1's θ_q = (1 − τ)·FOV_h / 4 from step 3's
    camera (intake.parallax.rotation_quantum); both bands ±25 % (I1's);
  * a frame is a candidate when any measure lies in its band; the window closes when any passes its band; the keyframe
    is the sharpest good candidate; a close with no candidate takes the two frames around it (arq2_select.select).

    python arq3_select.py <name> <scan dir> <out.json>"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

SERVER = "/workspace/stac-build/server"
HERE = Path(__file__).resolve().parent
for p in (SERVER, str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

DISPLACEMENT_DEG = 1.26        # USER 2026-11-22 OK; measured: the median of the measure where I1 closed by displacement
FRAMES = Path("/workspace/tmp/new_sequence/arq/frames")
PROBE = Path("/workspace/tmp/new_sequence/arq/probe")


def main() -> int:
    name, scan, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    import arq2_select as Q
    import arq3_map as MAP
    import arq3_probe as PR
    import v2_select as S
    from PIL import Image
    from intake.config import load_intake_config
    from intake.parallax import rotation_quantum
    from intake.quality import list_frames
    from view_range import view_range_m
    log = lambda m: print(m, flush=True)   # noqa: E731
    t0 = time.time()
    config = yaml.safe_load((Path(SERVER) / "config.yaml").read_text())
    icfg = load_intake_config(config)
    quant, band = float(icfg.parallax.parallax_quantile), float(icfg.parallax.keyframe_band_frac)
    fr = json.loads((FRAMES / f"{name}.json").read_text())
    good = {int(f): None for f in fr["frames"]}
    good.update({int(k): v for k, v in fr["bad"].items()})
    probe = json.loads((PROBE / f"{name}.json").read_text())
    metre = float(probe["metre"]["new"]["scale"])
    cam = probe["camera"]["new"]
    with Image.open(list_frames(scan / "frames")[0]) as im:
        native_w = im.size[0]
    K = np.array([[cam["fx"], 0.0, cam["cx"]], [0.0, cam["fy"], cam["cy"]], [0.0, 0.0, 1.0]])
    rot = rotation_quantum(K, native_w)
    video = MAP.video_in_metres(PR.load_video(scan / "intake" / "amb3r_video.npz"), metre)
    rng = float(view_range_m(config))
    ov = S.Overlap(video, rng)
    ms = Q.Measures(ov, video)
    score = S.Detail(scan / "frames", icfg.quality.fft_max_side)
    log(f"[select] {name}: {ov.n} frames, {len(fr['bad'])} bad; AMB3R at {metre:.4f} m per unit; overlap "
        f"{Q.OVERLAP_HOLE:.0%}–{Q.OVERLAP_REPEAT:.0%}; displacement {DISPLACEMENT_DEG}° (quantile {quant:g}); rotation "
        f"{rot['theta_q_deg']:.3f}° (FOV {rot['fov_h_deg']:.2f}°, fx {cam['fx']:.2f}); bands ±{band:.0%}; range {rng:g} m")
    chosen = Q.select(ms, ov, score, good, DISPLACEMENT_DEG, float(rot["theta_q_deg"]), band, quant, log)
    frames = [c["frame"] for c in chosen]
    by = {}
    for c in chosen:
        key = c["closed_by"] or c["rule"]
        by[key] = by.get(key, 0) + 1
    gaps = np.diff(frames)
    doc = {"session": name, "scan": str(scan),
           "rule": {"overlap": [Q.OVERLAP_HOLE, Q.OVERLAP_REPEAT], "overlap_measure": "v2_select.Overlap (footprint)",
                    "displacement_deg": DISPLACEMENT_DEG, "quantile": quant, "rotation_deg": rot["theta_q_deg"],
                    "fov_h_deg": rot["fov_h_deg"], "band": band, "range_m": rng, "metre": metre},
           "n_keyframes": len(frames), "frames": frames, "keyframes": chosen, "closed_by": by,
           "frames_between_keyframes": {"median": float(np.median(gaps)), "max": int(gaps.max())}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1))
    log(f"[select] {name}: {len(frames)} keyframes of {ov.n} frames in {time.time() - t0:.0f} s; closed by {by}; frames "
        f"between keyframes median {np.median(gaps):g}, max {gaps.max()} → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
