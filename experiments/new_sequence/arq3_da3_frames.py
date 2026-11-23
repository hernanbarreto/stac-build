"""DA3's metric depth for EVERY frame of a scan — the 'sensor' AMB3R's metric backend anchors each submap's scale to
(the test of the three nested improvements, USER 2026-11-22: "luego de esta probamos con las 3 mejoras anidadas").
The pipeline's own extractor (extract_da3_depth.py) over windows of WINDOW consecutive frames at 504 (the probe's
window size; DA3 per cube's resolution); each frame's depth kept as <out>/<stem>.npy (float16, DA3's grid).

    python arq3_da3_frames.py <scan dir> <out dir>"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SERVER = "/workspace/stac-build/server"
if SERVER not in sys.path:
    sys.path.insert(0, SERVER)
WINDOW = 16
RES = 504
MODEL = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"


def main() -> int:
    scan, out = Path(sys.argv[1]), Path(sys.argv[2])
    import da3_weights
    import repro
    import intake.walk as W
    from intake.quality import list_frames
    from intake.vram import IDENTITY_EXIT, OOM_EXIT
    log = lambda m: print(m, flush=True)   # noqa: E731
    t0 = time.time()
    names = [p.name for p in list_frames(scan / "frames")]
    wins = [names[a:a + WINDOW] for a in range(0, len(names), WINDOW)]
    if len(wins) > 1 and len(wins[-1]) == 1:                 # a lone frame measures no multi-view depth
        wins[-1].insert(0, wins[-2].pop())
    wdir = out / "_windows"
    wdir.mkdir(parents=True, exist_ok=True)
    spec = {"windows": [[f"frames/{n}" for n in w] for w in wins], "process_res": RES, "model_id": MODEL,
            "da3_environment": W.da3_identity(sys.executable, MODEL, log)}
    (wdir / "windows.json").write_text(json.dumps(spec))
    cmd = [sys.executable, str(Path(SERVER) / "extract_da3_depth.py"), "--image_dir", str(scan / "frames"),
           "--output_dir", str(wdir), "--model", MODEL, "--process_res", str(RES),
           "--windows_json", str(wdir / "windows.json"), "--windows_root", str(scan)]
    log(f"[da3] {len(names)} frames in {len(wins)} windows of ≤ {WINDOW} at {RES}")
    repro.require_exclusive_gpu(log=log)
    p = subprocess.run(cmd, env=da3_weights.hf_env(repro.deterministic_env()), text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if p.returncode in (OOM_EXIT, IDENTITY_EXIT) or p.returncode != 0:
        print(p.stdout[-3000:], flush=True)
        raise SystemExit(f"DA3 exited with code {p.returncode}")
    n = 0
    for k in range(len(wins)):
        with np.load(wdir / f"window_{k:04d}.npz") as z:
            if "is_metric" in z.files and not bool(z["is_metric"]):
                raise SystemExit(f"window {k} is not metric")
            fr = [int(x) for x in z["frames"]]
            if fr != [int(Path(nm).stem) for nm in wins[k]]:
                raise SystemExit(f"window {k} holds frames {fr[:3]}… not its own")
            for j, f in enumerate(fr):
                np.save(out / f"{Path(wins[k][j]).stem}.npy", np.asarray(z["depth"][j], np.float16))
                n += 1
        (wdir / f"window_{k:04d}.npz").unlink()
    shape = np.load(out / f"{Path(names[0]).stem}.npy").shape
    log(f"[da3] {n} frames' metric depth at {shape[1]}x{shape[0]} → {out} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
