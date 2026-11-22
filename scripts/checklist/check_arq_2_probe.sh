#!/usr/bin/env bash
# Line 2 of experiments/new_sequence/CHECKLIST_implementacion.md — step 3 of the architecture, THE DA3 PROBE on 16 of
# AMB3R's own keyframes. Read only; the same answer at every run. It reads what
# experiments/new_sequence/arq3_probe_run.py wrote in /workspace/tmp/new_sequence/arq/probe/ and says CLOSED when, for
# each of the three test sessions:
#   * the results come from the code as it is now (the sha256 of arq3_probe.py and arq3_probe_run.py);
#   * the probe's frames are 16 of AMB3R's own keyframes that are good frames (step 1), spread over that list, and
#     DA3 measured on exactly them;
#   * AMB3R's metre and the camera were measured;
# and it shows them beside the metre and the camera the session has today.
set -u
cd /workspace/stac-build/server || exit 1
exec /workspace/miniforge3/envs/da3/bin/python - <<'EOF'
import hashlib
import json
import sys
from pathlib import Path

HERE = Path("/workspace/stac-build/experiments/new_sequence")
sys.path.insert(0, str(HERE))
import arq3_probe as A

OUT = Path("/workspace/tmp/new_sequence/arq/probe")
FRAMES = Path("/workspace/tmp/new_sequence/arq/frames")
SESSIONS = ("bufferStop", "pccr_2026-08-31", "pccr_2026-08-24")
code = {n: hashlib.sha256((HERE / n).read_bytes()).hexdigest() for n in ("arq3_probe.py", "arq3_probe_run.py")}
n_probe = int(A.probe_config().focal_probe_frames)
ok = True


def line(good, text):
    global ok
    ok = ok and good
    print(f"  {'ok ' if good else 'NO '} {text}")


for name in SESSIONS:
    print(name)
    f = OUT / f"{name}.json"
    if not f.exists():
        line(False, f"no result ({f})")
        continue
    d = json.loads(f.read_text())
    line(d["code"] == code, "the results come from the code as it is now")
    fr = json.loads((FRAMES / f"{name}.json").read_text())
    good = set(fr["frames"]) - {int(k) for k in fr["bad"]}
    video = A.load_video(Path(d["scan"]) / "intake" / "amb3r_video.npz")
    want = A.probe_frames(video, good, n_probe)
    line(d["probe_frames"] == want and len(want) == n_probe,
         f"{len(d['probe_frames'])} probe frames = AMB3R's own keyframes that are good frames, spread over its "
         f"{len(video['kf'])}: {d['probe_frames']}")
    files = json.loads((OUT / name / "intake" / "focal_probe.json").read_text())["frames"]
    line([int(Path(x).stem) for x in files] == want, "DA3 measured exactly those frames")
    mt, mn = d["metre"]["today"], d["metre"]["new"]
    line(mn["frames"] >= 3, f"AMB3R's metre measured on {mn['frames']} frames")
    print(f"       metre (m per AMB3R unit): today {mt['scale']:.4f} (MAD {mt['mad_rel'] * 100:.1f} %) | new "
          f"{mn['scale']:.4f} (MAD {mn['mad_rel'] * 100:.1f} %) | {(mn['scale'] / mt['scale'] - 1) * 100:+.2f} %")
    ct, cn = d["camera"]["today"], d["camera"]["new"]
    print("       camera (px):  " + " | ".join(f"{k} today {ct[k]:.2f} new {cn[k]:.2f}" for k in ("fx", "fy", "cx", "cy")))
print("CLOSED" if ok else "NOT CLOSED")
EOF
