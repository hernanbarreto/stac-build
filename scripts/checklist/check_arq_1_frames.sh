#!/usr/bin/env bash
# Line 1 of experiments/new_sequence/CHECKLIST_implementacion.md — step 1 of the architecture, THE GOOD FRAMES.
# Read only; the same answer at every run. It reads what experiments/new_sequence/arq3_frames_run.py wrote in
# /workspace/tmp/new_sequence/arq/frames/ and says CLOSED when, for each of the five sessions:
#   * the results come from the code as it is now (the sha256 of arq3_frames.py and arq3_frames_run.py);
#   * they cover every frame on disk now;
#   * the card's path (the pipeline's) equals I0's own functions (the record) on every frame: the same decisions,
#     the same luma and burnt share, the same downscaled images bit for bit, the sharpness within 1e-9;
# and the bad frames are exactly the 12 of pccr 2026-08-24's two door crossings (video 2484-2490 and 3589-3593),
# rejected for their detail, and none of any other session.
set -u
cd /workspace/stac-build/server || exit 1
exec /workspace/miniforge3/envs/da3/bin/python - <<'EOF'
import hashlib
import json
from pathlib import Path

from intake.quality import list_frames

HERE = Path("/workspace/stac-build/experiments/new_sequence")
OUT = Path("/workspace/tmp/new_sequence/arq/frames")
SESSIONS = ("bufferStop", "observatorio", "pccr_2026-08-24", "pccr_2026-08-31", "zaragoza")
EXPECTED = {"pccr_2026-08-24": {f: "detail" for f in list(range(2484, 2491)) + list(range(3589, 3594))}}
code = {n: hashlib.sha256((HERE / n).read_bytes()).hexdigest() for n in ("arq3_frames.py", "arq3_frames_run.py")}
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
    on_disk = [int(p.stem) for p in list_frames(Path(d["scan"]) / "frames")]
    line(d["frames"] == on_disk, f"every frame on disk measured ({len(on_disk)})")
    e = d["equal"]
    line(e["frames"] and e["decisions_differ"] == 0, f"card = record: decisions that differ {e['decisions_differ']}")
    line(e["luma"] and e["burnt"], "card = record: luma and burnt share identical")
    line(e["small_images_differ"] == 0, f"card = record: downscaled images that differ {e['small_images_differ']}")
    line(e["fft_max_rel_diff"] <= 1e-9, f"card = record: sharpness within {e['fft_max_rel_diff']:.1e} (≤ 1e-9)")
    bad = {int(k): v for k, v in d["bad"].items()}
    want = EXPECTED.get(name, {})
    line(bad == want, f"bad frames {sorted(bad) if bad else 'none'}" + ("" if bad == want else f" — expected {sorted(want) or 'none'}"))
print("CLOSED" if ok else "NOT CLOSED")
EOF
