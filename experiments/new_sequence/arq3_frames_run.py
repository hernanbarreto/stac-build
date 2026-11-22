"""Line 1 of CHECKLIST_implementacion.md — step 1, the good frames — on the five sessions: every frame measured by the
card's path (the pipeline's) and by I0's own functions (the record), compared, and the bad frames written for the
check (scripts/checklist/check_arq_1_frames.sh). Read only on the sessions.

    python arq3_frames_run.py"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SERVER = "/workspace/stac-build/server"
for p in (SERVER, str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
P = Path(SERVER) / "projects"
OUT = Path("/workspace/tmp/new_sequence/arq/frames")
SESSIONS = (("bufferStop", "bufferStop/scans/2026-08-11/src_default"),
            ("observatorio", "observatorio/scans/2026-08-11/src_default"),
            ("pccr_2026-08-24", "pccr/scans/2026-08-24/src_default"),
            ("pccr_2026-08-31", "pccr/scans/2026-08-31/src_default"),
            ("zaragoza", "zaragoza/scans/2026-06-03/src_default"))


def sha256(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> int:
    import arq3_frames as A
    fft_max_side, clip_hi = A.quality_knobs()
    code = {"arq3_frames.py": sha256(HERE / "arq3_frames.py"), "arq3_frames_run.py": sha256(HERE / Path(__file__).name)}
    rec, gpu = {}, {}
    for name, rel in SESSIONS:                        # the record, on the CPU (before the card: its pool forks)
        frames, paths = A.frame_list(P / rel / "frames")
        t0 = time.time()
        luma, burnt, fft, dig = A.measures_numpy(paths, fft_max_side, clip_hi, workers=8)
        med = A.neighbour_median(fft, "numpy")
        rec[name] = dict(frames=frames, luma=luma, burnt=burnt, fft=fft, dig=dig,
                         why=A.decide(luma, burnt, fft, med), s=time.time() - t0)
        print(f"[record] {name}: {len(frames)} frames in {rec[name]['s']:.0f} s", flush=True)
    import torch
    from card import card
    with card("step 1 of the architecture: the good frames (line 1 of the implementation checklist)", session="pccr"):
        for name, rel in SESSIONS:                    # the card's path
            frames, paths = A.frame_list(P / rel / "frames")
            t0 = time.time()
            luma, burnt, fft, dig = A.measures_cuda(paths, fft_max_side, clip_hi, digests=True)
            med = A.neighbour_median(fft, "cuda")
            torch.cuda.synchronize()
            gpu[name] = dict(frames=frames, luma=luma, burnt=burnt, fft=fft, dig=dig,
                             why=A.decide(luma, burnt, fft, med), s=time.time() - t0)
            print(f"[card] {name}: {len(frames)} frames in {gpu[name]['s']:.0f} s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rel in SESSIONS:
        r, c = rec[name], gpu[name]
        same_frames = r["frames"] == c["frames"]
        rel_fft = np.abs(c["fft"] - r["fft"]) / np.abs(r["fft"]) if same_frames else np.array([np.inf])
        doc = {"session": name, "scan": str(P / rel), "code": code, "fft_max_side": fft_max_side, "clip_hi": clip_hi,
               "rule": {"luma_burnt": A.LUMA_BURNT, "burnt_share_max": A.BURNT_SHARE_MAX, "sharp_min": A.SHARP_MIN,
                        "sharp_neighbours": A.SHARP_NEIGHBOURS},
               "frames": c["frames"],
               "bad": {str(f): w for f, w in zip(c["frames"], c["why"]) if w},
               "equal": {"frames": same_frames,
                         "luma": bool(same_frames and np.array_equal(r["luma"], c["luma"])),
                         "burnt": bool(same_frames and np.array_equal(r["burnt"], c["burnt"])),
                         "small_images_differ": (sum(a != b for a, b in zip(r["dig"], c["dig"]))
                                                 if same_frames else -1),
                         "fft_max_rel_diff": float(rel_fft.max()),
                         "decisions_differ": sum(a != b for a, b in zip(r["why"], c["why"])) if same_frames else -1},
               "seconds": {"record": round(r["s"], 1), "card": round(c["s"], 1)}}
        (OUT / f"{name}.json").write_text(json.dumps(doc))
        e = doc["equal"]
        print(f"[frames] {name}: {len(c['frames'])} frames, bad {doc['bad'] or 'none'} | card = record: decisions "
              f"differ {e['decisions_differ']}, luma {e['luma']}, burnt {e['burnt']}, small images differ "
              f"{e['small_images_differ']}, fft max rel diff {e['fft_max_rel_diff']:.1e}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
