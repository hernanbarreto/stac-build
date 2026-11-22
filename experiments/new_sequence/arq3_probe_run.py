"""Line 2 of CHECKLIST_implementacion.md — step 3, the DA3 probe on 16 of AMB3R's own keyframes — on the three test
sessions, beside the camera and the metre they have today; written for the check (scripts/checklist/
check_arq_2_probe.sh). Read only on the sessions.

    python arq3_probe_run.py"""
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVER = "/workspace/stac-build/server"
for p in (SERVER, str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
P = Path(SERVER) / "projects"
OUT = Path("/workspace/tmp/new_sequence/arq/probe")
FRAMES = Path("/workspace/tmp/new_sequence/arq/frames")          # step 1's good frames (arq3_frames_run.py)
SESSIONS = (("bufferStop", "bufferStop/scans/2026-08-11/src_default"),
            ("pccr_2026-08-31", "pccr/scans/2026-08-31/src_default"),
            ("pccr_2026-08-24", "pccr/scans/2026-08-24/src_default"))


def sha256(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> int:
    import arq3_probe as A
    code = {n: sha256(HERE / n) for n in ("arq3_probe.py", "arq3_probe_run.py")}
    from card import card
    with card("step 3 of the architecture: the DA3 probe on AMB3R's own keyframes (line 2)", session="pccr"):
        for name, rel in SESSIONS:
            scan = P / rel
            fr = json.loads((FRAMES / f"{name}.json").read_text())
            good = set(fr["frames"]) - {int(k) for k in fr["bad"]}
            t0 = time.time()
            r = A.run(scan, OUT / name, good, sys.executable, log=print)
            today_cam = json.loads((scan / "intake" / "focal_probe.json").read_text())
            video = A.load_video(scan / "intake" / "amb3r_video.npz")
            today_metre = json.loads(str(video["metric_report"]))
            doc = {"session": name, "scan": str(scan), "code": code, "probe_frames": r["frames"],
                   "amb3r_keyframes": len(video["kf"]),
                   "camera": {"today": {k: today_cam[k] for k in ("fx", "fy", "cx", "cy", "spread_pct")},
                              "today_frames": today_cam["frames"], "new": r["camera"]},
                   "metre": {"today": today_metre, "new": r["metre"]},
                   "seconds": round(time.time() - t0, 1)}
            (OUT / f"{name}.json").write_text(json.dumps(doc))
            mt, mn = today_metre["scale"], r["metre"]["scale"]
            ct, cn = doc["camera"]["today"], r["camera"]
            print(f"[probe] {name}: metre {mt:.4f} → {mn:.4f} m per AMB3R unit ({(mn / mt - 1) * 100:+.2f} %, "
                  f"{r['metre']['frames']} frames, MAD {r['metre']['mad_rel'] * 100:.1f} %) | fx {ct['fx']:.2f} → "
                  f"{cn['fx']:.2f}, fy {ct['fy']:.2f} → {cn['fy']:.2f}, cx {ct['cx']:.2f} → {cn['cx']:.2f}, cy "
                  f"{ct['cy']:.2f} → {cn['cy']:.2f} ({doc['seconds']:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
