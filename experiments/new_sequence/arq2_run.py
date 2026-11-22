"""The keyframe selection of steps 1 and 5 (arq2_select.py) on the three test sessions, under the card's lease, then
the keyframes against the original runs and the test of 2026-11-21. Reads the sessions, writes only under OUT.

    python arq2_run.py"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SERVER = "/workspace/stac-build/server"
for p in (SERVER, str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
P = Path(SERVER) / "projects"
OUT = Path("/workspace/tmp/new_sequence/arq/select2")
SESSIONS = (("bufferStop", "bufferStop_prueba", "2026-08-11"), ("pccr3108", "pccr3108_prueba", "2026-08-31"),
            ("pccr2408", "pccr2408_prueba", "2026-08-24"))


def kf(scan: Path):
    return np.loadtxt(scan / "output" / "camera_frames.txt", dtype=np.int64)


def main() -> int:
    from card import card
    OUT.mkdir(parents=True, exist_ok=True)
    with card("keyframe selection of the architecture's steps 1 and 5", session="pccr2408_prueba"):
        for name, test, date in SESSIONS:
            orig = P / ("bufferStop" if name == "bufferStop" else "pccr") / "scans" / date / "src_default"
            tscan = P / test / "scans" / date / "src_default"
            rot = json.loads((orig / "frames" / "selected_frames.json").read_text())["rotation_quantum_deg"]
            print(f"\n===== {name}", flush=True)
            r = subprocess.run([sys.executable, str(HERE / "arq2_select.py"), str(tscan),
                                str(tscan / "intake" / "amb3r_video.npz"), str(orig / "frames" / "quality_features.json"),
                                repr(float(rot)), str(OUT / f"{name}.json")], cwd=SERVER, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for line in r.stdout.splitlines():
                if line.strip() and not line.startswith("[Config]"):
                    print(line, flush=True)
            if r.returncode != 0:
                print(f"{name}: the selection failed (exit {r.returncode})", flush=True)
                return 1
    print("\n===== the keyframes: original / test of 2026-11-21 / this selection", flush=True)
    for name, test, date in SESSIONS:
        orig = P / ("bufferStop" if name == "bufferStop" else "pccr") / "scans" / date / "src_default"
        a, b = kf(orig), kf(P / test / "scans" / date / "src_default")
        c = np.asarray(json.loads((OUT / f"{name}.json").read_text())["frames"], np.int64)
        print(f"{name}: {len(a)} / {len(b)} / {len(c)} keyframes; frames between keyframes max {np.diff(a).max()} / "
              f"{np.diff(b).max()} / {np.diff(c).max()}", flush=True)
        if name == "bufferStop":
            print(f"  video 425–450: {int(((a > 425) & (a < 450)).sum())} / {int(((b > 425) & (b < 450)).sum())} / "
                  f"{int(((c > 425) & (c < 450)).sum())} keyframes strictly between", flush=True)
        if name == "pccr2408":
            n = 13693
            edges = list(range(0, n, 685))
            edges[-1] = n
            for lo, hi in zip(edges[:-1], edges[1:]):
                na, nb, nc = (int(((x >= lo) & (x < hi)).sum()) for x in (a, b, c))
                print(f"  video {lo:5d}–{hi - 1:5d}: {na:4d} / {nb:4d} / {nc:4d}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
