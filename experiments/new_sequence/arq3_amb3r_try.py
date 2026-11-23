"""AMB3R-SLAM again on one session with some of its own settings changed (--override, AMB3R's own mechanism), into a
scratch folder — the pipeline's own launch and storage (reconstruction.amb3r_probe.run_amb3r_video, then make_metric
with step 3's DA3 probe) — and then the true loop measured as on the stored pass: the first keyframe against the last.
The session's stored pass is not touched. USER 2026-11-22: "bajale salad para ver que encuentra, y veamos como
corregir la deriva de escala" / "fijate ademas si amb3r-slam tiene alguna perilla que no estamos tocando bien".

    python arq3_amb3r_try.py <name> <scan dir> <out dir> <override> [<override> …]"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SERVER = "/workspace/stac-build/server"
for p in (SERVER, str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
PROBE = Path("/workspace/tmp/new_sequence/arq/probe")
SELECT = Path("/workspace/tmp/new_sequence/arq/select")


def loop(npz: Path, kf_first: int, kf_last: int):
    """AMB3R's depth of the first keyframe against the last one's, both ways (CoverageMap's band of one surface)."""
    import torch
    from reconstruction.amb3r_probe import load_video
    from reconstruction.coverage_fill import CoverageMap
    vm = load_video(npz)
    cm = CoverageMap(vm, "cuda", state=False, range_m=15.0)
    rec = np.asarray(vm["rec_index"])
    a, b = int(np.argmin(np.abs(rec - kf_first))), int(np.argmin(np.abs(rec - kf_last)))

    def agree(x, y):
        X = cm.X[cm.own(x)]
        Xc = X @ cm.R_w2c[y].T + cm.t_w2c[y]
        z = Xc[:, 2]
        zs = torch.where(z > 1e-6, z, torch.ones_like(z))
        u = torch.round(cm.K[y, 0, 0] * Xc[:, 0] / zs + cm.K[y, 0, 2]).long()
        v = torch.round(cm.K[y, 1, 1] * Xc[:, 1] / zs + cm.K[y, 1, 2]).long()
        ok = (z > 1e-3) & (u >= 0) & (u < cm.Ws) & (v >= 0) & (v < cm.Hs)
        d = cm.D[y][v.clamp(0, cm.Hs - 1), u.clamp(0, cm.Ws - 1)]
        ok &= d > 0
        q = d[ok] / z[ok] - 1.0
        q = q[q.abs() < 0.25]
        if len(q) < 50:
            return f"{len(q)} samples on one surface — too few"
        return (f"{len(q)} samples; median {float(q.median()) * 100:+.2f} %, half within "
                f"{float(q.abs().median()) * 100:.2f} %, 90 % within {float(q.abs().quantile(0.9)) * 100:.2f} %")
    C = cm.C.double()
    lines = [f"first {kf_first} / last {kf_last}: cameras {float((C[a] - C[b]).norm()):.2f} m apart | first in last: "
             f"{agree(a, b)} | last in first: {agree(b, a)}"]
    for x, y in ((463, 467), (463, 571), (463, 359), (463, 682)):    # the table of 2026-11-22 (bufferStop's frames)
        if max(x, y) <= int(rec.max()):
            rx, ry = int(np.argmin(np.abs(rec - x))), int(np.argmin(np.abs(rec - y)))
            lines.append(f"frame {x} in frame {y}: {agree(rx, ry)}")
    return "\n[try]   ".join(lines)


def main() -> int:
    name, scan, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    overrides = [a for a in sys.argv[4:] if not a.startswith(("--root=", "--extra="))]
    for a in sys.argv[4:]:                                   # the launcher drops this shell's environment
        if a.startswith("--root="):
            os.environ["STAC_AMB3R_ROOT"] = a[len("--root="):]
        elif a.startswith("--extra="):
            os.environ["STAC_AMB3R_EXTRA"] = a[len("--extra="):]
    from card import card
    from reconstruction import amb3r_probe as A3
    stored = A3.load_video(scan / "intake" / "amb3r_video.npz")
    ident = json.loads(str(stored["meta"]))["identity"]
    out.mkdir(parents=True, exist_ok=True)
    npz = out / "amb3r_video.npz"
    real = subprocess.Popen
    extra = os.environ.get("STAC_AMB3R_EXTRA", "")           # more of run.py's own arguments
    root = os.environ.get("STAC_AMB3R_ROOT")                 # another copy of AMB3R-SLAM (the test copy)
    if root:
        A3.AMB3R = Path(root)
        A3.LAUNCHER = HERE / "amb3r_launch_tests.py"

    def popen(cmd, *a, **k):                          # the pipeline's own command, with AMB3R's --override appended
        if isinstance(cmd, list) and len(cmd) == 3 and cmd[0] == "bash" and "amb3r_launch" in cmd[2]:
            cmd = cmd[:2] + [cmd[2] + (" " + extra if extra else "") + " --override " + " ".join(overrides)]
            print(f"[try] AMB3R{' (' + root + ')' if root else ''} with {extra} --override {' '.join(overrides)}",
                  flush=True)
        return real(cmd, *a, **k)
    A3.subprocess.Popen = popen
    with card(f"AMB3R-SLAM on {name} with {' '.join(overrides)} (test)", session="pccr"):
        A3.run_amb3r_video(scan / "frames", npz, model=str(ident["model"]), grid_wh=tuple(ident["grid_wh"]),
                           log=lambda m: print(m, flush=True))
        rep = A3.make_metric(npz, PROBE / name / "intake" / "focal_depth.npz", lambda m: print(m, flush=True))
        meta = json.loads(str(A3.load_video(npz)["meta"]))
        print(f"[try] {name}: {meta['keyframes_amb3r']} AMB3R keyframes, {meta['loops']} loop closure(s), reasons "
              f"{meta['loop_reject_reasons']}; {meta['time_s']:.0f} s; metre {rep['scale']:.4f} m per unit "
              f"(MAD {rep['mad_rel'] * 100:.1f} %)", flush=True)
        kf = json.loads((SELECT / f"{name}.json").read_text())["frames"]
        print(f"[try] {name}: {loop(npz, kf[0], kf[-1])}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
