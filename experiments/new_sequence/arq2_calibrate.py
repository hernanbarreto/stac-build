"""The validated runs' keyframes measured with step 5's three measures (arq2_select.Measures, on AMB3R's poses and
depth): for every two consecutive keyframes, the overlap, the displacement (the 0.9 quantile of the angle under which
each point the first sees is seen from the two cameras) and the rotation — grouped by the rule of I1 that closed the
second keyframe's window. Read only; under the card's lease.

    python arq2_calibrate.py"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
SERVER = "/workspace/stac-build/server"
for p in (SERVER, str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
P = Path(SERVER) / "projects"
SESSIONS = (("bufferStop", "bufferStop/scans/2026-08-11/src_default"), ("pccr3108", "pccr/scans/2026-08-31/src_default"),
            ("pccr2408", "pccr/scans/2026-08-24/src_default"))
OUT = Path("/workspace/tmp/new_sequence/arq/select2/calibration.json")


def q(a):
    a = np.asarray([x for x in a if np.isfinite(x)], np.float64)
    if not len(a):
        return None
    return {"n": int(len(a)), "p25": float(np.percentile(a, 25)), "median": float(np.median(a)),
            "p75": float(np.percentile(a, 75))}


def main() -> int:
    import arq2_select as A
    import v2_select as S
    from card import card
    from intake.config import load_intake_config
    from reconstruction import amb3r_probe as A3
    from view_range import view_range_m
    config = yaml.safe_load((Path(SERVER) / "config.yaml").read_text())
    quant = float(load_intake_config(config).parallax.parallax_quantile)
    rng = float(view_range_m(config))
    report = {}
    with card("calibration of step 5's measures on the validated runs", session="pccr2408_prueba"):
        for name, rel in SESSIONS:
            scan = P / rel
            video = A3.load_video(scan / "intake" / "amb3r_video.npz")
            ov = S.Overlap(video, rng)
            ms = A.Measures(ov, video)
            pos = {int(f): i for i, f in enumerate(ov.frame_no)}
            sel = json.loads((scan / "frames" / "selected_frames.json").read_text())
            fx = float(sel["parallax_quantum_px"]) / float(np.tan(np.deg2rad(float(sel["parallax_quantum_deg"]))))
            rows = []
            for k in sel.get("keyframes", []):
                a, b = k.get("anchor"), int(k["frame"])
                if a is None or k.get("parallax_px") is None or k.get("rotation_deg") is None:
                    continue
                o, par, rot = ms.against(pos[int(a)], [pos[b]], quant)
                old_par = float(np.rad2deg(np.arctan(float(k["parallax_px"]) / fx)))
                rows.append((str(k.get("closed_by")), float(o[0]), float(par[0]), float(rot[0]), old_par,
                             float(k["rotation_deg"]), b - int(a)))
            groups = {}
            for r in rows:
                groups.setdefault(r[0], []).append(r)

            def stats(rs):
                return {"overlap": q([r[1] for r in rs]), "displacement_deg": q([r[2] for r in rs]),
                        "rotation_deg": q([r[3] for r in rs]), "i1_parallax_deg": q([r[4] for r in rs]),
                        "i1_rotation_deg": q([r[5] for r in rs]),
                        "ratio_displacement": q([r[2] / r[4] for r in rs if r[4] > 0]),
                        "ratio_rotation": q([r[3] / r[5] for r in rs if r[5] > 0]),
                        "frames_apart": q([r[6] for r in rs])}
            rep = {"pairs": len(rows), "fx_native_px": fx, "all": stats(rows)}
            for g, rs in sorted(groups.items()):
                rep[f"closed_by_{g}"] = stats(rs)
            print(f"\n== {name}: {len(rows)} windows of I1 (anchor -> keyframe), fx {fx:.1f} px", flush=True)
            f = lambda d, s="": ("—" if d is None else f"{d['median']:.3g}{s} [{d['p25']:.3g}–{d['p75']:.3g}]")  # noqa: E731
            for k_, v in rep.items():
                if not isinstance(v, dict) or "overlap" not in v:
                    continue
                n = (v["overlap"] or {}).get("n", 0)
                print(f"  {k_:22s} ({n:4d}) displacement new {f(v['displacement_deg'], '°')} vs I1 {f(v['i1_parallax_deg'], '°')}"
                      f" ratio {f(v['ratio_displacement'])} | rotation new {f(v['rotation_deg'], '°')} vs I1 "
                      f"{f(v['i1_rotation_deg'], '°')} | overlap {f(v['overlap'])} | frames {f(v['frames_apart'])}",
                      flush=True)
            report[name] = rep
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    print(f"\n→ {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
