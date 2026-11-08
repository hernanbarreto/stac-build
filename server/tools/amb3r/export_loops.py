"""The session's REVISIT pairs as external loop candidates for AMB3R-SLAM (USER 2026-11-08: "uno para
ver dónde van los cierres, los detecte o no, los seleccionamos nosotros … la idea es decirle dónde
tiene que ajustar … si no coincide debe usar los que detecta más los nuestros").

The pairs are EXACTLY the ones F4 tracks across: precision/tracks.revisit_pairs over the structure
probe's kf_covisibility (two moments of the walk that see the same surface and that the walk does not
tie already) — no new criterion. Written as VIDEO FRAME NUMBER pairs [old, new]; the AMB3R tool maps
them onto its strided sequence (tools/amb3r/chain.py --loops).

Usage: python export_loops.py <session src_default dir> <out.json>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/stac-build/server")


def keyframe_numbers(ses: Path) -> list:
    sel = json.loads((ses / "frames" / "selected_frames.json").read_text())
    entries = sel.get("keyframes") or sel.get("selected") or sel.get("selected_files") or sel
    if isinstance(entries, dict):
        entries = entries.get("frames") or entries.get("keyframes")
    out = []
    for e in entries:
        if isinstance(e, dict):
            out.append(int(e["frame"]))
        else:
            out.append(int(Path(str(e)).stem))
    return out


def main() -> int:
    ses, out_p = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
    from precision.config import load_precision_config
    from precision.tracks import keyframe_windows, revisit_pairs
    kf = keyframe_numbers(ses)
    tcfg = load_precision_config(session_dir=ses).tracks
    windows = keyframe_windows(kf, tcfg.window_frames, tcfg.window_overlap_frac, tcfg.query_frames_per_window)
    cp = ses / "intake" / "cubes.json"
    plan_cv = json.loads(cp.read_text()).get("kf_covisibility") if cp.exists() else None
    if not plan_cv:
        raise SystemExit(f"{cp} holds no kf_covisibility — the structure probe's cut did not run on this session")
    pairs, rep = revisit_pairs([("structure probe (intake/cubes.json)", plan_cv)], kf, windows,
                               int(tcfg.window_frames) // 2)
    fp = sorted([int(kf[i]), int(kf[j])] if kf[i] < kf[j] else [int(kf[j]), int(kf[i])] for i, j in pairs)
    out_p.write_text(json.dumps({"source": str(ses), "rule": "precision.tracks.revisit_pairs (F4's own)",
                                 "pairs": fp, "report": rep}, indent=1))
    print(f"{len(fp)} revisit pair(s) (frame numbers, old→new) → {out_p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
