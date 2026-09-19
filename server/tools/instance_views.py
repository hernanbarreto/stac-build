"""One object, every keyframe that sees it: where its cloud points land.

For a single instance, this renders one panel per keyframe in which SAM3 drew
it. Each panel carries the mask outline and the instance's own cloud points
reprojected through THAT keyframe's pose, split by where each point was born:

    · points whose own origin IS this keyframe — the mask is literally their
      crop, so they must land inside it. They are the control: if they miss,
      the pose or the depth of this keyframe is wrong.
    · points born in any OTHER keyframe — the cross-view test. A correctly
      placed object puts these inside the mask too; a duplicate puts one
      visit's points inside and the other visit's somewhere else, and the
      distance between the two is the thing the correction has to close.

USER 2026-09-18: "mostrame el kf 0 con la reproyección de los puntos sobre la
máscara de un objeto en particular desde todas las vistas que se ve el objeto".

    python -m tools.instance_views --session <dir> --instance 203 [--max-views 16]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

OWN = (80, 230, 90)         # points born in this keyframe  (green)
OTHER = (70, 120, 250)      # points born elsewhere          (orange-red in BGR)
OUTLINE = (255, 255, 255)


def _frames_dir(session_dir: Path) -> Path:
    for c in (session_dir / "frames", session_dir / "frames_valid"):
        if c.is_dir() and any(c.glob("*.jpg")):
            return c
    raise SystemExit(f"no frames under {session_dir}")


def _image(frames: Path, fnum: int) -> Optional[np.ndarray]:
    for pat in (f"{fnum:06d}.jpg", f"{fnum}.jpg", f"{fnum:06d}.png"):
        p = frames / pat
        if p.exists():
            return cv2.imread(str(p))
    return None


def render(session_dir: Path, output_dir: Path, iid: int, out_dir: Path,
           max_views: int = 16, max_points: int = 60000, log=print) -> Path:
    from reconstruction.surface_fit.hole_audit import _Evidence
    from correction.session import load_session
    from segmentation.erase import _mask_obj_by_iid

    ev = _Evidence(output_dir, session_dir)
    if not ev.ok:
        raise SystemExit("could not load masks + cameras for this session")
    log(f"[views] {ev.space.describe()}")
    sess = load_session(output_dir)
    insts = {int(i.get("instance_id", i.get("id"))): i
             for i in json.loads((output_dir / "segmentation_result.json").read_text())["instances"]}
    inst = insts.get(int(iid))
    if inst is None:
        raise SystemExit(f"instance {iid} is not in segmentation_result.json")
    label = str(inst.get("label", "segment"))
    oid = _mask_obj_by_iid(output_dir).get(int(iid))
    if oid is None:
        raise SystemExit(f"instance {iid} has no mask object id")

    gi = np.asarray(inst.get("globalIndices") or [], np.int64)
    gi = gi[(gi >= 0) & (gi < len(sess.xyz))]
    if not len(gi):
        raise SystemExit(f"instance {iid} has no points")
    rng = np.random.default_rng(0)
    if len(gi) > max_points:
        gi = gi[rng.choice(len(gi), max_points, replace=False)]
    pts = sess.xyz[gi]
    born = np.asarray(sess.fg)[gi].astype(np.int64)

    mask_frames = [mf for mf, _k in ev.frames_for(int(oid))]
    views = sorted(mask_frames)
    if max_views and len(views) > max_views:
        idx = np.linspace(0, len(views) - 1, max_views).round().astype(int)
        views = [views[i] for i in sorted(set(idx.tolist()))]
    log(f"[views] {label}#{iid}: {len(gi):,} pts, mask in {len(mask_frames)} "
        f"keyframe(s), rendering {len(views)}")

    frames = _frames_dir(session_dir)
    panels: List[np.ndarray] = []
    rows: List[str] = []
    for mf in views:
        cf = ev.cloud_frame(int(mf))
        if cf is None:
            continue
        img = _image(frames, int(cf))
        c2w = ev.pose_at(int(mf))
        K = ev.K_at(int(mf))
        key = f"f{int(mf)}_o{int(oid)}"
        if img is None or c2w is None or K is None or key not in ev.masks.files:
            continue
        m = np.asarray(ev.masks[key]) > 0
        ih, iw = img.shape[:2]
        mh, mw = m.shape[:2]
        canvas = img.copy()

        c2w4 = np.eye(4)
        c2w4[:c2w.shape[0], :c2w.shape[1]] = c2w
        M = np.linalg.inv(c2w4)
        p = (M[:3, :3] @ pts.T).T + M[:3, 3]
        z = p[:, 2]
        front = z > 0.05
        u = np.full(len(pts), -1e9)
        v = np.full(len(pts), -1e9)
        u[front] = K[0, 0] * p[front, 0] / z[front] + K[0, 2]
        v[front] = K[1, 1] * p[front, 1] / z[front] + K[1, 2]
        px = np.rint(u * iw / ev.kw).astype(np.int64)
        py = np.rint(v * ih / ev.kh).astype(np.int64)
        inb = front & (px >= 0) & (px < iw) & (py >= 0) & (py < ih)
        is_own = born == int(cf)

        stat = {}
        for name, sel, col in (("own", inb & is_own, OWN),
                               ("other", inb & ~is_own, OTHER)):
            if not sel.any():
                stat[name] = (0, 0)
                continue
            xs, ys = px[sel], py[sel]
            canvas[ys, xs] = col
            mu = np.clip(xs * mw // iw, 0, mw - 1)
            mv = np.clip(ys * mh // ih, 0, mh - 1)
            stat[name] = (int(m[mv, mu].sum()), int(sel.sum()))

        big = cv2.resize(m.astype(np.uint8), (iw, ih),
                         interpolation=cv2.INTER_NEAREST).astype(bool)
        cnts, _ = cv2.findContours(big.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, cnts, -1, OUTLINE, 3)

        def _pc(t):
            return 100.0 * t[0] / t[1] if t[1] else float("nan")

        bar = np.zeros((44, iw, 3), np.uint8)
        cv2.putText(bar, f"kf {mf}  frame {cf}", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(bar, f"nacidos aqui {stat['own'][0]}/{stat['own'][1]} "
                         f"({_pc(stat['own']):.0f}%)", (8, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, OWN, 1)
        cv2.putText(bar, f"de otras vistas {stat['other'][0]}/{stat['other'][1]} "
                         f"({_pc(stat['other']):.0f}%)", (iw // 2, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, OTHER, 1)
        panels.append(np.vstack([canvas, bar]))
        rows.append(f"kf {mf:>3} (frame {cf:>4}): propios {_pc(stat['own']):>5.1f}% "
                    f"({stat['own'][1]:>6,})  |  otras vistas {_pc(stat['other']):>5.1f}% "
                    f"({stat['other'][1]:>6,})")

    if not panels:
        raise SystemExit("no renderable view for this instance")
    for r in rows:
        log("[views]   " + r)

    # contact sheet, 3 per row, every panel scaled to the same width
    W = 640
    scaled = []
    for p in panels:
        h = int(round(p.shape[0] * W / p.shape[1]))
        scaled.append(cv2.resize(p, (W, h)))
    H = max(s.shape[0] for s in scaled)
    scaled = [np.pad(s, ((0, H - s.shape[0]), (0, 0), (0, 0))) for s in scaled]
    per = 3
    sheet_rows = [np.hstack(scaled[i:i + per]) for i in range(0, len(scaled), per)]
    wmax = max(r.shape[1] for r in sheet_rows)
    sheet_rows = [np.pad(r, ((0, 0), (0, wmax - r.shape[1]), (0, 0)))
                  for r in sheet_rows]
    sheet = np.vstack(sheet_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    p_out = out_dir / f"inst_{iid:03d}_{label}_views.png"
    cv2.imwrite(str(p_out), sheet)
    log(f"[views] -> {p_out}")
    return p_out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", required=True)
    ap.add_argument("--instance", type=int, required=True)
    ap.add_argument("--max-views", type=int, default=16)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    sess = Path(a.session)
    out = Path(a.out) if a.out else sess / "output" / "instance_views"
    render(sess, sess / "output", int(a.instance), out, max_views=int(a.max_views))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
