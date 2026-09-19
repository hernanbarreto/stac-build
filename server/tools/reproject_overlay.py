"""See where the cloud lands: one image per keyframe with, for every segment
visible in it, the SILHOUETTE of its SAM3 mask and that instance's own cloud
points reprojected through the keyframe's pose — both in the same colour.

USER 2026-09-18: "quiero ver yo donde caen los puntos". Every number the
certification reports about a duplicate is an abstraction over exactly this:
a point of object X, seen from keyframe K, must land inside X's mask in K.
When it does not, either the point is misplaced or the pose is. This draws it
instead of summarising it, and it is generic — no assumption about the shape of
the walk or about where duplicates appear.

    python -m tools.reproject_overlay --session <dir> [--kf 0] [--limit N]
                                      [--out <dir>] [--point-size 2]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# distinct, high-contrast hues; the same colour is used for a segment's
# silhouette and for its reprojected points, so the eye pairs them directly
_PALETTE = [
    (66, 135, 245), (245, 96, 66), (66, 245, 132), (245, 221, 66),
    (196, 66, 245), (66, 245, 233), (245, 66, 155), (150, 245, 66),
    (245, 150, 66), (110, 66, 245), (66, 200, 245), (245, 66, 66),
    (100, 245, 180), (200, 200, 66), (245, 120, 200), (120, 180, 245),
]


def _colour(i: int) -> Tuple[int, int, int]:
    return _PALETTE[i % len(_PALETTE)]


def _frames_dir(session_dir: Path) -> Path:
    for c in (session_dir / "frames_valid", session_dir / "frames"):
        if c.is_dir() and any(c.glob("*.jpg")):
            return c
    raise SystemExit(f"no frames under {session_dir}")


def _frame_image(frames: Path, fnum: int) -> Optional[np.ndarray]:
    for pat in (f"{fnum:06d}.jpg", f"{fnum}.jpg", f"{fnum:06d}.png"):
        p = frames / pat
        if p.exists():
            return cv2.imread(str(p))
    return None


def _instances(output_dir: Path) -> List[dict]:
    p = output_dir / "segmentation_result.json"
    if not p.exists():
        raise SystemExit("no segmentation_result.json — run the segmentation first")
    return json.loads(p.read_text()).get("instances", [])


def _oid_map(output_dir: Path, instances: List[dict]) -> Dict[int, int]:
    """instance_id -> mask object id. The stored map when it exists; the
    documented instance_id - 1 otherwise (CLAUDE.md)."""
    try:
        from segmentation.erase import _mask_obj_by_iid
        m = _mask_obj_by_iid(output_dir)
        if m:
            return {int(k): int(v) for k, v in m.items()}
    except Exception:
        pass
    out = {}
    for i in instances:
        iid = i.get("instance_id", i.get("id"))
        if iid is not None:
            out[int(iid)] = int(iid) - 1
    return out


def _instance_mask(ev, sess, gi: np.ndarray, fnum: int, mask_frame: int,
                   fallback_oid: Optional[int]) -> Optional[np.ndarray]:
    """The instance's silhouette in one frame — the UNION of every SAM3 mask its
    own points were traced from.

    A single oid is not enough once the pipeline fuses fragments: pccr's floor
    arrives as 86 masks of one surface and the consolidation merges them into
    instance #44, which then claims exactly ONE mask id. Frame 1 carries the
    floor as masks o0..o31 and none of them is the one #44 claims, so the floor
    simply vanished from this view — and, far worse, from every downstream
    question of the form "where is this object's mask in keyframe K".

    The provenance answers it without any bookkeeping: every point knows the
    frame and the pixel it came from, so the masks covering those pixels ARE
    this instance's masks in that frame.

    ``fnum`` is the REAL video frame (what ``frame_global`` and the JPEG are
    keyed by); ``mask_frame`` is the same keyframe in the store's own space.
    Using fnum for both is how the first run of this tool drew keyframe 1's
    masks over keyframe 0's photograph.
    """
    names = sess.data.dtype.names or ()
    have_px = "pixel_row" in names and "pixel_col" in names
    own = gi[np.asarray(sess.fg)[gi] == int(fnum)] if have_px else gi[:0]
    acc = None
    if len(own):
        pr = np.asarray(sess.data["pixel_row"])[own].astype(np.int64)
        pc = np.asarray(sess.data["pixel_col"])[own].astype(np.int64)
        for k in ev.masks.files:
            mm = re.match(rf"^f{mask_frame}_o(\d+)$", k)
            if not mm:
                continue
            mask = np.asarray(ev.masks[k])
            if mask.ndim > 2:
                mask = mask[..., 0]
            mh, mw = mask.shape[:2]
            mv = np.clip((pr * mh // ev.kh).astype(np.int64), 0, mh - 1)
            mu = np.clip((pc * mw // ev.kw).astype(np.int64), 0, mw - 1)
            if not (mask[mv, mu] > 0).any():
                continue
            acc = (mask > 0) if acc is None else (acc | (mask > 0))
    if acc is None and fallback_oid is not None:
        k = f"f{mask_frame}_o{fallback_oid}"
        if k in ev.masks.files:
            mask = np.asarray(ev.masks[k])
            if mask.ndim > 2:
                mask = mask[..., 0]
            acc = mask > 0
    return None if acc is None else acc.astype(np.uint8)


def render(session_dir: Path, output_dir: Path, kf_indices: List[int],
           out_dir: Path, point_size: int = 0, max_points: int = 40000,
           log=print) -> List[Path]:
    from reconstruction.surface_fit.hole_audit import _Evidence
    from correction.session import load_session

    ev = _Evidence(output_dir, session_dir)
    if not ev.ok:
        raise SystemExit("could not load masks + cameras for this session")
    sess = load_session(output_dir)
    instances = _instances(output_dir)
    oid_of = _oid_map(output_dir, instances)
    frames = _frames_dir(session_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # frame number of each keyframe index, straight from the mask keys' frames
    # intersected with the session's own keyframe list
    kf_frames = [int(f) for f in sess.frames]
    rng = np.random.default_rng(0)
    written: List[Path] = []

    for kf in kf_indices:
        if kf < 0 or kf >= len(kf_frames):
            log(f"[overlay] keyframe {kf} out of range (0..{len(kf_frames) - 1})")
            continue
        fnum = kf_frames[kf]
        mask_frame = ev.space.to_mask(fnum)
        if mask_frame is None:
            log(f"[overlay] kf {kf} (frame {fnum}): no masks for this keyframe")
            continue
        img = _frame_image(frames, fnum)
        if img is None:
            log(f"[overlay] kf {kf}: no image for frame {fnum}")
            continue
        ih, iw = img.shape[:2]
        canvas = img.copy()
        legend: List[Tuple[Tuple[int, int, int], str, int, int]] = []
        outlines: List[Tuple[list, Tuple[int, int, int]]] = []

        for ci, inst in enumerate(instances):
            iid = inst.get("instance_id", inst.get("id"))
            if iid is None:
                continue
            gi_all = np.asarray(inst.get("globalIndices") or [], np.int64)
            gi_all = gi_all[(gi_all >= 0) & (gi_all < len(sess.xyz))]
            if not len(gi_all):
                continue
            m = _instance_mask(ev, sess, gi_all, int(fnum), int(mask_frame),
                               oid_of.get(int(iid)))
            if m is None or not m.any():
                continue
            col = _colour(ci)

            # ── the silhouette: a light tint so the frame still shows, and the
            # outline drawn LAST (below) so the points never bury it ──
            mh, mw = m.shape[:2]
            big = cv2.resize(m, (iw, ih), interpolation=cv2.INTER_NEAREST).astype(bool)
            canvas[big] = (0.80 * canvas[big] + 0.20 * np.array(col, np.float64)).astype(np.uint8)
            cnts, _ = cv2.findContours(big.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            outlines.append((cnts, col))

            # ── this instance's own cloud points, through THIS pose ──
            gi = gi_all
            if len(gi) > max_points:
                gi = gi[rng.choice(len(gi), max_points, replace=False)]
            pts = sess.xyz[gi]
            c2w = ev.cam.pose_map.get(int(fnum))
            K = ev.cam.K_for(int(fnum))
            if c2w is None or K is None:
                continue
            c2w4 = np.eye(4)
            c2w4[:c2w.shape[0], :c2w.shape[1]] = c2w
            M = np.linalg.inv(c2w4)
            p = (M[:3, :3] @ pts.T).T + M[:3, 3]
            z = p[:, 2]
            front = z > 0.05
            if not front.any():
                continue
            u = K[0, 0] * p[front, 0] / z[front] + K[0, 2]
            v = K[1, 1] * p[front, 1] / z[front] + K[1, 2]
            # K lives on the TRACE grid (ev.kw x ev.kh) — rescale to the image
            px = np.rint(u * iw / ev.kw).astype(np.int64)
            py = np.rint(v * ih / ev.kh).astype(np.int64)
            inb = (px >= 0) & (px < iw) & (py >= 0) & (py < ih)
            px, py = px[inb], py[inb]
            if not len(px):
                continue
            r = max(int(point_size), 0)
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if dx * dx + dy * dy > r * r:
                        continue
                    canvas[np.clip(py + dy, 0, ih - 1), np.clip(px + dx, 0, iw - 1)] = col
            # how many of them landed inside their own silhouette
            mu = np.clip((px * mw // iw), 0, mw - 1)
            mv = np.clip((py * mh // ih), 0, mh - 1)
            inside = int(m[mv, mu].sum())
            legend.append((col, str(inst.get("label", "?"))[:28] + f"#{iid}", inside, len(px)))

        # the outlines go on last: the eye needs the boundary to judge the dots
        for cnts, col in outlines:
            cv2.drawContours(canvas, cnts, -1, (255, 255, 255), 3)
            cv2.drawContours(canvas, cnts, -1, col, 2)

        # ── legend: colour, name, and the share that landed on its own mask ──
        if legend:
            pad, lh = 8, 18
            h = pad * 2 + lh * (len(legend) + 1)
            panel = np.zeros((h, 430, 3), np.uint8)
            cv2.putText(panel, f"kf {kf}  frame {fnum}   puntos DENTRO / proyectados",
                        (pad, pad + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            for r, (col, name, ins, tot) in enumerate(legend):
                y = pad + lh * (r + 1) + 12
                cv2.rectangle(panel, (pad, y - 9), (pad + 12, y + 2), col, -1)
                cv2.putText(panel, f"{name:<32} {ins:>6}/{tot:<6} {100 * ins / max(tot, 1):5.1f}%",
                            (pad + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
            ph, pw = panel.shape[:2]
            if pw < iw:
                panel = np.pad(panel, ((0, 0), (0, iw - pw), (0, 0)))
            else:
                panel = panel[:, :iw]
            canvas = np.vstack([canvas, panel])

        p_out = out_dir / f"kf_{kf:04d}_frame_{fnum:06d}.png"
        cv2.imwrite(str(p_out), canvas)
        written.append(p_out)
        tot_in = sum(l[2] for l in legend)
        tot_all = sum(l[3] for l in legend)
        log(f"[overlay] kf {kf} (frame {fnum}): {len(legend)} segmento(s), "
            f"{tot_in:,}/{tot_all:,} puntos dentro de su mascara "
            f"({100 * tot_in / max(tot_all, 1):.1f}%) -> {p_out.name}")
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", required=True)
    ap.add_argument("--kf", type=int, default=None, help="one keyframe index")
    ap.add_argument("--limit", type=int, default=None, help="render the first N keyframes")
    ap.add_argument("--out", default=None)
    ap.add_argument("--point-size", type=int, default=0)
    a = ap.parse_args(argv)
    session = Path(a.session)
    output = session / "output"
    out_dir = Path(a.out) if a.out else output / "reproject_overlay"
    if a.kf is not None:
        kfs = [a.kf]
    else:
        from correction.session import load_session
        n = len(load_session(output).frames)
        kfs = list(range(n if a.limit is None else min(a.limit, n)))
    render(session, output, kfs, out_dir, point_size=a.point_size)
    print(f"[overlay] {len(kfs)} imagen(es) en {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
