"""Steps 1 and 5 of the architecture guided by AMB3R's map (ARQUITECTURA_mapa.md), the keyframe selection alone.

STEP 1 — the good frames, from I0's per-frame features (frames/quality_features.json): a frame is bad when its mean
luma is above the burnt maximum, when its share of burnt pixels (≥ clip_hi) is above the saturation maximum (the stored
clip_frac counts both ends: a frame over the maximum is read again and only its burnt pixels are counted), or when its
detail (I0's fft) is under SHARP_MIN of the median of its SHARP_NEIGHBOURS neighbours on each side. Darkness rejects
nothing. AMB3R's pass is read whole; a bad frame can be neither a keyframe nor an F4 witness.

STEP 5 — the walk from the first good frame, every frame measured against the last keyframe on AMB3R's poses and depth
(on the card): the OVERLAP (v2_select.Overlap: the smaller share of what each sees inside the other's image, within the
range), the DISPLACEMENT (the angle under which each point the last keyframe sees, inside the candidate's image, is
seen from the two cameras — its PARALLAX_QUANTILE quantile), the ROTATION (the angle of the relative rotation of the two
cameras). A frame is a candidate when any measure lies in its band; the window closes when any measure passes its
band; the keyframe is the sharpest good candidate (v2_select.Detail: I0's detail on the frame with its light
equalised). A close with no candidate takes the two frames around it.

    python arq2_select.py <scan dir> <amb3r_video.npz> <quality_features.json> <rotation threshold deg> <out.json>"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

SERVER = "/workspace/stac-build/server"
HERE = Path(__file__).resolve().parent
for p in (SERVER, str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
import v2_select as S  # noqa: E402

OVERLAP_REPEAT, OVERLAP_HOLE = 0.96, 0.92
BATCH = 8
LUMA_BURNT = 220.0
BURNT_SHARE_MAX = 0.25
SHARP_MIN = 0.5
SHARP_NEIGHBOURS = 15
MIN_POINTS = 50


def good_frames(quality_json: Path, frames_dir: Path, clip_hi: int):
    """{frame number: None (good) or the reason it is bad}."""
    from intake.quality import frame_file, read_gray
    rows = json.loads(quality_json.read_text())["frames"]
    num = np.array([int(r["frame"]) for r in rows])
    luma = np.array([float(r["luma_mean"]) for r in rows])
    clip = np.array([float(r["clip_frac"]) for r in rows])
    fft = np.array([float(r["fft"]) for r in rows])
    n, W = len(rows), SHARP_NEIGHBOURS
    med = np.array([np.median(np.r_[fft[max(0, i - W):i], fft[i + 1:i + 1 + W]]) for i in range(n)])
    why = {}
    for i in range(n):
        f = int(num[i])
        if luma[i] > LUMA_BURNT:
            why[f] = "burnt"
        elif clip[i] > BURNT_SHARE_MAX and \
                float((read_gray(frame_file(frames_dir, f)) >= clip_hi).mean()) > BURNT_SHARE_MAX:
            why[f] = "saturated"
        elif fft[i] < SHARP_MIN * med[i]:
            why[f] = "detail"
        else:
            why[f] = None
    return why


class Measures:
    """Overlap, displacement and rotation of video positions against one keyframe, on the card."""

    def __init__(self, ov: "S.Overlap", video):
        import torch
        self.t, self.ov = torch, ov
        dev = ov.cmap.dev
        c2w = torch.as_tensor(np.asarray(video["pred"], np.float64), device=dev)
        self.C, self.R = c2w[:, :3, 3], c2w[:, :3, :3]

    def against(self, a: int, bs, quant: float):
        """(overlap, displacement, rotation) of every video position of ``bs`` against position ``a``.

        THE DISPLACEMENT is I1's parallax measured on AMB3R's points: each point the keyframe sees that falls inside
        the candidate's image is a ray in each camera; the rotation that best carries the keyframe's rays onto the
        candidate's (least squares on the unit sphere) is taken out, and what is left — the angle between the turned
        ray and the candidate's — is the parallax of that point; its ``quant`` quantile is the displacement. (The angle
        under which each point is seen from the two cameras is NOT it: measured on the validated runs it is 1.0 to
        2.4 times I1's reading depending on the scene — 2026-11-22, arq2_calibrate.py.)"""
        t, ov = self.t, self.ov
        dev = ov.cmap.dev
        o = ov.with_last(a, bs)
        idx = t.as_tensor([int(b) for b in bs], device=dev)
        Ra, Rb = self.R[a], self.R[idx]
        tr = t.einsum("ij,kij->k", Ra, Rb)                     # trace(Ra^T Rb)
        rot = t.rad2deg(t.arccos(t.clamp((tr - 1.0) / 2.0, -1.0, 1.0))).cpu().numpy()
        parts = [P for P, _w in ov.sees(a)]
        par = np.full(len(bs), np.inf)
        if parts:
            P = t.cat(parts).double()
            inside = ov._inside(idx, P.float())                # (len(bs), n)
            ra = (P - self.C[a]) @ Ra                          # the rays in camera a: R_a^T (X − C_a)
            ra = ra / t.linalg.norm(ra, dim=1, keepdim=True)
            for k in range(len(bs)):
                m = inside[k]
                if int(m.sum()) < MIN_POINTS:
                    continue
                rb = (P[m] - self.C[idx[k]]) @ Rb[k]
                rb = rb / t.linalg.norm(rb, dim=1, keepdim=True)
                xa = ra[m]
                U, _s, Vt = t.linalg.svd(rb.T @ xa)            # the rotation Q with Q·r_a ≈ r_b (Kabsch)
                d = t.sign(t.linalg.det(U @ Vt))
                D = t.diag(t.stack([t.ones((), dtype=d.dtype, device=dev), t.ones((), dtype=d.dtype, device=dev), d]))
                Q = U @ D @ Vt
                qa = xa @ Q.T
                res = t.rad2deg(t.atan2(t.linalg.norm(t.cross(qa, rb, dim=1), dim=1), (qa * rb).sum(1)))
                par[k] = float(t.quantile(res, quant))
        return o, par, rot


def select(ms: Measures, ov, score, good, q_par, q_rot, band, quant, log):
    n = ov.n
    fno = ov.frame_no
    ok = [good.get(int(fno[i])) is None for i in range(n)]
    first = next(i for i in range(n) if ok[i])
    chosen = [{"pos": first, "rule": "first_good_frame", "closed_by": None}]
    last, i, cands = first, first + 1, []
    lo_p, hi_p, lo_r, hi_r = q_par * (1 - band), q_par * (1 + band), q_rot * (1 - band), q_rot * (1 + band)
    t0 = time.time()
    while i < n:
        bs = list(range(i, min(n, i + BATCH)))
        o, par, rot = ms.against(last, bs, quant)
        restart = None
        for b, x, p, r in zip(bs, o, par, rot):
            x, p, r = float(x), float(p), float(r)
            closed_by = ("overlap" if x < OVERLAP_HOLE else "displacement" if p > hi_p else "rotation" if r > hi_r
                         else None)
            if closed_by is None:
                if ok[b] and (OVERLAP_HOLE <= x <= OVERLAP_REPEAT or lo_p <= p <= hi_p or lo_r <= r <= hi_r):
                    cands.append(b)
                continue
            if cands:
                best = max(cands, key=lambda c: (score(int(fno[c])), -c))
                chosen.append({"pos": best, "rule": "best_of_band", "closed_by": closed_by, "candidates": len(cands)})
                last = best
            else:
                picks = [y for y in (b - 1, b) if y > last and ok[y]]
                if not picks:
                    nxt = next((y for y in range(b + 1, n) if ok[y]), None)
                    picks = [nxt] if nxt is not None else []
                for y in picks:
                    chosen.append({"pos": y, "rule": "jump", "closed_by": closed_by, "candidates": 0})
                if not picks:                                  # no good frame left: the walk ends
                    restart = n
                    break
                last = picks[-1]
            restart = last + 1
            break
        if restart is None:
            i = bs[-1] + 1
        else:
            i, cands = restart, []
        if len(chosen) % 200 == 0 and restart is not None:
            log(f"[select] {len(chosen)} keyframes, video position {last} of {n} ({time.time() - t0:.0f} s)")
    if cands:
        best = max(cands, key=lambda c: (score(int(fno[c])), -c))
        chosen.append({"pos": best, "rule": "best_of_last_band", "closed_by": "end", "candidates": len(cands)})
    for c in chosen:
        c["frame"] = int(fno[c.pop("pos")])
    return chosen


def main() -> int:
    scan, npz, qjson, q_rot, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), float(sys.argv[4]), Path(sys.argv[5])
    from intake.config import load_intake_config
    from reconstruction import amb3r_probe as A3
    from view_range import view_range_m
    config = yaml.safe_load((Path(SERVER) / "config.yaml").read_text())
    icfg = load_intake_config(config)
    q_par, quant, band = float(icfg.parallax.parallax_quantum_deg), float(icfg.parallax.parallax_quantile), \
        float(icfg.parallax.keyframe_band_frac)
    log = lambda m: print(m, flush=True)   # noqa: E731
    t0 = time.time()
    good = good_frames(qjson, scan / "frames", int(icfg.quality.clip_hi))
    bad = {k: v for k, v in good.items() if v}
    reasons = {}
    for v in bad.values():
        reasons[v] = reasons.get(v, 0) + 1
    log(f"[step 1] {len(good) - len(bad)} good frames of {len(good)}; bad: {reasons or 'none'}"
        + (f" — frames {sorted(bad)}" if 0 < len(bad) <= 40 else ""))
    video = A3.load_video(npz)
    if "metric_scale" not in video:
        raise SystemExit(f"{npz} is not in metres")
    rng = float(view_range_m(config))
    ov = S.Overlap(video, rng)
    ms = Measures(ov, video)
    score = S.Detail(scan / "frames", icfg.quality.fft_max_side)
    log(f"[select] {ov.n} frames; overlap {OVERLAP_HOLE:.0%}–{OVERLAP_REPEAT:.0%}; displacement {q_par:.3f}° (quantile "
        f"{quant:g}, band ±{band:.0%}); rotation {q_rot:.3f}° (band ±{band:.0%}); range {rng:g} m")
    chosen = select(ms, ov, score, good, q_par, q_rot, band, quant, log)
    frames = [c["frame"] for c in chosen]
    by = {}
    for c in chosen:
        key = c["closed_by"] or c["rule"]
        by[key] = by.get(key, 0) + 1
    gaps = np.diff(frames)
    doc = {"rule": {"overlap": [OVERLAP_HOLE, OVERLAP_REPEAT], "displacement_deg": q_par, "quantile": quant,
                    "rotation_deg": q_rot, "band": band, "range_m": rng, "burnt_luma": LUMA_BURNT,
                    "burnt_share_max": BURNT_SHARE_MAX, "sharp_min": SHARP_MIN, "sharp_neighbours": SHARP_NEIGHBOURS},
           "bad_frames": {str(k): v for k, v in sorted(bad.items())}, "n_keyframes": len(frames), "frames": frames,
           "keyframes": chosen, "closed_by": by,
           "frames_between_keyframes": {"median": float(np.median(gaps)), "max": int(gaps.max())}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1))
    log(f"[select] {len(frames)} keyframes of {ov.n} frames in {time.time() - t0:.0f} s; closed by {by}; frames between "
        f"keyframes median {np.median(gaps):g}, max {gaps.max()} → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
