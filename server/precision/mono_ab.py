"""Phase 8 — the mono-detail A/B (claude_stac.txt, 2026-10-04): the same session with the flag off and on
(plus variants: context_scale, steps, H vs L), NOTHING published, every judge independent of PointDiT.

Per variant (``precision.depth_on_f5.compute`` — steps 1-4 of f6_bend — then the cloud stage's own
cleaner in a scratch directory):
  * the vote's own accounting (kept / contradicted / repaired / admitted / coverage), overall and
    inside the EDGE BAND — the band is image gradient energy above ``mono_detail.ab_edge_gradient_quantile``
    of the session's pixels, or a step of the CALIBRATED depth (the bent map, the flyers' window-spread
    bar), dilated one pixel; never a PointDiT product;
  * the held-out reprojection: F5's held-out landmarks (never fitted by anything) against the voted
    depth, median |dz| / z, global and in the band;
  * the flyers of the cleaned cloud by class a / b / c / d (precision/flyers.py, Phase 0's definition);
  * the hypotheses: tiles accepted / rejected, band pixels front / back / mixed_unresolved, points per
    `source` tier of the cleaned cloud;
  * ground truth: "pending" (no gauge / test role on these sessions);
  * seconds per frame (PointDiT, the whole stage) and the VRAM peak.
Writes ``<out>/mono_ab.json`` and ``mono_ab.csv``; prints the table, no verdict — the validation is
the user's eye on the overlays (``mono_detail.overlays``) and on the published cloud.

    python -m precision.mono_ab --session <dir> --out <dir> [--variants off on ctx2 steps4 L]
                                [--f5-files]   # read F5's own camera + poses (a session whose epochs
                                                 moved on, as pccr's hand-built ones) instead of load_inputs
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

LOG_TAG = "[mono-ab]"

VARIANTS: Dict[str, dict] = {
    "off": {"enabled": False},
    "on": {"enabled": True},
    "ctx2": {"enabled": True, "context_scale": 2.0},
    "steps4": {"enabled": True, "steps": 4},
    "L": {"enabled": True, "model": "L"},
    "inverse": {"enabled": True, "fit_space": "inverse"},
}


def f5_inputs(session_dir: Path):
    """F5's camera and poses from their own files (precision/camera_f5_r1.json, f5_camera_poses.txt),
    for a session whose live epoch is no longer F5's."""
    from precision import depth_sweep as DS
    from precision.camera import load_camera_json, undistort_maps
    out = session_dir / "output"
    cam_path = out / "precision" / "camera_f5_r1.json"
    if not cam_path.exists():
        cam_path = out / "camera.json"
    cam = load_camera_json(cam_path)
    kf, c2w = DS._read_poses(out / "precision" / "f5_camera_poses.txt", out / "camera_frames.txt")
    m1, m2, K = undistort_maps(cam)
    return DS.SweepInputs(cam=cam, K=K, wh=(cam.width, cam.height), maps=(m1, m2), kf=kf,
                          kf_w2c=np.linalg.inv(c2w), wit=[], wit_w2c=np.zeros((0, 4, 4)), tau_px=0.0,
                          heldout_rms_px=0.0, epochs={}, frames_dir=session_dir / "frames",
                          records_dir=out / "omega_run" / "results_output")


def edge_band(C, img_of: Callable[[int], np.ndarray], quantile: float, rng: np.random.Generator) -> Dict[int, np.ndarray]:
    """The judge's band per keyframe: image gradient energy above the session percentile OR a step of the
    calibrated (bent-before-refinement is not kept; the voted map is the calibrated depth here) depth."""
    from precision.flyers import texture_maps, window_spread
    H, W = C.H, C.W
    energies, spreads = {}, {}
    pool_e, pool_s = [], []
    for f in C.frames:
        g = np.asarray(img_of(f)).astype(np.float64).mean(2)
        e, _ = texture_maps(g, 3)
        energies[f] = e
        z = C.voted[f][0]
        _, _, sp = window_spread(z, 1)
        spreads[f] = (z, sp)
        pool_e.append(e.ravel()[rng.choice(e.size, min(20000, e.size), replace=False)])
        v = sp[z > 0]
        if len(v):
            pool_s.append(v[rng.choice(len(v), min(20000, len(v)), replace=False)])
    e_bar = float(np.percentile(np.concatenate(pool_e), quantile))
    s_bar = float(np.percentile(np.concatenate(pool_s), 95.0)) if pool_s else float("inf")
    from scipy.ndimage import binary_dilation
    band = {}
    for f in C.frames:
        z, sp = spreads[f]
        b = (energies[f] > e_bar) | ((z > 0) & (sp > s_bar))
        band[f] = binary_dilation(b, structure=np.ones((3, 3), bool))
    return band


def heldout_error(C, band: Dict[int, np.ndarray]) -> dict:
    """Median |dz| / z of F5's held-out landmarks against the VOTED depth (bilinear), global and in the band;
    a landmark whose pixel the vote left empty counts as 'not covered'."""
    from precision.depth_on_f5 import bilinear
    errs, errs_band, n_cov, n_all = [], [], 0, 0
    for i, f in enumerate(C.frames):
        h = C.obs[1][i]
        if not len(h):
            continue
        z = C.voted[f][0].astype(np.float64)
        zz = bilinear(z, h[:, 0], h[:, 1])
        ok = zz > 0
        n_all += len(h); n_cov += int(ok.sum())
        e = np.abs(zz[ok] - h[ok, 2]) / h[ok, 2]
        errs.append(e)
        r = np.clip(np.rint(h[ok, 1]).astype(int), 0, C.H - 1); c = np.clip(np.rint(h[ok, 0]).astype(int), 0, C.W - 1)
        errs_band.append(e[band[f][r, c]])
    E = np.concatenate(errs) if errs else np.zeros(0); EB = np.concatenate(errs_band) if errs_band else np.zeros(0)
    return {"median_rel": float(np.median(E)) if len(E) else None, "p90_rel": float(np.percentile(E, 90)) if len(E) else None,
            "band_median_rel": float(np.median(EB)) if len(EB) else None,
            "band_p90_rel": float(np.percentile(EB, 90)) if len(EB) else None,
            "landmarks": int(n_all), "covered": int(n_cov), "in_band": int(len(EB))}


def vote_in_band(C, band: Dict[int, np.ndarray]) -> dict:
    """The vote's outcome restricted to the band: share of band pixels (valid before the vote is not
    kept per pixel; what is measurable is the voted coverage of the band and the agree counts)."""
    n_band = n_out = 0
    agree = []
    for f in C.frames:
        z, a = C.voted[f]
        b = band[f]
        n_band += int(b.sum()); n_out += int((b & (z > 0)).sum())
        agree.append(a[b & (z > 0)])
    A = np.concatenate(agree) if agree else np.zeros(0)
    return {"band_pixels": n_band, "band_covered": n_out, "band_coverage": (n_out / n_band if n_band else None),
            "band_agree_median": float(np.median(A)) if len(A) else None}


def clean_cloud(C, pcfg, scratch: Path, log: Callable) -> Path:
    """The voted maps through the cloud stage's own cleaner, outside the session (the publish step's
    chunk writer, without the epoch)."""
    from config import cfg as raw_cfg
    from precision import corrected_cloud as CC
    from precision.epoch0_cloud import _write_ply_xyzrgb
    tmp = scratch / "tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "chunks").mkdir(parents=True)
    by_chunk: Dict[int, List[int]] = {}
    for f in C.frames:
        by_chunk.setdefault(C.chunk[f], []).append(f)
    K, c2w, uu, vv = C.K, C.c2w, C.uu, C.vv
    for k, fl in sorted(by_chunk.items()):
        L = {x: [] for x in ("xyz", "rgb", "fg", "pr", "pc", "cf")}
        for f in fl:
            zmap, amap = C.voted[f][0], C.voted[f][1]
            m = zmap > 0
            r, c = vv[m], uu[m]; zz = zmap[m].astype(np.float64)
            X = np.stack([(c - K[0, 2]) / K[0, 0] * zz, (r - K[1, 2]) / K[1, 1] * zz, zz], 1) @ c2w[f][:3, :3].T + c2w[f][:3, 3]
            img = CC._rgb_undistorted(C.inp.frames_dir, f, C.inp.maps)
            L["xyz"].append(X.astype(np.float32)); L["rgb"].append(img[r, c])
            L["fg"].append(np.full(len(r), f, np.int32)); L["pr"].append(r.astype(np.int16))
            L["pc"].append(c.astype(np.int16)); L["cf"].append(amap[m].astype(np.float32))
        _write_ply_xyzrgb(tmp / "chunks" / f"chunk_{k:03d}.ply", np.concatenate(L["xyz"]), np.concatenate(L["rgb"]))
        np.savez(tmp / "chunks" / f"chunk_{k:03d}_origins.npz", frame_global=np.concatenate(L["fg"]),
                 pixel_row=np.concatenate(L["pr"]), pixel_col=np.concatenate(L["pc"]), confidence=np.concatenate(L["cf"]))
    cleaned = scratch / "cleaned_cloud.ply"
    CC.clean(raw_cfg, tmp / "chunks", cleaned, log)
    shutil.rmtree(tmp, ignore_errors=True)
    return cleaned


def flyers_of(C, pcfg, cleaned: Path, log: Callable, workers: int) -> dict:
    """Phase 0's diagnosis on the cleaned cloud of this variant (frames, camera and poses = F5's)."""
    from correction.session import read_ply
    from precision import flyers as FL
    from precision.depth_on_f5 import source_column
    _, data = read_ply(cleaned)
    xyz = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float64)
    fr = np.asarray(data["frame_global"]).astype(np.int64); row = np.asarray(data["pixel_row"]).astype(np.int64)
    col = np.asarray(data["pixel_col"]).astype(np.int64)
    w2c = {f: np.linalg.inv(C.c2w[f]) for f in C.frames}
    energy, gray = FL.texture_scores(C.session_dir, C.inp.cam, C.frames, fr, row, col, int(pcfg.flyers.texture_window_px))
    cls, bars = FL.diagnose(xyz, fr, row, col, C.K, C.H, C.W, C.frames, w2c, pcfg.flyers, pcfg.bend.neighbors,
                            energy, gray, log=log, workers=workers)
    src = source_column(data, C.src_maps)
    tiers = {name: int((src == v).sum()) for v, name in ((1, "omega_bent"), (2, "mono_detail"), (3, "band_front"), (4, "band_back"))}
    rep = FL.counts(cls)
    rep["bars"] = bars
    rep["tiers"] = tiers
    rep["flyers_by_tier"] = {name: int(((src == v) & (cls > 0)).sum()) for v, name in
                             ((1, "omega_bent"), (2, "mono_detail"), (3, "band_front"), (4, "band_back"))}
    return rep


def run_variant(name: str, session_dir: Path, pcfg, inp, scratch: Path, log: Callable, workers: int,
                overlays: bool) -> dict:
    from precision.depth_on_f5 import compute
    from precision import corrected_cloud as CC
    over = dict(VARIANTS[name])
    if overlays and over.get("enabled"):
        over["overlays"] = True
    p = replace(pcfg, mono_detail=replace(pcfg.mono_detail, **over))
    t0 = time.time()
    C = compute(session_dir, p, log=log, inp=inp)
    t_compute = time.time() - t0
    rng = np.random.default_rng(0)
    img_of = lambda f: CC._rgb_undistorted(C.inp.frames_dir, f, C.inp.maps)  # noqa: E731
    band = edge_band(C, img_of, float(pcfg.mono_detail.ab_edge_gradient_quantile), rng)
    held = heldout_error(C, band)
    vband = vote_in_band(C, band)
    nv = max(C.vst["valid"], 1)
    vote = {k: (float(v) / nv if k not in ("tau", "valid") else v) for k, v in C.vst.items()}
    vote["coverage"] = C.cover
    sd = scratch / name
    sd.mkdir(parents=True, exist_ok=True)
    cleaned = clean_cloud(C, pcfg, sd, log)
    fl = flyers_of(C, pcfg, cleaned, log, workers)
    md = None
    if C.md_rep is not None:
        r = C.md_rep
        md = {"tiles": {"total": r.tiles, "accepted": r.accepted, "rejected_support": r.rejected_support,
                        "rejected_residual": r.rejected_residual, "residual_bar_rel": r.residual_bar},
              "pixels": r.totals, "seconds_pointdit": r.seconds_pointdit, "seconds_total": r.seconds_total,
              "params": r.params}
    rep = {"variant": name, "overrides": over, "seconds_compute": round(t_compute, 1),
           "seconds_per_frame": round(t_compute / max(C.N, 1), 2), "bend": {"window": int(C.wb), "held_out_A": C.score},
           "vote": vote, "vote_band": vband, "heldout": held, "flyers": fl, "mono_detail": md,
           "ground_truth": "pending (no gauge / test role on this session)", "cleaned_cloud": str(cleaned)}
    log(f"{LOG_TAG} {name}: held-out {held['median_rel'] * 100 if held['median_rel'] is not None else float('nan'):.2f} % "
        f"(band {held['band_median_rel'] * 100 if held['band_median_rel'] is not None else float('nan'):.2f} %), vote kept "
        f"{vote['kept'] * 100:.1f} % contradicted {vote['contradicted'] * 100:.1f} %, coverage {C.cover * 100:.1f} % "
        f"(band {vband['band_coverage'] * 100 if vband['band_coverage'] else 0:.1f} %), flyers {fl['flyer_share'] * 100:.2f} % "
        f"(a {fl['by_class']['mixed_edge']['share_of_flyers'] * 100:.1f} %), {t_compute / 60:.1f} min")
    return rep


def write_csv(path: Path, reps: List[dict]) -> None:
    rows = []
    for r in reps:
        fl = r["flyers"]; h = r["heldout"]; v = r["vote"]
        rows.append({"variant": r["variant"], "seconds_per_frame": r["seconds_per_frame"],
                     "heldout_median_pct": None if h["median_rel"] is None else round(100 * h["median_rel"], 3),
                     "heldout_band_median_pct": None if h["band_median_rel"] is None else round(100 * h["band_median_rel"], 3),
                     "vote_kept_pct": round(100 * v["kept"], 2), "vote_contradicted_pct": round(100 * v["contradicted"], 2),
                     "vote_repaired_pct": round(100 * v["repaired"], 2), "coverage_pct": round(100 * v["coverage"], 2),
                     "band_coverage_pct": None if r["vote_band"]["band_coverage"] is None else round(100 * r["vote_band"]["band_coverage"], 2),
                     "points": fl["points"], "flyers_pct": round(100 * fl["flyer_share"], 3),
                     **{f"flyers_{k}_pct_of_flyers": round(100 * d["share_of_flyers"], 2) for k, d in fl["by_class"].items()},
                     **{f"tier_{k}": n for k, n in fl["tiers"].items()},
                     "tiles_accepted": (r["mono_detail"] or {}).get("tiles", {}).get("accepted"),
                     "tiles_total": (r["mono_detail"] or {}).get("tiles", {}).get("total"),
                     "band_front": ((r["mono_detail"] or {}).get("pixels") or {}).get("band_front"),
                     "band_back": ((r["mono_detail"] or {}).get("pixels") or {}).get("band_back"),
                     "mixed_unresolved": ((r["mono_detail"] or {}).get("pixels") or {}).get("mixed_unresolved"),
                     "vram_peak_gb": (((r["mono_detail"] or {}).get("params") or {}).get("footprint") or {}).get("vram_peak_gb")})
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--session", required=True)
    ap.add_argument("--out", required=True, help="where mono_ab.json / .csv, the scratch clouds and the overlays go")
    ap.add_argument("--variants", nargs="+", default=["off", "on"], choices=sorted(VARIANTS))
    ap.add_argument("--f5-files", action="store_true", help="read F5's own camera + poses (precision/)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--overlays", action="store_true", help="per-keyframe PNGs of the 'on' variants under <out>/overlays_<variant>/")
    args = ap.parse_args(argv)
    from precision.config import load_precision_config
    pcfg = load_precision_config()
    session_dir = Path(args.session).resolve()
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    inp = f5_inputs(session_dir) if args.f5_files else None
    log = lambda m: print(m, flush=True)  # noqa: E731
    reps = []
    for name in args.variants:
        if args.overlays and VARIANTS[name].get("enabled"):
            import precision.mono_detail as MDm
            orig = MDm.run_stage
            def patched(*a, overlay_dir=None, _n=name, **k):   # overlays outside the session
                return orig(*a, overlay_dir=out / f"overlays_{_n}", **k)
            MDm.run_stage = patched
        try:
            reps.append(run_variant(name, session_dir, pcfg, inp, out / "scratch", log, args.workers, args.overlays))
        finally:
            if args.overlays and VARIANTS[name].get("enabled"):
                MDm.run_stage = orig
        (out / "mono_ab.json").write_text(json.dumps(reps, indent=1, default=float))
        write_csv(out / "mono_ab.csv", reps)
    print(f"\n{LOG_TAG} {'variant':8} {'held-out':>9} {'band':>7} {'kept':>6} {'contra':>7} {'cover':>6} {'flyers':>7} "
          f"{'a':>6} {'b':>6} {'c':>6} {'d':>6} {'s/frame':>8}")
    for r in reps:
        h = r["heldout"]; v = r["vote"]; fl = r["flyers"]["by_class"]
        print(f"{LOG_TAG} {r['variant']:8} {100 * (h['median_rel'] or 0):8.2f}% {100 * (h['band_median_rel'] or 0):6.2f}% "
              f"{100 * v['kept']:5.1f}% {100 * v['contradicted']:6.1f}% {100 * v['coverage']:5.1f}% "
              f"{100 * r['flyers']['flyer_share']:6.2f}% " + " ".join(f"{100 * fl[k]['share_of_flyers']:5.1f}%" for k in fl)
              + f" {r['seconds_per_frame']:8.2f}")
    print(f"{LOG_TAG} → {out / 'mono_ab.json'}, {out / 'mono_ab.csv'} (no verdict here: the validation is visual)")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.exit(main())
