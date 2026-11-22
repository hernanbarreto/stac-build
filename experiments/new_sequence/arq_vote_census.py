"""F6's vote on a published test session, counted pixel by pixel: for every candidate pixel in range, how many views
CONFIRM it and how many could see it — confirm + see through it (a view with a surface in front of the point cannot
see it: it is an occluder, not in the count). Nothing is written into the session.

The vote is the pipeline's own (precision.depth_on_f5.edge_keeping_vote → f6_torch.two_sided_vote) on the inputs F6
built: Omega's records, F5's camera and poses, each keyframe's depth factor as F6 applied it (depth_on_f5.json's
per_frame: s_k, bend — F4's tracks of these sessions are gone, so the bend is not fitted again), the confidence floor,
the co-visible witnesses of intake/cubes.json, the session's frozen configuration. The check: the pixels today's rule
keeps (≥ min_support confirmations, the majority, the occlusion veto) must equal the published run's "kept".

    python arq_vote_census.py <scan dir> <out.json>"""
import json
import sys
import time
from pathlib import Path

import numpy as np

SERVER = "/workspace/stac-build/server"
HERE = Path(__file__).resolve().parent
for p in (SERVER, str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
CAP = 1023


def main() -> int:
    scan, dst = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
    import torch
    from precision import corrected_cloud as CC
    from precision import depth_on_f5 as D
    from precision import depth_sweep as DS
    from precision import f6_torch as FT
    from precision.config import load_precision_config, raw_of
    from precision.epoch0_cloud import SKY_CONF
    log = lambda m: print(m, flush=True)   # noqa: E731
    t0 = time.time()
    pcfg = load_precision_config(session_dir=scan)
    if pcfg.mono_detail.enabled:
        raise SystemExit("this session's frozen configuration has PointDiT on — the census reproduces a run without it")
    rec = FT.require_cuda(int(pcfg.mono_detail.seed))
    log(f"[census] maps on {rec['device']} ({rec['card']})")
    raw_cfg = raw_of(pcfg)
    out = scan / "output"
    bc = pcfg.bend
    rep = json.loads((out / "precision" / "depth_on_f5.json").read_text())
    pf = rep["per_frame"]

    inp = DS.load_inputs(scan, pcfg)
    params = list(inp.cam.params)
    K = np.asarray(inp.K, np.float64)
    W, H = int(inp.wh[0]), int(inp.wh[1])
    frames = [int(f) for f in inp.kf]
    N = len(frames)
    w2c = np.asarray(inp.kf_w2c, np.float64)
    c2w = {f: np.linalg.inv(w2c[i]) for i, f in enumerate(frames)}
    lens = bool(np.any(inp.cam.dist()))
    with np.load(inp.records_dir / f"frame_{frames[0]}.npz") as z:
        rec_hw = tuple(int(x) for x in np.asarray(z["depth"]).shape)
    carry = lens or rec_hw != (H, W)
    final_coefs = {i: np.array([pf[str(f)]["s_k"], pf[str(f)]["bend"][0], pf[str(f)]["bend"][1]], np.float64)
                   for i, f in enumerate(frames)}
    log(f"[census] {N} keyframes, {W}x{H}, carry {carry}; depth factors from depth_on_f5.json")

    zo, conf, chunk = {}, {}, {}
    for f in frames:
        with np.load(inp.records_dir / f"frame_{f}.npz") as z:
            d = np.asarray(z["depth"], np.float32)
            cf = np.asarray(z["conf"], np.float32)
            if carry:
                d = CC.record_on_native(d, inp.cam, inp.maps)
                cf = CC.record_on_native(cf, inp.cam, inp.maps)
            zo[f] = d; conf[f] = cf; chunk[f] = int(z["chunk"]) if "chunk" in z.files else 0
    floor_norm = float(raw_cfg["reconstruction"]["simple"]["conf_min_norm"])
    thr = {}
    for k in sorted(set(chunk.values())):
        v = np.concatenate([conf[f][np.isfinite(conf[f]) & (conf[f] > SKY_CONF)].ravel()
                            for f in frames if chunk[f] == k])
        thr[k], _cmax = D.confidence_floor(v, floor_norm)
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    Dm = D.design(uu.ravel(), vv.ravel(), W, H)
    dep, valid, passed, zmed = {}, {}, {}, {}
    for i, f in enumerate(frames):
        valid[f] = np.isfinite(zo[f]) & (zo[f] > 0) & np.isfinite(conf[f]) & (conf[f] > SKY_CONF)
        if valid[f].any():
            zmed[f] = float(np.median(zo[f][valid[f]]))
        passed[f] = valid[f] & (conf[f] >= thr[chunk[f]])
        bent = (zo[f] * (Dm @ final_coefs[i]).reshape(H, W)).astype(np.float32)
        dep[f] = np.where(valid[f], bent, 0).astype(np.float32)
    del zo, conf

    t_dep, t_valid, t_passed, t_c2w, t_extra, _t_rep = D.load_turn_witnesses(
        out, frames, chunk, thr, zmed, final_coefs, c2w, inp, carry, rec_hw, Dm, H, W, bc.neighbors, log)
    dep.update(t_dep); valid.update(t_valid); passed.update(t_passed)
    c2w_vote = dict(c2w); c2w_vote.update(t_c2w)
    cv_extra, _cv_rep = D.covisible_witnesses(scan, frames, bc.neighbors, c2w, log)
    for f, ws in cv_extra.items():
        t_extra.setdefault(f, []).extend(ws)
    labels = D.mask_labels(out, H, W, log)
    if labels is not None and lens:
        labels = D.labels_on_undistorted(labels, inp.maps)
    from precision.camera import angle_px
    fusion = {"max_reproj_px": angle_px(float(K[0, 0]), float(pcfg.cloud.fusion_max_reproj_deg)),
              "max_reproj_deg": float(pcfg.cloud.fusion_max_reproj_deg),
              "max_depth_rel": float(pcfg.cloud.fusion_max_depth_rel),
              "tri_min_deg": float(pcfg.cloud.fusion_tri_min_deg),
              "tri_max_deg": float(pcfg.cloud.fusion_tri_max_deg),
              "mode": "adjust" if str(pcfg.cloud.fusion_rule) == "adjust" else "keep",
              "witness_max_m": float(pcfg.view_range_m),
              "repair": bool(pcfg.cloud.fusion_repair),
              "occlusion_veto": bool(pcfg.cloud.fusion_occlusion_veto),
              "majority": bool(pcfg.cloud.fusion_majority),
              "in_place": bool(pcfg.cloud.fusion_in_place)}
    if str(pcfg.cloud.fusion_rule) != "colmap":
        raise SystemExit(f"fusion rule {pcfg.cloud.fusion_rule!r}: the census is written for the COLMAP rule")
    no_birth = D.rotation_keyframes(scan, frames, log) if bool(pcfg.cloud.no_birth_without_parallax) else set()

    hist = {False: torch.zeros((CAP + 1) * (CAP + 1), dtype=torch.int64),
            True: torch.zeros((CAP + 1) * (CAP + 1), dtype=torch.int64)}
    real = FT.two_sided_vote

    def counting(*a, **k):
        v = real(*a, **k)
        f = frames[int(a[0])]
        if f in no_birth:
            return v
        candt = FT._t(a[4], torch.bool)
        m = candt & v["in_range"]
        conf_n = v["agree_q"][m].long().clamp(max=CAP)
        thru_n = v["thru_q"][m].long().clamp(max=CAP)
        occl = (v["contra"] - v["contra_fwd"]).clamp(min=0)[m]
        veto = occl > v["agree_q"][m]
        idx = conf_n * (CAP + 1) + thru_n
        for flag in (False, True):
            sel = veto == flag
            hist[flag] += torch.bincount(idx[sel], minlength=(CAP + 1) * (CAP + 1)).cpu()
        return v

    FT.two_sided_vote = counting
    _final, tau, tot = D.edge_keeping_vote(frames, dep, valid, passed, K, c2w_vote, labels, bc.neighbors,
                                           bc.tau_quantile, int(pcfg.cloud.repair_min_views),
                                           bool(pcfg.cloud.rescue_below_floor), log, bars_out={}, extra=t_extra,
                                           min_parallax_deg=float(pcfg.refine.min_tri_deg),
                                           min_support=int(pcfg.cloud.min_support_views), fusion=fusion,
                                           no_birth=no_birth)
    FT.two_sided_vote = real
    h_ok = hist[False].numpy().reshape(CAP + 1, CAP + 1)       # [confirm, see through], not vetoed
    h_veto = hist[True].numpy().reshape(CAP + 1, CAP + 1)
    a = np.arange(CAP + 1)[:, None]
    t = np.arange(CAP + 1)[None, :]
    ms = int(pcfg.cloud.min_support_views)
    kept_today = int(h_ok[(a >= ms) & (a > t)].sum())
    published = rep["vote"]
    valid_n = int(tot["valid"])
    doc = {"session": str(scan), "min_support": ms, "tau": float(tau), "valid": valid_n,
           "kept_published_run": int(published["kept"]), "kept_this_census": int(tot["kept"]),
           "kept_from_counts_today_rule": kept_today, "hist_not_vetoed": h_ok.tolist(), "hist_vetoed": h_veto.tolist()}
    log(f"[census] kept: published run {int(published['kept']):,} | this run's vote {int(tot['kept']):,} | "
        f"today's rule on the counts {kept_today:,} (of {valid_n:,} valid pixels)")
    den = a + t
    kept_mask = (a >= ms) & (a > t)
    w = h_ok[kept_mask]
    r = (a / np.maximum(den, 1))[np.broadcast_to(kept_mask, h_ok.shape)]
    dd = np.broadcast_to(den, h_ok.shape)[kept_mask]

    def wq(x, wts, q):
        o = np.argsort(x)
        c = np.cumsum(wts[o]) / max(wts.sum(), 1)
        return float(x[o][np.searchsorted(c, q)])
    log(f"[census] today's kept pixels: share of the views that could see them that confirm — p10 {wq(r, w, .1):.2f}, "
        f"p25 {wq(r, w, .25):.2f}, median {wq(r, w, .5):.2f}; views that could see them — p10 {wq(dd, w, .1):.0f}, "
        f"median {wq(dd, w, .5):.0f}, p90 {wq(dd, w, .9):.0f}")
    rows = []
    for X in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        for mn in (1, 2, 3, 5):
            keep = (a >= mn) & (a >= X * den) & (den > 0)
            n = int(h_ok[keep].sum())
            rows.append({"X": X, "min_confirm": mn, "kept": n, "share_of_valid": n / valid_n,
                         "vs_today": n / max(kept_today, 1)})
    doc["rules"] = rows
    for X in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        cells = [r_ for r_ in rows if r_["X"] == X]
        log(f"[census] X {X:.0%}: " + " | ".join(f"min {c['min_confirm']}: {c['share_of_valid'] * 100:.1f} % of valid "
                                                   f"(x{c['vs_today']:.2f} today)" for c in cells))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(doc))
    log(f"[census] → {dst} in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
