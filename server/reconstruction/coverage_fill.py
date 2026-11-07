"""THE COVERAGE FILL (USER 2026-11-06: "con esa información sacás un mapa de cobertura de lo escaneado, y qué frames
fueron cubriendo qué cosa … si hace falta completar con kf se completa, es eso nomás"; "donde no cubren o cubren poco";
"el intake es una propuesta primera de cobertura pero el mapa debe ser nuestra definitiva para tener cobertura
completa" — "sí, me parece bien" to the rule below).

THE MAP: every frame AMB3R-SLAM reconstructed over the whole video (reconstruction/amb3r_probe.py) with what it SAW —
its depth on the pixel-stride grid, its camera — so every point of the scanned scene with the frame it came from. A
keyframe SEES a point when the point falls inside its image (its whole horizontal and vertical field) at a depth
consistent with what the keyframe itself saw there, |z − d| ≤ tol·d (tol the session's own: 3σ of the relative
disagreement between consecutive keyframes on one surface — reconstruction/walk_structure.covisibility's rule, across
AMB3R's submaps as the walk crosses them); occluded or in front of what it saw, it does not. A point is COVERED when a
second keyframe sees it from a ray at least ``refine.min_tri_deg`` away from the first's (F5's bound for a landmark) — since
2026-11-07 (version 3, USER "ahora necesitamos cobertura") as many such views as F6 publishes a point with
(``precision.cloud.min_support_views``), the second view being the case of one:
one view, or two from the same place, measure nothing. A point no keyframe sees, or one alone, is WEAK. The session's
fill asks ONE second view since 2026-11-08 (workers/map_worker; USER "corremos todo así como vos proponés"): the fill to
F6's ten doubled pccr's keyframes, 313 → 686, and with them its cubes, its time and its witnesses' baseline; what is far
is covered by F6's witnesses chosen by their baseline (precision/depth_on_f5.covisible_witnesses).

THE FILL — the map is the definitive coverage, the intake's keyframes its first proposal: a frame AMB3R reconstructed
(usable by I0, not a keyframe's own record) whose own view — what it saw — holds more weak points it would see first, or
cover as the second view from far enough, than the measurement's own noise becomes a keyframe. The noise is the share
of consecutive keyframes' samples on one surface that the depth test itself calls inconsistent (the same samples tol is
measured on): below it a frame's gain cannot be told from the test failing. Greedy: the frame bringing the largest
share first (ties → the sharper by I0, then the earlier); what it sees enters the map and the others' shares are
measured again, until no frame brings more than the noise (a share only falls as keyframes are added: the search is
lazy and exact). What stays weak is DECLARED: no frame of the video sees it from a second place (a turn on the spot, a
corner glimpsed once). Every keyframe I1 chose stays. The added frames enter frames/selected_frames.json
(keyframes[].closed_by "coverage") and frames/witness_frames.json (every keyframe is a witness); the selection keeps I1's
own list (``coverage_fill.i1_selected_files``), so the fill always starts from I1's keyframes and an unchanged input is
not filled twice. intake/coverage_fill.json: the record — the map's coverage before and after, every frame added and
what it brought. On the card, deterministic: fixed grids, integer counts, sorted reductions, every tie by index (a
process of its own: the map worker never touches the card before Omega)."""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import time
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

FILL_NAME = "coverage_fill.json"
FILL_VERSION = 3                # 3: covered = the PUBLICATION's support (USER 2026-11-07: "ahora necesitamos cobertura"):
                               #    min_support views ≥ min_tri_deg from the first (2: one second view; the bar is the
                               #    measurement's own noise — 1: (1 − τ)/4 of a frame's view)
CLOSED_BY = "coverage"
PROVENANCE = "tool_measured"
CHUNK = 1 << 24                # points projected at once
ONE_SURFACE = 0.25             # |d/z − 1| under this: one surface — the samples the session's tolerance is measured on
                               # (and the band two moments that disagree are still on, reconstruction/cubes.py 7)


class CoverageFillError(RuntimeError):
    pass


def _frame_no(name) -> int:
    return int(Path(str(name)).stem)


class CoverageMap:
    """What every reconstructed frame saw, on the card, and which keyframes see and cover each of its points."""

    def __init__(self, video: Dict[str, np.ndarray], device: str, state: bool = True):
        """``state``: also the map's per-point state (which keyframes see and cover each point) — the fill's; the cut
        (reconstruction/cubes.py) reads only the points."""
        import torch
        self.t = torch
        self.dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        dev = self.dev
        D = torch.as_tensor(np.asarray(video["rec_depth"], np.float32), device=dev)
        self.F, self.Hs, self.Ws = (int(x) for x in D.shape)
        s = float(video["stride"])
        K = torch.as_tensor(np.asarray(video["rec_K"], np.float64), device=dev).clone()
        K[:, :2, :] /= s                                   # the model grid's K on the pixel-stride grid
        c2w = torch.as_tensor(np.asarray(video["rec_c2w"], np.float64), device=dev)
        R, C = c2w[:, :3, :3], c2w[:, :3, 3]
        self.R_w2c = R.transpose(1, 2).float()
        self.t_w2c = (-(R.transpose(1, 2) @ C[:, :, None])[:, :, 0]).float()
        self.C = C.float()
        self.K = K.float()
        self.D = D
        valid = D > 0
        self.n_per = valid.sum((1, 2)).to(torch.int64)
        self.off = torch.cat([torch.zeros(1, dtype=torch.int64, device=dev), torch.cumsum(self.n_per, 0)])
        N = int(self.off[-1])
        if N == 0:
            raise CoverageFillError("the whole-video pass holds no depth")
        self.X = torch.empty((N, 3), dtype=torch.float32, device=dev)
        vv, uu = torch.meshgrid(torch.arange(self.Hs, device=dev, dtype=torch.float64),
                                torch.arange(self.Ws, device=dev, dtype=torch.float64), indexing="ij")
        for f in range(self.F):
            m = valid[f]
            d = D[f][m].double()
            x = (uu[m] - K[f, 0, 2]) / K[f, 0, 0] * d
            y = (vv[m] - K[f, 1, 2]) / K[f, 1, 1] * d
            cam = torch.stack([x, y, d], 1)
            self.X[int(self.off[f]):int(self.off[f + 1])] = (cam @ R[f].T + C[f]).float()
        self.N = N
        if state:
            self.seen = torch.zeros(N, dtype=torch.bool, device=dev)
            self.covered = torch.zeros(N, dtype=torch.bool, device=dev)
            self.n_far = torch.zeros(N, dtype=torch.int32, device=dev)   # views ≥ the angle away from the first
            self.d1 = torch.zeros((N, 3), dtype=torch.float32, device=dev)

    def own(self, r: int):
        return slice(int(self.off[r]), int(self.off[r + 1]))

    def _sees(self, g: int, X, tol: float):
        t = self.t
        Xc = X @ self.R_w2c[g].T + self.t_w2c[g]
        z = Xc[:, 2]
        zs = t.where(z > 1e-6, z, t.ones_like(z))
        u = t.round(self.K[g, 0, 0] * Xc[:, 0] / zs + self.K[g, 0, 2]).long()
        v = t.round(self.K[g, 1, 1] * Xc[:, 1] / zs + self.K[g, 1, 2]).long()
        ok = (z > 1e-3) & (u >= 0) & (u < self.Ws) & (v >= 0) & (v < self.Hs)
        dg = self.D[g][v.clamp(0, self.Hs - 1), u.clamp(0, self.Ws - 1)]
        return ok & (dg > 0) & ((z - dg).abs() <= tol * dg)

    def tolerance(self, pairs: Sequence[Sequence[int]], log: Callable[[str], None]) -> Tuple[float, float]:
        """(tol, noise): the session's tolerance (walk_structure.covisibility's rule) on the record pairs ``pairs`` —
        consecutive KEYFRAMES' records: the spacing at which a keyframe is asked whether it sees a point, across
        AMB3R's submaps as the walk crosses them — and the share of those samples the test itself calls
        inconsistent (its own miss rate on surfaces both keyframes see)."""
        t = self.t
        rel = []
        for r, j in pairs:
            r, j = int(r), int(j)
            X = self.X[self.own(r)]
            if len(X) == 0:
                continue
            Xc = X @ self.R_w2c[j].T + self.t_w2c[j]
            z = Xc[:, 2]
            zs = t.where(z > 1e-6, z, t.ones_like(z))
            u = t.round(self.K[j, 0, 0] * Xc[:, 0] / zs + self.K[j, 0, 2]).long()
            v = t.round(self.K[j, 1, 1] * Xc[:, 1] / zs + self.K[j, 1, 2]).long()
            ok = (z > 1e-3) & (u >= 0) & (u < self.Ws) & (v >= 0) & (v < self.Hs)
            dj = self.D[j][v.clamp(0, self.Hs - 1), u.clamp(0, self.Ws - 1)]
            ok &= dj > 0
            q = dj[ok] / z[ok] - 1.0
            rel.append(q[q.abs() < ONE_SURFACE])
        rel = t.cat(rel) if rel else t.zeros(0, device=self.dev)
        if len(rel) < 100:
            raise CoverageFillError(f"only {len(rel)} samples of consecutive keyframes on one surface — the "
                                    f"session's tolerance cannot be measured")
        med = rel.median()
        tol = float(3.0 * 1.4826 * (rel - med).abs().median())
        noise = float(((rel - med).abs() > tol).double().mean())
        log(f"[coverage] the session's tolerance {tol * 100:.2f} % of the depth (3σ of {len(rel):,} samples of "
            f"consecutive keyframes on one surface); the test misses {noise * 100:.2f} % of them — the noise a "
            f"frame's gain must exceed")
        return tol, noise

    def add(self, g: int, tol: float, cos_min: float) -> None:
        """Keyframe record ``g`` sees what it sees: first sight, or the second view that covers a point."""
        t = self.t
        for a in range(0, self.N, CHUNK):
            b = min(self.N, a + CHUNK)
            X = self.X[a:b]
            m = self._sees(g, X, tol)
            if not bool(m.any()):
                continue
            idx = t.nonzero(m)[:, 0] + a
            dirg = self.C[g] - self.X[idx]
            dirg = dirg / dirg.norm(dim=1, keepdim=True).clamp_min(1e-12)
            was = self.seen[idx]
            first = idx[~was]
            self.d1[first] = dirg[~was]
            far = was & ~self.covered[idx] & ((self.d1[idx] * dirg).sum(1) <= cos_min)
            self.n_far[idx[far]] += 1
            self.covered[idx[far]] = self.n_far[idx[far]] >= int(self.support)
            self.seen[first] = True

    support = 1                  # views ≥ the angle away from a point's first view that cover it (fill() sets it)

    def brings(self, c: int, cos_min: float) -> int:
        """How many of what candidate record ``c`` saw no keyframe sees, or it would add a view to that is ≥ the angle
        away from its first while it has fewer than ``support`` of them."""
        sl = self.own(c)
        X = self.X[sl]
        if len(X) == 0:
            return 0
        dirc = self.C[c] - X
        dirc = dirc / dirc.norm(dim=1, keepdim=True).clamp_min(1e-12)
        seen, cov = self.seen[sl], self.covered[sl]
        gain = ~seen | (seen & ~cov & ((self.d1[sl] * dirc).sum(1) <= cos_min))
        return int(gain.sum())

    def summary(self) -> Dict[str, float]:
        n = float(self.N)
        seen, cov = int(self.seen.sum()), int(self.covered.sum())
        return {"points": self.N, "unseen": (self.N - seen) / n, "seen_once": (seen - cov) / n, "covered": cov / n}


def fill(video: Dict[str, np.ndarray], keyframe_frames: Sequence[int], quality_rows: Sequence[Dict], *,
         min_tri_deg: float, min_support: int = 1, device: str = "cuda", log: Callable[[str], None] = print) -> Dict:
    """The fill on the whole-video pass ``video`` (reconstruction/amb3r_probe.load_video) for I1's keyframes
    ``keyframe_frames`` (frame numbers); ``quality_rows``: I0's frames (frame, file, sharp_rank, usable). A point is
    COVERED when ``min_support`` keyframes see it from at least ``min_tri_deg`` away from the first that did — F6's
    publication rule (precision.cloud.min_support_views views that test its depth, USER 2026-11-07: pccr's racks were
    seen from ≥ 2° by a median of 14 reconstructed frames of the video and 4 keyframes, and the 10-view floor emptied
    them); 1 = the second view of version 2."""
    from reconstruction.amb3r_probe import source_records
    t0 = time.monotonic()
    cmap = CoverageMap(video, device)
    cmap.support = max(1, int(min_support))
    kf_src = sorted(set(int(r) for r in source_records(video, keyframe_frames)))
    tol, bar = cmap.tolerance(list(zip(kf_src[:-1], kf_src[1:])), log)
    cos_min = math.cos(math.radians(float(min_tri_deg)))
    rec_frame = np.asarray(video["frame_no"], np.int64)[np.asarray(video["rec_index"], np.int64)]
    for g in kf_src:
        cmap.add(g, tol, cos_min)
    before = cmap.summary()
    q = {int(r["frame"]): r for r in quality_rows}
    kf_set = set(int(f) for f in keyframe_frames)
    in_s = set(kf_src)
    n_per = cmap.n_per.cpu().numpy()
    heap = []
    cands = []
    for c in range(cmap.F):
        f = int(rec_frame[c])
        row = q.get(f)
        if c in in_s or f in kf_set or row is None or not row.get("usable") or n_per[c] == 0:
            continue
        cands.append(c)
        share = cmap.brings(c, cos_min) / float(n_per[c])
        if share > bar:
            heapq.heappush(heap, (-share, -float(row["sharp_rank"]), f, c))
    added: List[Dict] = []
    while heap:
        neg, nsharp, f, c = heapq.heappop(heap)
        share = cmap.brings(c, cos_min) / float(n_per[c])
        if share <= bar:
            continue                                    # it only falls from here
        if heap and (share, -nsharp, -f) < (-heap[0][0], -heap[0][1], -heap[0][2]):
            heapq.heappush(heap, (-share, nsharp, f, c))
            continue
        cmap.add(c, tol, cos_min)
        in_s.add(c)
        added.append({"frame": f, "file": q[f]["file"], "share": share, "points": int(n_per[c]),
                      "order": len(added)})
    after = cmap.summary()
    rest = max((cmap.brings(c, cos_min) / float(n_per[c]) for c in cands if c not in in_s), default=0.0)
    log(f"[coverage] the map: {cmap.N:,} points of {cmap.F} reconstructed frames; I1's {len(kf_set)} keyframes leave "
        f"{before['unseen'] * 100:.1f} % unseen and {before['seen_once'] * 100:.1f} % seen by fewer than "
        f"{cmap.support} view(s) ≥ {min_tri_deg:g}° from the first; {len(added)} frame(s) bringing more than the "
        f"noise ({bar * 100:.2f} % of their view) → {after['unseen'] * 100:.1f} % unseen, "
        f"{after['seen_once'] * 100:.1f} % under the support — DECLARED: no frame of the video brings it beyond the "
        f"noise (the most any frame left still brings: {rest * 100:.2f} %)"
        + (f"; added {[a['frame'] for a in added][:60]}" if added else ""))
    return {"tol_rel": tol, "noise_share": bar, "min_tri_deg": float(min_tri_deg), "min_support": cmap.support,
            "reconstructed_frames": cmap.F, "points": cmap.N, "before": before, "after": after,
            "most_left_share": rest, "added": sorted(added, key=lambda a: a["frame"]),
            "elapsed_s": round(time.monotonic() - t0, 3)}


# ── the selection on disk ─────────────────────────────────────────────────────────────────────

def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _base_selection(sel: Dict) -> Dict:
    """I1's own selection: the document without what an earlier fill added."""
    rec = sel.get("coverage_fill")
    if not rec:
        return sel
    base = {k: v for k, v in sel.items() if k != "coverage_fill"}
    base["selected_files"] = list(rec["i1_selected_files"])
    base["keyframes"] = [k for k in sel.get("keyframes", []) if k.get("closed_by") != CLOSED_BY]
    base["selected_count"] = len(base["selected_files"])
    return base


def _base_witness(wit: Dict) -> Dict:
    rec = wit.get("coverage_fill")
    if not rec:
        return wit
    base = {k: v for k, v in wit.items() if k != "coverage_fill"}
    base["frames"] = [w for w in wit.get("frames", []) if not w.get("coverage")]
    base["selected_files"] = list(rec["i1_selected_files"])
    base["selected_count"] = len(base["selected_files"])
    return base


def session_fill(scan_dir: Path, video_npz: Path, video_stamp: Dict, *, min_tri_deg: float, min_support: int = 1,
                 device: str = "cuda", log: Callable[[str], None] = print) -> Dict:
    """Fill the session's keyframes (module docstring); reused when its stamp — the whole-video pass, I1's selection,
    I0's report, this code and the bars — is the one on disk."""
    from intake.quality import _write_json_atomic
    from reconstruction.amb3r_probe import load_video
    scan_dir = Path(scan_dir)
    frames_dir = scan_dir / "frames"
    p_sel, p_wit = frames_dir / "selected_frames.json", frames_dir / "witness_frames.json"
    p_q = frames_dir / "quality_features.json"
    sel_doc, wit_doc = json.loads(p_sel.read_text()), json.loads(p_wit.read_text())
    base, wbase = _base_selection(sel_doc), _base_witness(wit_doc)
    stamp_src = {"version": FILL_VERSION, "video": video_stamp, "i1_selected_files": base["selected_files"],
                 "quality_features_sha256": _sha_file(p_q), "code": _sha_file(Path(__file__)),
                 "min_tri_deg": float(min_tri_deg), "min_support": int(min_support)}
    stamp = hashlib.sha256(json.dumps(stamp_src, sort_keys=True).encode()).hexdigest()
    rec_p = scan_dir / "intake" / FILL_NAME
    if (sel_doc.get("coverage_fill") or {}).get("stamp") == stamp and rec_p.exists() \
            and (wit_doc.get("coverage_fill") or {}).get("stamp") == stamp:
        log(f"[coverage] the fill reused ({len(sel_doc['coverage_fill']['added'])} keyframe(s) added to I1's "
            f"{len(base['selected_files'])}: the same pass, keyframes, quality report and code)")
        return json.loads(rec_p.read_text())
    quality = json.loads(p_q.read_text())
    kf = [_frame_no(f) for f in base["selected_files"]]
    res = fill(load_video(video_npz), kf, quality.get("frames") or [], min_tri_deg=min_tri_deg,
               min_support=min_support, device=device, log=log)
    added = res["added"]
    new_files = sorted(set(base["selected_files"]) | {a["file"] for a in added}, key=_frame_no)
    kfs = sorted(list(base.get("keyframes", [])) + [
        {"frame": a["frame"], "file": a["file"], "anchor": None, "parallax_px": None, "rotation_deg": None,
         "closed_by": CLOSED_BY, "coverage_share": a["share"]} for a in added], key=lambda k: int(k["frame"]))
    prev = None
    for k in kfs:                                       # an added keyframe's anchor: the keyframe before it
        if k.get("closed_by") == CLOSED_BY:
            k["anchor"] = prev
        prev = int(k["frame"])
    record = {"stamp": stamp, "i1_selected_files": list(base["selected_files"]),
              "added": [a["file"] for a in added]}
    doc = dict(base)
    doc.update({"selected_files": new_files, "selected_count": len(new_files), "keyframes": kfs,
                "coverage_fill": record})
    if doc.get("total_frames"):
        doc["reduction"] = 1.0 - len(new_files) / float(doc["total_frames"])
    w_frames = list(wbase.get("frames", []))
    have = {int(w["frame"]) for w in w_frames}
    for a in added:
        if a["frame"] not in have:
            w_frames.append({"frame": a["frame"], "file": a["file"], "reason": "keyframe", "parallax_px": None,
                             "coverage": True})
    w_frames.sort(key=lambda w: int(w["frame"]))
    wdoc = dict(wbase)
    w_files = [w["file"] for w in w_frames]
    wdoc.update({"frames": w_frames, "selected_files": w_files, "selected_count": len(w_files),
                 "coverage_fill": {"stamp": stamp, "i1_selected_files": list(wbase["selected_files"])}})
    doc["n_witness"] = len(w_files)
    full = {"version": FILL_VERSION, "provenance": PROVENANCE, "stamp": stamp, "inputs": stamp_src, **res,
            "n_keyframes_i1": len(base["selected_files"]), "n_keyframes": len(new_files)}
    _write_json_atomic(rec_p, full)
    _write_json_atomic(p_wit, wdoc)
    _write_json_atomic(p_sel, doc)
    log(f"[coverage] keyframes: I1's {len(base['selected_files'])} + {len(added)} by coverage = {len(new_files)} → "
        f"{p_sel.name}, {p_wit.name}, {FILL_NAME}")
    return full


def main(argv=None) -> int:
    """The fill in a process of its own, on the card (USER 2026-11-05: "ningún cálculo va en CPU"; the map worker
    never touches the card before Omega's exclusive check)."""
    ap = argparse.ArgumentParser(prog="python -m reconstruction.coverage_fill")
    ap.add_argument("--scan", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--video-stamp", required=True, help="the whole-video pass's stamp (a JSON file)")
    ap.add_argument("--min-tri-deg", type=float, required=True)
    ap.add_argument("--min-support", type=int, default=1,
                    help="views ≥ min-tri-deg from a point's first view that cover it (F6's support floor)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    session_fill(Path(a.scan), Path(a.video), json.loads(Path(a.video_stamp).read_text()),
                 min_tri_deg=a.min_tri_deg, min_support=a.min_support, device=a.device,
                 log=lambda m: print(m, flush=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
