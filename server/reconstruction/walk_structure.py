"""What a walk IS, measured on the structure probe (USER 2026-11-05: "una pasada de omega ahí ves cómo se distribuye
todo: ahh es recto, la covisibilidad es tanto, ok, lo puedo procesar en una pasada, o es un tramo con muchas vueltas").

The probe is ONE VGGT-Ω inference over every keyframe (vendor/VGGT-Long/stac_frames_pass.py, at the resolution where
they all fit the card). From its cameras and depths:

  c(i→j)   the share of keyframe i's sample (its pixels over the chunk confidence floor) that keyframe j sees at a
           consistent depth — |z_j(X) − d_j(π_j X)| ≤ tol·d_j, tol = 3σ of the relative disagreement between
           consecutive keyframes on one surface (the session's own) —; c_ij = min(c(i→j), c(j→i));
  ℓ(i)     the local run (the planner's majority scan at τ) and D = Σ 1/ℓ;
  revisits pairs that see each other although the walk LEFT in between (some keyframe between them sees neither);
           SALAD is the second witness — a non-local pair that is each other's best match above the loop bar —,
           because in one long pass the network drifts and a revisit may not show in the probe's geometry;
  zones    the keyframes the revisits join (union-find), as the RUNS of the walk they hold, each run widened by its
           margin so the zone carries context and shares keyframes with its neighbours — only the revisits the CHAIN
           of chunks does not already join: a pair is a zone's when no chunk holding one keyframe shares a frame with
           a chunk holding the other (the chunk graph's definition of a revisit, loop_utils/revisits.py). Two pieces
           inside one chunk, or across one seam, are reconstructed together already; and in ONE long pass the probe's
           geometry is loose (pccr: consecutive keyframes 10.9 % apart in depth), so a near pair 'with the walk gone
           in between' is often one keyframe that saw nothing — measured 2026-11-05: 6 zones on pccr, 5 of them such.

Deterministic: fixed strides, sorted reductions, no random draw."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

TAU = 0.3                 # the planner's co-visibility bar (chunk_covis.TAU)
SAMPLES = 1500            # pixels sampled per keyframe (a fixed stride over its valid pixels)
CONF_FLOOR = 0.10         # the chunk confidence floor (min + 0.10·(max − min)), reconstruction.simple.conf_min_norm
RUN_GAP = 3               # keyframes of one zone closer than this along the walk are one run
RUN_MARGIN = 2            # each run widened by this many keyframes on both sides (context + shared keyframes)


def _floor_mask(conf: np.ndarray, depth: np.ndarray) -> np.ndarray:
    lo, hi = float(conf.min()), float(conf.max())
    return (conf >= lo + CONF_FLOOR * (hi - lo)) & (conf > 1e-5) & (depth > 1e-3) & np.isfinite(depth)


def covisibility(c2w: np.ndarray, K: np.ndarray, depth: np.ndarray, conf: np.ndarray, *, device: str = "cuda",
                 log: Callable[[str], None] = print) -> Dict:
    """The directed co-visibility of every keyframe pair and the session's tolerance (module docstring)."""
    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    S, H, W = depth.shape
    valid = np.stack([_floor_mask(conf[i], depth[i]) for i in range(S)])
    D = torch.as_tensor(np.where(valid, depth, 0.0), dtype=torch.float32, device=dev)
    w2c = torch.as_tensor(np.linalg.inv(c2w.astype(np.float64)), dtype=torch.float32, device=dev)
    Kt = torch.as_tensor(K, dtype=torch.float32, device=dev)
    pts = []
    for i in range(S):
        vv, uu = np.nonzero(valid[i])
        if len(vv) == 0:
            pts.append(torch.zeros((0, 3), device=dev)); continue
        sel = np.linspace(0, len(vv) - 1, min(len(vv), SAMPLES)).astype(np.int64)
        u, v = uu[sel].astype(np.float64), vv[sel].astype(np.float64)
        d = depth[i][vv[sel], uu[sel]].astype(np.float64)
        k = K[i].astype(np.float64)
        cam = np.stack([(u - k[0, 2]) * d / k[0, 0], (v - k[1, 2]) * d / k[1, 1], d], 1)
        X = cam @ c2w[i, :3, :3].astype(np.float64).T + c2w[i, :3, 3].astype(np.float64)
        pts.append(torch.as_tensor(X, dtype=torch.float32, device=dev))

    def project(X, j):
        Xc = X @ w2c[j, :3, :3].T + w2c[j, :3, 3]
        z = Xc[:, 2]
        zs = torch.where(z > 1e-6, z, torch.ones_like(z))
        u = Kt[j, 0, 0] * Xc[:, 0] / zs + Kt[j, 0, 2]
        v = Kt[j, 1, 1] * Xc[:, 1] / zs + Kt[j, 1, 2]
        ui, vi = torch.round(u).long(), torch.round(v).long()
        ok = (z > 1e-3) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        dj = D[j][vi.clamp(0, H - 1), ui.clamp(0, W - 1)]
        return z, dj, ok & (dj > 0)

    # the session's tolerance: consecutive keyframes on one surface
    rel = []
    for i in range(S - 1):
        if len(pts[i]) == 0:
            continue
        z, dj, ok = project(pts[i], i + 1)
        r = (dj[ok] / z[ok] - 1.0)
        rel.append(r[r.abs() < 0.25])
    rel = torch.cat(rel) if rel else torch.zeros(0, device=dev)
    if len(rel) < 100:
        raise RuntimeError(f"walk structure: only {len(rel)} consecutive-keyframe samples on one surface — the probe "
                           "cannot measure the session's tolerance")
    med = rel.median()
    tol = float(3.0 * 1.4826 * (rel - med).abs().median())
    cov = torch.zeros((S, S), dtype=torch.float32, device=dev)
    jj = torch.arange(S, device=dev)
    for i in range(S):
        X = pts[i]
        n = len(X)
        if n == 0:
            continue
        # keyframe i's sample in EVERY keyframe at once
        Xc = torch.einsum("jab,nb->jna", w2c[:, :3, :3], X) + w2c[:, None, :3, 3]
        z = Xc[..., 2]
        zs = torch.where(z > 1e-6, z, torch.ones_like(z))
        u = Kt[:, 0, 0, None] * Xc[..., 0] / zs + Kt[:, 0, 2, None]
        v = Kt[:, 1, 1, None] * Xc[..., 1] / zs + Kt[:, 1, 2, None]
        ui, vi = torch.round(u).long(), torch.round(v).long()
        ok = (z > 1e-3) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        dj = D[jj[:, None].expand(-1, n), vi.clamp(0, H - 1), ui.clamp(0, W - 1)]
        cons = ok & (dj > 0) & ((z - dj).abs() <= tol * dj)
        cov[i] = cons.sum(1).float() / n
        cov[i, i] = 1.0
    c = torch.minimum(cov, cov.T).cpu().numpy().astype(np.float64)
    log(f"[structure] co-visibility of every pair of {S} keyframes (tolerance {tol * 100:.2f} % of the depth, "
        f"σ from {len(rel):,} consecutive samples)")
    return {"c": c, "directed": cov.cpu().numpy().astype(np.float64), "tol_rel": tol}


def local_runs(c: np.ndarray, tau: float = TAU) -> np.ndarray:
    """ℓ(i): the planner's majority scan (chunk_covis.covis_lengths) on this matrix."""
    K = len(c)
    ell = np.ones(K, int)
    for a in range(K):
        ps, fs, last = 0, 0, a
        for b in range(a + 1, K):
            if c[a, b] >= tau:
                ps += 1; last = b
            else:
                fs += 1
            if fs > ps:
                break
        ell[a] = max(1, last - a)
    return ell


def revisits(c: np.ndarray, ell: np.ndarray, salad: Optional[np.ndarray] = None, salad_bar: Optional[float] = None,
             tau: float = TAU) -> Dict[str, List]:
    """Geometric revisits (they see each other although the walk left in between) and SALAD's (each other's best
    non-local match above the loop bar)."""
    K = len(c)
    geo = []
    for i in range(K):
        for j in range(i + int(ell[i]) + 1, K):
            if c[i, j] >= tau:
                mid = np.arange(i + 1, j)
                if len(mid) and np.any((c[i, mid] < tau) & (c[mid, j] < tau)):
                    geo.append((i, j, float(c[i, j])))
    sal = []
    if salad is not None and salad_bar is not None:
        S = np.array(salad, np.float64)
        band = np.zeros_like(S, bool)
        for i in range(K):
            lo, hi = max(0, i - int(ell[i])), min(K, i + int(ell[i]) + 1)
            band[i, lo:hi] = True
        Sn = np.where(band | band.T, -np.inf, S)
        best_row = np.argmax(Sn, axis=1)
        for i in range(K):
            j = int(best_row[i])
            if j > i and np.isfinite(Sn[i, j]) and Sn[i, j] >= salad_bar and int(np.argmax(Sn[:, j])) == i:
                sal.append((i, j, float(S[i, j])))
    return {"geometric": geo, "salad": sal}


def zones_of(pairs: Sequence, K: int, gap: int = RUN_GAP, margin: int = RUN_MARGIN) -> List[Dict]:
    """The keyframes the revisits join, as zones of runs (module docstring)."""
    par = list(range(K))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    for i, j, _ in pairs:
        par[find(i)] = find(j)
    members: Dict[int, set] = {}
    for i, j, _ in pairs:
        for x in (i, j):
            members.setdefault(find(x), set()).add(x)
    out = []
    for m in sorted(members.values(), key=lambda s: min(s)):
        ix = sorted(m)
        runs = [[ix[0], ix[0]]]
        for x in ix[1:]:
            if x <= runs[-1][1] + gap:
                runs[-1][1] = x
            else:
                runs.append([x, x])
        runs = [[max(0, a - margin), min(K - 1, b + margin)] for a, b in runs]
        merged = [runs[0]]
        for a, b in runs[1:]:
            if a <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        if len(merged) >= 2:                     # a zone is a REVISIT: two or more pieces of the walk
            out.append(merged)
    # the runs' margins can make two zones share keyframes: a keyframe has ONE owner, so zones that
    # touch are one zone (their runs merged)
    changed = True
    while changed:
        changed = False
        for x in range(len(out)):
            for y in range(x + 1, len(out)):
                if any(a <= d + 1 and c <= b + 1 for a, b in out[x] for c, d in out[y]):
                    runs = sorted(out[x] + out[y])
                    merged = [list(runs[0])]
                    for a, b in runs[1:]:
                        if a <= merged[-1][1] + 1:
                            merged[-1][1] = max(merged[-1][1], b)
                        else:
                            merged.append([a, b])
                    out[x] = merged
                    del out[y]
                    changed = True
                    break
            if changed:
                break
    return [{"runs": m, "keyframes": sorted({k for a, b in m for k in range(a, b + 1)})}
            for m in sorted(out, key=lambda m: m[0][0]) if len(m) >= 2]


def joined_by_chain(i: int, j: int, ranges: Sequence[Sequence[int]]) -> bool:
    """True when a chunk holding keyframe i shares a frame with a chunk holding keyframe j ([start, end) ranges)."""
    ci = [(a, b) for a, b in ranges if a <= i < b]
    cj = [(a, b) for a, b in ranges if a <= j < b]
    return any(max(a, c) < min(b, d) for a, b in ci for c, d in cj)


def analyse(probe_npz: Path, *, salad_bar: Optional[float] = None, ranges: Optional[Sequence[Sequence[int]]] = None,
            device: str = "cuda", log: Callable[[str], None] = print) -> Dict:
    """The probe's structure. ``salad_bar``: SALAD's appearance bar; None = the bar the probe recorded (the loop
    detector's own: calibrated on the session's geometric revisits, else the configured one). ``ranges``: the plan's
    chunks — the zones are made of the revisits they do not already join (module docstring); None = every revisit."""
    z = np.load(probe_npz)
    if salad_bar is None and "salad_bar" in z.files:
        salad_bar = float(z["salad_bar"])
    cv = covisibility(z["c2w"], z["K"], z["depth"], z["conf"], device=device, log=log)
    c = cv["c"]
    ell = local_runs(c)
    rv = revisits(c, ell, z["salad"] if "salad" in z.files else None, salad_bar)
    pairs = rv["geometric"] + rv["salad"]
    kept = pairs if ranges is None else [p for p in pairs if not joined_by_chain(p[0], p[1], ranges)]
    zones = zones_of(kept, len(c))
    D = float((1.0 / np.maximum(ell, 1)).sum())
    log(f"[structure] D {D:.2f} over {len(c)} keyframes; revisits: {len(rv['geometric'])} by geometry, "
        f"{len(rv['salad'])} by SALAD" + ("" if ranges is None else
                                           f", {len(pairs) - len(kept)} of them joined by the chain already")
        + "; zones: " + (", ".join(f"{len(zz['keyframes'])} kf in runs {zz['runs']}" for zz in zones) or "none"))
    return {"keyframes": int(len(c)), "tol_rel": cv["tol_rel"], "tau": TAU, "D_probe": D,
            "local_run": ell.tolist(), "revisits": rv, "zone_revisits": [list(p) for p in kept],
            "ranges": None if ranges is None else [[int(a), int(b)] for a, b in ranges],
            "zones": zones, "salad_bar": salad_bar,
            "salad_bar_source": str(z["salad_bar_source"]) if "salad_bar_source" in z.files else None,
            "covisibility": c}


def figure(res: Dict, c2w: np.ndarray, out_png: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = res["covisibility"]; C = c2w[:, :3, 3]
    fig, ax = plt.subplots(1, 2, figsize=(16, 7.5))
    ax[0].imshow(c, cmap="magma", vmin=0, vmax=1, origin="upper")
    ax[0].set_title(f"{title}: co-visibility of every keyframe pair (structure probe)")
    for i, j, _ in res["revisits"]["geometric"]:
        ax[0].plot([j], [i], "c.", ms=2)
    for i, j, _ in res["revisits"]["salad"]:
        ax[0].plot([j], [i], "y+", ms=6)
    for i, j, _ in res.get("zone_revisits", []):
        ax[0].plot([i], [j], "g.", ms=3)                       # below the diagonal: the pairs that make the zones
    for a, b in res.get("ranges") or []:
        from matplotlib.patches import Rectangle
        ax[0].add_patch(Rectangle((a - 0.5, a - 0.5), b - a, b - a, fill=False, ec="w", lw=0.6))
    up = int(np.argmax(np.abs(np.mean(c2w[:, :3, 1], axis=0))))
    a0, a1 = [a for a in range(3) if a != up]
    ax[1].plot(C[:, a0], C[:, a1], "-", color="0.75", lw=1)
    inz = np.full(len(C), -1)
    for zi, zz in enumerate(res["zones"]):
        inz[zz["keyframes"]] = zi
    ax[1].scatter(C[inz < 0, a0], C[inz < 0, a1], s=8, c="0.5", label="straight")
    cols = plt.cm.tab10(np.arange(max(1, len(res["zones"]))) % 10)
    for zi, zz in enumerate(res["zones"]):
        ax[1].scatter(C[zz["keyframes"], a0], C[zz["keyframes"], a1], s=14, color=cols[zi],
                      label=f"zone {zi + 1}: runs {zz['runs']}")
    ax[1].scatter(C[0, a0], C[0, a1], marker="*", s=200, c="g", label="start")
    ax[1].set_aspect("equal"); ax[1].legend(fontsize=7)
    ax[1].set_title(f"{title}: the probe's walk from above (D {res['D_probe']:.1f})")
    fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)


def save(res: Dict, path: Path) -> None:
    doc = {k: v for k, v in res.items() if k != "covisibility"}
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    tmp.replace(path)
