"""
The cloud cleaner by PIECES (USER 2026-11-09: "todo en GPU … trozo por trozo, con la memoria independiente del
tamaño de la nube … carga de archivos en paralelo … avance en el log en cada paso … mismo resultado bit a bit").

``gpu_cloud_clean`` merged every chunk into host RAM (~50 B per point with its temporaries: zaragoza's 376 M points
took tens of GB and most of the time on one core). Here the cloud is never merged:

  * the chunk PLYs are READ IN PLACE (memory-mapped, headers parsed on threads); the origins (npz) are read per
    chunk when a step needs them;
  * which points are still in the cloud is a byte mask on the local scratch disk (``postprocessing.
    clean_scratch_dir``), one byte per input point; every pass walks the chunks in FILE ORDER;
  * a voxel pass scans the chunks once (per-axis cell histograms on the card), cuts the same slabs of whole cells
    as ``gpu_cloud_clean._voxel_keep`` (its ``_slab_bounds``, the same count), writes each slab's members to a scratch
    file and picks each slab's survivors on the card with ``_voxel_keep_dev`` itself;
  * the SOR runs slab by slab on the card with ``grid_knn`` over the slab plus a HALO of whole cells: a point's
    result is taken only when its k-th distance is CERTIFIED inside the halo (every point outside it lies farther),
    the others are re-run with a 4x wider halo until certified — the exact kNN of the whole cloud, point by point;
    the statistic (mean / std of the mean distances) is numpy's own reduction over the live points in the cloud's
    order, on a memory-mapped array (the same pairwise sums, the same bits);
  * the PLY is written chunk by chunk in the cloud's order, the same layout byte for byte.

``gpu_cloud_clean``'s in-memory path stays the PATH OF RECORD (``--in-memory``, and the ``--witness`` run, whose
witnesses read the whole merged cloud); ``tests/test_stream_clean.py`` holds the two equal byte for byte.
Stays on the CPU, declared: PLY / npz reading and writing, the byte masks, numpy's mean / std (the statistic of record).

Hernán Barreto - Ingerop IN3 Session IV - STAC
"""
from __future__ import annotations

import glob
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from reconstruction.gpu_cloud_clean import (LOAD_THREADS, _PLY_TYPES, _empty_cache, _slab_bounds,
                                            _voxel_keep_dev)

_MARGIN = 1e-6               # the same ulp guard as grid_knn: a certified gap is (1 - margin) x the cell bound


class CleanError(RuntimeError):
    pass


# ── the chunks, read in place ─────────────────────────────────────────────────

def _ply_map(path: str):
    """The PLY's vertex records memory-mapped (binary little endian), never copied."""
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise CleanError(f"{path}: not a PLY")
        n, props = 0, []
        while True:
            ln = f.readline().decode("ascii", "replace").strip()
            if ln.startswith("format") and "binary_little_endian" not in ln:
                raise CleanError(f"{path}: not binary_little_endian")
            elif ln.startswith("element vertex"):
                n = int(ln.split()[-1])
            elif ln.startswith("property") and n:
                _, t, name = ln.split()[:3]
                props.append((name, _PLY_TYPES[t]))
            elif ln == "end_header":
                break
        off = f.tell()
    dt = np.dtype(props)
    if n == 0:
        return np.zeros(0, dt)
    return np.memmap(path, dtype=dt, mode="r", offset=off, shape=(n,))


class Chunks:
    """The cloud as its chunk files in FILE ORDER (the order gpu_cloud_clean merged them): global index = chunk start
    + local index."""

    def __init__(self, files: Sequence[str], origin_files: Sequence[str]):
        self.files = list(files)
        with ThreadPoolExecutor(max_workers=max(1, min(LOAD_THREADS, len(self.files)))) as ex:
            self.maps = list(ex.map(_ply_map, self.files))
        self.sizes = np.array([len(m) for m in self.maps], np.int64)
        self.starts = np.concatenate([[0], np.cumsum(self.sizes)]).astype(np.int64)
        self.n = int(self.starts[-1])
        self.has_rgb = all(all(c in (m.dtype.names or ()) for c in ("red", "green", "blue")) for m in self.maps)
        self.origin_files = list(origin_files)
        if self.origin_files and len(self.origin_files) != len(self.files):
            raise CleanError(f"{len(self.origin_files)} origin file(s) for {len(self.files)} chunk(s) — every chunk "
                             f"carries its own origins")
        self.has_conf = False
        if self.origin_files:
            with np.load(self.origin_files[0]) as z:
                self.has_conf = "confidence" in z.files

    def xyz(self, c: int, local: Optional[np.ndarray] = None) -> np.ndarray:
        m = self.maps[c]
        if local is None:
            return np.stack([m["x"], m["y"], m["z"]], 1).astype(np.float32)
        return np.stack([m["x"][local], m["y"][local], m["z"][local]], 1).astype(np.float32)

    def gather_xyz(self, idx: np.ndarray) -> np.ndarray:
        """Coordinates of ascending global indices."""
        out = np.empty((len(idx), 3), np.float32)
        cs = np.searchsorted(self.starts, idx, side="right") - 1
        b = np.flatnonzero(np.diff(cs)) + 1
        for s, e in zip(np.concatenate([[0], b]), np.concatenate([b, [len(idx)]])):
            c = int(cs[s])
            out[s:e] = self.xyz(c, idx[s:e] - self.starts[c])
        return out

    def origins(self, c: int) -> Dict[str, np.ndarray]:
        with np.load(self.origin_files[c]) as z:
            rec = {k: z[k] for k in ("frame_global", "pixel_row", "pixel_col", "confidence") if k in z.files}
        if len(rec["frame_global"]) != int(self.sizes[c]):
            raise CleanError(f"Origin size mismatch: {os.path.basename(self.origin_files[c])} holds "
                             f"{len(rec['frame_global']):,} origins for {int(self.sizes[c]):,} points of "
                             f"{os.path.basename(self.files[c])} — cannot inject traceability")
        return rec


def _mask_file(scratch: str, name: str, n: int, fill: bool) -> np.memmap:
    m = np.memmap(os.path.join(scratch, f"{name}.mask"), dtype=np.bool_, mode="w+", shape=(max(n, 1),))
    m[:] = fill
    return m


def _members(mask: np.memmap, ch: Chunks, c: int) -> np.ndarray:
    return np.flatnonzero(np.asarray(mask[ch.starts[c]:ch.starts[c + 1]]))


# ── a voxel pass ──────────────────────────────────────────────────────────────

class Slabs:
    """The slabs of one voxel pass: the axis, the integer cell bounds, the cell size and each slab's SURVIVORS
    (ascending global indices, one scratch file per slab)."""

    def __init__(self, ax: int, bounds: np.ndarray, voxel: float, files: List[str], cmin: int, cmax: int):
        self.ax, self.bounds, self.voxel, self.files = ax, bounds, voxel, files
        self.cmin, self.cmax = cmin, cmax

    def members(self, t: int) -> np.ndarray:
        return np.fromfile(self.files[t], dtype=np.int64)


def voxel_pass(ch: Chunks, mask: np.memmap, voxel: float, *, tile_points: int, device: str, scratch: str,
               tag: str, log=print) -> Tuple[np.memmap, Slabs]:
    """``gpu_cloud_clean._voxel_keep`` over the points ``mask`` holds, by pieces: the same grid (world origin,
    float64 cells), the same slab axis and bounds, each slab's points in ascending global order handed to the same
    ``_voxel_keep_dev`` — the same survivors. Returns the new mask and the slabs (their survivors on scratch)."""
    import torch
    t0 = time.time()
    lo = np.full(3, np.inf, np.float32); hi = np.full(3, -np.inf, np.float32)
    hist: List[List[Tuple[np.ndarray, np.ndarray]]] = [[], [], []]
    n = 0

    def _scan(c):
        loc = _members(mask, ch, c)
        if len(loc) == 0:
            return 0, None
        p = torch.from_numpy(ch.xyz(c, loc)).to(device)
        mn, mx = p.min(0).values.cpu().numpy(), p.max(0).values.cpu().numpy()
        hs = []
        for a in range(3):
            u, k = torch.unique(torch.floor(p[:, a].double() / voxel).long(), return_counts=True)
            hs.append((u.cpu().numpy(), k.cpu().numpy()))
        return len(loc), (mn, mx, hs)
    with ThreadPoolExecutor(max_workers=max(1, min(LOAD_THREADS, len(ch.files)))) as ex:
        for c, (k, r) in enumerate(ex.map(_scan, range(len(ch.files)))):
            if r is not None:
                n += k
                lo = np.minimum(lo, r[0]); hi = np.maximum(hi, r[1])
                for a in range(3):
                    hist[a].append(r[2][a])
            log(f"  [{tag}] scan {c + 1}/{len(ch.files)}: {k:,} pts ({time.time() - t0:.0f} s)")
    new = _mask_file(scratch, f"{tag}_keep", ch.n, False)
    if n == 0:
        return new, Slabs(0, np.zeros(2, np.int64), voxel, [], 0, 0)
    lo64 = lo.astype(np.float64)
    shift = lo64 - np.floor(lo64 / voxel) * voxel
    log(f"  [voxel] grid anchored at the world origin ({voxel * 1000:g} mm cells, float64): "
        f"the cloud's minimum sits {np.round(shift * 1000, 3).tolist()} mm past its cell "
        f"boundaries — the offset of the old min-anchored grid, below one voxel")
    ax = int(np.argmax(hi.astype(np.float64) - lo64))
    cells = np.concatenate([u for u, _ in hist[ax]]); counts = np.concatenate([k for _, k in hist[ax]])
    ca0, cmax = int(cells.min()), int(cells.max())
    dense = np.zeros(cmax - ca0 + 1, np.int64)
    np.add.at(dense, cells - ca0, counts)
    n_tiles = max(1, -(-n // int(tile_points)))
    cum = np.cumsum(dense)                                  # == _slab_bounds' cumsum(bincount(ca - ca0))
    targets = [(t * n) // n_tiles for t in range(1, n_tiles)]
    inner = np.searchsorted(cum, targets, side="right")
    bounds = np.concatenate([[0], inner, [len(cum)]]).astype(np.int64) + ca0
    if n_tiles > 1:
        log(f"  [tiles] {n:,} pts → {n_tiles} slabs of whole {voxel * 1000:g} mm "
            f"cells (bound {int(tile_points):,} pts/slab)")
    # the members of each slab, appended chunk by chunk (file order) → every slab file is ascending
    part = [os.path.join(scratch, f"{tag}_slab{t:05d}.idx") for t in range(len(bounds) - 1)]
    fh = [open(p, "wb") for p in part]
    try:
        for c in range(len(ch.files)):
            loc = _members(mask, ch, c)
            if len(loc) == 0:
                continue
            xa = torch.from_numpy(ch.xyz(c, loc)[:, ax].copy()).to(device)
            ca = torch.floor(xa.double() / voxel).long()
            slab = (torch.searchsorted(torch.from_numpy(bounds).to(device), ca, right=True) - 1).cpu().numpy()
            g = loc + ch.starts[c]
            order = np.argsort(slab, kind="stable")
            sl_sorted = slab[order]
            cuts = np.flatnonzero(np.diff(sl_sorted)) + 1
            for s, e in zip(np.concatenate([[0], cuts]), np.concatenate([cuts, [len(order)]])):
                fh[int(sl_sorted[s])].write(g[order[s:e]].astype(np.int64).tobytes())
    finally:
        for f in fh:
            f.close()
    surv_files = [os.path.join(scratch, f"{tag}_surv{t:05d}.idx") for t in range(len(bounds) - 1)]
    origin = np.zeros(3, np.float64)
    t_v = time.time()
    for t in range(len(bounds) - 1):
        sl = np.fromfile(part[t], dtype=np.int64)
        os.remove(part[t])
        if len(sl) == 0:
            np.zeros(0, np.int64).tofile(surv_files[t])
            continue
        if n_tiles > 1:
            log(f"  [voxel] slab {t + 1}/{len(bounds) - 1}: {len(sl):,} pts ({time.time() - t_v:.0f} s)")
        xyz = ch.gather_xyz(sl)
        cl = np.empty((len(sl), 3), np.int64)
        for a in range(3):
            cl[:, a] = np.floor(xyz[:, a].astype(np.float64) / voxel).astype(np.int64)
        win = np.sort(sl[_voxel_keep_dev(xyz, cl, origin, voxel, device)])
        del xyz, cl
        new[win] = True
        win.tofile(surv_files[t])
    new.flush()
    _empty_cache(device)
    return new, Slabs(ax, bounds, voxel, surv_files, ca0, cmax)


# ── the SOR, slab by slab with a certified halo ──────────────────────────────

def sor_pass(ch: Chunks, sl: Slabs, n_cur: int, knn: int, n_sigma: float, reach: float, *, halo_m: float,
             device: str, query_block: int, candidate_budget: int, scratch: str, log=print):
    """``gpu_cloud_clean._sor_keep`` over the survivors of the voxel pass ``sl``, slab by slab. Returns the keep
    mask, mu, sd."""
    import torch
    from reconstruction.grid_knn import grid_knn
    if n_cur <= knn:
        raise ValueError(f"SOR needs more than knn={knn} points, got {n_cur}")
    kw = {"query_block": query_block, "candidate_budget": candidate_budget}
    v = sl.voxel
    hc0 = max(int(np.ceil(max(halo_m, reach) * (1.0 + _MARGIN) / v)) + 1, 1)
    n_slab = len(sl.files)
    cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def _slab(s):
        if s not in cache:
            idx = sl.members(s)
            xyz = ch.gather_xyz(idx)
            ca = np.floor(xyz[:, sl.ax].astype(np.float64) / v).astype(np.int64)
            cache[s] = (idx, xyz, ca)
            for k in [k for k in cache if abs(k - s) > 2 and k != s][:max(0, len(cache) - 5)]:
                del cache[k]
        return cache[s]

    res_files = []
    n_iso_total = 0
    t0 = time.time()
    for t in range(n_slab):
        idx_t, xyz_t, ca_t = _slab(t)
        b_lo, b_hi = int(sl.bounds[t]), int(sl.bounds[t + 1])
        md = np.full(len(idx_t), np.inf)
        iso = np.zeros(len(idx_t), bool)
        pending = np.arange(len(idx_t))
        hc = hc0
        first = True
        while len(pending):
            whole = (b_lo - hc <= sl.cmin) and (b_hi + hc > sl.cmax)
            parts = [xyz_t]
            for s in range(n_slab):
                if s == t or int(sl.bounds[s + 1]) <= b_lo - hc or int(sl.bounds[s]) >= b_hi + hc:
                    continue
                _i, x_s, c_s = _slab(s)
                keep = (c_s >= b_lo - hc) & (c_s < b_hi + hc)
                if keep.any():
                    parts.append(x_s[keep])
            dom = torch.from_numpy(np.concatenate(parts)).to(device)
            if first:
                # isolated = no neighbour within the reach: exact, the halo holds every point within it
                for q, ii, _d2 in grid_knn(dom, 1, None, radius=reach,
                                           queries=torch.arange(len(idx_t), device=device), **kw):
                    iso[q.cpu().numpy()] = (ii[:, 0] < 0).cpu().numpy()
                pending = np.flatnonzero(~iso)
                first = False
                if len(pending) == 0:
                    break
            unresolved = []
            for q, _ii, d2 in grid_knn(dom, knn, reach, queries=torch.from_numpy(pending).to(device), **kw):
                qn = q.cpu().numpy()
                d = np.sqrt(d2.cpu().numpy())
                s_ = d[:, 0].copy()
                for j in range(1, knn):
                    s_ += d[:, j]
                c = ca_t[qn]
                gap = (np.minimum(c - b_lo, b_hi - 1 - c) + hc) * v * (1.0 - _MARGIN)
                ok = np.ones(len(qn), bool) if whole else (d[:, knn - 1] <= gap)
                md[qn[ok]] = s_[ok] / knn
                unresolved.append(qn[~ok])
            del dom
            pending = np.sort(np.concatenate(unresolved)) if unresolved else np.zeros(0, np.int64)
            if len(pending):
                log(f"  [SOR] slab {t + 1}/{n_slab}: {len(pending):,} pt(s) whose {knn}-th neighbour may lie beyond "
                    f"a {hc * v * 1000:g} mm halo — again with {4 * hc * v * 1000:g} mm")
                hc *= 4
        n_iso_total += int(iso.sum())
        p = os.path.join(scratch, f"sor{t:05d}")
        md.tofile(p + ".md"); iso.tofile(p + ".iso")
        res_files.append((sl.files[t], p))                 # the slab's survivors file IS its index list
        if n_slab > 1:
            log(f"  [SOR] slab {t + 1}/{n_slab}: {len(idx_t):,} pts ({time.time() - t0:.0f} s)")
    _empty_cache(device)
    # the live points' mean distances in the CLOUD'S order (ascending global index), merged chunk by chunk
    n_live = n_cur - n_iso_total
    if n_live == 0:
        raise RuntimeError(f"SOR: no point has a neighbour within {reach} m")
    md_live = np.memmap(os.path.join(scratch, "md_live.f8"), dtype=np.float64, mode="w+", shape=(n_live,))
    res = [(np.memmap(fi, dtype=np.int64, mode="r") if os.path.getsize(fi) else np.zeros(0, np.int64), p)
           for fi, p in res_files]
    ptr = [0] * len(res)
    w = 0
    for c in range(len(ch.files)):
        end = ch.starts[c + 1]
        gi, mdv, isv = [], [], []
        for r, (ix, p) in enumerate(res):
            e = int(np.searchsorted(ix, end, side="left"))
            if e > ptr[r]:
                gi.append(np.asarray(ix[ptr[r]:e]))
                mdv.append(np.fromfile(p + ".md", dtype=np.float64, count=e - ptr[r], offset=8 * ptr[r]))
                isv.append(np.fromfile(p + ".iso", dtype=np.bool_, count=e - ptr[r], offset=ptr[r]))
                ptr[r] = e
        if not gi:
            continue
        gi = np.concatenate(gi); o = np.argsort(gi, kind="stable")
        m_ = np.concatenate(mdv)[o][~np.concatenate(isv)[o]]
        md_live[w:w + len(m_)] = m_
        w += len(m_)
    md_live.flush()
    if not np.isfinite(md_live).all():
        raise RuntimeError("SOR: a point has fewer than knn neighbours in the whole cloud")
    mu = float(md_live.mean())
    # numpy's std (_methods._var) with its temporary on scratch: arr - mean, squared in place, summed, / n, sqrt
    arrmean = np.add.reduce(md_live, keepdims=True) / n_live
    tmp = np.memmap(os.path.join(scratch, "md_dev.f8"), dtype=np.float64, mode="w+", shape=(n_live,))
    np.subtract(md_live, arrmean, out=tmp)
    np.multiply(tmp, tmp, out=tmp)
    sd = float(np.sqrt(np.add.reduce(tmp) / n_live))
    del tmp
    thr = mu + n_sigma * sd
    log(f"  [SOR] {n_iso_total:,} isolated pt(s) (no neighbour within "
        f"{reach * 1000:g} mm) dropped and kept out of mean/std")
    keep = _mask_file(scratch, "sor_keep", ch.n, False)
    for fi, p in res_files:
        ix = np.fromfile(fi, dtype=np.int64)
        md = np.fromfile(p + ".md", dtype=np.float64)
        isv = np.fromfile(p + ".iso", dtype=np.bool_)
        keep[ix[(~isv) & (md <= thr)]] = True
    keep.flush()
    return keep, mu, sd


# ── the run ───────────────────────────────────────────────────────────────────

def _count(mask: np.memmap, n: int) -> int:
    return int(np.count_nonzero(mask[:n])) if n else 0


def run(args, *, bounds: dict, sor_reach: float, halo_m: float, scratch_root: str, log=print) -> int:
    """The production recipe (no witnesses) by pieces — same steps, same log protocol, same bytes."""
    import torch
    dev = "cuda"
    t_pipe = time.time()
    chunk_files = sorted(glob.glob(os.path.join(args.input_dir, "chunk_*.ply")))
    if not chunk_files:
        chunk_files = sorted(glob.glob(os.path.join(args.input_dir, "*_pcd.ply")))
    origin_files = sorted(glob.glob(os.path.join(args.input_dir, "chunk_*_origins.npz")))
    log(f"[GPU-clean] torch {torch.__version__} on {torch.cuda.get_device_name(0)} — BY PIECES (stream_clean)")
    log(f"  Chunks: {len(chunk_files)}  |  Voxel: {args.voxel_size * 1000:.1f}mm  |  SOR: knn={args.sor_knn}, "
        f"σ={args.sor_sigma}")
    os.makedirs(scratch_root, exist_ok=True)
    scratch = tempfile.mkdtemp(prefix="clean_", dir=scratch_root)
    try:
        t0 = time.time()
        log(f"[Step 1/6] Loading and merging {len(chunk_files)} chunks...")
        ch = Chunks(chunk_files, origin_files)
        free = shutil.disk_usage(scratch).free
        need = ch.n * 40
        if free < need:
            raise CleanError(f"scratch {scratch_root}: {free / 2**30:.1f} GB free, the cleaning of {ch.n:,} points "
                             f"needs up to {need / 2**30:.1f} GB (40 B per point) — postprocessing.clean_scratch_dir")
        for c, p in enumerate(chunk_files):
            log(f"  [{c + 1}/{len(chunk_files)}] {os.path.basename(p)}: {int(ch.sizes[c]):,} pts "
                f"({time.time() - t0:.0f} s)")
        total_input = ch.n
        log(f"  ✅ Merged: {total_input:,} points ({time.time() - t0:.1f}s)\n")
        if ch.origin_files:
            inj = ["frame_global", "pixel_row", "pixel_col"] + (["confidence"] if ch.has_conf else [])
            log(f"  ✅ Injected scalar fields {inj} ({total_input:,} pts) (read chunk by chunk)\n")
        mask = _mask_file(scratch, "all", ch.n, True)
        n_cur = ch.n

        if args.conf_min_norm and args.conf_min_norm > 0:
            if not ch.has_conf:
                log("[GPU-clean] ❌ Confidence gate requested but no 'confidence' field")
                return 1
            t1c = time.time()
            confs = [ch.origins(c)["confidence"].astype(np.float32) for c in range(len(ch.files))]
            nz = [x for x in confs if len(x)]
            cmin = float(min(float(x.min()) for x in nz)); cmax = float(max(float(x.max()) for x in nz))
            thr = cmin + args.conf_min_norm * max(cmax - cmin, 1e-6)
            for c, x in enumerate(confs):
                mask[ch.starts[c]:ch.starts[c + 1]] = x >= thr
            n_new = _count(mask, ch.n)
            if n_new == 0:
                log("[GPU-clean] ❌ Confidence gate dropped everything")
                return 1
            log(f"[Step 1c] Confidence gate: norm>={args.conf_min_norm} → raw>={thr:.1f}  {n_cur:,} → {n_new:,} "
                f"({time.time() - t1c:.1f}s)\n")
            n_cur = n_new
            del confs

        if not args.skip_duplicates:
            t2 = time.time()
            log("[Step 2/6] Removing near-duplicate points (micro-voxel 0.1mm)...")
            mask, _ = voxel_pass(ch, mask, 1e-4, tile_points=bounds["tile_points"], device=dev, scratch=scratch,
                                 tag="dup", log=log)
            n_new = _count(mask, ch.n)
            log(f"  ✅ {n_cur:,} → {n_new:,} ({time.time() - t2:.1f}s)\n")
            n_cur = n_new
        else:
            log("[Step 2/6] Near-duplicate removal: SKIPPED\n")

        t3 = time.time()
        log(f"[Step 3/6] Voxel spatial subsampling ({args.voxel_size * 1000:.1f}mm)...")
        mask, slabs = voxel_pass(ch, mask, args.voxel_size, tile_points=bounds["tile_points"], device=dev,
                                 scratch=scratch, tag="voxel", log=log)
        n_new = _count(mask, ch.n)
        log(f"  ✅ {n_cur:,} → {n_new:,} ({(1 - n_new / n_cur) * 100:.1f}% reduction)")
        log(f"  ({time.time() - t3:.1f}s)\n")
        n_cur = n_new

        if not args.skip_sor:
            t4 = time.time()
            log(f"[Step 4/6] Statistical Outlier Removal (knn={args.sor_knn}, σ={args.sor_sigma})...")
            mask, mu, sd = sor_pass(ch, slabs, n_cur, args.sor_knn, args.sor_sigma, sor_reach, halo_m=halo_m,
                                    device=dev, query_block=bounds["knn_query_block"],
                                    candidate_budget=bounds["knn_candidate_budget"], scratch=scratch, log=log)
            n_new = _count(mask, ch.n)
            if n_new == 0:
                log("[GPU-clean] ❌ SOR dropped everything")
                return 1
            log(f"  ✅ {n_cur:,} → {n_new:,} ({n_cur - n_new:,} outliers, {(n_cur - n_new) / n_cur * 100:.1f}%) "
                f"[mean d {mu * 1000:.1f}mm σ {sd * 1000:.1f}mm]")
            log(f"  ({time.time() - t4:.1f}s)\n")
            n_cur = n_new
        else:
            log("[Step 4/6] SOR: SKIPPED\n")
        log("[Step 5/6] Noise filter: SKIPPED\n")

        if args.max_points > 0 and n_cur > args.max_points:
            tm = time.time()
            larger = args.voxel_size / ((args.max_points / n_cur) ** (1 / 3))
            log(f"  🔧 Capping to {args.max_points:,} pts (voxel={larger * 1000:.1f}mm)...")
            mask, _ = voxel_pass(ch, mask, larger, tile_points=bounds["tile_points"], device=dev, scratch=scratch,
                                 tag="cap", log=log)
            n_new = _count(mask, ch.n)
            log(f"  ✅ {n_cur:,} → {n_new:,}  ({time.time() - tm:.1f}s)\n")
            n_cur = n_new
        torch.cuda.empty_cache()
        log("[Step 6/6] Normal estimation: DISABLED (not written to PLY)\n")
        return _save(ch, mask, n_cur, args.output, total_input, t_pipe, log)
    except CleanError as e:
        log(f"[GPU-clean] ❌ {e}")
        return 1
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _save(ch: Chunks, mask: np.memmap, n_out: int, out: str, total_input: int, t_pipe: float, log) -> int:
    t_s = time.time()
    log("[Save] Writing binary PLY...")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    has_o = bool(ch.origin_files)
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n_out}", "property float x",
              "property float y", "property float z"]
    if ch.has_rgb:
        fields += [("r", "u1"), ("g", "u1"), ("b", "u1")]
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    if has_o and ch.has_conf:
        fields += [("confidence", "<f4")]
        header += ["property float confidence"]
    if has_o:
        fields += [("frame_global", "<i4"), ("pixel_row", "<i2"), ("pixel_col", "<i2")]
        header += ["property int frame_global", "property short pixel_row", "property short pixel_col"]
    header.append("end_header")
    dt = np.dtype(fields)
    tmp = out + ".partial"
    ok = False
    try:
        with open(tmp, "wb") as f:
            f.write(("\n".join(header) + "\n").encode("ascii"))
            for c in range(len(ch.files)):
                loc = _members(mask, ch, c)
                if len(loc) == 0:
                    continue
                xyz = ch.xyz(c, loc)
                if not np.isfinite(xyz).all():
                    log("[GPU-clean] ❌ non-finite XYZ — refusing to save")
                    return 1
                packed = np.empty(len(loc), dt)
                packed["x"], packed["y"], packed["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
                if ch.has_rgb:
                    m = ch.maps[c]
                    packed["r"] = m["red"][loc].astype(np.uint8)
                    packed["g"] = m["green"][loc].astype(np.uint8)
                    packed["b"] = m["blue"][loc].astype(np.uint8)
                if has_o:
                    o = ch.origins(c)
                    cols = {"frame_global": o["frame_global"][loc].astype(np.int32),
                            "pixel_row": o["pixel_row"][loc].astype(np.int16),
                            "pixel_col": o["pixel_col"][loc].astype(np.int16)}
                    if ch.has_conf:
                        cols["confidence"] = o["confidence"][loc].astype(np.float32)
                    for name, arr in cols.items():
                        if not np.isfinite(arr.astype(np.float64)).all():
                            log(f"[GPU-clean] ❌ scalar field '{name}' has non-finite values — refusing to save")
                            return 1
                        packed[name] = arr
                packed.tofile(f)
                log(f"  [save {c + 1}/{len(ch.files)}] {len(loc):,} pts ({time.time() - t_s:.0f} s)")
        os.replace(tmp, out)
        ok = True
    finally:
        if not ok and os.path.exists(tmp):
            os.remove(tmp)
    log(f"  ✅ {out}\n     {n_out:,} points | {os.path.getsize(out) / 1048576:.1f} MB | Binary PLY +origins")
    log(f"  ({time.time() - t_s:.1f}s)\n")
    log("=" * 65)
    log(f"  ✅ PIPELINE COMPLETE — {time.time() - t_pipe:.1f}s total")
    log("=" * 65)
    log(f"  Input:     {total_input:,} points ({len(ch.files)} chunks)")
    log(f"  Output:    {n_out:,} points")
    log(f"  Reduction: {(1 - n_out / total_input) * 100:.1f}%")
    log(f"  File:      {os.path.abspath(out)}")
    log("=" * 65)
    return 0
