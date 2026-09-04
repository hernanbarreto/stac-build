"""
GPU cloud cleaning (USER ORDER 2026-09-04: "porta ahora cloudcompy a gpu") —
torch/CUDA replacement for the CloudComPy postprocess stage, which is
single-core CPU (CloudCompare has no CUDA path) and crawled on pccr_v1's
521M merged points.

Replicates cloudcompy_postprocess.py semantics step by step:
  1. load + merge chunk_*.ply, inject origins (frame/pixel/confidence) —
     size mismatch FAILS (traceability is mandatory, no fallback);
  1c. confidence gate (min-max normalised threshold, same as the viewer);
  2. near-duplicate removal (micro-voxel 0.1 mm) when not skipped;
  3. voxel spatial subsampling — one SURVIVING point per voxel_size cell,
     the one nearest the cell centre (CloudCompare's resampleCloudSpatially
     is a greedy min-distance pick; the voxel-grid pick matches its density
     within the cell-diagonal bound and keeps REAL points, never averages —
     provenance survives);
  4. SOR (knn mean-distance, keep < mean + nSigma·std) via GPU grid-hash
     kNN. A point whose 27-cell neighbourhood holds no neighbours has
     mean distance +inf — exactly the isolated outlier SOR drops;
  5. noise filter NOT implemented — requesting it fails loudly (production
     skips it: skip_noise true since the O(n²) hang);
  6. normals disabled (as in the CPU script);
  final NaN gates + the exact same binary-PLY layout/field order.

stdout speaks the same "[Step X/6]" protocol the worker's progress parser
expects.

Hernán Barreto - Ingerop IN3 Session IV - STAC
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_PLY_TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8",
              "uchar": "u1", "uint8": "u1", "char": "i1", "short": "<i2",
              "ushort": "<u2", "int": "<i4", "uint": "<u4"}


def _read_ply(path: str):
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path}: not a PLY")
        n, props = 0, []
        while True:
            ln = f.readline().decode("ascii", "replace").strip()
            if ln.startswith("format") and "binary_little_endian" not in ln:
                raise ValueError(f"{path}: not binary_little_endian")
            elif ln.startswith("element vertex"):
                n = int(ln.split()[-1])
            elif ln.startswith("property") and n:
                _, t, name = ln.split()[:3]
                props.append((name, _PLY_TYPES[t]))
            elif ln == "end_header":
                break
        return np.fromfile(f, dtype=np.dtype(props), count=n)


def _tiles_for(n_pts: int, bytes_per_pt: int, extra: int = 0):
    """SIZE-ADAPTIVE (USER 2026-09-04: the system adapts to ANY cloud, it is
    never written for one in particular): number of spatial tiles so the
    working set fits 55% of the FREE VRAM."""
    import torch
    free_b, _ = torch.cuda.mem_get_info()
    budget = int(free_b * 0.55)
    return max(1, int(np.ceil((n_pts * bytes_per_pt + extra)
                              / max(budget, 1))))


def _slab_bounds(xyz: np.ndarray, cell: float, n_tiles: int):
    """(axis, snapped quantile bounds) — bounds land on the cell grid so a
    cell never spans two slabs."""
    ax = int(np.argmax(xyz.max(0) - xyz.min(0)))
    v = xyz[:, ax]
    lo = float(v.min())
    qs = np.quantile(v, np.linspace(0, 1, n_tiles + 1))
    qs = lo + np.round((qs - lo) / cell) * cell
    qs[0], qs[-1] = -np.inf, np.inf
    return ax, qs


def _voxel_keep_dev(dev_xyz, voxel: float, origin=None):
    """Resident: surviving point per voxel (nearest to cell centre).
    ``origin``: GLOBAL grid origin — tiles MUST share it or cell boundaries
    shift per tile (measured: 11/786k picks differed without it)."""
    import torch
    n = len(dev_xyz)
    dev = dev_xyz.device
    key = None
    d2 = torch.zeros(n, device=dev, dtype=torch.float32)
    for ax in range(3):
        v = dev_xyz[:, ax]
        mn = v.min() if origin is None else float(origin[ax])
        c = torch.floor((v - mn) / voxel).to(torch.int64)
        dim = int(c.max().item()) + 1
        key = c if key is None else key * dim + c
        d2 += (v - ((c.to(v.dtype) + 0.5) * voxel + mn)) ** 2
        del c
    uniq, inv = torch.unique(key, return_inverse=True)
    del key
    best = torch.full((len(uniq),), torch.inf, device=dev, dtype=d2.dtype)
    best.scatter_reduce_(0, inv, d2, reduce="amin")
    win = d2 <= best[inv] * (1 + 1e-6)
    del best
    idx = torch.arange(n, device=dev)
    first = torch.full((len(uniq),), n, device=dev, dtype=torch.int64)
    first.scatter_reduce_(0, inv[win], idx[win], reduce="amin")
    return first[first < n]


def _voxel_keep(xyz_np: np.ndarray, voxel: float):
    """Global indices surviving the voxel pick, tiled to fit any N (cells
    never span slabs → per-slab picks are globally exact, no halo needed)."""
    import torch
    n = len(xyz_np)
    n_tiles = _tiles_for(n, 44)
    if n_tiles == 1:
        d = torch.from_numpy(xyz_np).float().cuda()
        keep = _voxel_keep_dev(d, voxel).cpu().numpy()
        del d
        torch.cuda.empty_cache()
        return np.sort(keep)
    print(f"  [adaptive] {n:,} pts → {n_tiles} spatial tiles (VRAM budget)")
    gmin = xyz_np.min(0)
    ax, qs = _slab_bounds(xyz_np, voxel, n_tiles)
    v = xyz_np[:, ax]
    out = []
    for t in range(n_tiles):
        sl = np.flatnonzero((v >= qs[t]) & (v < qs[t + 1]))
        if len(sl) == 0:
            continue
        d = torch.from_numpy(xyz_np[sl]).float().cuda()
        keep_local = _voxel_keep_dev(d, voxel, origin=gmin).cpu().numpy()
        out.append(sl[keep_local])
        del d
        torch.cuda.empty_cache()
    return np.sort(np.concatenate(out))


def _sor_meand_dev(dev_xyz, core_local, knn: int, cell_h: float,
                   cand_per_cell: int = 48, block: int = 200_000):
    """Resident: mean kNN distance for the ``core_local`` points of one
    (slab+halo) tile. Returns a float32 CPU array aligned with core_local."""
    import torch
    dev = dev_xyz.device
    mins = dev_xyz.min(dim=0).values
    cell = torch.floor((dev_xyz - mins) / cell_h).to(torch.int64)
    dims = cell.max(dim=0).values + 2
    key = (cell[:, 0] * dims[1] + cell[:, 1]) * dims[2] + cell[:, 2]
    del cell
    order = torch.argsort(key)
    skey = key[order]
    sxyz = dev_xyz[order]
    uniq, counts = torch.unique_consecutive(skey, return_counts=True)
    starts = torch.cumsum(counts, 0) - counts
    offs = [(dx * dims[1] + dy) * dims[2] + dz
            for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    C = cand_per_cell
    nq = len(core_local)
    out = np.empty(nq, np.float32)
    ar = torch.arange(C, device=dev)
    tcore = torch.from_numpy(core_local).to(dev)
    for b0 in range(0, nq, block):
        cidx = tcore[b0:b0 + block]
        pk = key[cidx]
        px = dev_xyz[cidx]
        B = len(cidx)
        cand = torch.full((B, 27 * C), -1, device=dev, dtype=torch.int64)
        for oi, off in enumerate(offs):
            pos = torch.searchsorted(uniq, pk + off)
            pos = pos.clamp(max=len(uniq) - 1)
            hit = uniq[pos] == pk + off
            st = starts[pos]
            cnt = counts[pos].clamp(max=C)
            take = ar[None, :] < (cnt * hit)[:, None]
            cand[:, oi * C:(oi + 1) * C] = torch.where(
                take, st[:, None] + ar[None, :], -1)
        val = cand >= 0
        cxyz = sxyz[cand.clamp(min=0)]
        d2 = ((cxyz - px[:, None, :]) ** 2).sum(dim=2)
        d2 = torch.where(val, d2, torch.inf)
        k_eff = min(knn + 1, d2.shape[1])
        small = torch.topk(d2, k_eff, dim=1, largest=False).values
        # drop the self-distance (first ~0 column) and average the next knn
        small = torch.sqrt(small[:, 1:knn + 1])
        finite = torch.isfinite(small)
        cnt_f = finite.sum(dim=1).clamp(min=1)
        md = torch.where(finite, small, torch.zeros_like(small)).sum(dim=1) \
            / cnt_f
        md = torch.where(finite.any(dim=1), md,
                         torch.tensor(torch.inf, device=dev))
        out[b0:b0 + block] = md.float().cpu().numpy()
        del cand, val, cxyz, d2, small
    return out


def _sor_keep(xyz_np: np.ndarray, knn: int, n_sigma: float, cell_h: float):
    """Boolean keep mask (global stats), tiled with a one-cell halo so the
    slab borders keep their true neighbours — adapts to any N."""
    import torch
    n = len(xyz_np)
    n_tiles = _tiles_for(n, 44)
    mean_d = np.empty(n, np.float32)
    if n_tiles == 1:
        d = torch.from_numpy(xyz_np).float().cuda()
        mean_d[:] = _sor_meand_dev(d, np.arange(n, dtype=np.int64),
                                   knn, cell_h)
        del d
        torch.cuda.empty_cache()
    else:
        print(f"  [adaptive] {n:,} pts → {n_tiles} spatial tiles "
              f"(VRAM budget, halo {cell_h * 100:.0f}cm)")
        ax, qs = _slab_bounds(xyz_np, cell_h, n_tiles)
        v = xyz_np[:, ax]
        for t in range(n_tiles):
            halo = np.flatnonzero((v >= qs[t] - cell_h)
                                  & (v < qs[t + 1] + cell_h))
            if len(halo) == 0:
                continue
            core_mask = (v[halo] >= qs[t]) & (v[halo] < qs[t + 1])
            core_local = np.flatnonzero(core_mask).astype(np.int64)
            d = torch.from_numpy(xyz_np[halo]).float().cuda()
            mean_d[halo[core_local]] = _sor_meand_dev(
                d, core_local, knn, cell_h)
            del d
            torch.cuda.empty_cache()
    fin = np.isfinite(mean_d)
    mu = float(mean_d[fin].astype(np.float64).mean())
    sd = float(mean_d[fin].astype(np.float64).std())
    thr = mu + n_sigma * sd
    return mean_d <= thr, mu, sd


def main() -> int:
    try:                     # the worker streams our stdout line by line
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="GPU cloud cleaning (torch)")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--voxel-size", type=float, default=0.001)
    ap.add_argument("--sor-knn", type=int, default=6)
    ap.add_argument("--sor-sigma", type=float, default=1.0)
    ap.add_argument("--noise-radius", type=float, default=0.01)
    ap.add_argument("--noise-sigma", type=float, default=1.0)
    ap.add_argument("--conf-min-norm", type=float, default=0.0)
    ap.add_argument("--max-points", type=int, default=0)
    ap.add_argument("--skip-duplicates", action="store_true")
    ap.add_argument("--skip-sor", action="store_true")
    ap.add_argument("--skip-noise", action="store_true")
    ap.add_argument("--skip-normals", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "8")
    try:
        os.sched_setaffinity(0, set(range(min(8, os.cpu_count() or 8))))
    except Exception:  # noqa: BLE001
        pass
    import torch
    if not torch.cuda.is_available():
        print("[GPU-clean] ❌ CUDA not available — refusing (use the "
              "CloudComPy path: postprocessing.gpu_clean: false)")
        return 1
    if not args.skip_noise:
        print("[GPU-clean] ❌ noise filter requested but not implemented on "
              "GPU (production skips it) — set skip_noise or use the "
              "CloudComPy path")
        return 1
    dev = "cuda"
    t_pipe = time.time()

    chunk_files = sorted(glob.glob(os.path.join(args.input_dir,
                                                "chunk_*.ply")))
    if not chunk_files:
        chunk_files = sorted(glob.glob(os.path.join(args.input_dir,
                                                    "*_pcd.ply")))
    print(f"[GPU-clean] torch {torch.__version__} on "
          f"{torch.cuda.get_device_name(0)}")
    print(f"  Chunks: {len(chunk_files)}  |  Voxel: "
          f"{args.voxel_size * 1000:.1f}mm  |  SOR: knn={args.sor_knn}, "
          f"σ={args.sor_sigma}")

    # ── Step 1: load + merge + origins ──
    t0 = time.time()
    print(f"[Step 1/6] Loading and merging {len(chunk_files)} chunks...")
    xyz_l, rgb_l = [], []
    for i, p in enumerate(chunk_files):
        v = _read_ply(p)
        xyz_l.append(np.stack([v["x"], v["y"], v["z"]], 1))
        names = v.dtype.names
        if all(c in names for c in ("red", "green", "blue")):
            rgb_l.append(np.stack([v["red"], v["green"], v["blue"]], 1))
        print(f"  [{i + 1}/{len(chunk_files)}] {os.path.basename(p)}: "
              f"{len(v):,} pts")
    xyz = np.concatenate(xyz_l).astype(np.float32)
    del xyz_l
    rgb = (np.concatenate(rgb_l).astype(np.uint8)
           if len(rgb_l) == len(chunk_files) else None)
    del rgb_l
    total_input = len(xyz)
    print(f"  ✅ Merged: {total_input:,} points ({time.time() - t0:.1f}s)\n")

    origin_files = sorted(glob.glob(os.path.join(
        args.input_dir, "chunk_*_origins.npz")))
    fg = pr = pc = conf = None
    if origin_files:
        t1 = time.time()
        cols = {k: [] for k in ("frame_global", "pixel_row", "pixel_col",
                                "confidence")}
        for of in origin_files:
            d = np.load(of)
            for k in cols:
                if k in d:
                    cols[k].append(d[k])
        fg = np.concatenate(cols["frame_global"]).astype(np.int32)
        pr = np.concatenate(cols["pixel_row"]).astype(np.int16)
        pc = np.concatenate(cols["pixel_col"]).astype(np.int16)
        if cols["confidence"]:
            conf = np.concatenate(cols["confidence"]).astype(np.float32)
        if len(fg) != total_input:
            print(f"[GPU-clean] ❌ Origin size mismatch: {len(fg):,} vs "
                  f"cloud {total_input:,} — cannot inject traceability")
            return 1
        inj = ["frame_global", "pixel_row", "pixel_col"] + \
              (["confidence"] if conf is not None else [])
        print(f"  ✅ Injected scalar fields {inj} ({total_input:,} pts) "
              f"({time.time() - t1:.1f}s)\n")

    keep = np.arange(total_input, dtype=np.int64)

    def _apply(mask_or_idx):
        nonlocal xyz, rgb, fg, pr, pc, conf, keep
        xyz = xyz[mask_or_idx]
        keep = keep[mask_or_idx]
        rgb = rgb[mask_or_idx] if rgb is not None else None
        fg = fg[mask_or_idx] if fg is not None else None
        pr = pr[mask_or_idx] if pr is not None else None
        pc = pc[mask_or_idx] if pc is not None else None
        conf = conf[mask_or_idx] if conf is not None else None

    # ── Step 1c: confidence gate ──
    if args.conf_min_norm and args.conf_min_norm > 0:
        if conf is None:
            print("[GPU-clean] ❌ Confidence gate requested but no "
                  "'confidence' field")
            return 1
        t1c = time.time()
        cmin, cmax = float(conf.min()), float(conf.max())
        thr = cmin + args.conf_min_norm * max(cmax - cmin, 1e-6)
        m = conf >= thr
        if not m.any():
            print("[GPU-clean] ❌ Confidence gate dropped everything")
            return 1
        print(f"[Step 1c] Confidence gate: norm>={args.conf_min_norm} → "
              f"raw>={thr:.1f}  {total_input:,} → {int(m.sum()):,} "
              f"({time.time() - t1c:.1f}s)\n")
        _apply(m)

    # ── Step 2: near-duplicate micro-voxel ──
    if not args.skip_duplicates:
        t2 = time.time()
        n_b = len(xyz)
        print("[Step 2/6] Removing near-duplicate points "
              "(micro-voxel 0.1mm)...")
        ki = _voxel_keep(xyz, 1e-4)
        _apply(ki)
        print(f"  ✅ {n_b:,} → {len(xyz):,} ({time.time() - t2:.1f}s)\n")
    else:
        print("[Step 2/6] Near-duplicate removal: SKIPPED\n")

    # ── Step 3: voxel subsampling ──
    t3 = time.time()
    n_b = len(xyz)
    print(f"[Step 3/6] Voxel spatial subsampling "
          f"({args.voxel_size * 1000:.1f}mm)...")
    ki = _voxel_keep(xyz, args.voxel_size)
    _apply(ki)
    print(f"  ✅ {n_b:,} → {len(xyz):,} "
          f"({(1 - len(xyz) / n_b) * 100:.1f}% reduction)")
    print(f"  ({time.time() - t3:.1f}s)\n")

    # ── Step 4: SOR ──
    if not args.skip_sor:
        t4 = time.time()
        n_b = len(xyz)
        print(f"[Step 4/6] Statistical Outlier Removal "
              f"(knn={args.sor_knn}, σ={args.sor_sigma})...")
        m, mu, sd = _sor_keep(xyz, args.sor_knn, args.sor_sigma,
                              cell_h=max(args.voxel_size * 3.0, 0.01))
        if not m.any():
            print("[GPU-clean] ❌ SOR dropped everything")
            return 1
        _apply(m)
        print(f"  ✅ {n_b:,} → {len(xyz):,} ({n_b - len(xyz):,} outliers, "
              f"{(n_b - len(xyz)) / n_b * 100:.1f}%) "
              f"[mean d {mu * 1000:.1f}mm σ {sd * 1000:.1f}mm]")
        print(f"  ({time.time() - t4:.1f}s)\n")
    else:
        print("[Step 4/6] SOR: SKIPPED\n")

    print("[Step 5/6] Noise filter: SKIPPED\n")

    # ── max_points cap ──
    if args.max_points > 0 and len(xyz) > args.max_points:
        tm = time.time()
        n_b = len(xyz)
        larger = args.voxel_size / ((args.max_points / n_b) ** (1 / 3))
        print(f"  🔧 Capping to {args.max_points:,} pts "
              f"(voxel={larger * 1000:.1f}mm)...")
        ki = _voxel_keep(xyz, larger)
        _apply(ki)
        print(f"  ✅ {n_b:,} → {len(xyz):,}  ({time.time() - tm:.1f}s)\n")
    torch.cuda.empty_cache()

    print("[Step 6/6] Normal estimation: DISABLED (not written to PLY)\n")

    # ── NaN gates (same as the CPU script) ──
    if not np.isfinite(xyz).all():
        print("[GPU-clean] ❌ non-finite XYZ — refusing to save")
        return 1
    for name, arr in (("confidence", conf), ("frame_global", fg),
                      ("pixel_row", pr), ("pixel_col", pc)):
        if arr is not None and not np.isfinite(
                arr.astype(np.float64)).all():
            print(f"[GPU-clean] ❌ scalar field '{name}' has non-finite "
                  f"values — refusing to save")
            return 1

    # ── save: byte-identical layout to the CloudComPy writer ──
    t_s = time.time()
    print("[Save] Writing binary PLY...")
    out = args.output
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    header = ["ply", "format binary_little_endian 1.0",
              f"element vertex {len(xyz)}", "property float x",
              "property float y", "property float z"]
    if rgb is not None:
        fields += [("r", "u1"), ("g", "u1"), ("b", "u1")]
        header += ["property uchar red", "property uchar green",
                   "property uchar blue"]
    if conf is not None:
        fields += [("confidence", "<f4")]
        header += ["property float confidence"]
    if fg is not None:
        fields += [("frame_global", "<i4"), ("pixel_row", "<i2"),
                   ("pixel_col", "<i2")]
        header += ["property int frame_global", "property short pixel_row",
                   "property short pixel_col"]
    header.append("end_header")
    packed = np.empty(len(xyz), np.dtype(fields))
    packed["x"], packed["y"], packed["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    if rgb is not None:
        packed["r"], packed["g"], packed["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    if conf is not None:
        packed["confidence"] = conf
    if fg is not None:
        packed["frame_global"] = fg
        packed["pixel_row"] = pr
        packed["pixel_col"] = pc
    with open(out, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        packed.tofile(f)
    print(f"  ✅ {out}\n     {len(xyz):,} points | "
          f"{os.path.getsize(out) / 1048576:.1f} MB | Binary PLY +origins")
    print(f"  ({time.time() - t_s:.1f}s)\n")

    print("=" * 65)
    print(f"  ✅ PIPELINE COMPLETE — {time.time() - t_pipe:.1f}s total")
    print("=" * 65)
    print(f"  Input:     {total_input:,} points ({len(chunk_files)} chunks)")
    print(f"  Output:    {len(xyz):,} points")
    print(f"  Reduction: {(1 - len(xyz) / total_input) * 100:.1f}%")
    print(f"  File:      {os.path.abspath(out)}")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
