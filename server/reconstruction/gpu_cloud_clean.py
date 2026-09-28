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
  4. SOR (knn mean-distance, keep < mean + nSigma·std) via the EXACT GPU
     grid kNN of reconstruction.grid_knn — every point's statistic is over
     exactly knn neighbours, the isolated floater's included;
  DETERMINISM (2026-09-28): nothing here reads the card's free VRAM. The
  voxel pick cuts the cloud into slabs of WHOLE cells (one global origin,
  count from N and postprocessing.clean_bounds) and the SOR's kNN is exact —
  identical input gives a bit-identical cloud on any card, for any bound;
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


def _slab_bounds(ca: np.ndarray, n_tiles: int) -> np.ndarray:
    """Integer cell bounds [b_t, b_t+1) along the slab axis, placed where the
    cumulative point count crosses t·N/n_tiles — computed from the SAME integer
    cell index the voxel key uses, so a cell can never straddle two slabs."""
    cum = np.cumsum(np.bincount(ca))
    n = int(cum[-1])
    targets = [(t * n) // n_tiles for t in range(1, n_tiles)]
    inner = np.searchsorted(cum, targets, side="right")
    return np.concatenate([[0], inner, [len(cum)]]).astype(np.int64)


def _voxel_keep_dev(xyz_np: np.ndarray, cells_np: np.ndarray, origin: np.ndarray,
                    voxel: float, device: str) -> np.ndarray:
    """Local indices of the surviving point per voxel of one slab: the point
    nearest the cell centre (float64), ties to the lowest index — every cell of
    the slab is whole, so the pick is the global one."""
    import torch
    n = len(xyz_np)
    p = torch.from_numpy(np.ascontiguousarray(xyz_np)).to(device).double()
    c = torch.from_numpy(cells_np).to(device)
    cl = c - c.min(dim=0).values                     # slab-local key, same cells
    dims = [int(v) + 1 for v in cl.max(dim=0).values.tolist()]
    if dims[0] * dims[1] * dims[2] >= 2 ** 63:
        raise ValueError(f"voxel key overflow: {dims} cells of {voxel} m in one slab")
    key = (cl[:, 0] * dims[1] + cl[:, 1]) * dims[2] + cl[:, 2]
    del cl
    d2 = None
    for a in range(3):
        ctr = float(origin[a]) + (c[:, a].double() + 0.5) * voxel
        da = p[:, a] - ctr
        d2 = da * da if d2 is None else d2 + da * da
    del c, p
    uniq, inv = torch.unique(key, return_inverse=True)
    del key
    best = torch.full((len(uniq),), torch.inf, device=device, dtype=torch.float64)
    best.scatter_reduce_(0, inv, d2, reduce="amin")
    win = d2 == best[inv]
    del best, d2
    idx = torch.arange(n, device=device)
    first = torch.full((len(uniq),), n, device=device, dtype=torch.int64)
    first.scatter_reduce_(0, inv[win], idx[win], reduce="amin")
    return first.cpu().numpy()


def _voxel_keep(xyz_np: np.ndarray, voxel: float, *, tile_points: int,
                device: str) -> np.ndarray:
    """Global indices surviving the voxel pick. ONE grid (origin = the cloud's
    minimum, cells in float64); the cloud is cut into slabs of WHOLE cells along
    its longest axis, their count from N and the declared ``tile_points`` BOUND
    only — the same slabs on any card, and the same survivors for any count."""
    n = len(xyz_np)
    if n == 0:
        return np.empty(0, np.int64)
    origin = xyz_np.min(0).astype(np.float64)
    ax = int(np.argmax(xyz_np.max(0).astype(np.float64) - origin))
    ca = np.floor((xyz_np[:, ax].astype(np.float64) - origin[ax]) / voxel).astype(np.int64)
    n_tiles = max(1, -(-n // int(tile_points)))
    bounds = _slab_bounds(ca, n_tiles)
    if n_tiles > 1:
        print(f"  [tiles] {n:,} pts → {n_tiles} slabs of whole {voxel * 1000:g} mm "
              f"cells (bound {int(tile_points):,} pts/slab)")
    out = []
    for t in range(len(bounds) - 1):
        sl = (np.arange(n, dtype=np.int64) if n_tiles == 1
              else np.flatnonzero((ca >= bounds[t]) & (ca < bounds[t + 1])))
        if len(sl) == 0:
            continue
        cells = np.empty((len(sl), 3), np.int64)
        for a in range(3):
            cells[:, a] = (ca[sl] if a == ax else np.floor(
                (xyz_np[sl, a].astype(np.float64) - origin[a]) / voxel).astype(np.int64))
        out.append(sl[_voxel_keep_dev(xyz_np[sl], cells, origin, voxel, device)])
        del cells
    _empty_cache(device)
    return np.sort(np.concatenate(out))


def _sor_keep(xyz_np: np.ndarray, knn: int, n_sigma: float, cell_h: float, *,
              device: str, query_block: int, candidate_budget: int):
    """Boolean keep mask, mean distance to the EXACT ``knn`` nearest neighbours
    (reconstruction.grid_knn) — every statistic over exactly ``knn`` distances,
    the isolated floater included (its true distance, never +inf or the mean of
    the few a box happened to hold). ``cell_h`` only sizes the first search
    level. Global mu/sd in float64."""
    import torch
    from reconstruction.grid_knn import grid_knn
    n = len(xyz_np)
    if n <= knn:
        raise ValueError(f"SOR needs more than knn={knn} points, got {n}")
    pts = torch.from_numpy(np.ascontiguousarray(xyz_np)).to(device)
    mean_d = np.empty(n, np.float64)
    for q, idx, d2 in grid_knn(pts, knn, cell_h, query_block=query_block,
                               candidate_budget=candidate_budget):
        # the exact squared distances come back to the host: numpy's sqrt is
        # IEEE correctly rounded on every path (torch's CPU sqrt is not), and
        # the sum runs in a fixed order
        d = np.sqrt(d2.cpu().numpy())
        s = d[:, 0].copy()
        for j in range(1, knn):
            s += d[:, j]
        mean_d[q.cpu().numpy()] = s / knn
    del pts
    _empty_cache(device)
    if not np.isfinite(mean_d).all():
        raise RuntimeError("SOR: a point has fewer than knn neighbours in the whole cloud")
    mu = float(mean_d.mean())
    sd = float(mean_d.std())
    thr = mu + n_sigma * sd
    return mean_d <= thr, mu, sd


def _empty_cache(device: str) -> None:
    if str(device).startswith("cuda"):
        import torch
        torch.cuda.empty_cache()


def _clean_bounds() -> dict:
    """postprocessing.clean_bounds — memory BOUNDS, never decisions (the
    surviving set is bit-identical for any value). Missing key = FAIL."""
    from reconstruction.grid_knn import config_section
    return {k: int(v) for k, v in config_section(
        ("postprocessing", "clean_bounds"),
        ("tile_points", "knn_query_block", "knn_candidate_budget")).items()}


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
    ap.add_argument("--witness", action="store_true",
                    help="claude_stac.txt §6: mv_votes + provisional status on the merged "
                         "cloud BEFORE the net; voxel/SOR then run only on witness."
                         "clean_statuses (server config.yaml witness:)")
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "8")
    try:
        os.sched_setaffinity(0, set(range(min(8, os.cpu_count() or 8))))
    except Exception:  # noqa: BLE001
        pass
    server_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)
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
    # memory BOUNDS from config — the same slabs and blocks on any card; the
    # surviving set never depends on them nor on the card's free VRAM
    bounds = _clean_bounds()
    tile_pts = bounds["tile_points"]
    knn_kw = {"query_block": bounds["knn_query_block"],
              "candidate_budget": bounds["knn_candidate_budget"]}
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
    wit = None            # witness fields (dict of uint8 arrays) — §6, before the net

    def _apply(mask_or_idx):
        nonlocal xyz, rgb, fg, pr, pc, conf, keep, wit
        xyz = xyz[mask_or_idx]
        keep = keep[mask_or_idx]
        rgb = rgb[mask_or_idx] if rgb is not None else None
        fg = fg[mask_or_idx] if fg is not None else None
        pr = pr[mask_or_idx] if pr is not None else None
        pc = pc[mask_or_idx] if pc is not None else None
        conf = conf[mask_or_idx] if conf is not None else None
        if wit is not None:
            wit = {k: v[mask_or_idx] for k, v in wit.items()}

    # ── Step 1w: witnesses (§6.1/§6.3) on the merged cloud, BEFORE any point
    # can be dropped: mv_votes per point through the provenance with the final
    # poses; masks do not exist yet (segmentation comes later) → provisional
    # status. The net (voxel + SOR) then runs ONLY on clean_statuses; every
    # other point is kept untouched with its status — nothing silent.
    eligible = None
    if args.witness:
        if fg is None:
            print("[GPU-clean] ❌ --witness needs the provenance fields (origins) — "
                  "cannot compute per-point witnesses")
            return 1
        tw = time.time()
        from reconstruction.loops.config import load_loops_config
        from reconstruction.witness.frames import load_session_frames
        from reconstruction.witness.run import witness_fields, summarize
        from reconstruction.witness.status import status_mask
        wcfg = load_loops_config().witness
        out_dir = os.path.dirname(os.path.abspath(args.output))
        frames = load_session_frames(out_dir, log=lambda m: print(f"  [witness] {m}"))
        wit = witness_fields(xyz, fg, pr, pc, frames, wcfg, None, None, (), device=dev)
        s = summarize(wit)
        eligible = status_mask(wit["status"], wcfg.clean_statuses)
        print(f"[Step 1w] Witnesses on {total_input:,} pts ({len(frames)} keyframes): "
              f"{s['status_counts']} | mv_votes mean {s['mv_votes_mean']:.2f} | net eligible "
              f"{int(eligible.sum()):,} ({wcfg.clean_statuses}) ({time.time() - tw:.1f}s)\n")

        # ── Step 1d: the statuses the session does not want in the cloud ──
        # USER 2026-09-17, looking at the kit's Votes view on pccr: "la
        # eliminacion quiero que sea real porque no quiero que esos voladores
        # se usen para computar nada, ni para comparar ni para siquiera
        # calcular el OBB". This is the earliest point at which that is true:
        # the masks, the mask↔cloud matching, the OBBs, the reprojection
        # verdicts and the certification all read the cloud this step writes.
        #
        # What goes is named by STATUS, never by a number invented here:
        # `single_witness` is "observed, and fewer than
        # witness.rules.verified_min_mv_votes neighbouring keyframes agreed
        # with its depth" — a measurement that contradicts the point, not a
        # threshold. `unobserved` is NOT dropped by asking for it: no witness
        # could be computed there, and §6 is explicit that a zero would be a
        # claim. Whoever lists it in witness.drop_statuses says so on purpose.
        drop_names = tuple(wcfg.drop_statuses)
        if drop_names:
            from reconstruction.witness.status import STATUS_CODES
            doomed = status_mask(wit["status"], drop_names)
            n_drop = int(doomed.sum())
            if n_drop:
                shares = {n: int((wit["status"] == STATUS_CODES[n]).sum()) for n in drop_names}
                _apply(~doomed)
                eligible = status_mask(wit["status"], wcfg.clean_statuses)
                print(f"[Step 1d] Dropped {n_drop:,} pt(s) ({100 * n_drop / total_input:.2f}%) "
                      f"by status {shares} — witness.drop_statuses; "
                      f"{len(xyz):,} remain, net eligible {int(eligible.sum()):,}")
            else:
                print(f"[Step 1d] witness.drop_statuses={list(drop_names)}: no point carries "
                      f"those statuses — nothing dropped")

    def _net(fn_keep_idx):
        """Run a keep-index filter on the net-ELIGIBLE points only; the rest
        survive untouched. Returns a global boolean mask."""
        nonlocal eligible
        if eligible is None:
            return fn_keep_idx(xyz)
        sub = np.flatnonzero(eligible)
        ki = fn_keep_idx(xyz[sub])
        m = ~eligible
        m[sub[ki]] = True
        return m

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
        if eligible is not None:
            m = m | ~eligible          # the gate is part of the net: eligible points only
        if not m.any():
            print("[GPU-clean] ❌ Confidence gate dropped everything")
            return 1
        # the count BEFORE this gate is the live one — Step 1d may already have
        # removed points, and total_input is the merged input for the summary
        print(f"[Step 1c] Confidence gate: norm>={args.conf_min_norm} → "
              f"raw>={thr:.1f}  {len(xyz):,} → {int(m.sum()):,} "
              f"({time.time() - t1c:.1f}s)\n")
        _apply(m)
        if eligible is not None:
            eligible = eligible[m]

    # ── Step 2: near-duplicate micro-voxel ──
    if not args.skip_duplicates:
        t2 = time.time()
        n_b = len(xyz)
        print("[Step 2/6] Removing near-duplicate points "
              "(micro-voxel 0.1mm)...")
        m = _net(lambda p: _voxel_keep(p, 1e-4, tile_points=tile_pts, device=dev))
        _apply(m)
        if eligible is not None:
            eligible = eligible[m]
        print(f"  ✅ {n_b:,} → {len(xyz):,} ({time.time() - t2:.1f}s)\n")
    else:
        print("[Step 2/6] Near-duplicate removal: SKIPPED\n")

    # ── Step 3: voxel subsampling ──
    t3 = time.time()
    n_b = len(xyz)
    print(f"[Step 3/6] Voxel spatial subsampling "
          f"({args.voxel_size * 1000:.1f}mm)...")
    m = _net(lambda p: _voxel_keep(p, args.voxel_size, tile_points=tile_pts,
                                   device=dev))
    _apply(m)
    if eligible is not None:
        eligible = eligible[m]
    print(f"  ✅ {n_b:,} → {len(xyz):,} "
          f"({(1 - len(xyz) / n_b) * 100:.1f}% reduction)")
    print(f"  ({time.time() - t3:.1f}s)\n")

    # ── Step 4: SOR ──
    if not args.skip_sor:
        t4 = time.time()
        n_b = len(xyz)
        print(f"[Step 4/6] Statistical Outlier Removal "
              f"(knn={args.sor_knn}, σ={args.sor_sigma})...")
        stats = {}

        def _sor_idx(p):
            # cell_h sizes the FIRST level of the exact kNN only — performance,
            # never the answer
            mm, mu_, sd_ = _sor_keep(p, args.sor_knn, args.sor_sigma,
                                     cell_h=max(args.voxel_size * 3.0, 0.01),
                                     device=dev, **knn_kw)
            stats["mu"], stats["sd"] = mu_, sd_
            return np.flatnonzero(mm)

        m = _net(_sor_idx)
        mu, sd = stats["mu"], stats["sd"]
        if not m.any():
            print("[GPU-clean] ❌ SOR dropped everything")
            return 1
        _apply(m)
        if eligible is not None:
            eligible = eligible[m]
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
        m = _net(lambda p: _voxel_keep(p, larger, tile_points=tile_pts, device=dev))
        _apply(m)
        if eligible is not None:
            eligible = eligible[m]
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
    if wit is not None:
        for name in ("mv_votes", "mask_votes", "mask_conflicts", "status"):
            fields.append((name, "u1"))
            header.append(f"property uchar {name}")
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
    if wit is not None:
        for name in ("mv_votes", "mask_votes", "mask_conflicts", "status"):
            packed[name] = wit[name]
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
