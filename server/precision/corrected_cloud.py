"""F7-C — the CORRECTED CLOUD as the reconstruction's product, built on F6's depth.

USER 2026-09-29, on pccr, after the witness fusion (F7) delivered 9 M / 0.2 M-point
clouds judged worse than Omega's own: the complete Omega cloud re-projected with the
core's corrections — *"prácticamente corrigió la duplicidad del desk … y el piso"*,
*"hasta ahora la mejor nube"* — and *"que quede todo en el pipeline, nada a mano"*.
The same night (USER, plan validated): MAXIMUM COVERAGE from F6, no confirmation filter,
the contradiction majority as the flyer filter, and this recipe on top of F6's depth:

    point = c2w_F5 · ( z_F6(u, v) · K_F5⁻¹ [u, v, 1] )        (u, v) on F6's grid

- z_F6: F6's stored depth (``output/depth_native/frame_<f>.npz``) where its ``source`` is
  tier 0 (the plane sweep) or tier 1 (Omega × s_k, the prior at maximum coverage: every
  prior pixel the sweep did not confirm and no view contradicts);
- THE GRID: F6 works in the UNDISTORTED NATIVE frame of F0's maps with K_new = K_F5, so
  the unprojection is K_F5 itself on the camera's native (W, H) — never Omega's record
  grid (``K_on_grid``): a grid mismatch silently scales or shifts the cloud, so a frame
  not on that grid fails the step;
- THE METRIC FRAME: F6 carried Omega to F5's world with its measured s_k; its report
  must carry the session's camera epoch and its geometry epoch (or only NEW-CLOUD epochs
  after it — those move no camera) or the step fails naming the file. s_k is F6's own
  (report ``per_frame``), carried into this report;
- the confidence gate of the reconstruction (conf_percentile + conf_min_norm, the fork's
  rule, per Omega chunk, thresholds on Omega's own records), the record's confidence
  carried onto F6's grid through the same lens + grid map F6 carries its prior;
- then THE CLOUD STAGE'S OWN RECIPE: voxel + SOR (reconstruction/gpu_cloud_clean), the
  witness filter fed F6's frames (single_witness leaves), the SILHOUETTE filter
  (precision/silhouette_filter.py: SAM3's masks, both PLYs, the same rows), the scene
  consolidation with normals from F6's depth — and the octree, published as a NEW-CLOUD
  epoch through the same transaction as F7 (PLY + raw, origins v2, epoch record, atomic
  swap, ledger).

Provenance. Inside the step ``pixel_row``/``pixel_col`` are F6's pixel. At publication
they become the pixel of Omega's RECORD grid — the trace grid ``intrinsic.txt`` describes,
which the cloud stage and the correction stage index (the product's convention before
this change) — through the lens and ``native_to_grid``; ``origins.npz`` carries F6's exact
pixel (``pixel_u_und``/``pixel_v_und``), its tier (``source`` 0 | 1), ``n_consistent``,
``ncc`` and ``residual_rel``.

Measured on pccr (2026-09-29, the Omega × s_k version): epoch 0 (Omega) 28.5 M pts,
floor-layer thickness per 1 m cell 20.3 cm median → corrected 25.5 M / 12.9 cm → +
witness/MLS 21.8 M / 11.5 cm.

Runner step ``f7_cloud`` (after F6). ``precision.cloud.source`` selects this
(``omega_corrected``, default) or the witness fusion (``fusion``: F6 sweep → F7).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

CLOUD_REPORT = "corrected_cloud.json"
TX_TMP = "_tx_corrected_cloud"
LOG_TAG = "[corrected-cloud]"
PROVENANCE = "tool_measured"


class CorrectedCloudError(RuntimeError):
    """A structural impossibility of the step — always with the exact reason."""


def K_on_grid(K: np.ndarray, g) -> np.ndarray:
    """The native-grid camera on Omega's record grid (F0's GridMap: crop, scale, pad;
    ``scale_x/y`` = native pixels per grid pixel): u_g = (u − crop_x) / scale_x + pad_left."""
    Kg = np.array(K, np.float64, copy=True)
    sx, sy = float(g.scale_x), float(g.scale_y)
    Kg[0, 0] /= sx
    Kg[1, 1] /= sy
    Kg[0, 2] = (Kg[0, 2] - float(g.crop_x)) / sx + float(g.pad_left)
    Kg[1, 2] = (Kg[1, 2] - float(g.crop_y)) / sy + float(g.pad_top)
    return Kg


# ── F6's depth: the input, verified ──────────────────────────────────────

def load_f6(out: Path, inp) -> dict:
    """F6's report and its per-keyframe s_k, verified against the session: F6 must have
    run, on this camera epoch, and on this geometry epoch or one only NEW-CLOUD epochs
    followed (they move no camera). {"report", "path", "s_k" (per keyframe, NaN where F6
    did not sweep), "frames" (the keyframes with a depth map)}."""
    from correction.epoch import epoch_kind
    from precision.depth_sweep import DEPTH_DIRNAME, REPORT_NAME
    out = Path(out)
    rp = out / DEPTH_DIRNAME / REPORT_NAME
    if not rp.exists():
        raise CorrectedCloudError(f"{rp} is missing — the corrected cloud is built on F6's depth; "
                                  f"run F6 first (precision.depth_sweep, runner step f6_sweep)")
    rep = json.loads(rp.read_text())
    ge_now, ce_now = int(inp.epochs["geometry_epoch"]), int(inp.epochs["camera_epoch"])
    ge, ce = rep.get("geometry_epoch"), rep.get("camera_epoch")
    if ce is None or int(ce) != ce_now:
        raise CorrectedCloudError(f"{rp} was measured with camera epoch {ce}, the session camera is "
                                  f"epoch {ce_now} — re-run F6 on this camera")
    if ge is None or int(ge) > ge_now:
        raise CorrectedCloudError(f"{rp} carries geometry epoch {ge}, the session is at {ge_now} — "
                                  f"F6 did not measure this geometry; re-run F6")
    later = list(range(int(ge) + 1, ge_now + 1))
    kinds = {e: epoch_kind(out, e) for e in later}
    if any(k != "new_cloud" for k in kinds.values()):
        raise CorrectedCloudError(f"{rp} measured geometry epoch {ge}, the session is at {ge_now} and "
                                  f"the epochs in between are {kinds} — F6's depth belongs to poses "
                                  f"that moved since; re-run F6")
    per = rep.get("per_frame") or {}
    by_frame = rep.get("s_k_by_frame") or {str(f): r["s_k"] for f, r in per.items() if "s_k" in r}
    s_k = np.array([float(by_frame[str(f)]) if str(f) in by_frame else np.nan for f in inp.kf], np.float64)
    if not np.isfinite(s_k).any():
        raise CorrectedCloudError(f"{rp} carries no s_k — F6 did not measure Omega's scale")
    frames = [int(f) for f in inp.kf if str(f) in per]
    ddir = out / DEPTH_DIRNAME
    missing = [f for f in frames if not (ddir / f"frame_{f}.npz").exists()]
    if missing:
        raise CorrectedCloudError(f"{rp} lists keyframe(s) {missing[:8]} whose depth map is missing "
                                  f"from {ddir}")
    if not frames:
        raise CorrectedCloudError(f"{rp} swept no keyframe — there is no depth to build a cloud on")
    return {"report": rep, "path": rp, "s_k": s_k, "frames": frames, "dir": ddir}


def f6_frame(ddir: Path, f: int, wh) -> Dict[str, np.ndarray]:
    """One keyframe of F6: the depth that ENTERS (tier 0 / tier 1, 0 elsewhere) and the
    per-pixel evidence, on the undistorted native grid (verified)."""
    from precision.depth_sweep import SOURCE_PRIOR_FILL, SOURCE_SWEEP
    p = Path(ddir) / f"frame_{int(f)}.npz"
    with np.load(p) as z:
        d = np.asarray(z["depth"], np.float32)
        src = np.asarray(z["source"], np.uint8)
        rec = {k: np.asarray(z[k]) for k in ("n_consistent", "ncc", "residual_rel") if k in z.files}
    W, H = int(wh[0]), int(wh[1])
    if d.shape != (H, W) or src.shape != (H, W):
        raise CorrectedCloudError(f"{p} is {d.shape[1]}x{d.shape[0]}, the session camera's undistorted "
                                  f"native grid is {W}x{H} — F6's depth is not on the grid the "
                                  f"unprojection assumes")
    enter = ((src == SOURCE_SWEEP) | (src == SOURCE_PRIOR_FILL)) & np.isfinite(d) & (d > 0)
    return {"depth": np.where(enter, d, 0.0).astype(np.float32), "source": src, "enter": enter, **rec}


def record_on_native(arr: np.ndarray, cam, maps) -> np.ndarray:
    """An Omega record array (depth grid) carried onto F6's undistorted native grid —
    the lens (F0's maps) then the record grid (``native_to_grid``, nearest): the
    correspondence F6 uses for its prior. NaN where the record does not cover."""
    import cv2
    from precision.camera import native_to_grid
    from precision.depth_sweep import record_grid
    g = record_grid(cam, arr.shape)
    uv = native_to_grid(np.stack([maps[0], maps[1]], -1).astype(np.float64), g)
    return cv2.remap(np.asarray(arr, np.float32), uv[..., 0].astype(np.float32), uv[..., 1].astype(np.float32),
                     cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)


def to_record_grid(u_und: np.ndarray, v_und: np.ndarray, cam, maps, rec_hw) -> tuple:
    """F6's undistorted native pixel → the pixel of Omega's RECORD grid (the trace grid
    intrinsic.txt describes): the lens, then ``native_to_grid``, rounded and clipped."""
    from precision.camera import native_to_grid
    from precision.depth_sweep import record_grid
    g = record_grid(cam, tuple(rec_hw))
    u = np.asarray(u_und, np.int64)
    v = np.asarray(v_und, np.int64)
    src = np.stack([maps[0][v, u], maps[1][v, u]], -1).astype(np.float64)
    uv = native_to_grid(src, g)
    col = np.clip(np.rint(uv[:, 0]), 0, g.w - 1).astype(np.int64)
    row = np.clip(np.rint(uv[:, 1]), 0, g.h - 1).astype(np.int64)
    return row, col


def f6_frames(inp, f6: dict) -> Dict[int, dict]:
    """{frame: {depth (F6, tier 0/1), K (F5, undistorted native), T (c2w)}} — the frames
    the witness filter, the silhouette occlusion test and the trace normals are fed."""
    out: Dict[int, dict] = {}
    idx = {int(f): i for i, f in enumerate(inp.kf)}
    for f in f6["frames"]:
        fr = f6_frame(f6["dir"], f, inp.wh)
        out[int(f)] = {"depth": fr["depth"], "K": np.asarray(inp.K, np.float64),
                       "T": np.linalg.inv(inp.kf_w2c[idx[int(f)]])}
    return out


def trace_normals_from(frames: Dict[int, dict]) -> Callable:
    """Normals for the consolidation from the corrected depth maps (each point from
    its own keyframe's depth gradient, in world) — the fast, camera-oriented path."""
    from precision.depth_sweep import normals_from_depth
    cache: Dict[int, np.ndarray] = {}

    def fn(xyz, fg, pr, pc):
        n = np.full((len(fg), 3), np.nan)
        for f in np.unique(fg):
            fr = frames.get(int(f))
            if fr is None:
                continue
            if int(f) not in cache:
                cache[int(f)] = normals_from_depth(fr["depth"], fr["K"])
            m = fg == f
            H, W = fr["depth"].shape
            nc = cache[int(f)][np.clip(pr[m], 0, H - 1), np.clip(pc[m], 0, W - 1)]
            n[m] = nc @ np.asarray(fr["T"])[:3, :3].T
        bad = ~np.isfinite(n).all(1)
        if bad.mean() > 0.5:                       # the trace does not cover the cloud
            return None
        n[bad] = np.array([0.0, 1.0, 0.0])
        return n
    return fn


def _rgb_undistorted(frames_dir: Path, frame: int, maps) -> np.ndarray:
    import cv2
    from intake.content import frame_file
    bgr = cv2.imread(str(frame_file(frames_dir, frame)), cv2.IMREAD_COLOR)
    if bgr is None:
        raise CorrectedCloudError(f"frame image {frame_file(frames_dir, frame)} is unreadable")
    und = cv2.remap(bgr, maps[0], maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return und[..., ::-1]


def write_chunks(inp, f6: dict, raw_cfg: dict, tmp: Path, log: Callable) -> List[dict]:
    """F6's depth (tier 0 / tier 1) → one PLY + origins per Omega chunk (the cleaner's
    input), the reconstruction's confidence gate per chunk. pixel_row/col = F6's pixel."""
    from precision.depth_sweep import SOURCE_SWEEP
    from precision.epoch0_cloud import SKY_CONF, _write_ply_xyzrgb, conf_threshold
    simple = (raw_cfg.get("reconstruction") or {}).get("simple") or {}
    if "conf_percentile" not in simple or "conf_min_norm" not in simple:
        raise CorrectedCloudError("reconstruction.simple.conf_percentile / conf_min_norm are missing — "
                                  "the corrected cloud runs the reconstruction's own gate")
    pct, floor = simple["conf_percentile"], float(simple["conf_min_norm"] or 0.0)
    have = set(f6["frames"])
    by_chunk: Dict[int, List[int]] = {}
    for i, f in enumerate(inp.kf):
        with np.load(inp.records_dir / f"frame_{f}.npz") as z:
            by_chunk.setdefault(int(z["chunk"]) if "chunk" in z.files else 0, []).append(i)
    tmp.mkdir(parents=True, exist_ok=True)
    K = np.asarray(inp.K, np.float64)
    W, H = int(inp.wh[0]), int(inp.wh[1])
    v_all, u_all = np.mgrid[0:H, 0:W]
    chunks = []
    n_no_f6 = 0
    for k, members in sorted(by_chunk.items()):
        confs = []
        for i in members:
            with np.load(inp.records_dir / f"frame_{inp.kf[i]}.npz") as z:
                confs.append(np.asarray(z["conf"], np.float32).reshape(-1))
        thr = conf_threshold(np.concatenate(confs), pct, floor)
        xyz_l, rgb_l, fg_l, pr_l, pc_l, cf_l = [], [], [], [], [], []
        n_tier = {"tier0": 0, "tier1": 0}
        for i in members:
            f = int(inp.kf[i])
            if f not in have:
                n_no_f6 += 1
                continue
            fr = f6_frame(f6["dir"], f, inp.wh)
            with np.load(inp.records_dir / f"frame_{f}.npz") as z:
                c = record_on_native(np.asarray(z["conf"], np.float32), inp.cam, inp.maps)
            keep = fr["enter"] & np.isfinite(c) & (c > SKY_CONF) & (c >= thr)
            rr, cc = np.nonzero(keep)
            if not rr.size:
                continue
            zc = fr["depth"][rr, cc].astype(np.float64)
            Xc = np.stack([(u_all[rr, cc] - K[0, 2]) / K[0, 0] * zc, (v_all[rr, cc] - K[1, 2]) / K[1, 1] * zc,
                           zc], 1)
            c2w = np.linalg.inv(inp.kf_w2c[i])
            Xw = Xc @ c2w[:3, :3].T + c2w[:3, 3]
            rgb = _rgb_undistorted(inp.frames_dir, f, inp.maps)
            xyz_l.append(Xw.astype(np.float32)); rgb_l.append(rgb[rr, cc].astype(np.uint8))
            fg_l.append(np.full(rr.size, f, np.int32)); pr_l.append(rr.astype(np.int32))
            pc_l.append(cc.astype(np.int32)); cf_l.append(c[rr, cc])
            t0 = int((fr["source"][rr, cc] == SOURCE_SWEEP).sum())
            n_tier["tier0"] += t0
            n_tier["tier1"] += int(rr.size) - t0
        if not xyz_l:
            log(f"{LOG_TAG} chunk {k}: nothing of F6 above the confidence gate")
            continue
        xyz = np.concatenate(xyz_l)
        _write_ply_xyzrgb(tmp / f"chunk_{k:03d}.ply", xyz, np.concatenate(rgb_l))
        np.savez(tmp / f"chunk_{k:03d}_origins.npz", frame_global=np.concatenate(fg_l),
                 pixel_row=np.concatenate(pr_l).astype(np.int16), pixel_col=np.concatenate(pc_l).astype(np.int16),
                 confidence=np.concatenate(cf_l).astype(np.float32))
        chunks.append({"chunk": k, "n_keyframes": len(members), "conf_threshold": float(thr),
                       "raw_points": int(len(xyz)), **n_tier})
        log(f"{LOG_TAG} chunk {k}: {len(members)} keyframes, conf threshold {thr:.3f} → {len(xyz):,} pts "
            f"(tier 0 {n_tier['tier0']:,}, tier 1 {n_tier['tier1']:,})")
    if n_no_f6:
        log(f"{LOG_TAG} {n_no_f6} keyframe(s) without an F6 depth map (not swept) contribute nothing")
    if not chunks:
        raise CorrectedCloudError("no F6 pixel survived the confidence gate in any chunk")
    return chunks


def clean(raw_cfg: dict, tmp: Path, cleaned: Path, log: Callable) -> None:
    """The cloud stage's own cleaner (voxel + SOR, postprocessing keys), as a subprocess."""
    from precision.epoch0_cloud import clean_cmd
    cmd = clean_cmd(raw_cfg, tmp, cleaned)
    r = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[1]), capture_output=True, text=True)
    for ln in r.stdout.splitlines():
        if "✅" in ln or "❌" in ln:
            log(f"{LOG_TAG} {ln.strip()}")
    if r.returncode != 0 or not cleaned.exists():
        raise CorrectedCloudError(f"the cleaner failed (rc {r.returncode}): {r.stderr[-1500:]}")


def witness(tmp: Path, cleaned: Path, frames: Dict[int, dict], raw_cfg: dict, log: Callable) -> dict:
    """The witness filter on the cleaned cloud, fed the corrected frames; the statuses
    in witness.drop_statuses leave; mv_votes + status are written into the PLY."""
    from correction.session import read_ply
    from reconstruction.loops.config import load_loops_config
    from reconstruction.witness.run import witness_fields
    from reconstruction.witness.status import STATUS_CODES
    wcfg = load_loops_config(raw_cfg).witness
    header, data = read_ply(cleaned)
    names = data.dtype.names
    xyz = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float64)
    fields = witness_fields(xyz, np.asarray(data["frame_global"], np.int64), np.asarray(data["pixel_row"], np.int64),
                            np.asarray(data["pixel_col"], np.int64), frames, wcfg, device="cuda")
    st = fields["status"]
    counts = {k: int((st == v).sum()) for k, v in STATUS_CODES.items()}
    drop = np.isin(st, [STATUS_CODES[s] for s in wcfg.drop_statuses])
    keep = ~drop
    base = [(n, data.dtype[n].str) for n in names if n not in ("mv_votes", "status")]
    out = np.empty(int(keep.sum()), dtype=base + [("mv_votes", "u1"), ("status", "u1")])
    for n, _ in base:
        out[n] = data[n][keep]
    out["mv_votes"] = fields["mv_votes"][keep].astype(np.uint8)
    out["status"] = st[keep].astype(np.uint8)
    ply_t = {"<f4": "float", "<f8": "double", "|u1": "uchar", "<i4": "int", "<i2": "short", "<u2": "ushort",
             "|i1": "char", "<u4": "uint"}
    hdr = ["ply", "format binary_little_endian 1.0",
           "comment STAC corrected cloud (precision/corrected_cloud.py) - witnesses on F6's frames",
           f"element vertex {len(out)}"]
    hdr += [f"property {ply_t[out.dtype[n].str]} {n}" for n in out.dtype.names]
    hdr.append("end_header")
    for name in ("cleaned_cloud.ply", "cleaned_cloud_raw.ply"):
        with open(tmp / name, "wb") as fh:
            fh.write(("\n".join(hdr) + "\n").encode("ascii"))
            fh.write(out.tobytes())
    log(f"{LOG_TAG} witnesses ({wcfg.n_neighbors} neighbours, τ {wcfg.tau_rel}): {counts} → "
        f"{int(drop.sum()):,} leave ({list(wcfg.drop_statuses)}), {int(keep.sum()):,} stay")
    return {"status_counts": counts, "dropped": int(drop.sum()), "kept": int(keep.sum()),
            "drop_statuses": list(wcfg.drop_statuses)}


def consolidate(tmp: Path, frames: Dict[int, dict], raw_cfg: dict, log: Callable) -> dict:
    """The scene consolidation (postprocessing.scene_consolidate) in place, normals from
    the corrected depth maps."""
    from reconstruction.loops.config import load_loops_config
    from reconstruction.surface_fit.consolidate import scene_consolidate
    sc = (raw_cfg.get("postprocessing") or {}).get("scene_consolidate") or {}
    kw = {k: sc[k] for k in ("radius_m", "min_radius_m", "max_radius_m", "iterations", "normal_gate", "k") if k in sc}
    stats = scene_consolidate(tmp, excluded_statuses=load_loops_config(raw_cfg).witness.mls_excluded_statuses,
                              normals_fn=trace_normals_from(frames), **kw)
    log(f"{LOG_TAG} consolidated: normals {stats.get('normals')}, radius {stats.get('radius_m')}, "
        f"mean move {stats.get('mean_move_mm')} mm")
    return {k: v for k, v in stats.items() if not isinstance(v, (np.ndarray, list))}


def silhouettes(session_dir: Path, tmp: Path, frames: Dict[int, dict], inp, pcfg, raw_cfg: dict,
                log: Callable) -> dict:
    """The silhouette flyer filter (precision/silhouette_filter.py) on both PLYs."""
    from precision import silhouette_filter as SF
    params = SF.params_from(pcfg, raw_cfg)
    return SF.run_filter(Path(session_dir) / "output", tmp, frames, inp.kf, inp.kf_w2c, inp.K, inp.wh,
                         inp.maps, params, log=log)


def provenance_columns(tmp: Path, inp, f6: dict, log: Callable) -> Dict[str, np.ndarray]:
    """The v2 columns of the final rows from F6's own maps (tier, n_consistent, ncc,
    residual_rel, the exact undistorted pixel) — and, in BOTH PLYs, pixel_row/col moved
    from F6's pixel to Omega's record grid (the trace grid every later stage indexes)."""
    from correction.session import read_ply, write_ply
    import precision.provenance as PV
    header, data = read_ply(tmp / "cleaned_cloud.ply")
    header_r, raw = read_ply(tmp / "cleaned_cloud_raw.ply")
    if len(raw) != len(data) or not np.array_equal(raw["frame_global"], data["frame_global"]):
        raise CorrectedCloudError("cleaned_cloud.ply and cleaned_cloud_raw.ply do not hold the same rows")
    n = len(data)
    fg = np.asarray(data["frame_global"], np.int64)
    v = np.asarray(data["pixel_row"], np.int64)
    u = np.asarray(data["pixel_col"], np.int64)
    cols = {"pixel_u_und": u.copy(), "pixel_v_und": v.copy(),
            "source": np.zeros(n, PV.V2_DTYPES["source"]), "n_consistent": np.zeros(n, PV.V2_DTYPES["n_consistent"]),
            "ncc": np.full(n, np.nan, PV.V2_DTYPES["ncc"]),
            "residual_rel": np.full(n, np.nan, PV.V2_DTYPES["residual_rel"])}
    order = np.argsort(fg, kind="stable")
    cuts = np.flatnonzero(np.diff(fg[order])) + 1
    for seg in np.split(order, cuts):
        if not len(seg):
            continue
        fr = f6_frame(f6["dir"], int(fg[seg[0]]), inp.wh)
        r, c = v[seg], u[seg]
        if not fr["enter"][r, c].all():
            raise CorrectedCloudError(f"keyframe {int(fg[seg[0]])}: {int((~fr['enter'][r, c]).sum())} point(s) "
                                      f"sit on F6 pixels that are neither tier 0 nor tier 1")
        cols["source"][seg] = fr["source"][r, c]
        for k in ("n_consistent", "ncc", "residual_rel"):
            if k in fr:
                cols[k][seg] = fr[k][r, c]
    with np.load(inp.records_dir / f"frame_{inp.kf[0]}.npz") as z:
        rec_hw = tuple(int(x) for x in z["depth"].shape[:2])
    row, col = to_record_grid(u, v, inp.cam, inp.maps, rec_hw)
    for hdr, arr, name in ((header, data, "cleaned_cloud.ply"), (header_r, raw, "cleaned_cloud_raw.ply")):
        arr["pixel_row"] = row.astype(arr.dtype["pixel_row"])
        arr["pixel_col"] = col.astype(arr.dtype["pixel_col"])
        write_ply(tmp / name, hdr, arr)
    cols["camera_epoch"] = np.full(n, int(inp.epochs["camera_epoch"]), PV.V2_DTYPES["camera_epoch"])
    t = cols["source"]
    log(f"{LOG_TAG} provenance: {n:,} points — tier 0 {int((t == 0).sum()):,}, tier 1 {int((t == 1).sum()):,}; "
        f"pixel_row/col on Omega's record grid {rec_hw[1]}x{rec_hw[0]}, F6's pixel in origins.npz")
    return cols


def publish(session_dir: Path, tmp: Path, report: dict, log: Callable,
            columns: Optional[Dict[str, np.ndarray]] = None) -> dict:
    """The new-cloud epoch: the same transaction F7 runs (PLY + raw, origins v2, epoch
    record, octree, atomic swap, ledger). ``columns``: v2 columns measured by the caller
    (row-aligned with the PLY) — they win over the PLY's fields and the defaults."""
    from correction import ledger
    from correction.apply import TX_PREFIX, assert_no_interrupted_swap, next_epoch, swap_transaction
    from correction.epoch import EPOCH_FILE, current_epoch, make_epoch_record
    from correction.session import read_ply, write_ply
    import precision.provenance as PV
    out = Path(session_dir) / "output"
    assert_no_interrupted_swap(out)
    header, data = read_ply(tmp / "cleaned_cloud.ply")
    n = len(data)
    epoch_from, epoch_to = current_epoch(out), next_epoch(out)
    tx = out / f"{TX_PREFIX}{epoch_to}"
    if tx.exists():
        shutil.rmtree(tx)
    tx.mkdir(parents=True)
    arts: List[dict] = []

    def art(rel):
        arts.append({"rel": rel, "existed_before": (out / rel).exists()})
    write_ply(tx / "cleaned_cloud.ply", header, data); art("cleaned_cloud.ply")
    raw = tmp / "cleaned_cloud_raw.ply"
    if raw.exists():
        shutil.copy2(raw, tx / "cleaned_cloud_raw.ply")
    else:
        write_ply(tx / "cleaned_cloud_raw.ply", header, data)
    art("cleaned_cloud_raw.ply")
    cid = ledger.new_correction_id()
    names = data.dtype.names
    columns = columns or {}
    for k, v in columns.items():
        if len(v) != n:
            raise CorrectedCloudError(f"provenance column {k} has {len(v)} rows, the cloud {n}")
    o = {}
    for k in PV.V2_FIELDS:
        if k in columns:
            o[k] = np.asarray(columns[k])
        elif k in names:
            o[k] = np.asarray(data[k])
        elif k == "geometry_epoch":
            o[k] = np.full(n, epoch_to, PV.V2_DTYPES[k])
        else:
            o[k] = np.zeros(n, PV.V2_DTYPES[k])
    PV.write_origins(tx / PV.ORIGINS_NAME, o, {"stage": "corrected_cloud", "correction_id": cid, "epoch": epoch_to,
                                                "source_of_depth": report.get("source_of_depth", "omega x s_k"),
                                                "fields_measured": sorted(columns),
                                                "fields_from_ply": [k for k in PV.V2_FIELDS
                                                                    if k in names and k not in columns]})
    art(PV.ORIGINS_NAME)
    rep = dict(report, epoch_to=epoch_to, epoch_from=epoch_from, correction_id=cid, n_points=n)
    (tx / CLOUD_REPORT).write_text(json.dumps(rep, indent=1, default=float)); art(CLOUD_REPORT)
    if (out / "segmentation_result.json").exists():
        # the cloud stage projects the masks on this epoch (its own freshness test)
        (tx / "segmentation_result.json").write_text(json.dumps(
            {"instances": [], "pending": "projected by the cloud stage on this epoch", "geometry_epoch": epoch_to}))
        art("segmentation_result.json")
    for sib, dt in (("classification.npy", np.uint8), ("out_of_place.npy", bool)):
        if (out / sib).exists():
            np.save(tx / sib, np.zeros(n, dt)); art(sib)
    (tx / EPOCH_FILE).write_text(json.dumps(make_epoch_record(epoch_to, cid, epoch_from, kind="new_cloud"), indent=1))
    art(EPOCH_FILE)
    from potree_converter import convert_ply_to_potree
    ok = convert_ply_to_potree(Path(session_dir), force=True, ply_override=tx / "cleaned_cloud.ply",
                               potree_dir_override=tx / "potree")
    if not ok or not (tx / "potree" / "metadata.json").exists():
        shutil.rmtree(tx)
        raise CorrectedCloudError("the Potree build failed inside the transaction — nothing published")
    art("potree")
    swap_transaction(out, {"tx_dir": str(tx), "epoch_from": epoch_from, "epoch_to": epoch_to, "artifacts": arts}, log=log)
    ledger.record_run(out, correction_id=cid, epoch_from=epoch_from, epoch_to=epoch_to, kind="fuse",
                      operator="auto", instance_ids=[], visits=[], observability=[], anchors=[],
                      diagnosis=[{"stage": "corrected_cloud"}], gates=[], overrides={}, verdict="applied",
                      report_path=CLOUD_REPORT)
    log(f"{LOG_TAG} published epoch {epoch_to} ({n:,} points; epoch {epoch_from} kept, selectable)")
    return rep


def run_corrected_cloud(session_dir: Path, pcfg, log: Callable = print,
                        progress: Optional[Callable[[float, str], None]] = None) -> dict:
    """The step: F6's depth → chunks → clean → witnesses → silhouettes → consolidate →
    provenance → publish."""
    from config import cfg as raw_cfg
    from precision import depth_sweep as DS
    t0 = time.time()
    session_dir = Path(session_dir)
    out = session_dir / "output"
    ccfg = pcfg.cloud

    def _p(pct, msg):
        log(f"{LOG_TAG} {msg}")
        if progress:
            progress(pct, msg)
    _p(2, "loading F5's camera and poses, F6's depth")
    inp = DS.load_inputs(session_dir, pcfg)
    f6 = load_f6(out, inp)
    s_k = f6["s_k"]
    ok = np.isfinite(s_k)
    _p(6, f"F6 ({f6['path']}): {len(f6['frames'])} of {len(inp.kf)} keyframes swept, tier-1 rule "
          f"{f6['report'].get('tier1_rule', 'confirmation (a report older than the rule switch)')}, "
          f"s_k median {np.median(s_k[ok]):.4f}, range {s_k[ok].min():.4f}–{s_k[ok].max():.4f}")
    tmp = out / TX_TMP
    if tmp.exists():
        shutil.rmtree(tmp)
    chunks_dir = tmp / "chunks"
    rep6 = f6["report"]
    report = {"version": 2, "provenance": PROVENANCE, "stage": "corrected_cloud", **inp.epochs,
              "source_of_depth": "F6 depth_native: tier 0 (plane sweep) + tier 1 (Omega x s_k)",
              "f6": {"report": str(f6["path"].relative_to(out)), "geometry_epoch": rep6.get("geometry_epoch"),
                     "camera_epoch": rep6.get("camera_epoch"), "tier1_rule": rep6.get("tier1_rule"),
                     "contradiction": rep6.get("contradiction"), "percent": rep6.get("percent"),
                     "n_swept": len(f6["frames"])},
              "camera": [float(v) for v in inp.cam.params] if hasattr(inp.cam, "params") else None,
              "grid": {"width": int(inp.wh[0]), "height": int(inp.wh[1]), "space": "undistorted native (F0 maps)"},
              "per_frame": {str(f): {"s_k": float(s_k[i])} for i, f in enumerate(inp.kf) if np.isfinite(s_k[i])},
              "s_k": {"median": float(np.median(s_k[ok])), "min": float(s_k[ok].min()), "max": float(s_k[ok].max())},
              "recipe": ["F6 depth: tier 0 + tier 1 (maximum coverage, contradiction vote)",
                         "confidence gate (conf_percentile + conf_min_norm per Omega chunk)",
                         "voxel + SOR (postprocessing, reconstruction/gpu_cloud_clean)"]}
    try:
        _p(12, "F6's depth with F5's camera and poses → chunks")
        report["chunks"] = write_chunks(inp, f6, raw_cfg, chunks_dir, log)
        report["raw_points"] = int(sum(c["raw_points"] for c in report["chunks"]))
        _p(35, "the cloud stage's cleaner (voxel + SOR)")
        cleaned = tmp / "cleaned_cloud.ply"
        clean(raw_cfg, chunks_dir, cleaned, log)
        shutil.rmtree(chunks_dir, ignore_errors=True)
        frames = f6_frames(inp, f6)
        if ccfg.witness_filter:
            _p(50, "witnesses on F6's frames")
            report["witness"] = witness(tmp, cleaned, frames, raw_cfg, log)
            report["recipe"].append("witness filter (F6's frames; witness.drop_statuses leave)")
        else:
            shutil.copy2(cleaned, tmp / "cleaned_cloud_raw.ply")
        if ccfg.silhouette_filter:
            _p(62, "silhouettes: SAM3's masks from the other keyframes")
            report["silhouette"] = silhouettes(session_dir, tmp, frames, inp, pcfg, raw_cfg, log)
            report["recipe"].append("silhouette filter (precision/silhouette_filter.py)")
        if ccfg.consolidate:
            _p(72, "scene consolidation (normals from F6's depth)")
            report["consolidate"] = consolidate(tmp, frames, raw_cfg, log)
            report["recipe"].append("scene_consolidate (trace normals from F6's depth)")
        _p(82, "provenance from F6's maps")
        cols = provenance_columns(tmp, inp, f6, log)
        t = cols["source"]
        report["tiers_in_cloud"] = {"tier0": int((t == 0).sum()), "tier1": int((t == 1).sum())}
        _p(85, "publishing the epoch (octree, atomic swap)")
        rep = publish(session_dir, tmp, report, log, columns=cols)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    rep["seconds"] = round(time.time() - t0, 1)
    (out / CLOUD_REPORT).write_text(json.dumps(rep, indent=1, default=float))
    _p(100, f"corrected cloud is epoch {rep['epoch_to']} ({rep['n_points']:,} pts, {rep['seconds']} s)")
    return rep


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--session", required=True)
    args = ap.parse_args(argv)
    from precision.config import load_precision_config
    run_corrected_cloud(Path(args.session), load_precision_config())
    return 0


if __name__ == "__main__":
    sys.exit(main())
