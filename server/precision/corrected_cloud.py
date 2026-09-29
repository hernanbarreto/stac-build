"""F7-C — the CORRECTED OMEGA CLOUD as the reconstruction's product.

USER 2026-09-29, on pccr, after the witness fusion (F7) delivered 9 M / 0.2 M-point
clouds judged worse than Omega's own: the complete Omega cloud re-projected with the
core's corrections — *"prácticamente corrigió la duplicidad del desk … y el piso"*,
*"hasta ahora la mejor nube"* — and *"que quede todo en el pipeline, nada a mano"*.

    point = c2w_F5 · ( s_k · z_omega(u, v) · K_F5⁻¹ [u, v, 1] )

- z_omega: Omega's own per-keyframe depth (omega_run/results_output), the most complete
  measurement of the scene this capture allows;
- s_k: the scale of that depth measured against F5's triangulated landmarks, pooled
  along the walk over the gauge's knot spacing (the same measurement F6 makes);
- K_F5, c2w_F5: the session camera and poses after F5 (R1 on pccr: fx 364 → 392, the
  loop of the walk closed from 77 cm to ~15 cm at the objects);
- the confidence gate of the reconstruction (conf_percentile + conf_min_norm, the
  fork's rule, per Omega chunk), then THE CLOUD STAGE'S OWN RECIPE: voxel + SOR
  (reconstruction/gpu_cloud_clean), the witness filter fed the corrected frames
  (single_witness leaves), the scene consolidation with normals from the corrected
  depth maps — and the octree, published as a NEW-CLOUD epoch through the same
  transaction as F7 (PLY + raw, origins v2, epoch record, atomic swap, ledger).

Measured on pccr (2026-09-29): epoch 0 (Omega) 28.5 M pts, floor-layer thickness per
1 m cell 20.3 cm median → corrected 25.5 M / 12.9 cm → + witness/MLS 21.8 M / 11.5 cm.

Runner step ``f7_cloud`` (after F5). ``precision.cloud.source`` selects this
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


def corrected_frames(inp, s_k: np.ndarray) -> Dict[int, dict]:
    """{frame: {depth (Omega × s_k), K (on the record grid), T (c2w)}} — the frames
    the witness filter and the trace normals are fed."""
    from precision.depth_sweep import record_grid
    out: Dict[int, dict] = {}
    for i, f in enumerate(inp.kf):
        with np.load(inp.records_dir / f"frame_{f}.npz") as z:
            d = np.asarray(z["depth"], np.float64)
        d = np.where(np.isfinite(d) & (d > 0), d * float(s_k[i]), 0.0).astype(np.float32)
        Kg = K_on_grid(inp.K, record_grid(inp.cam, d.shape))
        out[int(f)] = {"depth": d, "K": Kg, "T": np.linalg.inv(inp.kf_w2c[i])}
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


def write_chunks(inp, s_k: np.ndarray, raw_cfg: dict, tmp: Path, log: Callable) -> List[dict]:
    """Omega's records → one PLY + origins per Omega chunk (the cleaner's input), the
    reconstruction's confidence gate per chunk."""
    import cv2
    from precision.epoch0_cloud import SKY_CONF, _write_ply_xyzrgb, conf_threshold
    from precision.depth_sweep import record_grid
    simple = (raw_cfg.get("reconstruction") or {}).get("simple") or {}
    if "conf_percentile" not in simple or "conf_min_norm" not in simple:
        raise CorrectedCloudError("reconstruction.simple.conf_percentile / conf_min_norm are missing — "
                                  "the corrected cloud runs the reconstruction's own gate")
    pct, floor = simple["conf_percentile"], float(simple["conf_min_norm"] or 0.0)
    by_chunk: Dict[int, List[int]] = {}
    for i, f in enumerate(inp.kf):
        with np.load(inp.records_dir / f"frame_{f}.npz") as z:
            by_chunk.setdefault(int(z["chunk"]) if "chunk" in z.files else 0, []).append(i)
    tmp.mkdir(parents=True, exist_ok=True)
    chunks = []
    for k, members in sorted(by_chunk.items()):
        confs = []
        for i in members:
            with np.load(inp.records_dir / f"frame_{inp.kf[i]}.npz") as z:
                confs.append(np.asarray(z["conf"], np.float32).reshape(-1))
        thr = conf_threshold(np.concatenate(confs), pct, floor)
        xyz_l, rgb_l, fg_l, pr_l, pc_l, cf_l = [], [], [], [], [], []
        for i in members:
            f = inp.kf[i]
            with np.load(inp.records_dir / f"frame_{f}.npz") as z:
                d = np.asarray(z["depth"], np.float64)
                c = np.asarray(z["conf"], np.float32)
            keep = np.isfinite(d) & (d > 0) & (c > SKY_CONF) & (c >= thr)
            rr, cc = np.nonzero(keep)
            if not rr.size:
                continue
            Kg = K_on_grid(inp.K, record_grid(inp.cam, d.shape))
            zc = d[rr, cc] * float(s_k[i])
            Xc = np.stack([(cc - Kg[0, 2]) / Kg[0, 0] * zc, (rr - Kg[1, 2]) / Kg[1, 1] * zc, zc], 1)
            c2w = np.linalg.inv(inp.kf_w2c[i])
            Xw = Xc @ c2w[:3, :3].T + c2w[:3, 3]
            img = cv2.imread(str(inp.frames_dir / f"{f:06d}.jpg"), cv2.IMREAD_COLOR)
            if img is None:
                raise CorrectedCloudError(f"frame image {inp.frames_dir / f'{f:06d}.jpg'} is missing")
            if img.shape[:2] != d.shape:
                img = cv2.resize(img, (d.shape[1], d.shape[0]), interpolation=cv2.INTER_AREA)
            xyz_l.append(Xw.astype(np.float32)); rgb_l.append(img[rr, cc][:, ::-1].astype(np.uint8))
            fg_l.append(np.full(rr.size, f, np.int32)); pr_l.append(rr.astype(np.int32))
            pc_l.append(cc.astype(np.int32)); cf_l.append(c[rr, cc])
        if not xyz_l:
            log(f"{LOG_TAG} chunk {k}: nothing above the confidence gate")
            continue
        xyz = np.concatenate(xyz_l)
        _write_ply_xyzrgb(tmp / f"chunk_{k:03d}.ply", xyz, np.concatenate(rgb_l))
        np.savez(tmp / f"chunk_{k:03d}_origins.npz", frame_global=np.concatenate(fg_l),
                 pixel_row=np.concatenate(pr_l).astype(np.int16), pixel_col=np.concatenate(pc_l).astype(np.int16),
                 confidence=np.concatenate(cf_l).astype(np.float32))
        chunks.append({"chunk": k, "n_keyframes": len(members), "conf_threshold": float(thr), "raw_points": int(len(xyz))})
        log(f"{LOG_TAG} chunk {k}: {len(members)} keyframes, conf threshold {thr:.3f} → {len(xyz):,} pts")
    if not chunks:
        raise CorrectedCloudError("no point survived the confidence gate in any chunk")
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
           "comment STAC corrected Omega cloud (precision/corrected_cloud.py) - witnesses on corrected frames",
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


def publish(session_dir: Path, tmp: Path, report: dict, log: Callable) -> dict:
    """The new-cloud epoch: the same transaction F7 runs (PLY + raw, origins v2, epoch
    record, octree, atomic swap, ledger)."""
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
    source = getattr(PV, "SOURCE_PRIOR_FILL", 0)
    o = {}
    for k in PV.V2_FIELDS:
        if k in names:
            o[k] = np.asarray(data[k])
        elif k == "source":
            o[k] = np.full(n, source, PV.V2_DTYPES[k])
        elif k == "geometry_epoch":
            o[k] = np.full(n, epoch_to, PV.V2_DTYPES[k])
        else:
            o[k] = np.zeros(n, PV.V2_DTYPES[k])
    PV.write_origins(tx / PV.ORIGINS_NAME, o, {"stage": "corrected_cloud", "correction_id": cid, "epoch": epoch_to,
                                                "source_of_depth": "omega x s_k",
                                                "fields_from_ply": [k for k in PV.V2_FIELDS if k in names]})
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
    """The step: s_k → chunks → clean → witnesses → consolidate → publish."""
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
    _p(2, "loading F5's camera, poses and tracks")
    inp = DS.load_inputs(session_dir, pcfg)
    n_kf = len(inp.kf)
    lm = DS.landmark_samples(session_dir, pcfg, inp)
    chain = DS.keyframe_chainage(session_dir, inp.kf)
    s_k, s_n = DS.keyframe_scales(lm, n_kf, int(pcfg.depth.min_scale_samples), chainage=chain,
                                  window_m=float(pcfg.gauge.knot_walk_m))
    if not np.isfinite(s_k).all():
        raise CorrectedCloudError("s_k could not be measured on every keyframe — F5's landmarks do not cover the walk")
    _p(8, f"s_k over {n_kf} keyframes: median {np.median(s_k):.4f}, range {s_k.min():.4f}–{s_k.max():.4f}"
          f"{'' if chain is not None else ' (no walk: pooled by keyframe index)'}")
    tmp = out / TX_TMP
    if tmp.exists():
        shutil.rmtree(tmp)
    chunks_dir = tmp / "chunks"
    report = {"version": 1, "provenance": PROVENANCE, "stage": "corrected_cloud", **inp.epochs,
              "camera": [float(v) for v in inp.cam.params] if hasattr(inp.cam, "params") else None,
              "per_frame": {str(f): {"s_k": float(s_k[i]), "n_samples": int(s_n[i])} for i, f in enumerate(inp.kf)},
              "s_k": {"median": float(np.median(s_k)), "min": float(s_k.min()), "max": float(s_k.max())},
              "recipe": ["confidence gate (conf_percentile + conf_min_norm per Omega chunk)",
                         "voxel + SOR (postprocessing, reconstruction/gpu_cloud_clean)"]}
    try:
        _p(12, "Omega's depth × s_k with F5's camera and poses → chunks")
        report["chunks"] = write_chunks(inp, s_k, raw_cfg, chunks_dir, log)
        report["raw_points"] = int(sum(c["raw_points"] for c in report["chunks"]))
        _p(35, "the cloud stage's cleaner (voxel + SOR)")
        cleaned = tmp / "cleaned_cloud.ply"
        clean(raw_cfg, chunks_dir, cleaned, log)
        shutil.rmtree(chunks_dir, ignore_errors=True)
        frames = corrected_frames(inp, s_k) if (ccfg.witness_filter or ccfg.consolidate) else {}
        if ccfg.witness_filter:
            _p(55, "witnesses on the corrected frames")
            report["witness"] = witness(tmp, cleaned, frames, raw_cfg, log)
            report["recipe"].append("witness filter (corrected frames; witness.drop_statuses leave)")
        else:
            shutil.copy2(cleaned, tmp / "cleaned_cloud_raw.ply")
        if ccfg.consolidate:
            _p(70, "scene consolidation (normals from the corrected depth maps)")
            report["consolidate"] = consolidate(tmp, frames, raw_cfg, log)
            report["recipe"].append("scene_consolidate (trace normals from the corrected depth)")
        _p(85, "publishing the epoch (octree, atomic swap)")
        rep = publish(session_dir, tmp, report, log)
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
