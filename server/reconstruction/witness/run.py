"""The witness stage on a session (§6.1–6.3): mv_votes, mask_votes,
mask_conflicts and status for every point of cleaned_cloud.ply (and the raw
cloud), written as scalar fields. Nothing is deleted. As an epoch (§8) the
fields land through the correction package's transaction; the merge-time
call inside gpu_cloud_clean (before voxel/SOR) uses ``witness_fields`` on
arrays directly.

    python -m reconstruction.witness.run --session <dir> [--no-epoch]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from reconstruction.witness.fields import WITNESS_FIELDS, add_fields, write_fields
from reconstruction.witness.frames import load_session_frames
from reconstruction.witness.mask_votes import compute_mask_votes, load_mask_store
from reconstruction.witness.mv_votes import mv_votes_per_frame, point_mv_votes
from reconstruction.witness.status import assign_status, status_counts

REPORT_JSON = "witness_report.json"


def cpu_torch_threads(n: int, device=None) -> None:
    """Bound torch's CPU thread pool when the witnesses run on the CPU."""
    import torch
    on_cpu = (device is not None and str(device).startswith("cpu")) or not torch.cuda.is_available()
    if on_cpu and torch.get_num_threads() > int(n):
        torch.set_num_threads(int(n))


def dynamic_instance_ids(output_dir) -> List[int]:
    """Instances Qwen classified ``dynamic`` (§4.4, loop_semantics.json)."""
    p = Path(output_dir) / "loop_semantics.json"
    if not p.exists():
        return []
    classes = json.loads(p.read_text()).get("classes") or {}
    return [int(k) for k, v in classes.items() if v.get("class") == "dynamic"]


def witness_fields(xyz: np.ndarray, fg: np.ndarray, pr: np.ndarray, pc: np.ndarray,
                   frames: Dict[int, dict], wcfg, instances: Optional[List[dict]] = None,
                   store=None, dynamic_ids=(), device=None) -> Dict[str, np.ndarray]:
    """All four fields for an arbitrary cloud with provenance. Without a
    segmentation (instances/store None) the mask fields are zero and the
    status is provisional (multi-view only)."""
    cpu_torch_threads(wcfg.cpu_threads, device)
    per_frame = mv_votes_per_frame(frames, wcfg.n_neighbors, wcfg.tau_rel, device=device)
    mv, observed = point_mv_votes(per_frame, fg, pr, pc)
    n = len(fg)
    if instances and store is not None:
        mvt, mct = compute_mask_votes(xyz, instances, frames, store, wcfg.mask_erosion_px,
                                      wcfg.occlusion_tol_rel, device=device)
    else:
        mvt, mct = np.zeros(n, np.uint8), np.zeros(n, np.uint8)
    dyn = np.zeros(n, bool)
    if instances and dynamic_ids:
        dyn_set = set(int(i) for i in dynamic_ids)
        for inst in instances:
            iid = int(inst.get("instance_id", inst.get("id")))
            if iid in dyn_set:
                gi = np.asarray(inst.get("globalIndices") or [], np.int64)
                gi = gi[(gi >= 0) & (gi < n)]
                dyn[gi] = True
    status = assign_status(mv, observed, mvt, mct, dyn, wcfg.rules)
    return {"mv_votes": mv, "mask_votes": mvt, "mask_conflicts": mct, "status": status}


def summarize(fields: Dict[str, np.ndarray]) -> dict:
    n = len(fields["status"])
    counts = status_counts(fields["status"])
    return {"n_points": int(n), "status_counts": counts,
            "status_fraction": {k: (v / n if n else 0.0) for k, v in counts.items()},
            "mv_votes_mean": float(np.mean(fields["mv_votes"])) if n else 0.0,
            "mask_votes_mean": float(np.mean(fields["mask_votes"])) if n else 0.0,
            "mask_conflicts_total": int(np.sum(fields["mask_conflicts"], dtype=np.int64))}


def run_witnesses(output_dir, cfg=None, frames: Optional[Dict[int, dict]] = None,
                  device=None, log: Callable[[str], None] = print, epoch: bool = True,
                  operator: str = "auto", correction_cfg=None) -> dict:
    """Compute the witnesses of the session's cloud and persist them. With
    ``epoch`` the fields land as a geometry epoch through the correction
    package (transaction + ledger, kind ``witness``); otherwise they are
    written straight into the two PLYs (epoch-less contexts: the merge
    stage, tests). Returns the report (also output/witness_report.json)."""
    from correction.session import load_session
    from reconstruction.loops.config import load_loops_config
    cfg = cfg or load_loops_config()
    wcfg = cfg.witness
    output_dir = Path(output_dir)
    t0 = time.time()
    session = load_session(output_dir)
    frames = frames or load_session_frames(output_dir, log)
    res_path = output_dir / "segmentation_result.json"
    instances = json.loads(res_path.read_text()).get("instances") or [] if res_path.exists() else []
    store = load_mask_store(output_dir) if instances else None
    fields = witness_fields(session.xyz, session.fg, session.data["pixel_row"],
                            session.data["pixel_col"], frames, wcfg, instances, store,
                            dynamic_instance_ids(output_dir), device=device)
    report = {"version": 1, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
              "params": {"n_neighbors": wcfg.n_neighbors, "tau_rel": wcfg.tau_rel,
                         "mask_erosion_px": wcfg.mask_erosion_px,
                         "occlusion_tol_rel": wcfg.occlusion_tol_rel,
                         "rules": dataclasses.asdict(wcfg.rules)},
              "n_keyframes_with_depth": len(frames), "n_instances": len(instances),
              "masks": store is not None, **summarize(fields),
              "provenance": "tool_measured"}
    if epoch:
        _witness_epoch(output_dir, session, fields, report, operator, log, correction_cfg)
    else:
        write_fields(output_dir / "cleaned_cloud.ply", fields)
        raw = output_dir / "cleaned_cloud_raw.ply"
        if raw.exists():
            write_fields(raw, fields)
    report["elapsed_s"] = round(time.time() - t0, 1)
    (output_dir / REPORT_JSON).write_text(json.dumps(report, indent=1, default=float))
    c = report["status_counts"]
    log(f"[witness] {report['n_points']:,} pts: verified {c['verified']:,}, single_witness "
        f"{c['single_witness']:,}, mask_conflict {c['mask_conflict']:,}, dynamic {c['dynamic']:,}, "
        f"unobserved {c['unobserved']:,} | mv_votes mean {report['mv_votes_mean']:.2f} | "
        f"conflicts {report['mask_conflicts_total']:,} ({report['elapsed_s']}s)")
    return report


def _witness_epoch(output_dir: Path, session, fields, report, operator, log, ccfg=None):
    """A status change is an epoch (§8): identity transforms, the clouds
    re-staged with the new fields, ledger record kind ``witness``."""
    from correction.apply import stage_transaction, swap_transaction, assert_no_interrupted_swap
    from correction.config import load_correction_config
    from correction import ledger
    assert_no_interrupted_swap(output_dir)
    ccfg = ccfg or load_correction_config()
    h2, d2 = add_fields(session.header, session.data, fields)
    rh, rd = session.raw_header, session.raw_data
    if rd is not None:
        if len(rd) != len(d2):
            raise RuntimeError(f"cleaned_cloud_raw.ply has {len(rd)} points, cleaned_cloud.ply "
                               f"{len(d2)} — the witnesses index the same points; the raw cloud "
                               f"is out of sync")
        rh, rd = add_fields(rh, rd, fields)
    sess2 = dataclasses.replace(session, header=h2, data=d2, raw_header=rh, raw_data=rd)
    N = session.n_kf
    R_kf = np.tile(np.eye(3), (N, 1, 1)); t_kf = np.zeros((N, 3)); k_kf = np.ones(N)
    cid = ledger.new_correction_id()
    tx = stage_transaction(sess2, ccfg, R_kf, t_kf, k_kf, correction_id=cid,
                           scale_diag_new=None, floor_npz=None, log=log, progress=None)
    swap_transaction(output_dir, tx, log=log)
    rep_path = output_dir / "corrections" / f"report_{cid}.json"
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.write_text(json.dumps(report, indent=1, default=float))
    ledger.record_run(output_dir, correction_id=cid, epoch_from=tx["epoch_from"],
                      epoch_to=tx["epoch_to"], kind="witness", operator=operator,
                      instance_ids=[], visits=[], observability=[], diagnosis=[], anchors=[],
                      gates=[{"name": "witness_fields", "passed": True,
                              "value": report["status_counts"], "threshold": None}],
                      overrides={}, report_path=str(rep_path.relative_to(output_dir)))
    report["correction_id"] = cid
    report["epoch"] = tx["epoch_to"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="witness stage (§6) on a session")
    ap.add_argument("--session", required=True)
    ap.add_argument("--no-epoch", action="store_true")
    args = ap.parse_args(argv)
    run_witnesses(Path(args.session) / "output", epoch=not args.no_epoch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
