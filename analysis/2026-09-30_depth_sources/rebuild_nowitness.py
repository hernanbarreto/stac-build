"""pccr 2026-09-30 (USER: "reconstruí … sin tantas épocas: sólo la cero y esta nueva como
epoch 1"): the corrected cloud WITHOUT the witness filter (it dropped 5.2 M points, 28 %,
and the user sees holes), then the cloud stage's projection and the certify stage's
correction, then the epochs are compacted to 0 (Omega) + 1 (the new one).
  1. back to F5's poses:      correction.run.run_select(epoch 4) — epochs 3/4 moved no camera
  2. the cloud:               precision.corrected_cloud.run_corrected_cloud, witness_filter off
  3. the projection:          segmentation.pipeline.map_segmentation_to_cloud
  4. the correction:          reconstruction.certify.run.certify_session
  5. compact:                 keep _epoch_0 + live; live renumbered 1 (parent 0, new_cloud);
                              stamps re-written; the ledger gets a 'compact' record
"""
import json
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, "/workspace/stac-build/server")
S = Path("/workspace/stac-build/server/projects/pccr/scans/2026-08-31/src_default")
O = S / "output"


def log(m, level="info"):
    print(f"[rebuild {time.strftime('%H:%M:%S')}] {m}", flush=True)


def live_epoch():
    return int(json.loads((O / "geometry_epoch.json").read_text())["epoch"])


def compact():
    """Keep _epoch_0 and the live epoch; the live one becomes epoch 1."""
    before = live_epoch()
    freed = 0
    for d in O.glob("_epoch_*"):
        if d.name != "_epoch_0" and d.is_dir():
            freed += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            shutil.rmtree(d)
    for d in O.glob("_tx_epoch_*"):
        shutil.rmtree(d, ignore_errors=True)
    for f in (O / "corrections").glob("epoch_*.npz"):
        f.unlink()
    rec = json.loads((O / "geometry_epoch.json").read_text())
    rec.update(epoch=1, parent_epoch=0, kind="new_cloud",
               compacted_from=before, compacted_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    (O / "geometry_epoch.json").write_text(json.dumps(rec, indent=1))
    # the stamps every consumer compares with the live epoch
    org = O / "origins.npz"
    if org.exists():
        with np.load(org, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
        if "geometry_epoch" in d:
            d["geometry_epoch"] = np.full_like(d["geometry_epoch"], 1)
        np.savez(org, **d)
    for name, key in (("corrected_cloud.json", "epoch_to"), ("segmentation_result.json", "geometry_epoch")):
        p = O / name
        if p.exists():
            doc = json.loads(p.read_text())
            if key in doc:
                doc[key] = 1
                p.write_text(json.dumps(doc))
    # selecting epoch 0 swaps only what its manifest lists: the new epoch's own artifacts
    # (segmentation, OBB store, levelling, provenance) join it as "absent in epoch 0", so
    # they step aside when 0 is shown and come back with 1
    man_p = O / "_epoch_0" / "_manifest.json"
    man = json.loads(man_p.read_text())
    have = {a["rel"] for a in man["artifacts"]}
    for rel in ("cleaned_cloud_raw.ply", "origins.npz", "corrected_cloud.json",
                "segmentation_result.json", "seg_broadcast.json", "scene_r.db",
                "classification.npy", "out_of_place.npy", "floor_transform.npz",
                "floor_level.json"):
        if rel not in have and (O / rel).exists():
            man["artifacts"].append({"rel": rel, "existed_before": False})
    man["epoch_to"] = 1
    man_p.write_text(json.dumps(man))
    with open(O / "corrections.jsonl", "a") as f:
        f.write(json.dumps({"type": "compact", "epoch_from": before, "epoch_to": 1,
                            "kept": [0, 1], "folded": list(range(1, before + 1)),
                            "reason": "USER 2026-09-30: only the original (0) and the new one (1)",
                            "at": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
    log(f"compacted: epochs {sorted([0, before])} → 0 + 1 ({freed / 1e9:.1f} GB of intermediate epochs removed)")


def main() -> int:
    t0 = time.time()
    from config import cfg as raw_cfg
    from correction.run import run_select
    from workers.base import stop_semantic_service_verified
    stop_semantic_service_verified(None, stage="rebuild", log=log)

    log(f"1/5 back to F5's poses: epoch {live_epoch()} → 4")
    run_select(O, 4, "auto", log=log)

    log("2/5 f7_cloud WITHOUT the witness filter")
    from precision.config import load_precision_config
    from precision.corrected_cloud import run_corrected_cloud
    p = load_precision_config()
    p = replace(p, cloud=replace(p.cloud, witness_filter=False))
    rep = run_corrected_cloud(S, p, log=log)
    log(f"   epoch {rep['epoch_to']}: {rep['n_points']:,} pts")

    log("3/5 projecting the SAM3 masks (cloud stage)")
    from segmentation.pipeline import map_segmentation_to_cloud
    res = O / "segmentation_result.json"
    if res.exists() and "pending" in json.loads(res.read_text()):
        res.unlink()
    d = map_segmentation_to_cloud(O)
    if d.get("error"):
        log(f"❌ projection: {d['error']}")
        return 1
    log(f"   {len(d.get('instances', []))} instances, coverage {d.get('coverage')}")

    log("4/5 the correction (certify stage)")
    from reconstruction.loops.config import load_loops_config
    from reconstruction.certify.run import certify_session
    acta = certify_session(S, cfg=load_loops_config(raw_cfg), operator="pipeline", log=log,
                           progress=lambda a, b: None)
    log(f"   {acta.get('stop_reason')} — epoch {acta.get('epoch_initial')} → {acta.get('epoch_final')}")

    log("5/5 compacting the epochs to 0 + 1")
    compact()
    n = json.loads((O / "potree" / "metadata.json").read_text()).get("points")
    log(f"DONE in {(time.time() - t0) / 60:.1f} min — epoch 1 live ({n:,} pts), epoch 0 selectable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
