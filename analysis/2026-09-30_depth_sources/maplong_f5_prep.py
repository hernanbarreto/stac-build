"""MapAnything-long (VGGT-Long + MapAnything, SALAD loop closure, 120/60) fed the FULL prior per
keyframe: DA3 depth conditioned on F5 (output/da3_posed_depth, window-scale solved) + its K + F5's
pose — the pipeline's hybrid_cond path with F5 in place of ARKit. Outside the pipeline."""
import json, sys
from pathlib import Path
import numpy as np, yaml, cv2
RES = None                                 # USER 2026-09-30: keep the native resolution
sys.path.insert(0, "/workspace/stac-build/server")
S = Path("/workspace/stac-build/server/projects/pccr/scans/2026-08-31/src_default")
O = S / "output"
PRI = O / "maplong_f5_priors"; PRI.mkdir(exist_ok=True)
for _f in PRI.glob('frame_*.npz'): _f.unlink()
names = []
for p in sorted((O / "da3_posed_depth").glob("kf_*.npz")):
    z = np.load(p)
    fn = int(z["frame"])
    d = np.where(z["valid"], z["depth"], 0).astype(np.float32)
    H, W = d.shape
    f = 1.0 if RES is None else RES / max(H, W)
    h, w = int(round(H * f)), int(round(W * f))
    K = z["K"].astype(np.float64).copy()
    K[0, :] *= w / W; K[1, :] *= h / H                # pixel-edge convention: DA3's own resize
    np.savez(PRI / f"frame_{fn}.npz", depth=cv2.resize(d, (w, h), interpolation=cv2.INTER_NEAREST),
             intrinsics=K.astype(np.float32), extrinsics=np.linalg.inv(z["c2w"]).astype(np.float64),
             conf=cv2.resize(z["conf"].astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST))
    names.append(f"{fn:06d}.jpg")
(PRI / "selected_frames.json").write_text(json.dumps({"selected_files": names}))
from config import cfg
from workers.map_worker import _build_vggt_config
vc = _build_vggt_config(dict(cfg))
vc["Model"]["da3_priors_dir"] = str(PRI)
vc["Model"]["chunk_size"], vc["Model"]["overlap"] = 60, 30   # USER 2026-09-30: 120/60 does not fit on the card
vc["Model"]["da3_prior_use_poses"] = True          # the full-prior path (hybrid_cond forces it too)
(PRI / "vggt_long_config.yaml").write_text(yaml.dump(vc, default_flow_style=False))
m = vc["Model"]
print(f"{len(names)} priors; chunk {m['chunk_size']}/{m['overlap']}, loop {m['loop_enable']}, "
      f"prior conf pct {m['da3_prior_conf_percentile']}, map conf pct {m.get('map_conf_percentile')}, "
      f"cloud conf coef {m['Pointcloud_Save']['conf_threshold_coef']}, poses {m['da3_prior_use_poses']}")
