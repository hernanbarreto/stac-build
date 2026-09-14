"""RGB + mask + reprojection of both copies, side by side, for eyeballing."""
import sys, json
sys.path.insert(0, "/workspace/stac-build/server")
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
from pathlib import Path
from PIL import Image

SESS = Path("/workspace/stac-build/server/projects/pccr/scans/2026-08-31/src_default")
OUT = SESS / "output"
DST = Path("/workspace/stac-build/server/reproj_check"); DST.mkdir(exist_ok=True)

from segmentation.pipeline import _load_ply_origins, _mask_frame_lookup
from segmentation.erase import _mask_obj_by_iid
from segmentation.shape_proposer import _visible_in_frame
from reconstruction.surface_fit.hole_audit import _Evidence

A_ID, B_ID = int(sys.argv[1]), int(sys.argv[2])
FRAMES = [int(x) for x in sys.argv[3:]]

xyz, fg, _pr, _pc = _load_ply_origins(OUT / "cleaned_cloud.ply")
res = json.loads((OUT / "segmentation_result.json").read_text())
by = {int(i.get("instance_id", i["id"])): i for i in res["instances"]}
oid_of = _mask_obj_by_iid(OUT)
ev = _Evidence(OUT, SESS)
z = np.load(OUT / "seg_masks.npz", allow_pickle=True)
c2m = _mask_frame_lookup(OUT, z["frames"].tolist(), sorted({int(f) for f in np.unique(fg)}))

def pts_of(iid):
    gi = np.asarray(by[iid]["globalIndices"], dtype=np.int64)
    return xyz[gi[(gi >= 0) & (gi < len(xyz))]]

PA, PB = pts_of(A_ID), pts_of(B_ID)
print(f"copia A = instancia {A_ID} ({len(PA):,} pts) | copia B = instancia {B_ID} ({len(PB):,} pts)")

for f in FRAMES:
    img_p = SESS / "frames" / f"{f:06d}.jpg"
    if not img_p.exists():
        print(f"frame {f}: no existe {img_p.name}"); continue
    img = Image.open(img_p).convert("RGB")
    W, H = img.size; mh, mw = H, W
    canvas = np.asarray(img).astype(np.int16).copy()

    # mask of whichever copy this frame belongs to
    painted = []
    for iid, col, name in ((A_ID, (0, 255, 0), "A"), (B_ID, (0, 128, 255), "B")):
        oid = oid_of.get(iid)
        key = f"f{c2m.get(f, f)}_o{oid}" if oid is not None else None
        if key and key in z.files:
            m = np.asarray(z[key]) > 0
            if m.shape != (mh, mw):
                import cv2
                m = cv2.resize(m.astype(np.uint8), (mw, mh), interpolation=cv2.INTER_NEAREST) > 0
            edge = m ^ np.roll(m, 1, 0) | (m ^ np.roll(m, 1, 1))
            canvas[edge] = col
            painted.append(f"mascara {name} ({int(m.sum())} px)")

    # reprojections
    for P, col, name in ((PA, (255, 0, 0), "A"), (PB, (255, 255, 0), "B")):
        r = _visible_in_frame(ev, f, mh, mw, P)
        if r is None: continue
        mu, mv, vis = r
        if vis.sum() == 0: continue
        canvas[mv[vis], mu[vis]] = col
        painted.append(f"reproy {name} ({int(vis.sum())} px)")

    out = DST / f"frame_{f:06d}.png"
    Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8)).save(out)
    print(f"frame {f:>6} -> {out.name}   [{', '.join(painted) or 'nada visible'}]")
print(f"\nCOLORES: contorno verde = mascara de {A_ID} | contorno celeste = mascara de {B_ID}")
print(f"         puntos ROJOS = reproyeccion de {A_ID} | puntos AMARILLOS = reproyeccion de {B_ID}")
