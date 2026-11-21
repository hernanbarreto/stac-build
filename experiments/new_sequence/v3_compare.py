"""VERSION 3 — the two clouds compared "como la sábana": each cloud painted by how far it lies from the other.

USER 2026-11-21: "lo que vamos a hacer es una comparacion de ambas nubes como la sabana, la pintura para ver las
diferencias entre ambas nubes". Cloud A = F6 without PointDiT, cloud B = F6 with PointDiT, same session, same cameras.

For every point of one cloud, the distance to the NEAREST point of the other (exact nearest neighbour, on the card).
The colour is the deviation sheet's scale: green at 0, yellow at the tolerance, red at twice the tolerance and beyond
(tolerance 1 cm: red from 2 cm — proposed to the user with the plan).

Two painted sessions are written, each a copy of the test session whose cloud file keeps its points and takes the
colours, with its octree rebuilt by the pipeline's own converter:

    <session>_pintada_A   cloud A's points; red = A has a point where B has none within 2 cm (what PointDiT removes
                          or moves)
    <session>_pintada_B   cloud B's points; red = B has a point where A has none within 2 cm (what PointDiT adds
                          or moves)

    python v3_compare.py paint <cloud A ply> <cloud B ply> <painted scan dir> A|B
    (the painted scan dir is a copy of the session at the epoch whose cloud it holds)"""
import json
import sys
from pathlib import Path

import numpy as np

SERVER = "/workspace/stac-build/server"
if SERVER not in sys.path:
    sys.path.insert(0, SERVER)
TOLERANCE_M = 0.01


def nearest(Q: np.ndarray, P: np.ndarray, batch: int = 500_000) -> np.ndarray:
    """For every point of Q, the distance to the nearest point of P — exact, on the card."""
    import open3d.core as o3c
    dev = o3c.Device("CUDA:0")
    nns = o3c.nns.NearestNeighborSearch(o3c.Tensor(np.ascontiguousarray(P, np.float32), device=dev))
    nns.knn_index()
    out = np.empty(len(Q), np.float64)
    for a in range(0, len(Q), batch):
        _idx, d2 = nns.knn_search(o3c.Tensor(np.ascontiguousarray(Q[a:a + batch], np.float32), device=dev), 1)
        out[a:a + batch] = np.sqrt(np.maximum(d2.cpu().numpy().reshape(-1), 0.0))
    return out


def sheet_colours(d: np.ndarray, tol: float) -> np.ndarray:
    """green (0) → yellow (tol) → red (2 tol and beyond)."""
    t = np.clip(d / (2.0 * tol), 0.0, 1.0)
    g, y, r = np.array([0, 200, 0.0]), np.array([255, 220, 0.0]), np.array([230, 0, 0.0])
    lo = t[:, None] * 2.0
    hi = (t[:, None] - 0.5) * 2.0
    c = np.where(t[:, None] <= 0.5, g + (y - g) * lo, y + (r - y) * hi)
    return np.rint(c).astype(np.uint8)


def xyz(d) -> np.ndarray:
    return np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float64)


def paint(ply_a: Path, ply_b: Path, scan: Path, which: str) -> int:
    from correction.session import read_ply, write_ply
    from potree_converter import convert_ply_to_potree
    _ha, A = read_ply(ply_a)
    _hb, B = read_ply(ply_b)
    own_ply, other = (ply_a, B) if which == "A" else (ply_b, A)
    header, own = read_ply(scan / "output" / "cleaned_cloud.ply")
    ref = A if which == "A" else B
    if len(own) != len(ref) or not np.array_equal(own["x"], ref["x"]):
        raise SystemExit(f"{scan} does not hold cloud {which} ({len(own):,} points on disk, {len(ref):,} in {own_ply})")
    d = nearest(xyz(own), xyz(other))
    rgb = sheet_colours(d, TOLERANCE_M)
    own = own.copy()
    own["red"], own["green"], own["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    target = scan / "output" / "cleaned_cloud.ply"
    tmp = target.with_name(target.name + ".painted")
    write_ply(tmp, header, own)
    tmp.replace(target)
    raw = scan / "output" / "cleaned_cloud_raw.ply"
    if raw.exists():
        import shutil
        shutil.copy2(target, raw)
    ok = convert_ply_to_potree(scan, force=True)
    stats = {"cloud": which, "points": int(len(d)), "tolerance_m": TOLERANCE_M,
             "within_5mm": float((d <= 0.005).mean()), "within_1cm": float((d <= 0.01).mean()),
             "within_2cm": float((d <= 0.02).mean()), "beyond_2cm": float((d > 0.02).mean()),
             "beyond_5cm": float((d > 0.05).mean()), "median_m": float(np.median(d)),
             "p95_m": float(np.percentile(d, 95)), "octree": bool(ok)}
    (scan / "output" / "painted_comparison.json").write_text(json.dumps(stats, indent=1))
    print(f"[compare] cloud {which}: {len(d):,} points painted by their distance to the other cloud — within 5 mm "
          f"{stats['within_5mm'] * 100:.1f} %, within 1 cm {stats['within_1cm'] * 100:.1f} %, within 2 cm "
          f"{stats['within_2cm'] * 100:.1f} %, beyond 2 cm (red) {stats['beyond_2cm'] * 100:.1f} %, beyond 5 cm "
          f"{stats['beyond_5cm'] * 100:.1f} %; median {stats['median_m'] * 1000:.1f} mm; octree rebuilt: {ok}", flush=True)
    return 0 if ok else 1


def main() -> int:
    if sys.argv[1] == "paint":
        return paint(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]).resolve(), sys.argv[5])
    raise SystemExit("usage: v3_compare.py paint <A.ply> <B.ply> <painted scan dir> A|B")


if __name__ == "__main__":
    sys.exit(main())
