"""After a test session's cloud is published: remove what the run left behind that nothing reads any more.

    python arq_clean.py <scan dir> [--go]

Without --go it only lists. Refuses unless the session's cloud is published (precision.product.product_is_live).

REMOVED (each named with its readers, read in server/ on 2026-11-21):
  output/_restart_epochs            the way back to epoch 0 — a precision chain started again from its first step
  output/da3_cubes                  the DA3 windows of the cubes — the cube sizes and the scale, both applied
  output/da3_run                    every keyframe's DA3 depth — the scale step, applied
  output/maplong_run/_tmp_results_* the cubes as inferred / placed — the cube stage, done
  output/maplong_run/pcd, uncert, context_views    the fork's scratch
  output/maplong_run/sky_masks      the sky masks Omega's passes read
  output/precision/tracks.npz       F4's tracks — F5 and F6, done
  output/chunk_*.ply / *_origins.npz / *_meta.json   Omega's cube files (the worker's own end removes them too)
KEPT: the published cloud and its octree, the cameras, Omega's per-keyframe records (the reconstruction's identity and
the depth the VLM stage reads), every report, the frames, the video, AMB3R's pass, the cube plan.
A directory that is a link to the pod's local disk (reconstruction/fork_scratch.py) is removed with its target."""
import shutil
import sys
from pathlib import Path

SERVER = "/workspace/stac-build/server"
if SERVER not in sys.path:
    sys.path.insert(0, SERVER)

DIRS = ("_restart_epochs", "da3_cubes", "da3_run", "maplong_run/_tmp_results_unaligned",
        "maplong_run/_tmp_results_aligned", "maplong_run/_tmp_results_loop", "maplong_run/pcd", "maplong_run/uncert",
        "maplong_run/context_views", "maplong_run/sky_masks")
FILES = ("precision/tracks.npz",)
GLOBS = ("chunk_*.ply", "chunk_*_origins.npz", "chunk_*_meta.json")


def tree_bytes(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main() -> int:
    scan = Path(sys.argv[1]).resolve()
    go = "--go" in sys.argv[2:]
    out = scan / "output"
    from precision.product import product_is_live
    live, why = product_is_live(out)
    if not live:
        raise SystemExit(f"the session's cloud is not published ({why}) — nothing is removed")
    total = 0
    for rel in DIRS:
        p = out / rel
        if not (p.exists() or p.is_symlink()):
            continue
        target = p.resolve() if p.is_symlink() else None
        n = tree_bytes(target if target is not None and target.exists() else p) if p.exists() else 0
        total += n
        print(f"{'removed' if go else 'would remove'} {rel}/  {n / 1e6:.0f} MB" + (f"  (→ {target})" if target else ""))
        if go:
            if target is not None:
                p.unlink()
                if target.exists():
                    shutil.rmtree(target)
            else:
                shutil.rmtree(p)
    keep = out / "precision" / "KEEP_TRACKS"           # F4's tracks stay while this file exists (its text says why)
    files = [out / f for f in FILES if not (f == "precision/tracks.npz" and keep.exists())] \
        + [f for g in GLOBS for f in sorted(out.glob(g))]
    if keep.exists():
        print(f"kept precision/tracks.npz ({keep.name}: {keep.read_text().strip()[:160]})")
    for p in files:
        if p.is_file():
            n = p.stat().st_size
            total += n
            print(f"{'removed' if go else 'would remove'} {p.relative_to(out)}  {n / 1e6:.0f} MB")
            if go:
                p.unlink()
    print(f"{'freed' if go else 'to free'}: {total / 1e9:.2f} GB")
    left = sorted(((tree_bytes(p), p) for p in list(out.iterdir()) + [scan / "frames", scan / "intake"]
                   + [p for p in scan.iterdir() if p.is_file()]), reverse=True)
    print("what the scan holds now (over 20 MB): " + ", ".join(
        f"{p.relative_to(scan)} {n / 1e6:.0f} MB" for n, p in left if n > 20e6))
    return 0


if __name__ == "__main__":
    sys.exit(main())
