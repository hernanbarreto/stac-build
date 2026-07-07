# STAC-Builder — Phase 3 findings CLI.
#   python -m phase3_findings.cli --scene <store.db> --session <session_dir> \
#          [--output <run_output_dir>] [--stride N] [--max-frames N]
# Hernán Barreto - Ingerop IN3 Session IV - STAC
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from phase3_findings.detect import FindingDetector  # noqa: E402
from phase_r.instance_store import InstanceStore  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Detect 3D-anchored visual findings (Phase 3)")
    ap.add_argument("--scene", required=True, help="instance store (.db)")
    ap.add_argument("--session", required=True, help="session dir (has frames/)")
    ap.add_argument("--output", default=None, help="run output dir (segmentation + depth)")
    ap.add_argument("--stride", type=int, default=1, help="keyframe subsampling stride")
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--backend", default="qwen_local")
    args = ap.parse_args()

    cfg = {"phase3": {"sample_stride": args.stride, "max_frames": args.max_frames}}
    det = FindingDetector(args.scene, args.session, output_dir=args.output,
                          config=cfg, backend=args.backend)
    stats = det.run(on_progress=lambda p, m: print(f"[findings {p:3d}%] {m}", flush=True))
    print(f"\nRESULT: {stats.summary()}")

    st = InstanceStore(args.scene)
    print("\nFINDINGS (all born 'proposed', await human validation):")
    for f in st.list_findings():
        p = f["point3d"]
        loc = f"({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})" if p else "unanchored"
        corr = " [corr-residual↑]" if f["correlated_residual"] else ""
        print(f"  #{f['finding_id']} {f['type']:18s} sev={f['severity']:6s} "
              f"conf={f['confidence']:.2f} inst={f['instance_id']} {loc}{corr}  "
              f"{f['description'][:60]}")
    st.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
