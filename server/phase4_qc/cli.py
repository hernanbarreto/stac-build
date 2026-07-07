# STAC-Builder — Phase 4 capture-QC CLI.
#   Pre-filter (ingestion):
#     python -m phase4_qc.cli prefilter --session <dir> --out <qc_dir> [--no-vlm]
#   Coverage report (post-reconstruction):
#     python -m phase4_qc.cli coverage --scene <store.db> --session <dir> \
#            --output <run_output> --out <cov_dir>
# Hernán Barreto - Ingerop IN3 Session IV - STAC
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))


def _prefilter(a) -> int:
    from phase4_qc.prefilter import CaptureQC
    cfg = {"phase4": {"sample_stride": a.stride, "escalate_to_vlm": not a.no_vlm}}
    qc = CaptureQC(a.session, config=cfg, backend=a.backend)
    man = qc.run(on_progress=lambda p, m: print(f"[qc {p:3d}%] {m}", flush=True))
    paths = qc.write(man, a.out)
    print(f"\nRESULT: {man.summary()}")
    print(f"  manifest : {paths['manifest']}")
    print(f"  drop-log : {paths['drop_log']} ({paths['n_dropped']} dropped)")
    return 0


def _coverage(a) -> int:
    from phase4_qc.coverage import CoverageReport
    cfg = {"phase4": {"voxel_size_m": a.voxel, "describe": not a.no_vlm,
                      "max_zones": a.max_zones}}
    cov = CoverageReport(a.scene, a.session, output_dir=a.output, config=cfg,
                         backend=a.backend)
    rep = cov.run(on_progress=lambda p, m: print(f"[cov {p:3d}%] {m}", flush=True))
    paths = cov.write(rep, a.out)
    print(f"\nRESULT: {rep['n_zones']} zones over {rep['total_points']} points")
    print(f"  report    : {paths['report']}")
    print(f"  checklist : {paths['checklist']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 4 capture QC")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prefilter", help="ingestion pre-filter (blur/exposure/occlusion)")
    p.add_argument("--session", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--no-vlm", action="store_true", help="classical only; ambiguous kept+flagged")
    p.add_argument("--backend", default="qwen_local")
    p.set_defaults(fn=_prefilter)

    c = sub.add_parser("coverage", help="post-reconstruction coverage + recapture checklist")
    c.add_argument("--scene", required=True)
    c.add_argument("--session", required=True)
    c.add_argument("--output", default=None)
    c.add_argument("--out", required=True)
    c.add_argument("--voxel", type=float, default=0.10)
    c.add_argument("--max-zones", type=int, default=12)
    c.add_argument("--no-vlm", action="store_true")
    c.add_argument("--backend", default="qwen_local")
    c.set_defaults(fn=_coverage)

    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
