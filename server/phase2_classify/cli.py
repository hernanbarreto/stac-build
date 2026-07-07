# STAC-Builder — Phase 2 classification CLI.
#   python -m phase2_classify.cli --scene <store.db> --session <session_dir>
# Hernán Barreto - Ingerop IN3 Session IV - STAC
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from phase2_classify.classify import SegmentClassifier  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Classify/enrich store instances (Phase 2)")
    ap.add_argument("--scene", required=True, help="instance store (.db)")
    ap.add_argument("--session", required=True, help="session dir (has frames/)")
    ap.add_argument("--backend", default="qwen_local")
    args = ap.parse_args()
    clf = SegmentClassifier(args.scene, args.session, backend=args.backend)
    stats = clf.run(on_progress=lambda p, m: print(f"[classify {p:3d}%] {m}", flush=True))
    print(f"\nRESULT: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
