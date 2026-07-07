# STAC-Builder — Phase 6 report CLI.
#   python -m phase6_report.cli --scene <store.db> --session <dir> --out <dir> \
#          [--lang es|fr|both] [--coverage <coverage_report.json>] [--timestamp ISO]
# Hernán Barreto - Ingerop IN3 Session IV - STAC
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parents[1]
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from phase6_report.report import ReportBuilder  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a supervision report draft (Phase 6)")
    ap.add_argument("--scene", required=True, help="instance store (.db)")
    ap.add_argument("--session", default=None, help="session dir (for finding figures)")
    ap.add_argument("--out", required=True, help="output dir (report + assets)")
    ap.add_argument("--lang", default="both", choices=["es", "fr", "both"])
    ap.add_argument("--coverage", default=None, help="Phase 4 coverage_report.json")
    ap.add_argument("--timestamp", default=None, help="ISO timestamp (default: now, UTC)")
    args = ap.parse_args()

    ts = args.timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    langs = ["es", "fr"] if args.lang == "both" else [args.lang]
    for lang in langs:
        rb = ReportBuilder(args.scene, session_dir=args.session, lang=lang,
                           coverage_json=args.coverage, timestamp=ts)
        res = rb.write(args.out)
        print(f"[{lang}] {res['report']}  ({res['n_assets']} assets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
