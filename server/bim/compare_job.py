"""THE BIM COMPARISON IN ITS OWN PROCESS.

    python -m bim.compare_job --session-dir DIR --matches FILE [--tolerance-mm 50] [--manual]

A scan of today is compared in minutes, not in the seconds of March's (pccr 2026-08-31: 18.8 M
points against 281 elements): run inside the server it held one of its threads and its HTTP answer
for all that time — and an answer that takes more than 100 s never arrives through the pod's
proxy. The server launches this, answers at once and is asked for the state (main.py,
/api/bim/compare … /status). What it says on its output, one line each (the exporter's protocol):

    PROGRESS <percent> <what it is doing>
    RESULT <json>       the comparison's summary — its numbers are in sabana_meta.json, its sheet on disk
                        (and the sheet's octree built, so that the viewer shows it at once)
    ERROR <why>         … and a non-zero exit

CPU only (the caller hides the card): the distances are numpy's, in threads.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--session-dir", required=True, help="the scan's directory (…/scans/<date>/src_<source>)")
    ap.add_argument("--matches", required=True, help="a JSON file: the matches (bim.scan_data.Scan.match)")
    ap.add_argument("--tolerance-mm", type=float, default=50.0)
    ap.add_argument("--manual", action="store_true", help="the scan's own alignment (its floor transform), no registration")
    args = ap.parse_args(argv)

    def progress(pct, detail: str = "") -> None:
        print(f"PROGRESS {int(pct)} {detail}", flush=True)

    try:
        from bim.comparison import run_comparison
        matches = json.loads(Path(args.matches).read_text())
        if not isinstance(matches, list) or not matches:
            raise ValueError("no matches: nothing of the scan is paired with an element of the IFC")
        result = run_comparison(args.session_dir, matches, tolerance=args.tolerance_mm / 1000.0,
                                progress_callback=lambda pct, detail="": progress(min(int(pct) * 95 // 100, 95), detail),
                                skip_registration=args.manual)
        if not result.get("ok"):
            raise RuntimeError(result.get("error") or "the comparison gave no result")
        # the sheet's octree, now: the viewer then shows it at once (it would build it at the first look)
        progress(97, "Building the sheet's octree...")
        from bim.comparison import project_dirs
        from potree_converter import convert_sabana_to_potree
        result["summary"]["octree"] = bool(convert_sabana_to_potree(project_dirs(args.session_dir)[1]))
        progress(100, "Comparison complete")
    except Exception as e:  # noqa: BLE001 — said on the protocol's line, with the trace on stderr
        import traceback
        traceback.print_exc()
        print(f"ERROR {type(e).__name__}: {e}", flush=True)
        return 1
    print("RESULT " + json.dumps({k: result[k] for k in ("date", "tolerance_mm", "transform", "summary")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
