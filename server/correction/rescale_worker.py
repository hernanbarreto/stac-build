"""The real rescale of a scan, in its own process (the backend launches it: calibration_api).

    python -m correction.rescale_worker --session <id> [--scan <date/source>] --scale <s> [--source <json>]

CPU work only (the cloud is multiplied, the octree converted; then the scan's IFCs written again
and its stored comparison removed — bim/after_rescale.py): the card and the chat's engines are not touched. Prints its progress; the last line is ``RESULT <json report>`` (exit 0) or
``ERROR <message>`` (exit 1)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent.parent
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--scan", default=None)
    ap.add_argument("--scale", type=float, required=True)
    ap.add_argument("--source", default="{}")
    a = ap.parse_args(argv)
    try:
        from correction.rescale import apply_scale
        from project_paths import resolve_session
        ctx = resolve_session(str(SERVER_DIR), a.session, scan_key=a.scan or None)
        paths = getattr(ctx, "paths", None)
        key = f"{ctx.date}/{ctx.source_name}" if hasattr(ctx, "date") and hasattr(ctx, "source_name") else None
        report = apply_scale(ctx.output_dir, a.scale, source=json.loads(a.source), paths=paths, scan_key=key,
                             log=lambda m: print(m, flush=True),
                             progress=lambda pct, msg: print(f"PROGRESS {int(pct)} {msg}", flush=True))
        # what the new scale left stale (bim/after_rescale.py): the scan's IFCs written again, its stored comparison
        # removed — the rescale itself stands whatever happens here (its epoch is in place)
        if key:
            from bim.after_rescale import renew
            print("PROGRESS 96 the scan's IFCs and comparison at the new scale", flush=True)
            report["renewed"] = renew(a.session, key, ctx.ifcs_dir, ctx.bim_comparison_dir,
                                      log=lambda m: print(m, flush=True))
    except Exception as e:  # noqa: BLE001 — told to the caller as what it is
        import traceback
        traceback.print_exc()
        print(f"ERROR {type(e).__name__}: {e}", flush=True)
        return 1
    print("RESULT " + json.dumps(report, default=float, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
