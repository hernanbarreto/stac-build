#!/usr/bin/env python3
"""Merge one project's scans INTO another (USER DESIGN 2026-09-06).

    python scripts/merge_projects.py --into pccr --from pccr_v1 [--reference 2026-09-03/default] [--dry-run] [--force]

A project is one place; scans from different days belong together.
pccr_v1 (older, complete) + pccr (newer, partial) → ONE project `pccr`:
every scan day folder of --from is MOVED under --into (same filesystem →
instant, artifacts untouched), project.json is merged (scans, ifcs), the
composition reference is set (default: the OLDEST scan, or --reference),
and the emptied source project is left as a shell for the user to delete.

Refuses to run while any reconstruction/Potree process is alive (a move
under a running pipeline would corrupt it) unless --force. Idempotent:
scan days already present in --into are skipped with a report.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

SERVER = Path(__file__).resolve().parents[1] / "server"
sys.path.insert(0, str(SERVER))


def _busy() -> list:
    """Names of live processes that must not be running during a move."""
    out = []
    for pat in ("vggt_long", "run_mapanything", "PotreeConverter",
                "gpu_cloud_clean", "uvicorn main:app"):
        r = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True)
        if r.returncode == 0:
            out.append(pat)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--into", required=True, help="target project slug (kept)")
    ap.add_argument("--from", dest="src", required=True,
                    help="source project slug (its scans move into --into)")
    ap.add_argument("--reference", default=None,
                    help="composition reference scan key 'date/source' "
                         "(default: oldest scan of the merged project)")
    ap.add_argument("--rename-day", action="append", default=[],
                    metavar="PROJECT:OLD=NEW",
                    help="give a scan day its REAL capture date, e.g. "
                         "pccr_v1:2026-09-03=2026-06-15 (folder dates are "
                         "processing dates, not capture dates). Repeatable.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="run even if pipeline/backend processes are alive")
    args = ap.parse_args()

    from config import PROJECTS_DIR
    from project_paths import ProjectPaths
    from project_scans import sync_scans, set_reference

    into = ProjectPaths(str(PROJECTS_DIR), args.into)
    src = ProjectPaths(str(PROJECTS_DIR), args.src)
    if not src.project_json.exists():
        print(f"✗ source project {args.src} not found"); return 2
    if not into.project_json.exists():
        print(f"✗ target project {args.into} not found"); return 2

    busy = _busy()
    if busy and not args.force:
        print("✗ REFUSED — processes alive that could be using these folders: "
              f"{busy}. Stop the backend/pipeline first (or --force).")
        return 3

    # real capture dates (folder names are processing dates)
    renames = {}
    for spec in args.rename_day:
        try:
            proj, pair = spec.split(":", 1)
            old, new = pair.split("=", 1)
            renames[(proj, old)] = new
        except ValueError:
            print(f"✗ bad --rename-day '{spec}' (want PROJECT:OLD=NEW)"); return 2
    # rename days INSIDE the target project first (e.g. pccr's own scan)
    for (proj, old), new in renames.items():
        if proj == args.into and into.scan_day_dir(old).exists():
            if into.scan_day_dir(new).exists():
                print(f"✗ {args.into}: day {new} already exists"); return 2
            print(f"  ↻ {args.into}: day {old} → {new}")
            if not args.dry_run:
                into.scan_day_dir(old).rename(into.scan_day_dir(new))

    moves = []
    for date in src.list_scan_days():
        new_date = renames.get((args.src, date), date)
        for source in src.list_sources(date):
            s_dir = src.source_dir(date, source)
            t_dir = into.source_dir(new_date, source)
            if t_dir.exists():
                print(f"  = {new_date}/{source}: already in {args.into} — skipped")
                continue
            moves.append((new_date, source, s_dir, t_dir))

    print(f"Merge {args.src} → {args.into}: {len(moves)} scan(s) to move")
    for date, source, s_dir, t_dir in moves:
        print(f"  → {date}/{source}: {s_dir} → {t_dir}")
    if args.dry_run:
        print("(dry run — nothing moved)"); return 0

    for date, source, s_dir, t_dir in moves:
        t_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(s_dir), str(t_dir))
        # scan-day json (if any) rides along
        sj = src.scan_day_json(date)
        if sj.exists() and not into.scan_day_json(date).exists():
            shutil.move(str(sj), str(into.scan_day_json(date)))

    # merge project.json (ifcs + any extra keys the source carried)
    m_into = into.load_project_meta()
    m_src = src.load_project_meta()
    for ifc in m_src.get("ifcs", []) or []:
        if ifc not in m_into.setdefault("ifcs", []):
            m_into["ifcs"].append(ifc)
    m_into.setdefault("merged_from", []).append(
        {"project": args.src, "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
         "scans": [f"{d}/{s}" for d, s, _a, _b in moves]})
    into.save_project_meta(m_into)
    meta = sync_scans(into)                       # scans list + default reference
    ref = args.reference or meta["composition"].get("reference")
    if ref:
        set_reference(into, ref)

    # anything under the moved scans still naming the old slug (chat notes,
    # prefs, reports) is reported, never rewritten blindly
    hits = []
    for date, source, _s, t_dir in moves:
        for p in t_dir.rglob("*"):
            if p.suffix in (".json", ".txt", ".md", ".yaml") and p.stat().st_size < 50_000_000:
                try:
                    if args.src in p.read_text(errors="ignore"):
                        hits.append(str(p.relative_to(into.project_dir)))
                except Exception:  # noqa: BLE001
                    pass
    print(f"✓ moved {len(moves)} scan(s); reference = {ref}")
    if hits:
        print(f"⚠ {len(hits)} file(s) still mention '{args.src}' (review, not rewritten):")
        for h in hits[:30]:
            print("   ", h)
    print(f"ℹ {src.project_dir} left as an empty shell — delete it when you are sure.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
