#!/usr/bin/env python3
"""
STAC Build — Legacy Session Migration
========================================

Migrates existing flat sessions from scans/{name}/ to the new
hierarchical project structure in projects/{name}/.

Old layout:
    scans/pccr-10022026/
        frames/
        output/
            chunk_*.ply
            cleaned_cloud.ply
            potree/
            floor_transform.npz
            segmentation*.json
            seg_masks.npz
            ...
        *.ifc
        sabana*.{npz,json,bin,ply}
        coverage_history/

New layout:
    projects/pccr-10022026/
        project.json
        ifcs/
            *.ifc
        scans/
            YYYY-MM-DD/
                scan_day.json
                src_legacy/
                    source.json
                    frames/        ← moved from old frames/
                    output/        ← moved from old output/
        merged/
            merged_cloud.ply       ← symlink to source output/cleaned_cloud.ply
            potree/                ← moved from old output/potree/
            floor_transform.npz    ← moved from old output/floor_transform.npz
        coverage_history/          ← moved from old coverage_history/
        segmentation/              ← segmentation files moved here
        bim_comparison/            ← sabana files moved here

Usage:
    python migrate_sessions.py                    # dry-run (preview)
    python migrate_sessions.py --execute          # actually move files
    python migrate_sessions.py --execute --session pccr-10022026  # single session

Authors: Hernán Barreto — Ingerop IN3
"""

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path


# ── Configuration ──────────────────────────────────────────────

SERVER_DIR = Path(__file__).parent
SCANS_DIR = SERVER_DIR / "scans"
PROJECTS_DIR = SERVER_DIR / "projects"

# Files that belong at project root / bim_comparison level
SABANA_FILES = [
    "sabana.npz", "sabana_meta.json",
    "sabana_cloud.ply", "sabana_potree",
    "sabana_positions.bin", "sabana_colors.bin",
]

# Segmentation files to move from output/ to segmentation/
SEG_FILES = [
    "segmentation.json", "segmentation_result.json",
    "seg_masks.npz", "seg_broadcast.json",
]

# VLM analysis files at session root (written by VLM worker)
VLM_FILES = [
    "scene_analysis.json", "vlm_analysis.json",
]


def detect_session_date(session_path: Path) -> str:
    """
    Try to detect the scan date from the session.
    1. Parse from session name if it contains a date
    2. Use modification time of the frames directory
    3. Fallback to today
    """
    name = session_path.name

    # Try common date patterns in session names
    # e.g. "2026-01-31-22-23-52" or "live_1770423147"
    if len(name) >= 10 and name[:4].isdigit():
        try:
            return datetime.strptime(name[:10], "%Y-%m-%d").strftime("%Y-%m-%d")
        except ValueError:
            pass

    # Try Unix timestamp (e.g., "live_1770423147")
    if "_" in name:
        parts = name.split("_")
        for part in parts:
            if part.isdigit() and len(part) >= 10:
                try:
                    ts = int(part)
                    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                except (ValueError, OSError):
                    pass

    # Use frames dir mtime
    frames_dir = session_path / "frames"
    if frames_dir.exists():
        mtime = frames_dir.stat().st_mtime
        return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")

    # Fallback
    return datetime.now().strftime("%Y-%m-%d")


def migrate_session(session_path: Path, execute: bool = False) -> dict:
    """
    Migrate a single session from flat to hierarchical layout.
    Returns a summary dict.
    """
    session_name = session_path.name
    project_dir = PROJECTS_DIR / session_name
    scan_date = detect_session_date(session_path)
    source_name = "legacy"

    summary = {
        "session": session_name,
        "scan_date": scan_date,
        "actions": [],
        "errors": [],
    }

    def log(msg):
        summary["actions"].append(msg)
        print(f"  {msg}")

    def err(msg):
        summary["errors"].append(msg)
        print(f"  ❌ {msg}")

    print(f"\n{'='*60}")
    print(f"  Migrating: {session_name}")
    print(f"  Detected date: {scan_date}")
    print(f"  Target: projects/{session_name}/scans/{scan_date}/src_legacy/")
    print(f"{'='*60}")

    if project_dir.exists():
        err(f"Target already exists: {project_dir}")
        return summary

    # ── Create directory structure ──
    source_dir = project_dir / "scans" / scan_date / f"src_{source_name}"
    dirs_to_create = [
        project_dir,
        project_dir / "ifcs",
        project_dir / "scans" / scan_date,
        source_dir,
        project_dir / "merged",
        project_dir / "merged" / "potree",
        project_dir / "coverage_history" / "snapshots",
        project_dir / "segmentation",
        project_dir / "bim_comparison",
    ]

    for d in dirs_to_create:
        log(f"mkdir {d.relative_to(PROJECTS_DIR)}")
        if execute:
            d.mkdir(parents=True, exist_ok=True)

    # ── Move frames/ ──
    old_frames = session_path / "frames"
    new_frames = source_dir / "frames"
    if old_frames.exists():
        frame_count = len(list(old_frames.glob("*.jpg")))
        log(f"move frames/ ({frame_count} images) → src_legacy/frames/")
        if execute:
            shutil.move(str(old_frames), str(new_frames))

    # ── Move output/ ──
    old_output = session_path / "output"
    new_output = source_dir / "output"
    if old_output.exists():
        # First, extract files that go to merged/ or segmentation/
        # Potree → merged/potree/
        old_potree = old_output / "potree"
        if old_potree.exists():
            log(f"move output/potree/ → merged/potree/")
            if execute:
                new_potree = project_dir / "merged" / "potree"
                if new_potree.exists():
                    shutil.rmtree(str(new_potree))
                shutil.move(str(old_potree), str(new_potree))

        # floor_transform → merged/
        old_floor = old_output / "floor_transform.npz"
        if old_floor.exists():
            log(f"copy output/floor_transform.npz → merged/")
            if execute:
                shutil.copy2(str(old_floor), str(project_dir / "merged" / "floor_transform.npz"))

        # cleaned_cloud → merged/merged_cloud.ply (copy, keep original in source output)
        old_cloud = old_output / "cleaned_cloud.ply"
        if old_cloud.exists():
            size_mb = old_cloud.stat().st_size / (1024 * 1024)
            log(f"symlink output/cleaned_cloud.ply → merged/merged_cloud.ply ({size_mb:.1f} MB)")
            if execute:
                merged_cloud = project_dir / "merged" / "merged_cloud.ply"
                # Create symlink to save disk space
                os.symlink(
                    os.path.relpath(str(new_output / "cleaned_cloud.ply"), str(merged_cloud.parent)),
                    str(merged_cloud)
                )

        # Segmentation files → segmentation/
        for seg_file in SEG_FILES:
            old_seg = old_output / seg_file
            if old_seg.exists():
                log(f"copy output/{seg_file} → segmentation/")
                if execute:
                    shutil.copy2(str(old_seg), str(project_dir / "segmentation" / seg_file))

        # Move the rest of output/
        log(f"move output/ → src_legacy/output/")
        if execute:
            shutil.move(str(old_output), str(new_output))

    # ── Move IFC files ──
    ifc_files = list(session_path.glob("*.ifc"))
    for ifc in ifc_files:
        log(f"move {ifc.name} → ifcs/")
        if execute:
            shutil.move(str(ifc), str(project_dir / "ifcs" / ifc.name))

    # ── Move sábana files ──
    for sabana_name in SABANA_FILES:
        old_sabana = session_path / sabana_name
        if old_sabana.exists():
            if old_sabana.is_dir():
                log(f"move {sabana_name}/ → bim_comparison/")
                if execute:
                    shutil.move(str(old_sabana), str(project_dir / "bim_comparison" / sabana_name))
            else:
                log(f"move {sabana_name} → bim_comparison/")
                if execute:
                    shutil.move(str(old_sabana), str(project_dir / "bim_comparison" / sabana_name))

    # ── Move VLM analysis files (session root → source dir) ──
    for vlm_name in VLM_FILES:
        old_vlm = session_path / vlm_name
        if old_vlm.exists():
            log(f"move {vlm_name} → src_legacy/")
            if execute:
                shutil.move(str(old_vlm), str(source_dir / vlm_name))

    # ── Move coverage_history/ ──
    old_coverage = session_path / "coverage_history"
    if old_coverage.exists():
        log(f"move coverage_history/ → coverage_history/")
        if execute:
            new_coverage = project_dir / "coverage_history"
            # Remove the empty one we created
            if new_coverage.exists():
                shutil.rmtree(str(new_coverage))
            shutil.move(str(old_coverage), str(new_coverage))

    # ── Create metadata files ──
    # project.json
    project_meta = {
        "name": session_name,
        "slug": session_name,
        "location": "",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "migrated_from": f"scans/{session_name}",
        "ifcs": [f.name for f in ifc_files],
        "scan_days": [scan_date],
    }
    log(f"create project.json")
    if execute:
        (project_dir / "project.json").write_text(
            json.dumps(project_meta, indent=2, ensure_ascii=False)
        )

    # scan_day.json
    scan_day_meta = {
        "scan_date": scan_date,
        "operator": "unknown",
        "zone": "full_site",
        "notes": f"Migrated from legacy session {session_name}",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    log(f"create scans/{scan_date}/scan_day.json")
    if execute:
        (project_dir / "scans" / scan_date / "scan_day.json").write_text(
            json.dumps(scan_day_meta, indent=2, ensure_ascii=False)
        )

    # source.json
    source_meta = {
        "source_name": source_name,
        "video_path": None,
        "operator": "unknown",
        "status": "done",
        "migrated_from": f"scans/{session_name}",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    log(f"create src_legacy/source.json")
    if execute:
        (source_dir / "source.json").write_text(
            json.dumps(source_meta, indent=2, ensure_ascii=False)
        )

    # ── Clean up empty old session dir ──
    if execute:
        # Check if old session dir is now empty
        remaining = list(session_path.iterdir())
        if not remaining:
            log(f"remove empty scans/{session_name}/")
            session_path.rmdir()
        else:
            remaining_names = [f.name for f in remaining]
            log(f"⚠️ scans/{session_name}/ still has: {remaining_names}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Migrate legacy STAC sessions to project hierarchy")
    parser.add_argument("--execute", action="store_true", help="Actually move files (default: dry-run)")
    parser.add_argument("--session", type=str, help="Migrate a single session by name")
    args = parser.parse_args()

    if not SCANS_DIR.exists():
        print(f"No scans directory found at {SCANS_DIR}")
        return

    # Find sessions to migrate
    if args.session:
        session_path = SCANS_DIR / args.session
        if not session_path.exists():
            print(f"Session not found: {args.session}")
            return
        sessions = [session_path]
    else:
        sessions = sorted([
            d for d in SCANS_DIR.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])

    if not sessions:
        print("No sessions to migrate")
        return

    mode = "EXECUTE" if args.execute else "DRY RUN"
    print(f"\n{'#'*60}")
    print(f"  STAC Session Migration — {mode}")
    print(f"  Sessions: {len(sessions)}")
    print(f"  From: {SCANS_DIR}")
    print(f"  To:   {PROJECTS_DIR}")
    print(f"{'#'*60}")

    if not args.execute:
        print("\n  ⚠️  This is a DRY RUN. No files will be moved.")
        print("  Run with --execute to actually migrate.\n")

    # Create projects dir
    if args.execute:
        PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for session_path in sessions:
        result = migrate_session(session_path, execute=args.execute)
        results.append(result)

    # Summary
    print(f"\n{'#'*60}")
    print(f"  SUMMARY")
    print(f"{'#'*60}")
    total_actions = sum(len(r["actions"]) for r in results)
    total_errors = sum(len(r["errors"]) for r in results)
    print(f"  Sessions: {len(results)}")
    print(f"  Actions:  {total_actions}")
    print(f"  Errors:   {total_errors}")
    if total_errors:
        for r in results:
            for e in r["errors"]:
                print(f"    ❌ [{r['session']}] {e}")

    if not args.execute:
        print(f"\n  Re-run with --execute to apply these changes.")


if __name__ == "__main__":
    main()
