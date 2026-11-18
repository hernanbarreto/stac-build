"""Write again, from the pod's hub caches, the ``hfcache`` rows of models.tsv.

    python3 capture_models.py            (capture.sh runs it)

A model that some component fetched through ``from_pretrained`` lands in a hub cache without
anyone writing a row for it. The pod has TWO caches — /workspace/hf_cache (the backend exports it
as HF_HOME) and /workspace/.cache/huggingface (HF_HOME of the pod's template: what a shell, a
stage script or a test launched without the backend's environment uses) — and a rebuild restores
exactly the snapshots each holds: the repo, the PINNED revision, every file, its sha256 when the
hub stores it as LFS (the blob's own name), and whether refs/main names that revision.

The other rows (plain files with their sources: Qwen's local directory, SAM 3.1, GenRecon's
checkpoints, DINOv3 …) are written by hand, one per file, and are left as they are: a weight
outside a hub cache needs a human to say where it came from. inventory.py names any large file
that no row covers.
"""
from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOTS = [Path("/workspace/hf_cache"), Path("/workspace/.cache/huggingface")]


def cache_rows(notes):
    rows = []
    for root in ROOTS:
        hub = root / "hub"
        if not hub.is_dir():
            continue
        for repo_dir in sorted(hub.glob("models--*")):
            repo = repo_dir.name[len("models--"):].replace("--", "/", 1)
            main = (repo_dir / "refs" / "main").read_text().strip() if (repo_dir / "refs" / "main").is_file() else ""
            snaps = repo_dir / "snapshots"
            for snap in sorted(snaps.iterdir()) if snaps.is_dir() else []:
                for f in sorted(p for p in snap.rglob("*") if not p.is_dir()):
                    blob = os.path.basename(os.path.realpath(f))
                    sha = blob if len(blob) == 64 else "-"       # (an LFS blob is named by its sha256)
                    dest = str(root) + ("#main" if main == snap.name else "")
                    rows.append(("hfcache", dest, f"{repo}@{snap.name}:{f.relative_to(snap).as_posix()}", sha,
                                 str(os.path.getsize(f)), notes.get(repo, "")))
    return rows


def main() -> None:
    path = HERE / "models.tsv"
    head, kept, notes = [], [], {}
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            head.append(line)
            continue
        cols = (line.split("\t") + [""] * 6)[:6]
        if cols[0] == "hfcache":
            repo = cols[2].split("@")[0]
            if cols[5]:
                notes[repo] = cols[5]
        else:
            kept.append(line)
    rows = cache_rows(notes)
    path.write_text("\n".join(head + kept + ["\t".join(r) for r in rows]) + "\n")
    print(f"models.tsv: {len(kept)} file rows kept, {len(rows)} hub-cache rows written from {', '.join(str(r) for r in ROOTS)}")


if __name__ == "__main__":
    main()
