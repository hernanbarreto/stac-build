"""Every file of a conda env that NO package put there, and every conda-installed source file that
was edited afterwards.

    python3 audit_env_files.py <env prefix> <env name> <locks dir>

The locks describe an env as its packages (conda's exact list, pip's exact wheels). That is the
whole env only if nothing else lives in it — a file copied in by hand, a tool installed by another
installer, a model downloaded into site-packages, a library patched in place would be lost by a
rebuild without anyone noticing. This walks the prefix and names them:

  <env>.orphans.tsv          path (relative to the prefix), bytes — files owned by neither a conda
                             package (conda-meta/*.json) nor a pip distribution (its RECORD)
  <env>.conda-modified.tsv   path, package — text files of a conda package whose sha256 is no
                             longer the one conda recorded (capture_env.py does the same for pip's)

Left out on purpose, as what any install regenerates: bytecode (__pycache__, *.pyc), conda's own
bookkeeping (conda-meta/, .stac_rebuild), and what pip itself writes for a distribution
(direct_url.json, INSTALLER, REQUESTED, RECORD of an editable install).
Standard library only; runs with any python ≥ 3.8 (the env's own is not needed).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

AUDITED = (".py", ".pth", ".json", ".yaml", ".yml", ".cfg", ".toml", ".txt", ".sh")
REGENERATED_DIRS = {"__pycache__"}


def conda_owned(prefix: Path):
    """{relative path: (package, sha256 or None)} of every file conda's packages installed."""
    owned = {}
    for meta in sorted((prefix / "conda-meta").glob("*.json")):
        try:
            data = json.loads(meta.read_text())
        except (OSError, ValueError):
            continue
        name = data.get("name", meta.stem)
        hashes = {p.get("_path"): p.get("sha256_in_prefix") or p.get("sha256")
                  for p in (data.get("paths_data") or {}).get("paths", [])
                  if not p.get("prefix_placeholder")}            # (a file conda rewrites at install has its own hash)
        for rel in data.get("files", []):
            owned[rel] = (name, hashes.get(rel))
    return owned


def pip_owned(prefix: Path):
    """{relative path} of every file a pip distribution recorded (RECORD paths are relative to
    site-packages; '..' entries reach bin/, share/ …)."""
    owned = set()
    for site in list(prefix.glob("lib/python3*/site-packages")):
        for info in list(site.glob("*.dist-info")) + list(site.glob("*.egg-info")):
            owned.add(os.path.relpath(info, prefix))               # (the metadata directory itself)
            record = info / "RECORD"
            if record.is_file():
                with open(record, newline="") as f:
                    for row in csv.reader(f):
                        if row:
                            owned.add(os.path.normpath(os.path.relpath(site / row[0], prefix)))
            for extra in ("installed-files.txt",):
                listing = info / extra
                if listing.is_file():
                    for line in listing.read_text().splitlines():
                        owned.add(os.path.normpath(os.path.relpath(info / line.strip(), prefix)))
    return owned


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(prefix: str, env: str, locks: str) -> None:
    root = Path(prefix)
    conda = conda_owned(root)
    pip = pip_owned(root)
    orphans, modified = [], []
    for directory, dirs, files in os.walk(root):
        rel_dir = os.path.relpath(directory, root)
        dirs[:] = [d for d in dirs if d not in REGENERATED_DIRS and not (rel_dir == "." and d in ("conda-meta", "pkgs"))]
        for name in files:
            rel = os.path.normpath(os.path.join(rel_dir, name))
            if name.endswith(".pyc") or rel == ".stac_rebuild":
                continue
            if rel in conda:
                package, recorded = conda[rel]
                if recorded and name.endswith(AUDITED) and rel not in pip:
                    full = root / rel
                    if full.is_file() and not full.is_symlink() and sha256_of(full) != recorded:
                        modified.append(f"{rel}\t{package}")
                continue
            if rel in pip or any(rel.startswith(p + os.sep) for p in pip if p.endswith(("-info",))):
                continue
            full = root / rel
            try:
                size = full.lstat().st_size
            except OSError:
                size = -1
            orphans.append(f"{rel}\t{size}")
    out = Path(locks)
    (out / f"{env}.orphans.tsv").write_text("".join(f"{line}\n" for line in sorted(orphans)))
    (out / f"{env}.conda-modified.tsv").write_text("".join(f"{line}\n" for line in sorted(modified)))
    print(f"{env}: {len(orphans)} file(s) no package owns, {len(modified)} conda file(s) edited after their install")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
