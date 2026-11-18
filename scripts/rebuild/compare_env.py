"""Two conda envs, file by file: is the rebuilt one the original?

    python3 compare_env.py <original prefix> <rebuilt prefix> [--list N]

verify_env.py says "the same packages"; this says "the same files". Every file of both prefixes is
compared by content, and every difference falls in a class:

  identical        the same bytes
  path-only        the same bytes once each env's own path is taken out — a script's shebang, a
                   conda file with the prefix written in it. They are identical when the rebuild
                   is at the original's path, which a real rebuild is (/workspace/miniforge3).
  bookkeeping      the installers' own records of an install: conda-meta/ (dates, the package
                   cache's path), a distribution's RECORD / INSTALLER / REQUESTED / direct_url.json
  DIFFERENT        other content — named, with its package when one owns it
  ONLY IN ORIGINAL / ONLY IN REBUILT

Bytecode (__pycache__, *.pyc) is left out: python writes it again, with the date in it.
Exit status 1 when anything is DIFFERENT or only on one side. Standard library only.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

BOOKKEEPING_NAMES = ("RECORD", "INSTALLER", "REQUESTED", "direct_url.json")


def walk(prefix: Path):
    out = {}
    for root, dirs, files in os.walk(prefix):
        rel_root = os.path.relpath(root, prefix)
        dirs[:] = [d for d in dirs if d != "__pycache__" and not (rel_root == "." and d == "pkgs")]
        for name in files:
            if name.endswith(".pyc"):
                continue
            rel = os.path.normpath(os.path.join(rel_root, name))
            out[rel] = os.path.join(root, name)
        for d in list(dirs):
            full = os.path.join(root, d)
            if os.path.islink(full):
                out[os.path.normpath(os.path.join(rel_root, d))] = full
                dirs.remove(d)
    return out


def content(path: str) -> bytes:
    if os.path.islink(path):
        return b"link:" + os.readlink(path).encode()
    with open(path, "rb") as f:
        return f.read()


def bookkeeping(rel: str) -> bool:
    parts = rel.split(os.sep)
    if parts[0] == "conda-meta" or rel == ".stac_rebuild":
        return True
    return len(parts) >= 2 and parts[-2].endswith((".dist-info", ".egg-info")) and parts[-1] in BOOKKEEPING_NAMES


def without(data: bytes, prefix: str) -> bytes:
    """The content with the env's own path removed (and the NUL padding conda writes after a
    replaced prefix in a binary)."""
    token = prefix.encode()
    if token not in data:
        return data
    pieces = data.split(token)
    return b"<PREFIX>".join(p.lstrip(b"\0") if i else p for i, p in enumerate(pieces)).rstrip(b"\0")


def main(original: str, rebuilt: str, show: int) -> int:
    a, b = walk(Path(original)), walk(Path(rebuilt))
    counts = {}
    listed = {"DIFFERENT": [], "ONLY IN ORIGINAL": [], "ONLY IN REBUILT": []}

    def count(cls, rel=None):
        counts[cls] = counts.get(cls, 0) + 1
        if rel is not None and cls in listed:
            listed[cls].append(rel)

    for rel in sorted(set(a) | set(b)):
        if rel not in b:
            count("bookkeeping" if bookkeeping(rel) else "ONLY IN ORIGINAL", rel)
            continue
        if rel not in a:
            count("bookkeeping" if bookkeeping(rel) else "ONLY IN REBUILT", rel)
            continue
        sa, sb = os.lstat(a[rel]), os.lstat(b[rel])
        if sa.st_size == sb.st_size and not os.path.islink(a[rel]) and sa.st_size > (64 << 20):
            # (a large file: by digest, in blocks)
            ha, hb = hashlib.sha256(), hashlib.sha256()
            for path, h in ((a[rel], ha), (b[rel], hb)):
                with open(path, "rb") as f:
                    for block in iter(lambda: f.read(1 << 22), b""):
                        h.update(block)
            count("identical" if ha.digest() == hb.digest() else "DIFFERENT", rel)
            continue
        ca, cb = content(a[rel]), content(b[rel])
        if ca == cb:
            count("identical")
        elif bookkeeping(rel):
            count("bookkeeping")
        elif without(ca, os.path.normpath(original)) == without(cb, os.path.normpath(rebuilt)):
            count("path-only")
        else:
            count("DIFFERENT", rel)
    order = ("identical", "path-only", "bookkeeping", "DIFFERENT", "ONLY IN ORIGINAL", "ONLY IN REBUILT")
    print("  " + ", ".join(f"{k} {counts.get(k, 0)}" for k in order))
    bad = 0
    for cls, rels in listed.items():
        bad += len(rels)
        for rel in rels[:show]:
            print(f"   {cls:17s} {rel}")
        if len(rels) > show:
            print(f"   {cls:17s} … and {len(rels) - show} more")
    return 1 if bad else 0


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    show = int(sys.argv[sys.argv.index("--list") + 1]) if "--list" in sys.argv else 25
    if "--list" in sys.argv:
        args.remove(str(show))
    if len(args) != 2:
        sys.exit(__doc__)
    sys.exit(main(args[0], args[1], show))
