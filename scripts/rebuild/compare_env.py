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
    """The content with the env's own path taken out. In a binary conda overwrites a long
    placeholder with the prefix and pads what is left of the STRING with NULs, so a shorter or
    longer prefix moves the padding: every run of NULs is read as one."""
    token = prefix.encode()
    if token not in data:
        return data
    import re
    data = data.replace(token, b"<PREFIX>")
    # a script's first line: pip writes `#!<python>` — or, when that path is longer than a kernel
    # accepts in a shebang (127 bytes: a deep test tree, never /workspace/miniforge3), a /bin/sh line
    # that execs it. The same launcher either way.
    data = re.sub(rb"\A#!/bin/sh\n'''exec' \"(<PREFIX>[^\"]*)\" \"\$0\" \"\$@\" #'''\n", rb"#!\1\n", data)
    return re.sub(rb"\0+", b"\0", data)


def classify(rel, a, b, original, rebuilt):
    """The class of one path present on one side or both."""
    if rel not in b:
        return "bookkeeping" if bookkeeping(rel) else "ONLY IN ORIGINAL"
    if rel not in a:
        return "bookkeeping" if bookkeeping(rel) else "ONLY IN REBUILT"
    sa, sb = os.lstat(a[rel]), os.lstat(b[rel])
    if sa.st_size == sb.st_size and not os.path.islink(a[rel]) and sa.st_size > (64 << 20):
        ha, hb = hashlib.sha256(), hashlib.sha256()                # (a large file: by digest, in blocks)
        for path, h in ((a[rel], ha), (b[rel], hb)):
            with open(path, "rb") as f:
                for block in iter(lambda: f.read(1 << 22), b""):
                    h.update(block)
        return "identical" if ha.digest() == hb.digest() else "DIFFERENT"
    ca, cb = content(a[rel]), content(b[rel])
    if ca == cb:
        return "identical"
    if bookkeeping(rel):
        return "bookkeeping"
    if without(ca, os.path.normpath(original)) == without(cb, os.path.normpath(rebuilt)):
        return "path-only"
    return "DIFFERENT"


def main(original: str, rebuilt: str, show: int) -> int:
    from concurrent.futures import ThreadPoolExecutor
    a, b = walk(Path(original)), walk(Path(rebuilt))
    rels = sorted(set(a) | set(b))
    # (tens of thousands of small files, one side on a network volume: many at a time)
    with ThreadPoolExecutor(max_workers=32) as pool:
        classes = list(pool.map(lambda rel: classify(rel, a, b, original, rebuilt), rels))
    counts = {}
    listed = {"DIFFERENT": [], "ONLY IN ORIGINAL": [], "ONLY IN REBUILT": []}
    for rel, cls in zip(rels, classes):
        counts[cls] = counts.get(cls, 0) + 1
        if cls in listed:
            listed[cls].append(rel)
    order = ("identical", "path-only", "bookkeeping", "DIFFERENT", "ONLY IN ORIGINAL", "ONLY IN REBUILT")
    print("  " + ", ".join(f"{k} {counts.get(k, 0)}" for k in order))
    bad = 0
    for cls, found in listed.items():
        bad += len(found)
        for rel in found[:show]:
            print(f"   {cls:17s} {rel}")
        if len(found) > show:
            print(f"   {cls:17s} … and {len(found) - show} more")
    return 1 if bad else 0


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    show = int(sys.argv[sys.argv.index("--list") + 1]) if "--list" in sys.argv else 25
    if "--list" in sys.argv:
        args.remove(str(show))
    if len(args) != 2:
        sys.exit(__doc__)
    sys.exit(main(args[0], args[1], show))
