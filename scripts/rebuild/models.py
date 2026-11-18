"""Every model weight of the pod, put back where it was from where it came — checked by sha256.

    python3 models.py fetch   [--jobs N] [--only KIND,KIND]      download / link / copy what is missing
    python3 models.py verify                                     check every file against the manifest
    python3 models.py list                                       what the manifest holds, by source

The manifest is ``models.tsv`` beside this file (kind, dest, source, sha256, bytes, note). A file
already in place with the right sha256 is never fetched again; a file whose sha256 is wrong is an
ERROR, never kept. Standard library + curl: no environment is needed, so the weights download
while the environments are being built.

Environment: STAC_ROOT, HF_HOME (the cache), HF_TOKEN (the gated repos: SAM 3 / 3.1, VGGT-Omega),
DINOV3_URL (Meta's signed link — any of the two files' links, or the ``*`` form: it expires, ask
for a new one at https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/),
STAC_MIRROR (optional: a directory or an http(s) base holding files named by their sha256 — tried
FIRST, for what the official source may no longer give).
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("STAC_ROOT") or HERE.parent.parent)
HF_HOME = Path(os.environ.get("HF_HOME") or "/workspace/hf_cache")
TMP = Path(os.environ.get("STAC_REBUILD_CACHE") or "/tmp/stac_rebuild") / "models"
MIRROR = os.environ.get("STAC_MIRROR", "").rstrip("/")
TUM_PARTS = 32


def rows():
    out = []
    for line in (HERE / "models.tsv").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        kind, dest, source, sha, size, note = (line.split("\t") + [""] * 6)[:6]
        out.append({"kind": kind, "dest": dest, "source": source, "sha": sha, "size": int(size or 0), "note": note})
    return out


def path_of(dest: str) -> Path:
    return Path(dest) if dest.startswith("/") else ROOT / dest


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def git_blob_id(path: Path) -> str:
    h = hashlib.sha1(b"blob %d\0" % path.stat().st_size)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def curl(url: str, dest: Path, headers=(), extra=()) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["curl", "-fL", "-sS", "--retry", "8", "--retry-delay", "5", "--retry-all-errors", "--connect-timeout", "30",
           "-C", "-", "-o", str(dest), *extra]
    for h in headers:
        cmd += ["-H", h]
    run = subprocess.run(cmd + [url], capture_output=True, text=True)
    if run.returncode != 0:
        # (416: the partial file is already whole — the check that follows decides)
        if "416" not in run.stderr:
            raise RuntimeError(f"{url.split('?')[0]}: {run.stderr.strip()[-300:] or 'download failed'}")


def hf_parts(source: str):
    repo, rest = source.split("@", 1)
    revision, name = rest.split(":", 1)
    return repo, revision, name


def hf_headers():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return [f"Authorization: Bearer {token}"] if token else []


def from_mirror(row, part: Path) -> bool:
    """The file from the mirror when one is configured and holds it (named by its sha256)."""
    if not MIRROR or row["sha"] in ("", "-"):
        return False
    try:
        if MIRROR.startswith("http"):
            curl(f"{MIRROR}/{row['sha']}", part)
        else:
            src = Path(MIRROR) / row["sha"]
            if not src.exists():
                return False
            shutil.copy2(src, part)
        return part.exists() and sha256_of(part) == row["sha"]
    except RuntimeError:
        part.unlink(missing_ok=True)
        return False


def tum_ranged(url: str, part: Path, size: int) -> None:
    """GenRecon's server gives ~0.25 MB/s a connection: the file in TUM_PARTS ranges at once."""
    chunk = -(-size // TUM_PARTS)

    def piece(i: int) -> Path:
        start, end = i * chunk, min(size, (i + 1) * chunk) - 1
        p = part.with_name(f"{part.name}.{i:02d}")
        for _ in range(8):
            have = p.stat().st_size if p.exists() else 0
            if have >= end - start + 1:
                break
            subprocess.run(["curl", "-fsS", "--connect-timeout", "30", "--max-time", "7200", "-r", f"{start + have}-{end}", url],
                           stdout=open(p, "ab"), stderr=subprocess.DEVNULL)
        if p.stat().st_size != end - start + 1:
            raise RuntimeError(f"{url}: range {i} holds {p.stat().st_size} bytes of {end - start + 1}")
        return p

    with ThreadPoolExecutor(max_workers=TUM_PARTS) as pool:
        pieces = list(pool.map(piece, range(TUM_PARTS)))
    with open(part, "wb") as out:
        for p in pieces:
            with open(p, "rb") as f:
                shutil.copyfileobj(f, out, 1 << 24)
    for p in pieces:
        p.unlink()


def dinov3_url(name: str) -> str:
    """Meta's signed link for one file. DINOV3_URL is the link Meta mails: with a ``*`` for the
    file, or the link of one of the files (its file part is swapped — the signature covers all)."""
    base = os.environ.get("DINOV3_URL", "")
    if not base:
        raise RuntimeError("DINOV3_URL is not set (.env): ask Meta for a link at "
                           "https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/")
    if "*" in base:
        return base.replace("*", name)
    head, _, query = base.partition("?")
    origin = re.match(r"^(https?://[^/]+)/", head).group(1)
    return f"{origin}/{name}?{query}"


def hf_cache_place(row, blob_src: Path) -> None:
    """Put a downloaded file into the hub cache the way huggingface_hub lays it out: blobs/<id>,
    snapshots/<revision>/<path> → the blob, and refs/main when today's cache has it."""
    repo, revision, name = hf_parts(row["source"])
    base = HF_HOME / "hub" / ("models--" + repo.replace("/", "--"))
    blob_id = row["sha"] if row["sha"] not in ("", "-") else git_blob_id(blob_src)
    blob = base / "blobs" / blob_id
    blob.parent.mkdir(parents=True, exist_ok=True)
    if not blob.exists():
        shutil.move(str(blob_src), str(blob))
    else:
        blob_src.unlink(missing_ok=True)
    link = base / "snapshots" / revision / name
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(os.path.relpath(blob, link.parent))
    if row["dest"] == "ref=main":
        (base / "refs").mkdir(exist_ok=True)
        (base / "refs" / "main").write_text(revision)


def hf_cache_file(row) -> Path:
    repo, revision, name = hf_parts(row["source"])
    return HF_HOME / "hub" / ("models--" + repo.replace("/", "--")) / "snapshots" / revision / name


def target(row) -> Path:
    return hf_cache_file(row) if row["kind"] == "hfcache" else path_of(row["dest"])


def in_place(row, deep: bool) -> bool:
    p = target(row)
    if row["kind"] == "link":
        return p.is_symlink() and os.readlink(p).rstrip("/") == row["source"].rstrip("/")
    if not p.exists():
        return False
    if row["kind"] == "copy":
        src = path_of(row["source"])
        return src.exists() and p.read_bytes() == src.read_bytes()
    if row["size"] and p.stat().st_size != row["size"]:
        return False
    if row["sha"] in ("", "-") or not deep:
        return True
    return sha256_of(p) == row["sha"]


def fetch_one(row) -> str:
    kind, p = row["kind"], target(row)
    if kind in ("link", "copy", "convert"):
        return "later"
    if in_place(row, deep=True):
        return "present"
    part = TMP / (row["sha"] if row["sha"] not in ("", "-") else re.sub(r"[^A-Za-z0-9._-]", "_", row["source"]))
    part.parent.mkdir(parents=True, exist_ok=True)
    if not from_mirror(row, part):
        if kind in ("hf", "hfcache"):
            repo, revision, name = hf_parts(row["source"])
            curl(f"https://huggingface.co/{repo}/resolve/{revision}/{name}", part, hf_headers())
        elif kind == "url":
            curl(row["source"], part)
        elif kind == "targz":
            archive = part.with_name(part.name + ".tar.gz")
            curl(row["source"], archive)
            with tarfile.open(archive) as tar:
                member = next(m for m in tar.getmembers() if Path(m.name).name == p.name)
                with tar.extractfile(member) as src, open(part, "wb") as out:
                    shutil.copyfileobj(src, out, 1 << 24)
            archive.unlink()
        elif kind == "tum":
            tum_ranged(row["source"], part, row["size"])
        elif kind == "dinov3":
            curl(dinov3_url(row["source"]), part)
        else:
            raise RuntimeError(f"unknown kind {kind!r}")
    if row["sha"] not in ("", "-"):
        got = sha256_of(part)
        if got != row["sha"]:
            part.unlink()
            raise RuntimeError(f"{row['source'].split('?')[0]}: sha256 {got}, the manifest says {row['sha']}")
    elif row["size"] and part.stat().st_size != row["size"]:
        raise RuntimeError(f"{row['source']}: {part.stat().st_size} bytes, the manifest says {row['size']}")
    if kind == "hfcache":
        hf_cache_place(row, part)
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(part), str(p))
    return "fetched"


def place_links_and_copies(all_rows) -> list:
    failed = []
    for row in all_rows:
        p = path_of(row["dest"])
        try:
            if row["kind"] == "link" and not in_place(row, False):
                p.parent.mkdir(parents=True, exist_ok=True)
                if p.is_symlink() or p.exists():
                    p.unlink()
                p.symlink_to(row["source"])
            elif row["kind"] == "copy" and not in_place(row, False):
                p.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path_of(row["source"]), p)
        except Exception as e:                                  # reported with the rest, by name
            failed.append((row, str(e)))
    return failed


def fetch(jobs: int, only) -> int:
    all_rows = [r for r in rows() if not only or r["kind"] in only]
    work = [r for r in all_rows if r["kind"] not in ("link", "copy", "convert")]
    # the slow server first and alone on its connections; the rest in parallel
    results = {}

    def run(row):
        try:
            return row, fetch_one(row), ""
        except Exception as e:
            return row, "FAILED", str(e)

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for row, state, why in pool.map(run, sorted(work, key=lambda r: (r["kind"] != "tum", -r["size"]))):
            results[id(row)] = (state, why)
            label = row["dest"] if row["kind"] != "hfcache" else row["source"]
            print(f"  {state:8s} {row['size'] / 1e9:7.2f} GB  {label}" + (f"\n           {why}" if why else ""), flush=True)
    failed = [(r, results[id(r)][1]) for r in work if results[id(r)][0] == "FAILED"]
    failed += place_links_and_copies(all_rows)
    done = sum(results[id(r)][0] in ("present", "fetched") for r in work)
    print(f"models: {done} of {len(work)} file(s) in place, {len(failed)} FAILED")
    for row, why in failed:
        print(f"  FAILED {row['dest'] if row['kind'] != 'hfcache' else row['source']}: {why}")
    return 1 if failed else 0


def verify() -> int:
    bad = 0
    for row in rows():
        ok = in_place(row, deep=True)
        if row["kind"] == "convert" and not target(row).exists():
            ok = False
        bad += not ok
        if not ok:
            print(f"  MISSING OR WRONG  [{row['kind']}] {row['dest'] if row['kind'] != 'hfcache' else row['source']}  {row['note']}")
    total = len(rows())
    print(f"models: {total - bad} of {total} entries verified" + ("" if not bad else f" — {bad} MISSING OR WRONG"))
    return 1 if bad else 0


def listing() -> int:
    by = {}
    for row in rows():
        by.setdefault(row["kind"], []).append(row)
    for kind, group in sorted(by.items()):
        print(f"{kind:8s} {len(group):3d} entries {sum(r['size'] for r in group) / 1e9:7.2f} GB")
    return 0


if __name__ == "__main__":
    argv = sys.argv[1:]
    jobs = int(argv[argv.index("--jobs") + 1]) if "--jobs" in argv else 6
    only = set(argv[argv.index("--only") + 1].split(",")) if "--only" in argv else None
    if argv[:1] == ["fetch"]:
        sys.exit(fetch(jobs, only))
    if argv[:1] == ["verify"]:
        sys.exit(verify())
    if argv[:1] == ["list"]:
        sys.exit(listing())
    sys.exit(__doc__)
