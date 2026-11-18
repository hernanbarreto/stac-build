"""The exact files pip must fetch to rebuild an environment — and the proof that they are the
ones installed here.

    python wheels.py resolve <env> <locks dir> <pip report.json> [<pip report.json> …]
    python wheels.py check   <env> <locks dir> <cache dir> [--jobs N]
    python wheels.py fetch   <env> <locks dir> <cache dir> <out dir> [--jobs N]

resolve   turns pip's own resolution (``pip install --dry-run --no-deps --report``, run by the
          env's pip on ``<env>.pip.txt``) into ``<env>.wheels.tsv``: name, version, sha256, size,
          url, kind (wheel / sdist). From then on the rebuild asks no index anything: it downloads
          these URLs and checks these digests.
check     downloads every file of ``<env>.wheels.tsv`` (content-addressed in ``<cache dir>``),
          checks its sha256, and compares the wheel's own RECORD with the digest of the files
          installed in the live env (``<env>.records.tsv``). Writes the verdict as the last column:
          ``identical`` — the rebuilt package will hold the same files byte for byte;
          ``differs``   — another build of that version is installed here (named, never hidden);
          ``sdist``     — compiled at install: only its version can be held equal.
fetch     downloads and checks the same files into ``<out dir>`` (what rebuild.sh installs from).

Standard library only: it runs with any python ≥ 3.8, the bare system's included.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import unquote, urlsplit

COLUMNS = ("name", "version", "sha256", "size", "url", "kind")


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def filename_of(url: str) -> str:
    return unquote(Path(urlsplit(url).path).name)


# ── resolve ────────────────────────────────────────────────────────────────────────────────────
def resolve(env: str, locks: Path, reports) -> int:
    rows = {}
    for report in reports:
        data = json.loads(Path(report).read_text())
        for item in data.get("install", []):
            info = item["download_info"]
            archive = info.get("archive_info", {})
            sha = archive.get("hashes", {}).get("sha256") or archive.get("hash", "").split("=")[-1]
            if not sha:
                raise SystemExit(f"{env}: pip reported no sha256 for {info.get('url')}")
            name, version = item["metadata"]["name"], item["metadata"]["version"]
            kind = "wheel" if filename_of(info["url"]).endswith(".whl") else "sdist"
            rows[norm(name)] = (name, version, sha, "0", info["url"].split("#")[0], kind)
    wanted = [line.strip() for line in (locks / f"{env}.pip.txt").read_text().splitlines() if line.strip()]
    missing = [w for w in wanted if norm(w.split("==")[0]) not in rows]
    if missing:
        raise SystemExit(f"{env}: pip resolved nothing for {missing}")
    for w in wanted:
        name, version = w.split("==")
        got = rows[norm(name)][1]
        if got != version:
            raise SystemExit(f"{env}: {name} is pinned at {version} and pip resolved {got}")
    write(locks / f"{env}.wheels.tsv", [rows[k] for k in sorted(rows)])
    print(f"{env}: {len(rows)} file(s) resolved ({sum(r[5] == 'sdist' for r in rows.values())} sdist)")
    return 0


def write(path: Path, rows) -> None:
    path.write_text("".join("\t".join(str(c) for c in row) + "\n" for row in rows))


def read(path: Path):
    return [line.split("\t") for line in path.read_text().splitlines() if line.strip()]


# ── download ───────────────────────────────────────────────────────────────────────────────────
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, sha: str, cache: Path) -> Path:
    """The file of ``url`` in the cache, its sha256 checked (a wrong or partial file is fetched again)."""
    dest = cache / sha[:2] / sha / filename_of(url)
    if dest.exists() and sha256_of(dest) == sha:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(4):
        run = subprocess.run(["curl", "-sSfL", "--retry", "5", "--retry-delay", "3", "--connect-timeout", "30",
                              "-C", "-", "-o", str(part), url], capture_output=True, text=True)
        if run.returncode == 0 and sha256_of(part) == sha:
            part.replace(dest)
            return dest
        if run.returncode != 0 and attempt == 3:
            raise RuntimeError(f"{url}: {run.stderr.strip() or 'download failed'}")
        part.unlink(missing_ok=True)                       # (a corrupt or foreign partial: start over)
    raise RuntimeError(f"{url}: the file's sha256 is not {sha}")


def fetch_all(rows, cache: Path, jobs: int):
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(lambda r: download(r[4], r[2], cache), rows))


# ── check ──────────────────────────────────────────────────────────────────────────────────────
def wheel_digest(path: Path) -> str:
    """The digest capture_env.py computes for an INSTALLED distribution, from the wheel itself."""
    with zipfile.ZipFile(path) as z:
        record = next(n for n in z.namelist() if n.endswith(".dist-info/RECORD") and n.count("/") == 1)
        rows = []
        for line in z.read(record).decode().splitlines():
            parts = line.rsplit(",", 2)
            if len(parts) != 3 or not parts[1]:
                continue
            p = parts[0].strip('"')
            m = re.match(r"^[^/]+\.data/(purelib|platlib)/(.+)$", p)
            if m:
                p = m.group(2)                             # (installed straight into site-packages)
            elif re.match(r"^[^/]+\.data/", p):
                continue                                   # (scripts / data / headers: outside it)
            if ".dist-info/" in p or p.endswith(".pyc"):
                continue
            rows.append((p, parts[1]))
    h = hashlib.sha256()
    for p, file_hash in sorted(rows):
        h.update(f"{p}\t{file_hash}\n".encode())
    return h.hexdigest()[:32]


EMPTY = hashlib.sha256().hexdigest()[:32]                    # (a distribution with no file of its own)


def same(path: Path, installed_digest: str) -> bool:
    got = wheel_digest(path)
    return got == installed_digest or (installed_digest == "-" and got == EMPTY)


def other_builds(env: str, locks: Path, name: str, version: str):
    """[(url, sha256)] — the builds of that version on PyTorch's index of the env's CUDA label.
    A package pulled in by ``pip install torch --index-url …/whl/cu128`` came from THAT index, and
    its build of a version is not always PyPI's (triton: two different files, one version)."""
    labels = sorted(set(re.findall(r"\+(cu\d+)$", (locks / f"{env}.pip.txt").read_text(), re.M))) or ["cu128"]
    out = []
    for label in labels:
        page = subprocess.run(["curl", "-fsSL", "--max-time", "60", f"https://download.pytorch.org/whl/{label}/{norm(name)}/"],
                              capture_output=True, text=True).stdout
        for href, sha in re.findall(r'href="([^"#]+)#sha256=([0-9a-f]{64})"', page):
            file = filename_of(href)
            parts = file[:-4].split("-")
            if not file.endswith(".whl") or len(parts) < 5 or parts[1] != version:
                continue
            if "x86_64" not in parts[-1] and parts[-1] != "any":
                continue
            out.append((href if href.startswith("http") else "https://download.pytorch.org" + href, sha))
    return out


def check(env: str, locks: Path, cache: Path, jobs: int) -> int:
    rows = [r[:6] for r in read(locks / f"{env}.wheels.tsv")]
    installed = {norm(r[0]): r for r in read(locks / f"{env}.records.tsv")}
    files = fetch_all(rows, cache, jobs)
    py = None
    for r in installed.values():
        m = re.search(r"\b(cp3\d+)-", r[4] if len(r) > 4 else "")
        if m:
            py = m.group(1)
            break
    out, differs = [], []
    for row, path in zip(rows, files):
        row[3] = str(path.stat().st_size)
        want = installed[norm(row[0])][3]
        if row[5] == "sdist":
            verdict = "sdist"
        else:
            verdict = "identical" if same(path, want) else "differs"
        if verdict == "differs":
            # the same version from the index the package really came from
            for url, sha in other_builds(env, locks, row[0], row[1]):
                tag = filename_of(url)[:-4].split("-")[2]
                if sha == row[2] or (tag.startswith("cp3") and py and tag != py):
                    continue
                candidate = download(url, sha, cache)
                if same(candidate, want):
                    row[2], row[3], row[4], verdict = sha, str(candidate.stat().st_size), url, "identical"
                    break
        if verdict == "differs":
            differs.append(f"{row[0]}=={row[1]}  resolved {filename_of(row[4])}  installed tags "
                           f"{installed[norm(row[0])][4] if len(installed[norm(row[0])]) > 4 else '?'}")
        out.append(row + [verdict])
    write(locks / f"{env}.wheels.tsv", out)
    total = sum(int(r[3]) for r in out) / 1e9
    print(f"{env}: {len(out)} file(s), {total:.2f} GB — identical {sum(r[6] == 'identical' for r in out)}, "
          f"sdist {sum(r[6] == 'sdist' for r in out)}, DIFFERS {len(differs)}")
    for line in differs:
        print(f"   DIFFERS  {line}")
    return 0


def fetch(env: str, locks: Path, cache: Path, out_dir: Path, jobs: int) -> int:
    rows = read(locks / f"{env}.wheels.tsv")
    files = fetch_all(rows, cache, jobs)
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in files:
        dest = out_dir / path.name
        if not dest.exists():
            try:
                dest.hardlink_to(path)
            except OSError:
                shutil.copy2(path, dest)
    print(f"{env}: {len(files)} file(s) in {out_dir}")
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 16
    if "--jobs" in sys.argv:
        args.remove(str(jobs))
    if len(args) >= 4 and args[0] == "resolve":
        sys.exit(resolve(args[1], Path(args[2]), args[3:]))
    if len(args) == 4 and args[0] == "check":
        sys.exit(check(args[1], Path(args[2]), Path(args[3]), jobs))
    if len(args) == 5 and args[0] == "fetch":
        sys.exit(fetch(args[1], Path(args[2]), Path(args[3]), Path(args[4]), jobs))
    sys.exit(__doc__)
