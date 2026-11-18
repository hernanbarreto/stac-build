"""What pip installed in THIS environment, told apart by how it can be installed again.

Run with the environment's own python (scripts/rebuild/capture.sh does it for every env):

    <env>/bin/python scripts/rebuild/capture_env.py <env name> <locks dir>

`pip freeze` alone is not a recipe: it lists conda's packages too (as ``pkg @ file:///home/conda/…``)
and says nothing of what was compiled on this machine. This reads every distribution's own record:

  INSTALLER         conda → already in ``<env>.conda-explicit.txt``, left out here
  direct_url.json   a wheel by URL, a git commit, or a LOCAL DIRECTORY (editable or built here)
  WHEEL             a tag ``linux_x86_64`` (not manylinux) = possibly built on this machine
  RECORD            the sha256 of every installed file

and writes, in ``<locks dir>``:

  <env>.pip.txt         ``name==version`` for everything pip took from an index
  <env>.pip-url.txt     wheels by URL and git commits, as pip takes them (``name @ url``)
  <env>.pip-local.tsv   what came from a directory of this machine: name, version, how, the
                        directory — each one has its own build step in the rebuild
  <env>.records.tsv     name, version, origin (index / url / git / local), the digest of the
                        files the distribution installed, its wheel tags — verify_env.py recomputes
                        the digest in the rebuilt env: the same digest = the same files, byte for byte
  <env>.scripts.tsv     the launchers pip wrote in bin/ and the interpreter name each starts with
  <env>.overlaps.tsv    winner, loser, files: two packages that installed the same path — the
                        winner (its content is the file's today) is installed LAST by the rebuild
  <env>.modified.tsv    source files that no longer match their RECORD and that no other package
                        overwrote: edited in place after the install (a hot patch a rebuild would
                        silently lose — it must be saved under scripts/rebuild/overlays/<env>/)
"""
from __future__ import annotations

import base64
import hashlib
import json
import sys
from importlib import metadata
from pathlib import Path

AUDITED = (".py", ".json", ".yaml", ".yml", ".cfg", ".toml", ".txt", ".pth")


def record_rows(dist):
    """[(path, 'sha256=…')] of the files a distribution installed inside site-packages, the
    volatile ones left out."""
    text = dist.read_text("RECORD")
    if text is None:
        return None
    rows = []
    for line in text.splitlines():
        parts = line.rsplit(",", 2)
        if len(parts) != 3 or not parts[1]:
            continue                                       # (no hash: a .pyc, RECORD itself)
        path = parts[0].strip('"')
        if path.startswith("..") or ".dist-info/" in path or path.endswith(".pyc"):
            continue                                       # (bin/ scripts carry the env's own shebang;
                                                           #  the dist-info is the installer's own)
        rows.append((path, parts[1]))
    return sorted(rows)


def digest(rows) -> str:
    h = hashlib.sha256()
    for path, file_hash in rows:
        h.update(f"{path}\t{file_hash}\n".encode())
    return h.hexdigest()[:32]


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return "sha256=" + base64.urlsafe_b64encode(h.digest()).decode().rstrip("=")


def all_rows(dist):
    """{path: recorded hash} of EVERY hashed file of a distribution inside site-packages."""
    text = dist.read_text("RECORD") or ""
    out = {}
    for line in text.splitlines():
        parts = line.rsplit(",", 2)
        if len(parts) == 3 and parts[1] and not parts[0].startswith("..") and ".dist-info/" not in parts[0]:
            out[parts[0].strip('"')] = parts[1]
    return out


def overlaps(dists):
    """[(winner, loser, files)] — two distributions that installed the SAME path with different
    content: the one whose recorded hash is the file's today was installed last (pillow-simd over
    pillow). A rebuild installs the winner last too, or the env is not the same."""
    owners = {}
    for name, dist in dists:
        for path, recorded in all_rows(dist).items():
            owners.setdefault(path, []).append((name, recorded, dist))
    pairs = {}
    for path, group in owners.items():
        if len(group) < 2 or len({recorded for _n, recorded, _d in group}) < 2:
            continue
        real = Path(group[0][2].locate_file(path))
        if not real.exists():
            continue
        actual = file_hash(real)
        winners = [n for n, recorded, _d in group if recorded == actual]
        for winner in winners:
            for loser, recorded, _d in group:
                if recorded != actual:
                    pairs[(winner, loser)] = pairs.get((winner, loser), 0) + 1
    return sorted((w, l, n) for (w, l), n in pairs.items())


def describe(dist):
    """(origin, detail) — origin: conda / index / url / git / local."""
    installer = (dist.read_text("INSTALLER") or "").strip().lower()
    if installer == "conda":
        return "conda", ""
    direct = dist.read_text("direct_url.json")
    if direct:
        info = json.loads(direct)
        url = info.get("url", "")
        if "vcs_info" in info:
            if url.startswith("file://"):
                return "local", "editable\t" + url
            sub = f"#subdirectory={info['subdirectory']}" if info.get("subdirectory") else ""
            return "git", f"git+{url}@{info['vcs_info'].get('commit_id', '')}{sub}"
        if url.startswith("file://"):
            if "dir_info" in info:
                return "local", ("editable" if info["dir_info"].get("editable") else "built") + "\t" + url
            return "wheel-file", url                        # a wheel downloaded first, then installed
        archive = info.get("archive_info", {})
        file_digest = archive.get("hash") or "sha256=" + archive.get("hashes", {}).get("sha256", "")
        return "url", url + (f"#{file_digest}" if file_digest.split("=")[-1] else "")
    if dist.read_text("RECORD") is None:
        where = str(getattr(dist, "_path", ""))
        if "site-packages" in where:
            return "conda", ""                              # (an .egg-info conda's package left there)
        return "local", "develop\t" + where                # (setup.py develop)
    return "index", ""


def main(env: str, out_dir: str) -> None:
    out = Path(out_dir)
    index, by_url, local, records, modified, kept, audit = [], [], [], [], [], [], []
    seen = set()
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        origin, detail = describe(dist)
        if origin == "conda":
            continue
        version = dist.version
        kept.append((name, dist))
        if origin in ("index", "wheel-file"):
            index.append(f"{name}=={version}")
        elif origin in ("url", "git"):
            by_url.append(f"{name} @ {detail}")
        else:
            local.append(f"{name}\t{version}\t{detail}")
        rows = record_rows(dist)
        tags = ",".join(line.split(":", 1)[1].strip() for line in (dist.read_text("WHEEL") or "").splitlines()
                        if line.startswith("Tag:"))
        records.append(f"{name}\t{version}\t{origin}\t{digest(rows) if rows else '-'}\t{tags or '-'}")
        for path, recorded in rows or []:
            if path.endswith(AUDITED):
                audit.append((name, version, path, Path(dist.locate_file(path)), recorded))

    def audited(task):
        name, version, path, real, recorded = task
        if not real.exists():
            return f"{name}\t{version}\tMISSING\t{path}"
        return f"{name}\t{version}\tEDITED\t{path}" if file_hash(real) != recorded else None

    # (tens of thousands of small files on a network volume: read them many at a time)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=32) as pool:
        modified = [line for line in pool.map(audited, audit) if line]

    def write(suffix, lines):
        (out / f"{env}.{suffix}").write_text("".join(f"{line}\n" for line in sorted(lines, key=str.lower)))

    # the launchers pip wrote in bin/: `pip install …` gives them #!<prefix>/bin/python3.11, `python -m pip
    # install …` gives #!<prefix>/bin/python — the same interpreter under two names. Which one each has is
    # recorded, so a rebuild writes the same first line.
    import os
    prefix = os.path.normpath(sys.prefix)
    scripts = []
    for name, dist in kept:
        for f in dist.files or []:
            full = os.path.normpath(str(dist.locate_file(f)))
            if os.path.dirname(full) != os.path.join(prefix, "bin") or not os.path.isfile(full):
                continue
            with open(full, "rb") as fh:
                first = fh.readline().strip()
            if first.startswith(b"#!" + prefix.encode() + b"/bin/"):
                scripts.append(f"bin/{os.path.basename(full)}\t{first.rsplit(b'/', 1)[-1].decode()}")
    write("scripts.tsv", sorted(set(scripts)))
    over = overlaps(kept)
    # (a file of a package overwritten by another one's is not a hand edit)
    losers = {loser for _w, loser, _n in over}
    modified = [m for m in modified if m.split("\t")[0] not in losers]
    write("overlaps.tsv", [f"{w}\t{l}\t{n}" for w, l, n in over])
    write("pip.txt", index)
    write("pip-url.txt", by_url)
    write("pip-local.tsv", local)
    write("records.tsv", records)
    write("modified.tsv", modified)
    print(f"{env}: {len(index)} from an index, {len(by_url)} by URL / git, {len(local)} from a local "
          f"directory; {len(over)} overlap(s); {len(modified)} file(s) edited or missing after their install")


def asis(env: str, out_dir: str) -> None:
    """``<env>.asis-files.tsv`` — the paths (relative to the env's prefix) a rebuild cannot download:
    the files and metadata of the distributions installed from a directory or compiled on the pod
    (a source distribution, a wheel that is not the index's), and the files no package owns
    (``<env>.orphans.tsv``, audit_env_files.py). backup.py archives them; restored after the pip
    step they make the env's compiled parts the SAME bytes, not a new compilation."""
    import os
    import re
    out = Path(out_dir)

    def norm(n):
        return re.sub(r"[-_.]+", "-", n).lower()

    names = {norm(line.split("\t")[0]) for line in (out / f"{env}.pip-local.tsv").read_text().splitlines() if line.strip()}
    wheels = out / f"{env}.wheels.tsv"
    if wheels.exists():
        for line in wheels.read_text().splitlines():
            cols = line.split("\t")
            if len(cols) >= 6 and (cols[5] == "sdist" or (len(cols) >= 7 and cols[6] != "identical")):
                names.add(norm(cols[0]))
    # a git commit is built at install too (its WHEEL names the setuptools of that day)
    by_url = out / f"{env}.pip-url.txt"
    if by_url.exists():
        names |= {norm(line.split(" @ ")[0]) for line in by_url.read_text().splitlines() if " @ git+" in line}
    prefix = os.path.normpath(sys.prefix)
    paths = set()
    for dist in metadata.distributions():
        if norm(dist.metadata["Name"] or "") not in names:
            continue
        for f in dist.files or []:
            full = os.path.normpath(str(dist.locate_file(f)))
            if full.startswith(prefix + os.sep) and not full.endswith(".pyc") and os.path.lexists(full):
                paths.add(os.path.relpath(full, prefix))
        info = getattr(dist, "_path", None)
        if info is not None and os.path.isdir(str(info)) and os.path.normpath(str(info)).startswith(prefix + os.sep):
            for root, _dirs, files in os.walk(str(info)):
                for name in files:
                    paths.add(os.path.relpath(os.path.join(root, name), prefix))
    orphans = out / f"{env}.orphans.tsv"
    if orphans.exists():
        for line in orphans.read_text().splitlines():
            rel = line.split("\t")[0]
            if rel and not rel.startswith(("lib/node_modules/", "bin/")):     # (npm's global packages: npm puts them back)
                paths.add(rel)
    (out / f"{env}.asis-files.tsv").write_text("".join(f"{p}\n" for p in sorted(paths)))
    print(f"{env}: {len(paths)} file(s) of {len(names)} distribution(s) to archive as they are")


def scripts(env: str, out_dir: str) -> None:
    """After a rebuild's pip step: every launcher of ``<env>.scripts.tsv`` starts with the
    interpreter name it had on the original pod (see main)."""
    import os
    prefix = os.path.normpath(sys.prefix)
    head = b"#!" + prefix.encode() + b"/bin/"
    fixed = missing = 0
    listing = Path(out_dir) / f"{env}.scripts.tsv"
    for line in listing.read_text().splitlines() if listing.exists() else []:
        rel, interpreter = line.split("\t")
        full = os.path.join(prefix, rel)
        if not os.path.isfile(full):
            missing += 1
            continue
        with open(full, "rb") as fh:
            data = fh.read()
        first, _, rest = data.partition(b"\n")
        if first.startswith(head) and first[len(head):].strip() != interpreter.encode():
            mode = os.stat(full).st_mode
            with open(full, "wb") as fh:
                fh.write(head + interpreter.encode() + b"\n" + rest)
            os.chmod(full, mode)
            fixed += 1
    print(f"{env}: {fixed} launcher(s) set to the interpreter name they had" + (f", {missing} not installed here" if missing else ""))


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "asis":
        asis(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 4 and sys.argv[1] == "scripts":
        scripts(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        sys.exit(__doc__)
