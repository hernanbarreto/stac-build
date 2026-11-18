"""Is THIS environment the one the lock describes? Run with the env's own python:

    <env>/bin/python scripts/rebuild/verify_env.py <env name> <locks dir>

For every distribution pip installed on the original pod (``<env>.records.tsv``) it recomputes,
here, the digest capture_env.py took there (the RECORD of the installed wheel: every file's path
and sha256):

  identical   the same wheel: the same files, byte for byte
  DIFFERS     the version is right and the files are not (another build of that version)
  VERSION     another version is installed
  MISSING     not installed
  built       compiled on each machine (a source distribution, an editable tree, an extension):
              only its presence and version can be held equal

then two things a RECORD does not tell: that the files patched in place on the original pod
(``overlays/<env>/``) are the ones on disk, and that where two packages wrote the same path the
same one won (``<env>.overlaps.tsv``).

Exit status 1 when anything is DIFFERS / VERSION / MISSING / a wrong overlay / another winner.
"""
from __future__ import annotations

import re
import sys
import sysconfig
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_env import describe, digest, overlaps, record_rows  # noqa: E402


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def main(env: str, locks: str) -> int:
    locks_dir = Path(locks)
    here, kept = {}, []
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        if name and norm(name) not in here:
            here[norm(name)] = dist
            if describe(dist)[0] != "conda":
                kept.append((name, dist))
    sdists = set()
    wheels = locks_dir / f"{env}.wheels.tsv"
    if wheels.exists():
        for line in wheels.read_text().splitlines():
            cols = line.split("\t")
            if len(cols) >= 6 and cols[5] == "sdist":
                sdists.add(norm(cols[0]))

    counts, bad = {}, []
    for line in (locks_dir / f"{env}.records.tsv").read_text().splitlines():
        name, version, origin, want = line.split("\t")[:4]
        dist = here.get(norm(name))
        if dist is None:
            state = "MISSING"
        elif dist.version != version:
            state = f"VERSION ({dist.version} here)"
        elif origin == "local" or norm(name) in sdists or want == "-":
            state = "built"
        else:
            state = "identical" if digest(record_rows(dist) or []) == want else "DIFFERS"
        key = state.split(" ")[0]
        counts[key] = counts.get(key, 0) + 1
        if key in ("MISSING", "VERSION", "DIFFERS"):
            bad.append(f"   {state:10s} {name}=={version}")

    overlay_root = Path(__file__).resolve().parent / "overlays" / env
    if overlay_root.is_dir():
        site = Path(sysconfig.get_paths()["purelib"])
        for src in sorted(p for p in overlay_root.rglob("*") if p.is_file()):
            rel = src.relative_to(overlay_root)
            ok = (site / rel).exists() and (site / rel).read_bytes() == src.read_bytes()
            counts["overlay" if ok else "OVERLAY-WRONG"] = counts.get("overlay" if ok else "OVERLAY-WRONG", 0) + 1
            if not ok:
                bad.append(f"   OVERLAY    {rel} is not the patched file of scripts/rebuild/overlays/{env}/")

    lock_over = locks_dir / f"{env}.overlaps.tsv"
    want_pairs = {tuple(line.split("\t")[:2]) for line in lock_over.read_text().splitlines() if line.strip()} if lock_over.exists() else set()
    have_pairs = {(w, l) for w, l, _n in overlaps(kept)}
    for winner, loser in sorted(want_pairs - have_pairs):
        bad.append(f"   OVERLAP    {winner} must win over {loser} (as on the original pod) and does not")
    for winner, loser in sorted(have_pairs - want_pairs):
        bad.append(f"   OVERLAP    {winner} overwrote {loser} here and did not on the original pod")

    summary = ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
    print(f"{env}: {summary}" + (f", overlaps as recorded ({len(want_pairs)})" if want_pairs == have_pairs else ""))
    for line in bad:
        print(line)
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
