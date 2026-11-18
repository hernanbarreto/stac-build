"""The rebuild kit (scripts/rebuild/) holds together, and — on the pod — still describes the pod.

USER 2026-11-18: "debe quedar exactamente como está ahora y cada cosa que avancemos debemos
actualizarlo para que siempre quede actualizado a la perfección". The kit is a description of the
pod: an install, a clone, a new weight that nobody captured makes it false without any error. The
second half of this file is what turns that into a failing test (the fast comparison: package
lists, vendor commits, the weights' presence — `python scripts/rebuild/drift.py` is the whole one).
"""
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
KIT = REPO / "scripts" / "rebuild"
LOCKS = KIT / "locks"
sys.path.insert(0, str(KIT))

ON_THE_POD = Path("/workspace/miniforge3/envs").is_dir() and REPO == Path("/workspace/stac-build")


def _envs():
    return re.search(r"^STAC_ENVS=\(([^)]*)\)", (KIT / "lib.sh").read_text(), re.M).group(1).split()


def _norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


# ── the kit by itself ──────────────────────────────────────────────────────────────────────────
def test_every_script_parses():
    for script in sorted(KIT.glob("*.sh")):
        run = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
        assert run.returncode == 0, f"{script.name}: {run.stderr}"
    for script in sorted(KIT.glob("*.py")):
        compile(script.read_text(), str(script), "exec")


def test_every_env_has_its_locks():
    for env in _envs():
        conda = LOCKS / f"{env}.conda-explicit.txt"
        assert conda.exists() and "@EXPLICIT" in conda.read_text(), f"{env}: no conda lock"
        urls = [l for l in conda.read_text().splitlines() if l.startswith("https")]
        assert urls and all(re.search(r"#[0-9a-f]{32}$", u) for u in urls), f"{env}: a conda package without its md5"
        if env == "nodejs":
            continue
        pins = {(_norm(l.split("==")[0]), l.split("==")[1]) for l in (LOCKS / f"{env}.pip.txt").read_text().split()}
        rows = [l.split("\t") for l in (LOCKS / f"{env}.wheels.tsv").read_text().splitlines() if l.strip()]
        assert {(_norm(r[0]), r[1]) for r in rows} == pins, f"{env}: wheels.tsv is not the pins of pip.txt — capture.sh {env}"
        for r in rows:
            assert re.fullmatch(r"[0-9a-f]{64}", r[2]) and r[4].startswith("https://"), f"{env}: {r[0]} has no file"
            assert len(r) >= 7 and r[6] in ("identical", "sdist", "differs"), f"{env}: {r[0]} was never checked (CHECK=1)"
        # what a rebuild cannot download is what the backup set archives
        archived = {_norm(m.group(1)) for line in (LOCKS / f"{env}.asis-files.tsv").read_text().splitlines()
                    for m in [re.search(r"/([^/]+)-[^/-]+\.dist-info/", line)] if m}
        for r in rows:
            if r[6] != "identical":
                assert _norm(r[0]) in archived, f"{env}: {r[0]} is not downloadable as installed and is not archived"


def test_overlays_are_the_edited_files():
    """A file edited in place inside an env is saved under overlays/<env>/ — and nothing else is."""
    for env in _envs():
        modified = LOCKS / f"{env}.modified.tsv"
        edited = {l.split("\t")[3] for l in modified.read_text().splitlines() if l.strip()} if modified.exists() else set()
        root = KIT / "overlays" / env
        saved = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} if root.is_dir() else set()
        assert edited == saved, f"{env}: edited in place {sorted(edited)}, saved as overlays {sorted(saved)}"


def test_the_model_manifest_is_complete_rows():
    import models
    rows = models.rows()
    assert rows
    for row in rows:
        assert row["kind"] in ("hf", "hfcache", "url", "targz", "dinov3", "tum", "link", "copy", "convert"), row
        if row["kind"] in ("hf", "url", "targz", "dinov3", "tum"):
            assert re.fullmatch(r"[0-9a-f]{64}", row["sha"]) and row["size"] > 0, f"{row['dest']}: no sha256 / size"
        if row["kind"] in ("hf", "hfcache"):
            repo, revision, name = models.hf_parts(row["source"])
            assert re.fullmatch(r"[0-9a-f]{40}", revision), f"{row['source']}: the revision is not pinned"
        if row["kind"] == "hfcache":
            assert row["dest"].partition("#")[0].startswith("/"), f"{row['source']}: no cache root"
    # a sidecar STAC writes beside a weight is not a file of the model's repository
    assert not [r for r in rows if r["dest"].endswith(".stac_sha256.json")]


def test_vendor_pins_and_bundles():
    text = (KIT / "steps_vendors.sh").read_text()
    rows = re.findall(r'^\s*"([A-Za-z0-9._-]+)\|(https://[^|]+)\|([0-9a-f]{40})\|([^|]*)\|([0-9a-f]{40})\|([^|]*)\|(yes|no)"', text, re.M)
    assert len(rows) >= 8
    for name, _url, base, branch, final, patch, _rec in rows:
        if branch != "-":
            assert (KIT / "set" / "git" / f"stac-build__vendor__{name}.bundle").exists(), f"{name}: its own commits are in no bundle"
        else:
            assert base == final, f"{name}: no bundle and two commits"
        if patch != "-":
            assert (REPO / patch).exists(), f"{name}: {patch} is missing"


def test_the_inventory_names_no_unexplained_large_file():
    import inventory
    rows = inventory.read(LOCKS / "inventory.tsv")
    classes = {r[0] for r in rows}
    assert {"asis", "weight", "download", "private", "data", "git-head"} <= classes
    assert not inventory.large_asis([(r[0], r[1], r[2], r[3], r[4] if len(r) > 4 else "") for r in rows])
    # a private file's digest is never written in the committed inventory
    assert all(r[3] == "-" or r[3].endswith("files") for r in rows if r[0] == "private")


# ── the kit against the pod it describes ───────────────────────────────────────────────────────
@pytest.mark.skipif(not ON_THE_POD, reason="the comparison needs the pod the kit was captured from")
def test_the_pod_is_still_the_kit():
    """The fast comparison (no file is hashed): every env's conda and pip packages, every vendor
    tree's commit and diff, every weight's presence and size, the hub caches' content. A failure
    here = something was installed, cloned or downloaded and not captured:
    `bash scripts/rebuild/capture.sh`, commit scripts/rebuild/, copy the backup set off the pod."""
    import drift
    problems = [p for env in drift.envs() for p in drift.env_fast(env)] + drift.vendors() + drift.weights()
    assert not problems, "the pod is no longer what scripts/rebuild describes:\n  " + "\n  ".join(problems[:40])
