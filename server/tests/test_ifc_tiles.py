"""bim/ifc_tiles.py — the viewer's tiles of a project's IFC (USER 2026-11-28: "tipo los tiles del glb?" / "si, mejora todo"
/ "debe funcionar siempre incluso para futuras sesiones"): made when absent, made again for a new version of the file
(uploaded again, regenerated, rescaled in place), a failure recorded for its version and not retried in a loop, files
served only under the version on disk. The converter is stood in for by a small script (the real one is node's,
tools/glb/ifc_tiles.mjs, checked on real IFCs)."""
import json
import os
import stat
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bim import ifc_tiles  # noqa: E402

FAKE = r'''
import json, os, sys, time
ifc, out = sys.argv[1], sys.argv[2]
if os.environ.get("FAKE_FAIL"):
    print("[ifc_tiles] FAILED: the fake was told to fail"); sys.exit(1)
time.sleep(float(os.environ.get("FAKE_SLEEP", "0")))
st = os.stat(ifc)
os.makedirs(out, exist_ok=True)
open(os.path.join(out, "tileset.json"), "w").write('{"root": {}}')
open(os.path.join(out, "elements.json"), "w").write('{"parts": [], "hierarchy": []}')
open(os.path.join(out, "tile_0.glb"), "wb").write(b"glTF")
json.dump({"source_bytes": st.st_size, "source_mtime_ns": str(st.st_mtime_ns)}, open(os.path.join(out, "tiles_report.json"), "w"))
'''


@pytest.fixture
def fake(tmp_path, monkeypatch):
    script = tmp_path / "tools" / "ifc_tiles.mjs"            # the name the job's liveness check looks for
    script.parent.mkdir()
    script.write_text(FAKE)
    node = tmp_path / "node"
    node.write_text(f"#!/bin/sh\nshift\nexec {sys.executable} \"$@\"\n")      # (node's --max-old-space-size dropped)
    node.chmod(node.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(ifc_tiles, "SCRIPT", script)
    monkeypatch.setattr(ifc_tiles, "_node", lambda: str(node))
    ifcs = tmp_path / "ifcs"
    ifcs.mkdir()
    (ifcs / "a.ifc").write_text("ISO-10303-21; version 1")
    return ifcs


def wait(ifcs, name, timeout=20.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = ifc_tiles.state(ifcs, name)
        if st["state"] != "building":
            return st
        time.sleep(0.1)
    raise AssertionError("still building")


def test_absent_then_built_then_ready(fake):
    assert ifc_tiles.state(fake, "a.ifc") == {"state": "absent"}
    assert ifc_tiles.ensure(fake, "a.ifc")["state"] == "building"
    st = wait(fake, "a.ifc")
    assert st["state"] == "ready" and st["version"] == ifc_tiles.version_of(fake / "a.ifc")
    # the finished job's record goes; asking again starts nothing
    assert not (fake / ".tiles" / "a.ifc.building.json").exists()
    assert ifc_tiles.ensure(fake, "a.ifc")["state"] == "ready"


def test_a_new_version_of_the_ifc_is_tiled_again(fake):
    ifc_tiles.ensure(fake, "a.ifc")
    v1 = wait(fake, "a.ifc")["version"]
    time.sleep(0.01)
    (fake / "a.ifc").write_text("ISO-10303-21; version 2, longer")   # uploaded again / regenerated / rescaled
    assert ifc_tiles.state(fake, "a.ifc")["state"] == "absent"
    assert ifc_tiles.file_path(fake, "a.ifc", v1, "tileset.json") is None     # the old version is never served
    ifc_tiles.ensure(fake, "a.ifc")
    st = wait(fake, "a.ifc")
    assert st["state"] == "ready" and st["version"] != v1


def test_a_failure_is_recorded_for_its_version_and_not_retried(fake, monkeypatch):
    monkeypatch.setenv("FAKE_FAIL", "1")
    ifc_tiles.ensure(fake, "a.ifc")
    st = wait(fake, "a.ifc")
    assert st["state"] == "failed" and "told to fail" in st["error"]
    assert ifc_tiles.ensure(fake, "a.ifc")["state"] == "failed"           # not started again for the same file
    monkeypatch.delenv("FAKE_FAIL")
    time.sleep(0.01)
    (fake / "a.ifc").write_text("ISO-10303-21; mended")                   # a new version is tried
    assert ifc_tiles.ensure(fake, "a.ifc")["state"] == "building"
    assert wait(fake, "a.ifc")["state"] == "ready"


def test_files_only_of_the_version_on_disk_and_inside_the_tiles(fake):
    ifc_tiles.ensure(fake, "a.ifc")
    v = wait(fake, "a.ifc")["version"]
    assert ifc_tiles.file_path(fake, "a.ifc", v, "tile_0.glb").read_bytes() == b"glTF"
    assert ifc_tiles.file_path(fake, "a.ifc", "0-0", "tile_0.glb") is None
    assert ifc_tiles.file_path(fake, "a.ifc", v, "../../a.ifc") is None
    assert ifc_tiles.file_path(fake, "a.ifc", v, "missing.glb") is None


def test_building_while_the_job_runs_and_missing_ifc(fake, monkeypatch):
    monkeypatch.setenv("FAKE_SLEEP", "1.5")
    assert ifc_tiles.ensure(fake, "a.ifc")["state"] == "building"
    assert ifc_tiles.ensure(fake, "a.ifc")["state"] == "building"         # one job, not two
    assert wait(fake, "a.ifc")["state"] == "ready"
    assert ifc_tiles.state(fake, "nope.ifc") == {"state": "missing"}


def test_remove_takes_the_tiles_and_their_records(fake):
    ifc_tiles.ensure(fake, "a.ifc")
    wait(fake, "a.ifc")
    ifc_tiles.remove(fake, "a.ifc")
    assert not (fake / ".tiles" / "a.ifc").exists()
    assert not list((fake / ".tiles").glob("a.ifc*"))
