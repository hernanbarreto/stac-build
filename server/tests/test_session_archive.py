"""A session as one ZIP: what goes out comes back the SAME session, and nothing else is let in.

USER 2026-11-18: "una función en cada sesión para descargarla como zip, debe descargar todo lo
necesario para que pueda volver a subirse y correr exactamente igual … subir un archivo zip que va a
comprobar que tenga la estructura correcta de los datos que contiene el zip y si está todo ok la sube
y la descomprime, elimina el zip y se queda con la sesión perfectamente utilizable."
"""
import io
import json
import os
import sqlite3
import time
import zipfile
from pathlib import Path

import pytest

import session_archive as A


class Unseekable:
    """What the download writes into: write() and nothing else (the response's stream)."""

    def __init__(self):
        self.parts = []

    def write(self, data):
        self.parts.append(bytes(data))
        return len(data)

    def flush(self):
        pass

    def getvalue(self):
        return b"".join(self.parts)


def make_session(root: Path, sid: str = "site_a") -> Path:
    """A project of the real layout, with everything a real one has that could be lost: links,
    empty folders, a database with SQLite's side files, lock files, odd dates and modes — and
    the things that must NOT travel."""
    p = root / sid
    scan = p / "scans" / "2026-01-02" / "src_default"
    (scan / "frames").mkdir(parents=True)
    (scan / "output" / "potree").mkdir(parents=True)
    (scan / "output" / "_before_x").mkdir()
    (p / "ifcs").mkdir()                                         # (empty: must come back)
    (p / "segmentation").mkdir()
    (p / "merged").mkdir()
    (p / "exports").mkdir()
    (p / "project.json").write_text(json.dumps({"name": sid, "slug": sid, "scan_days": ["2026-01-02"]}))
    (scan / "source.json").write_text("{}")
    (scan / "source_video.mp4").write_bytes(os.urandom(70_000))
    for i in range(5):
        (scan / "frames" / f"{i:06d}.jpg").write_bytes(os.urandom(3000 + i))
    (scan / "output" / "cleaned_cloud.ply").write_bytes(b"ply\n" + os.urandom(2_500_000))
    (scan / "output" / "potree" / "metadata.json").write_text(json.dumps({"points": 123}))
    (scan / "output" / "potree" / "octree.bin").write_bytes(os.urandom(300_000))
    (scan / "output" / "segmentation_result.json").write_text(json.dumps({"instances": list(range(2000))}))
    (scan / "output" / ".potree.lock").write_bytes(b"")           # (an empty lock file: part of the tree)
    db = scan / "output" / "scene_r.db"
    con = sqlite3.connect(db)
    con.execute("create table t (a integer, b text)")
    con.executemany("insert into t values (?, ?)", [(i, "x" * 50) for i in range(500)])
    con.commit()
    con.close()
    (scan / "output" / "scene_r.db-wal").write_bytes(b"")         # (empty: SQLite's own, not sent)
    (scan / "output" / "scene_r.db-shm").write_bytes(b"\0" * 32768)
    (p / "exports" / "old_export.zip").write_bytes(os.urandom(5000))
    os.symlink("../scans/2026-01-02/src_default/output/potree", p / "merged" / "potree")
    os.symlink("../scans/2026-01-02/src_default/output/cleaned_cloud.ply", p / "merged" / "merged_cloud.ply")
    # what must not travel
    (scan / "output" / "__pycache__").mkdir()
    (scan / "output" / "__pycache__" / "m.cpython-310.pyc").write_bytes(b"x")
    (p / ".ipynb_checkpoints").mkdir()
    (p / ".ipynb_checkpoints" / "n.ipynb").write_text("{}")
    # dates and modes that a copy would lose
    os.utime(scan / "output" / "cleaned_cloud.ply", ns=(1_700_000_000_123_456_789, 1_700_000_000_123_456_789))
    os.utime(scan / "frames" / "000003.jpg", ns=(946_684_800_000_000_001, 946_684_800_000_000_001))
    os.chmod(scan / "source_video.mp4", 0o600)
    os.chmod(scan / "output" / "segmentation_result.json", 0o666)
    return p


def zip_of(project: Path, sid: str, dest: Path) -> dict:
    stream = Unseekable()
    manifest = A.write_zip(project, sid, stream)
    dest.write_bytes(stream.getvalue())
    return manifest


def same_content(a: Path, b: Path) -> None:
    sa, sb = A.tree_state(a), A.tree_state(b)
    assert sa == sb
    for rel, state in sa.items():
        if isinstance(state, tuple):
            assert (a / rel).read_bytes() == (b / rel).read_bytes(), rel


# ── out and back ───────────────────────────────────────────────────────────────────────────────
def test_a_session_comes_back_the_same(tmp_path):
    src = make_session(tmp_path / "pod_a")
    z = tmp_path / "s.zip"
    manifest = zip_of(src, "site_a", z)
    other = tmp_path / "pod_b"
    other.mkdir()
    report = A.extract(z, other)
    assert report["session_id"] == "site_a" and report["files"] == manifest["files"] and report["bytes"] == manifest["bytes"]
    same_content(src, other / "site_a")
    back = other / "site_a"
    # the links are links, the empty folders are there, dates to the nanosecond, modes
    assert os.readlink(back / "merged" / "potree") == "../scans/2026-01-02/src_default/output/potree"
    assert (back / "merged" / "potree" / "metadata.json").read_text() == json.dumps({"points": 123})
    assert (back / "ifcs").is_dir() and not list((back / "ifcs").iterdir())
    scan = back / "scans" / "2026-01-02" / "src_default"
    assert (scan / "output" / "cleaned_cloud.ply").stat().st_mtime_ns == 1_700_000_000_123_456_789
    assert (scan / "frames" / "000003.jpg").stat().st_mtime_ns == 946_684_800_000_000_001
    assert (scan / "source_video.mp4").stat().st_mode & 0o777 == 0o600
    assert (scan / "output" / ".potree.lock").stat().st_size == 0
    # the database is the database
    assert sqlite3.connect(scan / "output" / "scene_r.db").execute("select count(*), sum(a) from t").fetchone() == (500, 124750)
    # what any use writes again did not travel
    assert not (scan / "output" / "__pycache__").exists() and not (back / ".ipynb_checkpoints").exists()
    assert not (scan / "output" / "scene_r.db-shm").exists() and not (scan / "output" / "scene_r.db-wal").exists()
    # nothing is left beside the session
    assert sorted(p.name for p in other.iterdir()) == ["site_a"]


def test_the_stream_needs_no_seek_and_text_is_deflated(tmp_path):
    src = make_session(tmp_path)
    z = tmp_path / "s.zip"
    zip_of(src, "site_a", z)
    with zipfile.ZipFile(z) as zf:
        assert zf.testzip() is None
        kinds = {i.filename: i.compress_type for i in zf.infolist()}
        assert zf.infolist()[0].filename == A.MANIFEST                        # the manifest is read first
        assert kinds["site_a/scans/2026-01-02/src_default/output/cleaned_cloud.ply"] == zipfile.ZIP_STORED
        assert kinds["site_a/scans/2026-01-02/src_default/output/segmentation_result.json"] == zipfile.ZIP_DEFLATED
        m = json.loads(zf.read(A.MANIFEST))
        assert m["format"] == "stac-session" and m["version"] == 1 and m["scans"] == ["2026-01-02/default"]


def test_another_name_and_a_name_that_exists(tmp_path):
    src = make_session(tmp_path / "pod_a")
    z = tmp_path / "s.zip"
    zip_of(src, "site_a", z)
    other = tmp_path / "pod_b"
    other.mkdir()
    assert A.extract(z, other, name="site_copy")["session_id"] == "site_copy"
    same_content(src, other / "site_copy")
    with pytest.raises(A.ArchiveError, match="already exists"):
        A.extract(z, other, name="site_copy")
    with pytest.raises(A.ArchiveError, match="letters, numbers"):
        A.validate(z, other, name="../evil")
    assert sorted(p.name for p in other.iterdir()) == ["site_copy"]


# ── what is not let in ─────────────────────────────────────────────────────────────────────────
def rewrite(z: Path, out: Path, change) -> None:
    """A copy of the ZIP with `change(name, data, manifest) -> (name, data) | None` applied to
    every entry; the manifest is written again from the (possibly edited) dict."""
    with zipfile.ZipFile(z) as src, zipfile.ZipFile(out, "w") as dst:
        manifest = json.loads(src.read(A.MANIFEST))
        entries = []
        for info in src.infolist():
            if info.filename == A.MANIFEST:
                continue
            res = change(info.filename, src.read(info.filename), manifest)
            if res is not None:
                entries.append((info, res))
        dst.writestr(A.MANIFEST, json.dumps(manifest))
        for info, (name, data) in entries:
            new = zipfile.ZipInfo(name, date_time=info.date_time)
            new.external_attr = info.external_attr
            dst.writestr(new, data)


@pytest.fixture()
def good(tmp_path):
    src = make_session(tmp_path / "pod_a")
    z = tmp_path / "good.zip"
    zip_of(src, "site_a", z)
    target = tmp_path / "pod_b"
    target.mkdir()
    return z, target, tmp_path


def refused(z, target, match):
    with pytest.raises(A.ArchiveError, match=match):
        A.extract(z, target)
    assert list(target.iterdir()) == [], "a refused ZIP must leave nothing beside the sessions"


def test_not_a_zip_and_a_cut_zip(good):
    z, target, tmp = good
    junk = tmp / "junk.zip"
    junk.write_bytes(b"this is not a zip" * 100)
    refused(junk, target, "not a ZIP")
    cut = tmp / "cut.zip"
    cut.write_bytes(z.read_bytes()[: z.stat().st_size // 2])
    refused(cut, target, "not a ZIP")


def test_a_zip_that_is_not_a_session(good):
    z, target, tmp = good
    plain = tmp / "plain.zip"
    with zipfile.ZipFile(plain, "w") as zf:
        zf.writestr("readme.txt", "hello")
    refused(plain, target, "not a STAC session")
    other = tmp / "other.zip"
    rewrite(z, other, lambda n, d, m: (m.update(format="something-else"), (n, d))[1])
    refused(other, target, "another format")
    newer = tmp / "newer.zip"
    rewrite(z, newer, lambda n, d, m: (m.update(version=99), (n, d))[1])
    refused(newer, target, "format version 99")


def test_a_missing_file_an_extra_one_and_a_wrong_size(good):
    z, target, tmp = good
    frame = "site_a/scans/2026-01-02/src_default/frames/000002.jpg"
    missing = tmp / "missing.zip"
    rewrite(z, missing, lambda n, d, m: None if n == frame else (n, d))
    refused(missing, target, "incomplete")
    extra = tmp / "extra.zip"
    with zipfile.ZipFile(z) as src, zipfile.ZipFile(extra, "w") as dst:
        for info in src.infolist():
            dst.writestr(info, src.read(info.filename))
        dst.writestr("site_a/scans/2026-01-02/src_default/output/planted.sh", "#!/bin/sh")
    refused(extra, target, "does not list")
    outside = tmp / "outside.zip"
    with zipfile.ZipFile(z) as src, zipfile.ZipFile(outside, "w") as dst:
        for info in src.infolist():
            dst.writestr(info, src.read(info.filename))
        dst.writestr("another_session/project.json", "{}")
    refused(outside, target, "does not list")
    shorter = tmp / "shorter.zip"
    rewrite(z, shorter, lambda n, d, m: (n, d[:-1]) if n == frame else (n, d))
    refused(shorter, target, "bytes in the ZIP")


def test_unsafe_paths_and_links(good):
    z, target, tmp = good

    def escape(n, d, m):
        for e in m["entries"]:
            if e["p"].endswith("000001.jpg"):
                e["p"] = "../../escaped.jpg"
        return n, d
    bad = tmp / "escape.zip"
    rewrite(z, bad, escape)
    refused(bad, target, "unsafe path")

    def link_out(n, d, m):
        for e in m["entries"]:
            if e["p"] == "merged/potree":
                e["l"] = "../../../etc"
        return n, d
    bad2 = tmp / "link.zip"
    rewrite(z, bad2, link_out)
    refused(bad2, target, "points outside the session")

    def absolute(n, d, m):
        for e in m["entries"]:
            if e["p"] == "merged/potree":
                e["l"] = "/etc/passwd"
        return n, d
    bad3 = tmp / "abs.zip"
    rewrite(z, bad3, absolute)
    refused(bad3, target, "points outside the session")


def test_the_shape_of_a_session(good):
    z, target, tmp = good
    no_meta = tmp / "no_meta.zip"

    def drop_meta(n, d, m):
        m["entries"] = [e for e in m["entries"] if e["p"] != "project.json"]
        m["files"] = sum(1 for e in m["entries"] if "s" in e)
        m["bytes"] = sum(e.get("s", 0) for e in m["entries"])
        return None if n == "site_a/project.json" else (n, d)
    rewrite(z, no_meta, drop_meta)
    refused(no_meta, target, "no project.json")

    def bad_day(n, d, m):
        for e in m["entries"]:
            e["p"] = e["p"].replace("scans/2026-01-02/", "scans/january/")
        return n.replace("scans/2026-01-02/", "scans/january/"), d
    wrong = tmp / "wrong_day.zip"
    rewrite(z, wrong, bad_day)
    refused(wrong, target, "not a scan day")


def test_a_damaged_file_leaves_nothing(good):
    z, target, tmp = good
    data = bytearray(z.read_bytes())
    with zipfile.ZipFile(z) as zf:
        info = zf.getinfo("site_a/scans/2026-01-02/src_default/output/cleaned_cloud.ply")
        at = info.header_offset + 30 + len(info.filename.encode()) + len(info.extra) + 1000
    data[at] ^= 0xFF                                                         # one byte of a stored cloud
    hurt = tmp / "hurt.zip"
    hurt.write_bytes(bytes(data))
    refused(hurt, target, "damaged")


# ── what is not sent ───────────────────────────────────────────────────────────────────────────
def test_a_session_being_written_is_not_sent(tmp_path):
    src = make_session(tmp_path)
    scan = src / "scans" / "2026-01-02" / "src_default"
    (scan / "output" / "scene_r.db-wal").write_bytes(b"pending writes")
    with pytest.raises(A.ArchiveError, match="not yet written into its database"):
        A.manifest_of(src, "site_a")
    (scan / "output" / "scene_r.db-wal").write_bytes(b"")
    (scan / "frames.extracting").mkdir()
    with pytest.raises(A.ArchiveError, match="being extracted"):
        A.manifest_of(src, "site_a")


def test_a_file_that_changes_while_it_is_sent_ends_the_stream(tmp_path):
    src = make_session(tmp_path)
    manifest = A.manifest_of(src, "site_a")
    cloud = src / "scans" / "2026-01-02" / "src_default" / "output" / "cleaned_cloud.ply"
    time.sleep(0.01)
    cloud.write_bytes(cloud.read_bytes() + b"one more point")
    with pytest.raises(A.ArchiveError, match="changed while"):
        A.write_zip(src, "site_a", Unseekable(), manifest=manifest)


def test_not_a_session(tmp_path):
    (tmp_path / "x").mkdir()
    with pytest.raises(A.ArchiveError, match="no project.json"):
        A.plan(tmp_path / "x", "x")


# ── the routes ─────────────────────────────────────────────────────────────────────────────────
@pytest.fixture()
def client(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import session_archive_api as R
    projects = tmp_path / "projects"
    projects.mkdir()
    make_session(projects)
    busy = {}
    log = []
    R.configure(projects, busy=lambda sid: busy.get(sid), audit_log=lambda *a, **k: log.append((a, k)), commit="abc1234")
    R._TICKETS.clear(), R._READING.clear(), R._IMPORTS.clear()
    app = FastAPI()
    app.include_router(R.router)
    app.dependency_overrides[R._manager] = lambda: {"username": "tester", "role": "admin"}
    return TestClient(app), projects, busy, log, R


def upload(c, R, data: bytes, name=None, part=None):
    opened = c.post("/api/session_import", json={"filename": "s.zip", "size": len(data), "name": name}).json()
    job, step = opened["job"], part or opened["part_bytes"]
    for off in range(0, len(data), step):
        r = c.put(f"/api/session_import/{job}/part?offset={off}", content=data[off:off + step])
        assert r.status_code == 200, r.text
    assert c.post(f"/api/session_import/{job}/finish").status_code == 200
    for _ in range(600):
        st = c.get(f"/api/session_import/{job}").json()
        if st["phase"] in ("done", "failed"):
            return st
        time.sleep(0.05)
    raise AssertionError("the import did not end")


def test_download_then_upload_through_the_routes(client):
    c, projects, busy, log, R = client
    plan = c.get("/api/session_archive/site_a/plan").json()
    assert plan["files"] > 10 and plan["scans"] == ["2026-01-02/default"]
    t = c.post("/api/session_archive/site_a/ticket").json()
    assert t["filename"] == "site_a_stac_session.zip"
    r = c.get(t["url"])
    assert r.status_code == 200 and r.headers["content-disposition"].endswith('"site_a_stac_session.zip"')
    data = r.content
    assert int(r.headers["x-stac-session-bytes"]) == plan["bytes"]
    assert c.get(t["url"]).status_code == 403                                 # a ticket is good once
    assert R._READING == {}                                                   # … and the session is free again
    st = upload(c, R, data, name="site_b", part=100_000)                      # in many parts
    assert st["phase"] == "done" and st["session_id"] == "site_b" and st["pct"] == 100.0
    same_content(projects / "site_a", projects / "site_b")
    assert not list((projects / ".incoming").iterdir()), "the ZIP is removed once the session is in place"
    assert json.loads(zipfile.ZipFile(io.BytesIO(data)).read(A.MANIFEST))["stac_commit"] == "abc1234"
    assert [a[0][0] for a in log] == ["archive.download", "archive.import"]


def test_the_routes_refuse(client):
    c, projects, busy, log, R = client
    assert c.get("/api/session_archive/nope/plan").status_code == 404
    busy["site_a"] = "a pipeline job of site_a is running"
    assert c.post("/api/session_archive/site_a/ticket").status_code == 409
    busy.clear()
    assert c.get("/api/session_archive/site_a/download?ticket=invented").status_code == 403
    # a name that exists is refused before a byte is uploaded
    assert c.post("/api/session_import", json={"filename": "s.zip", "size": 10, "name": "site_a"}).status_code == 409
    assert c.post("/api/session_import", json={"filename": "s.zip", "size": 10, "name": "bad name"}).status_code == 400
    # a part at the wrong offset is answered with what the server holds
    job = c.post("/api/session_import", json={"filename": "s.zip", "size": 10}).json()["job"]
    assert c.put(f"/api/session_import/{job}/part?offset=0", content=b"12345").json()["received"] == 5
    r = c.put(f"/api/session_import/{job}/part?offset=0", content=b"12345")
    assert r.status_code == 409 and r.json()["detail"]["received"] == 5
    assert c.post(f"/api/session_import/{job}/finish").status_code == 409     # not whole yet
    assert c.put(f"/api/session_import/{job}/part?offset=5", content=b"1234567890").status_code == 400   # more than declared
    assert c.get(f"/api/session_import/{job}").json()["received"] == 5         # … and nothing of it was kept
    assert c.delete(f"/api/session_import/{job}").json()["phase"] == "failed"
    assert not list((projects / ".incoming").glob("*.zip"))


def test_a_bad_upload_fails_clean(client):
    c, projects, busy, log, R = client
    st = upload(c, R, b"definitely not a zip" * 50)
    assert st["phase"] == "failed" and "not a ZIP" in st["error"]
    assert sorted(p.name for p in projects.iterdir()) == [".incoming", "site_a"]
    assert not list((projects / ".incoming").iterdir())
    # the same session twice: the second is refused by name, nothing changes
    data = c.get(c.post("/api/session_archive/site_a/ticket").json()["url"]).content
    st = upload(c, R, data)
    assert st["phase"] == "failed" and "already exists" in st["error"]
    assert sorted(p.name for p in projects.iterdir()) == [".incoming", "site_a"]


def test_a_session_being_downloaded_counts_as_read_until_the_reader_leaves(client):
    """While the ZIP is being read the session must not be rewritten (the pipeline, a rename, a
    delete, a video ask reading()); a reader that leaves half-way frees it and stops the writer."""
    c, projects, busy, log, R = client
    manifest = A.manifest_of(projects / "site_a", "site_a")
    gen = R._stream_session("site_a", projects / "site_a", manifest)
    assert R.reading("site_a") is None                                        # (nothing is read before the first byte is asked)
    first = next(gen)
    assert first[:2] == b"PK" and "being downloaded" in R.reading("site_a")
    gen.close()                                                               # the client went away
    assert R.reading("site_a") is None
    import threading
    assert not [th for th in threading.enumerate() if th.name == "session-zip-site_a" and th.is_alive()]
    # a file that changes under the writer ends the stream with an error, not with a cut ZIP
    cloud = projects / "site_a" / "scans" / "2026-01-02" / "src_default" / "output" / "cleaned_cloud.ply"
    stale = A.manifest_of(projects / "site_a", "site_a")
    time.sleep(0.01)
    cloud.write_bytes(cloud.read_bytes() + b"x")
    with pytest.raises(RuntimeError, match="changed while"):
        b"".join(R._stream_session("site_a", projects / "site_a", stale))
    assert R.reading("site_a") is None
