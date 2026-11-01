"""repro.sha256_file_memo: the digest of an immutable file computed once and reused while the file's identity
(device, inode, size, mtime_ns) holds; a rewritten file is hashed again; an unreadable sidecar is ignored."""
import json
import os
import time

from repro import sha256_file, sha256_file_memo


def test_memo_reuses_and_invalidates(tmp_path, monkeypatch):
    f = tmp_path / "w.bin"
    f.write_bytes(os.urandom(1 << 16))
    want = sha256_file(f)
    calls = []
    import repro
    real = repro.sha256_file

    def counting(p):
        calls.append(str(p))
        return real(p)
    monkeypatch.setattr(repro, "sha256_file", counting)
    assert sha256_file_memo(f) == want and len(calls) == 1
    assert sha256_file_memo(f) == want and len(calls) == 1            # memoised
    side = f.with_name(f.name + ".stac_sha256.json")
    assert json.loads(side.read_text())["sha256"] == want
    time.sleep(0.01)
    f.write_bytes(os.urandom(1 << 16))                                 # rewritten: hashed again
    assert sha256_file_memo(f) == real(f) and len(calls) == 2
    side.write_text("{not json")                                        # a broken sidecar: hashed again
    assert sha256_file_memo(f) == real(f) and len(calls) == 3
    # through a symlink: the memo lives beside the resolved file
    link = tmp_path / "link.bin"
    link.symlink_to(f)
    assert sha256_file_memo(link) == real(f) and len(calls) == 3
