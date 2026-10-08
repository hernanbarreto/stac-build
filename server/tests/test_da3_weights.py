"""DA3's weights: ONE Hugging Face cache for every launcher, a PINNED revision, sha256-verified at
load (server/da3_weights.py, docs/plan_determinismo.md points 32 and 43)."""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))

import da3_weights                                              # noqa: E402

MODEL = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"


def test_every_launcher_exports_the_one_cache():
    """The backend (start.sh) and vLLM (serve_semantic.sh) export HF_HOME; the DA3 subprocesses
    get it from da3_weights.hf_env whoever launched them — the SAME directory everywhere."""
    for sh in ("start.sh", "serve_semantic.sh"):
        txt = (ROOT / "scripts" / sh).read_text()
        assert f"export HF_HOME={da3_weights.HF_HOME}" in txt, sh
    env = da3_weights.hf_env({"HF_HUB_CACHE": "/elsewhere", "HUGGINGFACE_HUB_CACHE": "/x", "PATH": "/bin"})
    assert env["HF_HOME"] == da3_weights.HF_HOME and env["HF_HUB_OFFLINE"] == "1"
    assert "HF_HUB_CACHE" not in env and "HUGGINGFACE_HUB_CACHE" not in env and env["PATH"] == "/bin"
    # every code path that launches DA3 passes hf_env (walk, focal, vram's calibration, anchors)
    server = ROOT / "server"
    for name in ("intake/walk.py", "intake/focal.py", "intake/vram.py", "reconstruction/da3_anchor.py"):
        assert "da3_weights.hf_env(" in (server / name).read_text(), name
    # the extractor loads the VERIFIED snapshot, never a hub id resolving refs/main
    src = (server / "extract_da3_depth.py").read_text()
    assert "da3_weights.verified_snapshot(" in src
    assert "from_pretrained(args.model)" not in src and "from_pretrained(str(snap))" in src


def test_the_pin_names_a_revision_and_every_file_sha():
    p = da3_weights.pinned(MODEL)
    assert len(p["revision"]) == 40 and set(p["sha256"]) == {"config.json", "model.safetensors"}
    assert all(len(s) == 64 for s in p["sha256"].values())
    ident = da3_weights.identity(MODEL)
    assert ident == {"model_id": MODEL, "revision": p["revision"], "sha256": dict(sorted(p["sha256"].items()))}
    with pytest.raises(da3_weights.DA3WeightsError, match="no pinned revision"):
        da3_weights.pinned("someone/else")
    d = da3_weights.snapshot_dir(MODEL, "/cache")
    assert d == Path("/cache/hub/models--depth-anything--DA3NESTED-GIANT-LARGE-1.1/snapshots") / p["revision"]


def test_verification_fails_on_a_missing_snapshot_or_other_bytes(tmp_path, monkeypatch):
    with pytest.raises(da3_weights.DA3WeightsError, match="does not exist"):
        da3_weights.verified_snapshot(MODEL, tmp_path, log=lambda m: None)
    d = da3_weights.snapshot_dir(MODEL, tmp_path)
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    with pytest.raises(da3_weights.DA3WeightsError, match="model.safetensors is missing"):
        da3_weights.verified_snapshot(MODEL, tmp_path, log=lambda m: None)
    (d / "model.safetensors").write_bytes(b"not the weights")
    with pytest.raises(da3_weights.DA3WeightsError, match="hashes to sha256") as e:
        da3_weights.verified_snapshot(MODEL, tmp_path, log=lambda m: None)
    assert "another file than the validated one" in str(e.value)
    # the pinned bytes pass
    import repro
    monkeypatch.setitem(da3_weights.PINNED, MODEL, {
        "revision": da3_weights.pinned(MODEL)["revision"],
        "sha256": {"config.json": repro.sha256_file(d / "config.json"),
                   "model.safetensors": repro.sha256_file(d / "model.safetensors")}})
    logs = []
    assert da3_weights.verified_snapshot(MODEL, tmp_path, log=logs.append) == d
    assert any("verified" in m and "2 file(s)" in m for m in logs)


@pytest.mark.skipif(not Path(da3_weights.HF_HOME).is_dir(), reason="the pod's HF cache is not here")
def test_the_pods_cache_holds_the_pinned_snapshot():
    """The real cache: the pinned revision is the snapshot every validated run loaded; config.json
    hashes to its pin and the weights blob is named by its sha256 (the LFS object id) — read
    without hashing 6.7 GB."""
    import repro
    d = da3_weights.snapshot_dir(MODEL)
    assert d.is_dir(), d
    pin = da3_weights.pinned(MODEL)["sha256"]
    assert repro.sha256_file(d / "config.json") == pin["config.json"]
    blob = os.readlink(d / "model.safetensors")
    assert Path(blob).name == pin["model.safetensors"], blob
    assert (d / "model.safetensors").stat().st_size > 6 * 1024 ** 3
