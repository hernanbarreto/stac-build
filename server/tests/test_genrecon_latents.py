"""tools/genrecon/latents.py — the sampled latents on disk, the point a GenRecon run continues from when its
decode failed: what is read is what was written, a record that does not describe the same chunks is refused by
name, and a file is whole or absent (USER 2026-11-12: "la regla es que se continúa desde el punto de falla")."""
import sys
from pathlib import Path

import pytest
import torch

_TOOLS = Path(__file__).resolve().parent.parent / "tools" / "genrecon"
sys.path.insert(0, str(_TOOLS))
import latents as L  # noqa: E402


def _fields(n=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    shape, tex, coords = [], [], []
    for c in range(n):
        rows = 5 + c
        xyz = torch.randint(0, 32, (rows, 3), generator=g, dtype=torch.int32)
        co = torch.cat([torch.zeros(rows, 1, dtype=torch.int32), xyz], 1)
        shape.append((torch.randn(rows, 32, generator=g), co))
        tex.append((torch.randn(rows, 32, generator=g), co.clone()))
        coords.append(co.clone())
    return {"shape": shape, "tex": tex, "coords_list": coords,
            "relative_translations": [torch.tensor([0.5 * c, 0.0, 0.0]) for c in range(n)],
            "m_c2o": [torch.eye(4) * (c + 1) for c in range(n + 2)],
            "chunk_indices": [0, 2, 4][:n], "coords_resolution": 32, "pipeline_type": "512"}


def test_what_is_read_is_what_was_written(tmp_path):
    f = _fields()
    p = L.save(tmp_path / "latents.pt", **f)
    assert p.is_file() and not (tmp_path / "latents.pt.tmp").exists()
    r = L.load(p)
    assert r["version"] == L.VERSION and r["chunk_indices"] == [0, 2, 4]
    assert r["coords_resolution"] == 32 and r["pipeline_type"] == "512"
    for k in ("shape", "tex"):
        for (fa, ca), (fb, cb) in zip(f[k], r[k]):
            assert torch.equal(fa, fb) and torch.equal(ca, cb) and fb.dtype == fa.dtype and cb.dtype == ca.dtype
    assert all(torch.equal(a, b) for a, b in zip(f["coords_list"], r["coords_list"]))
    assert all(torch.equal(a, b) for a, b in zip(f["relative_translations"], r["relative_translations"]))
    assert tuple(r["m_c2o"].shape) == (5, 4, 4) and torch.equal(r["m_c2o"][2], torch.eye(4) * 3)
    assert "3 chunks" in L.summary(r) and "18 latent voxels" in L.summary(r)


def test_a_record_that_does_not_describe_the_same_chunks_is_refused_by_name(tmp_path):
    f = _fields()
    f["tex"] = f["tex"][:2]
    with pytest.raises(ValueError, match="2 entries of 'tex' for 3 chunks"):
        L.save(tmp_path / "a.pt", **f)
    assert not (tmp_path / "a.pt").exists() and not (tmp_path / "a.pt.tmp").exists()
    f = _fields()
    f["tex"][1] = (f["tex"][1][0], f["tex"][1][1].flip(0))
    with pytest.raises(ValueError, match="chunk 2's texture latent is not on its shape latent's voxels"):
        L.save(tmp_path / "a.pt", **f)
    f = _fields()
    f["chunk_indices"] = [0, 2, 9]
    with pytest.raises(ValueError, match="outside the 5 chunks of m_c2o"):
        L.save(tmp_path / "a.pt", **f)
    f = _fields()
    f["shape"][0] = (f["shape"][0][0][:-1], f["shape"][0][1])
    with pytest.raises(ValueError, match="'shape' of chunk 0 holds feats"):
        L.save(tmp_path / "a.pt", **f)


def test_a_file_of_another_version_or_a_missing_one_is_refused(tmp_path):
    rec = L.payload(**_fields())
    rec["version"] = L.VERSION + 1
    torch.save(rec, tmp_path / "new.pt")
    with pytest.raises(ValueError, match="this code reads version"):
        L.load(tmp_path / "new.pt")
    del rec["m_c2o"]
    torch.save(rec, tmp_path / "cut.pt")
    with pytest.raises(ValueError, match="no m_c2o"):
        L.load(tmp_path / "cut.pt")
    torch.save([1, 2], tmp_path / "list.pt")
    with pytest.raises(ValueError, match="not a latents record"):
        L.load(tmp_path / "list.pt")
    with pytest.raises(FileNotFoundError, match="no sampled latents"):
        L.load(tmp_path / "absent.pt")
