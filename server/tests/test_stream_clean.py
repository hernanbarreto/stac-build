"""The cleaner by pieces (reconstruction/stream_clean, USER 2026-11-09: "mismo resultado bit a bit, con test") writes
the SAME BYTES as gpu_cloud_clean's in-memory path of record: several voxel slabs, an SOR halo too narrow on purpose
(points re-run with wider halos), duplicates, floaters, the confidence gate, the near-duplicate pass and the cap. On
the card (the cleaner refuses without one); seconds."""
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the cleaner runs on the card")


def _write_chunks(d, rng):
    k0 = 0
    for c in range(4):
        n = 30000
        # two walls, a floor, duplicates and floaters, the chunks overlapping in space
        a = rng.uniform(0, 3.0, n); b = rng.uniform(0, 1.0, n)
        kind = rng.integers(0, 3, n)
        x = np.where(kind == 0, a, np.where(kind == 1, a, 0.4 + 0.002 * rng.standard_normal(n)))
        y = np.where(kind == 2, b, np.where(kind == 0, 0.0 + 0.002 * rng.standard_normal(n), b))
        z = np.where(kind == 1, 1.0 + 0.002 * rng.standard_normal(n), rng.uniform(0, 1.0, n))
        xyz = np.stack([x, y, z], 1).astype(np.float32)
        xyz[:500] = xyz[500:1000]                                             # exact duplicates
        xyz[1000:1040] = rng.uniform([-2, -2, -2], [5, 3, 3], (40, 3))          # floaters
        xyz[1040:1060] = rng.uniform([8, 0, 0], [8.2, 0.05, 0.05], (20, 3))     # a sparse far cluster: wide halos
        rec = np.empty(n, [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        rec["x"], rec["y"], rec["z"] = xyz.T
        rec["red"], rec["green"], rec["blue"] = rng.integers(0, 256, (3, n), dtype=np.uint8)
        hdr = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\n"
               "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n" % n)
        with open(d / f"chunk_{c:03d}.ply", "wb") as f:
            f.write(hdr.encode()); rec.tofile(f)
        np.savez(d / f"chunk_{c:03d}_origins.npz", frame_global=np.arange(k0, k0 + n, dtype=np.int32),
                 pixel_row=rng.integers(0, 1080, n).astype(np.int16), pixel_col=rng.integers(0, 1920, n).astype(np.int16),
                 confidence=rng.uniform(0, 10, n).astype(np.float32))
        k0 += n


def _run(monkeypatch, tmp_path, argv):
    import reconstruction.gpu_cloud_clean as G
    import reconstruction.grid_knn as GK
    real = GK.config_section

    def fake(path, keys):
        if tuple(path) == ("postprocessing",) and tuple(keys) == ("clean_scratch_dir",):
            return {"clean_scratch_dir": str(tmp_path / "scratch")}
        if tuple(path) == ("postprocessing", "clean_bounds") and tuple(keys) == ("sor_halo_m",):
            return {"sor_halo_m": 0.002}                        # narrow on purpose: halos must grow
        return real(path, keys)
    monkeypatch.setattr(GK, "config_section", fake)
    monkeypatch.setattr(G, "_clean_bounds", lambda: {"tile_points": 20000, "knn_query_block": 4096,
                                                     "knn_candidate_budget": 1 << 20})
    monkeypatch.setattr(sys, "argv", ["gpu_cloud_clean"] + argv)
    assert G.main() == 0


@pytest.mark.parametrize("extra", [
    ["--skip-duplicates"],                                                         # the production recipe
    ["--conf-min-norm", "0.1", "--max-points", "20000"],                            # gate + duplicates + cap
])
def test_by_pieces_writes_the_same_bytes_as_the_path_of_record(monkeypatch, tmp_path, extra):
    rng = np.random.default_rng(7)
    src = tmp_path / "chunks"; src.mkdir()
    _write_chunks(src, rng)
    base = ["--input-dir", str(src), "--voxel-size", "0.005", "--sor-knn", "8", "--sor-sigma", "2.0",
            "--skip-noise", "--skip-normals"] + extra
    a, b = tmp_path / "record.ply", tmp_path / "pieces.ply"
    _run(monkeypatch, tmp_path, base + ["--output", str(a), "--in-memory"])
    _run(monkeypatch, tmp_path, base + ["--output", str(b)])
    ra, rb = a.read_bytes(), b.read_bytes()
    assert len(ra) > 1000 and ra == rb, "the cleaner by pieces must write the path of record's bytes"
    assert not any((tmp_path / "scratch").iterdir()), "the scratch is emptied"


def test_the_statistic_on_scratch_is_numpys_own():
    """mean / std over a memory-mapped array, std rebuilt from numpy's own steps: the same bits."""
    import tempfile, os
    rng = np.random.default_rng(3)
    x = rng.gamma(2.0, 0.003, 3_000_001)
    with tempfile.TemporaryDirectory() as d:
        m = np.memmap(os.path.join(d, "a"), dtype=np.float64, mode="w+", shape=x.shape); m[:] = x
        t = np.memmap(os.path.join(d, "b"), dtype=np.float64, mode="w+", shape=x.shape)
        am = np.add.reduce(m, keepdims=True) / len(x)
        np.subtract(m, am, out=t); np.multiply(t, t, out=t)
        sd = float(np.sqrt(np.add.reduce(t) / len(x)))
        assert float(m.mean()) == float(x.mean()) and sd == float(x.std())
