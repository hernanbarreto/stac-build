"""Every pipeline stage runs in the manager's DAEMONIC multiprocessing child, and a daemon may not start
children: station 2026-11-02's certification died on its first mask read ("daemonic processes are not
allowed to have children") — correction.mask_store's process pool. The mask store's preload and the chunk
check's depth-map reads run inside a daemonic process (threads there) and give what they give outside."""
import multiprocessing as mp

import numpy as np

from segmentation import pipeline as P


def _store(out, rng):
    members = {f"f{f}_o{o}": (rng.random((6, 9)) > 0.5) for o in range(1, 6) for f in range(4)}
    members["scaled_res"] = np.array([6, 9])
    from segmentation import mask_space
    members[mask_space.KEYFRAMES_KEY] = np.arange(4, dtype=np.int32)
    members[mask_space.NPZ_KEY] = mask_space.declaration(mask_space.SPACE_KEYFRAME)
    P.write_canonical_store(out / "seg_masks.npz", members)


def _in_daemon(q, out, npz_paths):
    try:
        from correction import mask_store
        st = mask_store.MaskStore(out, threads=3).preload()
        bits = {k: v[0].tobytes() for k, v in sorted(st._bits.items())}
        from precision.chunk_check import load_depth_maps
        maps = load_depth_maps(npz_paths, ["d"], workers=3)
        q.put(("ok", bits, [m["d"].tobytes() for m in maps]))
    except BaseException as e:  # noqa: BLE001
        q.put(("error", repr(e), None))


def test_preload_and_depth_reads_run_in_a_daemonic_stage_process(tmp_path):
    rng = np.random.default_rng(2)
    out = tmp_path / "output"
    out.mkdir()
    _store(out, rng)
    paths = []
    for i in range(5):
        p = tmp_path / f"m{i}.npz"
        np.savez(p, d=rng.random((4, 4)))
        paths.append(p)
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    proc = ctx.Process(target=_in_daemon, args=(q, out, paths), daemon=True)
    proc.start()
    status, bits, maps = q.get(timeout=120)
    proc.join(timeout=60)
    assert status == "ok", bits
    # the same as outside a daemon (forked processes there)
    from correction import mask_store
    from precision.chunk_check import load_depth_maps
    st = mask_store.MaskStore(out, threads=3).preload()
    assert bits == {k: v[0].tobytes() for k, v in sorted(st._bits.items())}
    assert maps == [m["d"].tobytes() for m in load_depth_maps(paths, ["d"], workers=3)]
