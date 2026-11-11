"""The SAM3 mask store inflated on the device (segmentation/mask_store_gpu.py, USER 2026-11-11: "Nada debe
hacerse en cpu"): nvCOMP's Deflate on the card gives np.load's masks bit for bit (a synthetic store with
stored and deflated members, and a slice of a real one when present); the morphology on the device is
scipy's dilation and cv2's erosion, pixel for pixel."""
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmentation.mask_store_gpu import DeviceMaskStore, dilate_cross_t, erode_like_cv2_t  # noqa: E402

CUDA = torch.cuda.is_available()


def _store(path, rng, n=40, h=37, w=29):
    members = {f"f{k % 7}_o{k}": (rng.random((h, w)) > 0.6).astype(np.uint8 if k % 2 else bool) for k in range(n)}
    members["obj_ids"] = np.arange(n)
    np.savez_compressed(path, **members)
    # one member re-written STORED (no compression): the reader takes both kinds
    with zipfile.ZipFile(path, "a", compression=zipfile.ZIP_STORED) as zf:
        import io
        b = io.BytesIO()
        np.lib.format.write_array(b, (rng.random((h, w)) > 0.5).astype(np.uint8))
        zf.writestr("f9_o999.npy", b.getvalue())
    return path


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if CUDA else []))
def test_the_device_store_is_np_loads(tmp_path, device):
    p = _store(tmp_path / "seg_masks.npz", np.random.default_rng(0))
    st = DeviceMaskStore(p, device)
    keys = [k for k in st.files if k.startswith("f")]
    got = st.stack(keys).cpu().numpy()
    with np.load(p) as z:
        ref = np.stack([np.asarray(z[k]) != 0 for k in keys])
    assert got.dtype == bool and np.array_equal(got, ref) and "f9_o999" in st


@pytest.mark.skipif(not CUDA, reason="nvCOMP inflates on the card")
def test_a_real_store_inflates_bit_for_bit_on_the_card():
    p = Path(__file__).resolve().parents[1] / "projects/pccr/scans/2026-08-31/src_default/output/seg_masks.npz"
    if not p.exists():
        pytest.skip("pccr 3108's store is not on this disk")
    st = DeviceMaskStore(p, "cuda")
    keys = [k for k in st.files if k.startswith("f")][::97][:600]
    with np.load(p) as z:
        ref = np.stack([np.asarray(z[k]) != 0 for k in keys])
    assert np.array_equal(st.stack(keys).cpu().numpy(), ref)


@pytest.mark.parametrize("it", [1, 2, 3])
def test_the_device_erosion_is_cv2s(it):
    rng = np.random.default_rng(it)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for _ in range(5):
        m = np.zeros((41, 57), np.uint8)
        m[rng.integers(0, 20):rng.integers(21, 41), rng.integers(0, 25):rng.integers(26, 57)] = 1
        m[0, :] = 1                                                     # touching the border
        ref = cv2.erode(m, k, iterations=it).astype(bool)
        got = erode_like_cv2_t(torch.as_tensor(m), k, it).numpy()
        assert np.array_equal(ref, got)


def test_the_stacked_dilation_is_per_mask():
    rng = np.random.default_rng(4)
    st = torch.as_tensor(rng.random((6, 23, 31)) > 0.9)
    one = torch.stack([dilate_cross_t(st[i], 2) for i in range(6)])
    assert torch.equal(dilate_cross_t(st, 2), one)
