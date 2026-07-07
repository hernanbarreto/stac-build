# STAC-Builder — unit tests for geometric temporal association (Phase 1).
# Synthetic pinhole geometry: a fronto-parallel square seen from two translated
# cameras must merge into ONE instance; a far-away detection must NOT merge.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from segmentation.autoprompt.associate import associate_detections  # noqa: E402
from segmentation.autoprompt.detector import Detection  # noqa: E402

IMG_W, IMG_H = 640, 480
K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])


def _project_point(Xw, c2w):
    cam = np.linalg.inv(c2w) @ np.append(Xw, 1.0)
    z = cam[2]
    u = K[0, 0] * cam[0] / z + K[0, 2]
    v = K[1, 1] * cam[1] / z + K[1, 2]
    return u, v


def _square_box(center, half, c2w):
    """Normalized xyxy of a fronto-parallel square projected into c2w."""
    cx, cy, cz = center
    corners = [
        (cx - half, cy - half, cz), (cx + half, cy - half, cz),
        (cx + half, cy + half, cz), (cx - half, cy + half, cz),
    ]
    us, vs = [], []
    for c in corners:
        u, v = _project_point(np.array(c), c2w)
        us.append(u); vs.append(v)
    return (min(us) / IMG_W, min(vs) / IMG_H, max(us) / IMG_W, max(vs) / IMG_H)


def _c2w(tx=0.0, ty=0.0, tz=0.0):
    T = np.eye(4)
    T[:3, 3] = [tx, ty, tz]
    return T


def test_same_object_two_frames_merges():
    c2w_i = _c2w()               # camera at origin, +Z forward
    c2w_j = _c2w(tx=0.5)         # moved 0.5 m right
    center = (0.0, 0.0, 4.0)     # square 4 m ahead
    box_i = _square_box(center, 0.5, c2w_i)
    box_j = _square_box(center, 0.5, c2w_j)

    dets = {
        10: [Detection(label="column", box=box_i, confidence=0.9, frame_id=10)],
        20: [Detection(label="column", box=box_j, confidence=0.8, frame_id=20)],
    }
    pose_map = {10: c2w_i, 20: c2w_j}
    depth_provider = lambda fid: (np.full((IMG_H, IMG_W), 4.0, np.float32), K)

    insts = associate_detections(
        dets, pose_map, lambda fid: K, lambda fid: (IMG_W, IMG_H),
        depth_provider=depth_provider, iou_threshold=0.3,
    )
    assert len(insts) == 1, f"expected 1 merged instance, got {len(insts)}"
    assert insts[0].n_views == 2
    assert insts[0].label == "column"


def test_different_objects_do_not_merge():
    c2w_i = _c2w()
    c2w_j = _c2w(tx=0.5)
    near = _square_box((0.0, 0.0, 4.0), 0.5, c2w_i)
    far = _square_box((6.0, 0.0, 4.0), 0.5, c2w_j)  # 6 m to the side -> different
    dets = {
        10: [Detection(label="column", box=near, confidence=0.9, frame_id=10)],
        20: [Detection(label="column", box=far, confidence=0.8, frame_id=20)],
    }
    pose_map = {10: c2w_i, 20: c2w_j}
    depth_provider = lambda fid: (np.full((IMG_H, IMG_W), 4.0, np.float32), K)
    insts = associate_detections(
        dets, pose_map, lambda fid: K, lambda fid: (IMG_W, IMG_H),
        depth_provider=depth_provider, iou_threshold=0.3,
    )
    assert len(insts) == 2, f"expected 2 separate instances, got {len(insts)}"


def test_different_labels_never_merge():
    c2w_i = _c2w()
    c2w_j = _c2w(tx=0.5)
    center = (0.0, 0.0, 4.0)
    box_i = _square_box(center, 0.5, c2w_i)
    box_j = _square_box(center, 0.5, c2w_j)
    dets = {
        10: [Detection(label="column", box=box_i, confidence=0.9, frame_id=10)],
        20: [Detection(label="pipe", box=box_j, confidence=0.8, frame_id=20)],
    }
    pose_map = {10: c2w_i, 20: c2w_j}
    insts = associate_detections(
        dets, pose_map, lambda fid: K, lambda fid: (IMG_W, IMG_H),
        depth_provider=lambda fid: (np.full((IMG_H, IMG_W), 4.0, np.float32), K),
    )
    assert len(insts) == 2


def test_no_poses_fallback_keeps_separate():
    dets = {
        10: [Detection(label="column", box=(0.1, 0.1, 0.3, 0.9), confidence=0.9, frame_id=10)],
        20: [Detection(label="column", box=(0.1, 0.1, 0.3, 0.9), confidence=0.8, frame_id=20)],
    }
    # No poses -> geometric linking impossible -> never merge by appearance.
    insts = associate_detections(dets, None, None, lambda fid: (IMG_W, IMG_H))
    assert len(insts) == 2


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("PASS", name)
    print("all association tests passed")
