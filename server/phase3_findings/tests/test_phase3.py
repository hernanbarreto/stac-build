# STAC-Builder — Phase 3 unit tests (synthetic, no model / no I/O of frames).
#
# Covers the deterministic core: single-pixel 3D anchoring round-trip, the
# localized residual proxy (flat vs. rough surface), multi-view dedup by
# type + 3D proximity, severity bump, and finding-JSON parsing robustness.
#
# Hernán Barreto - Ingerop IN3 Session IV - STAC

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from phase_r.geometry import unproject_pixel_to_world  # noqa: E402
from phase_r.instance_store import InstanceStore  # noqa: E402
from phase3_findings.detect import (  # noqa: E402
    FindingDetector, RawFinding, _parse_findings,
)

K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])


def _detector(store_path):
    # construct without touching frames/masks (run() loads those lazily)
    return FindingDetector(store_path, session_dir="/nonexistent",
                           output_dir="/nonexistent", config={"phase3": {}})


# ── 3D anchoring round-trip ─────────────────────────────────────────
def test_unproject_roundtrip_identity_pose():
    # world point in front of an identity camera; project, then unproject
    Pw = np.array([0.3, -0.2, 2.0])
    u = K[0, 0] * Pw[0] / Pw[2] + K[0, 2]
    v = K[1, 1] * Pw[1] / Pw[2] + K[1, 2]
    depth = np.full((480, 640), np.nan, np.float32)
    depth[int(round(v)), int(round(u))] = Pw[2]
    out = unproject_pixel_to_world(u, v, depth, K, np.eye(4), win=0)
    assert out is not None and np.linalg.norm(out - Pw) < 1e-3


def test_unproject_uses_pose():
    Pw = np.array([0.0, 0.0, 2.0])
    depth = np.full((480, 640), 2.0, np.float32)
    c2w = np.eye(4); c2w[:3, 3] = [10.0, 5.0, -1.0]  # translated camera
    out = unproject_pixel_to_world(320, 240, depth, K, c2w, win=0)
    assert np.linalg.norm(out - (Pw + c2w[:3, 3])) < 1e-6


def test_unproject_none_when_no_depth():
    depth = np.zeros((480, 640), np.float32)
    assert unproject_pixel_to_world(320, 240, depth, K, np.eye(4), win=0) is None


# ── localized residual proxy ────────────────────────────────────────
def test_local_residual_flat_vs_rough():
    with tempfile.TemporaryDirectory() as d:
        sp = os.path.join(d, "s.db")
        store = InstanceStore(sp)
        store.upsert_instance(1, "wall")
        store.upsert_instance(2, "wall")
        rng = np.random.default_rng(0)
        xy = rng.uniform(-0.5, 0.5, size=(400, 2))
        flat = np.column_stack([xy, np.zeros(len(xy))])            # z=0 plane
        rough = np.column_stack([xy, rng.normal(0, 0.12, len(xy))])  # ~12cm rms
        store.set_points(1, flat)
        store.set_points(2, rough)
        store.close()
        det = _detector(sp)
        pt = np.array([0.0, 0.0, 0.0])
        rf = det._local_residual(1, pt)
        rr = det._local_residual(2, pt)
        det.store.close()
        assert rf is not None and rf < 0.01           # flat -> tiny residual
        assert rr is not None and rr > det.residual_thresh_m  # rough -> above thresh


def test_local_residual_off_cloud_is_none():
    with tempfile.TemporaryDirectory() as d:
        sp = os.path.join(d, "s.db")
        store = InstanceStore(sp)
        store.upsert_instance(1, "wall")
        store.set_points(1, np.zeros((50, 3)))
        store.close()
        det = _detector(sp)
        assert det._local_residual(1, np.array([100.0, 100.0, 100.0])) is None
        det.store.close()


# ── multi-view dedup ────────────────────────────────────────────────
def _rf(typ, pt, conf=0.8):
    return RawFinding(type=typ, box_xywh=(0.1, 0.1, 0.1, 0.1), severity="medium",
                      description="", confidence=conf, frame_id=0,
                      instance_id=1, point3d=np.asarray(pt, float))


def test_dedup_merges_same_type_near():
    with tempfile.TemporaryDirectory() as d:
        det = _detector(os.path.join(d, "s.db"))
        raws = [_rf("crack", [0, 0, 0], 0.7), _rf("crack", [0.05, 0, 0], 0.9),
                _rf("crack", [1.0, 0, 0], 0.6)]  # third is far -> separate
        merged = det._dedup(raws)
        det.store.close()
        assert len(merged) == 2
        # the merged near-cluster keeps the higher confidence representative
        near = [m for m in merged if m.point3d[0] < 0.5]
        assert len(near) == 1 and abs(near[0].confidence - 0.9) < 1e-9


def test_dedup_keeps_different_types_apart():
    with tempfile.TemporaryDirectory() as d:
        det = _detector(os.path.join(d, "s.db"))
        merged = det._dedup([_rf("crack", [0, 0, 0]), _rf("moisture", [0, 0, 0])])
        det.store.close()
        assert len(merged) == 2


def test_dedup_drops_unanchored():
    with tempfile.TemporaryDirectory() as d:
        det = _detector(os.path.join(d, "s.db"))
        r = _rf("crack", [0, 0, 0]); r.point3d = None
        merged = det._dedup([r])
        det.store.close()
        assert merged == []


# ── severity + parsing ──────────────────────────────────────────────
def test_severity_bump_saturates():
    assert FindingDetector._bump("low") == "medium"
    assert FindingDetector._bump("medium") == "high"
    assert FindingDetector._bump("high") == "high"


def test_parse_findings_variants():
    assert _parse_findings('{"findings": []}') == []
    assert _parse_findings("garbage no json") == []
    assert len(_parse_findings('prose {"findings":[{"type":"crack"}]} tail')) == 1


if __name__ == "__main__":
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("PASS", name); n += 1
    print(f"\n{n} tests passed")
