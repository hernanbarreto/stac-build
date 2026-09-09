"""USER 2026-09-09: the loop closure is applied per CHUNK (SE(3) blocks,
seam-weighted), never smeared linearly over keyframes that were right."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from correction import distribute                                 # noqa: E402
from correction.run import run_objects                            # noqa: E402
from correction.ledger import load_epoch_npz                      # noqa: E402
from reconstruction.chunk_plan import chunk_ranges                # noqa: E402
from tests.synth_correction import build_scene, make_correction_cfg  # noqa: E402


def _plan(n_kf=30, size=12, ov=6):
    return {"version": 1, "phase": "chunked-metric", "n_keyframes": n_kf,
            "chunk_size": size, "overlap": ov,
            "chunk_ranges": [[a, b] for a, b in chunk_ranges(n_kf, size, ov)]}


def test_chunk_graph_keeps_inner_chunks_rigid_and_weights_bad_seam():
    plan = _plan()                    # ranges [0,12),[6,18),[12,24),[18,30)
    sol = [{"anchor_kf": 26, "kf_span": [24, 29], "R": np.eye(3),
            "t": np.array([1.0, 0.0, 0.0]), "k": 1.0}]
    # seam 2 (chunks 2→3) glued badly: it must absorb most of the closure
    R, t, k, rep = distribute.distribute_chunks(30, 3, sol, plan,
                                                [0.01, 0.01, 0.98])
    assert rep["mode"] == "chunk_graph"
    assert np.allclose(t[0:4], 0)                          # reference: identity
    ch = {c["chunk"]: c for c in rep["chunks"]}
    assert ch[0]["t_m"] == 0.0
    assert abs(ch[1]["t_m"] - 0.01) < 1e-6                  # seam-0 share only
    assert abs(ch[2]["t_m"] - 0.02) < 1e-6                  # + seam-1 share
    assert abs(ch[3]["t_m"] - 1.0) < 1e-6                   # bad seam 2: 98 %
    # keyframes shared by chunks 1 and 2 stay within those tiny shares —
    # the correct middle sector is NOT smeared with the closure
    assert np.all(t[12:18, 0] < 0.03)
    # the overlap [18,24) absorbs the closure monotonically; after it, full
    assert np.all(np.diff(t[18:24, 0]) > 0)
    assert np.allclose(t[24:30], [1.0, 0.0, 0.0])


def test_run_uses_chunk_plan_when_present(tmp_path):
    scene = build_scene(tmp_path, drift_yaw_deg=1.5, drift_t=(0.25, 0.0, 0.1),
                        chunk_plan=_plan())
    rep = run_objects(scene.output_dir, [1, 2], "test",
                      cfg=make_correction_cfg(**{"gates.mode": "advisory"}))
    assert rep["status"] == "pending", rep.get("rejection_reason")
    assert rep["distribution"]["mode"] == "chunk_graph"
    d = rep["distribution"]
    ch = {c["chunk"]: c for c in d["chunks"]}
    # the target chunk carries the full closure; the reference chunk none
    assert ch[0]["t_m"] == 0.0
    assert abs(ch[len(ch) - 1]["t_m"] - d["anchors"][0]["t_m"]) < 1e-6
    assert len(d["seam_weights"]) == len(ch) - 1
