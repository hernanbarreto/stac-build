"""Manhattan world (USER 2026-10-15): every object is marked by default, the user unmarks; the marked
planes snap EXACTLY to the scene frame (floor horizontal, walls vertical and on one of two axes),
coplanar pieces merge, the deviation is recorded, the cloud is untouched."""
import json

import numpy as np
import pytest

from reconstruction.surface_fit.manhattan import regularize_manhattan, scene_frame
from reconstruction.surface_fit.plane import PlaneModel
from segmentation import manhattan as M


def _rot(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    a = np.deg2rad(deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def _plane_pts(normal, d, extent, n, seed):
    """Points on n·x + d = 0 with 3 mm noise."""
    rng = np.random.default_rng(seed)
    n_ = np.asarray(normal, float) / np.linalg.norm(normal)
    ref = np.array([1.0, 0, 0]) if abs(n_[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(n_, ref); u /= np.linalg.norm(u); v = np.cross(n_, u)
    s = rng.uniform(-extent[0], extent[0], n); t = rng.uniform(-extent[1], extent[1], n)
    pts = (-d) * n_ + s[:, None] * u + t[:, None] * v + rng.normal(0, 0.003, (n, 1)) * n_
    return pts


def _fit(pts):
    """A PlaneModel from the points' own PCA — deterministic (Open3D's RANSAC in fit_plane draws from
    an unseeded engine; the regularization, not the fit, is under test here)."""
    from reconstruction.surface_fit.plane import _plane_basis
    pts = np.asarray(pts, float)
    ctr = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - ctr, full_matrices=False)
    n = vt[-1] / np.linalg.norm(vt[-1])
    if n @ np.array([0, 0, 1.0]) < 0 and abs(n[2]) > 0.5:
        n = -n
    d = float(-n @ ctr)
    u, v = _plane_basis(n, np.array([0, 0, 1.0]))
    return PlaneModel(normal=n, d=d, origin=ctr.copy(), u=u, v=v, rms=0.003, inlier_frac=1.0, n_points=len(pts))


def test_marks_default_on_and_only_exceptions_are_stored(tmp_path):
    assert M.is_manhattan(tmp_path, 7) and M.flags_for(tmp_path, [1, 2]) == {1: True, 2: True}
    M.set_flag(tmp_path, 7, False)
    assert not M.is_manhattan(tmp_path, 7) and M.is_manhattan(tmp_path, 8)
    doc = json.loads((tmp_path / M.FILE).read_text())
    assert doc["off"] == [7] and doc["provenance"] == "human_validated"
    M.set_flag(tmp_path, 7, True)
    assert M.load(tmp_path)["off"] == []
    rows = M.annotate(tmp_path, [{"instance_id": 7}, {"id": 9}])
    assert rows[0]["manhattan"] is True and rows[1]["manhattan"] is True


def test_the_marked_planes_snap_exactly_to_the_measured_frame():
    up_true = _rot([1, 0, 0], 2.0) @ np.array([0, 0, 1.0])            # the floor, 2° off the hint
    az = np.deg2rad(7.0)                                                # the walls' dominant azimuth
    nx = np.array([np.cos(az), np.sin(az), 0.0])
    nz = np.array([-np.sin(az), np.cos(az), 0.0])
    floor = _plane_pts(up_true, 0.0, (4, 3), 20000, 1)
    wall_a = _plane_pts(_rot([0, 0, 1], 1.2) @ nx, -3.0, (2, 1.2), 12000, 2)      # 1.2° off its axis
    wall_b = _plane_pts(_rot([0, 0, 1], -0.8) @ nz, 2.0, (2, 1.2), 9000, 3)
    wall_b2 = _plane_pts(nz, 2.004, (1, 1.2), 4000, 4)                           # a 4 mm coplanar piece
    ramp = _plane_pts(_rot([1, 0, 0], 30.0) @ np.array([0, 0, 1.0]), -1.0, (1, 1), 5000, 5)  # not Manhattan
    models = [_fit(p) for p in (floor, wall_a, wall_b, wall_b2, ramp)]
    pts = [floor, wall_a, wall_b, wall_b2, ramp]
    out, rep = regularize_manhattan(models, [len(p) for p in pts], pts, up_hint=np.array([0, 0, 1.0]),
                                    snap_deg=15.0, coplanar_tol_mm=10.0, ids=[1, 2, 3, 4, 5])
    fr = rep["frame"]
    up, X, Z = (np.asarray(fr[k]) for k in ("up", "X", "Z"))
    assert abs(up @ X) < 1e-12 and abs(up @ Z) < 1e-12 and abs(X @ Z) < 1e-12, "an orthonormal frame"
    assert fr["up_from_instance"] == 1 and abs(fr["up_tilt_deg"] - 2.0) < 0.2
    assert abs(fr["azimuth_deg"] - 7.0) < 0.5 and fr["n_walls"] == 3
    objs = {o["instance_id"]: o for o in rep["objects"]}
    assert objs[1]["snapped"] and objs[1]["axis"] == "up" and abs(abs(out[0].normal @ up) - 1) < 1e-12
    assert objs[2]["snapped"] and objs[2]["axis"] == "X" and abs(abs(out[1].normal @ X) - 1) < 1e-12
    assert 0.5 < objs[2]["moved_deg"] < 2.5, "the degrees the wall's normal moved are recorded"
    assert objs[3]["axis"] == "Z" and objs[4]["axis"] == "Z"
    # one plane: the same offset along the shared axis (each piece keeps its own normal sign)
    assert np.allclose(np.abs(out[2].normal @ out[3].normal), 1.0)
    assert abs(abs(out[2].d) - abs(out[3].d)) < 1e-12 and objs[3]["coplanar_group"] == [3, 4] and rep["coplanar_groups"] == [[3, 4]]
    assert not objs[5]["snapped"] and 28 < objs[5]["off_manhattan_deg"] < 32 and "kept as measured" in objs[5]["reason"]
    assert np.allclose(out[4].normal, models[4].normal), "an off-Manhattan plane keeps its measured normal"
    # the snapped offsets sit on the points (robust median), the basis follows the frame
    assert abs(np.median(floor @ out[0].normal + out[0].d)) < 1e-9
    assert np.allclose(out[1].v, up) or np.allclose(out[1].v, -up)
    assert np.allclose(np.abs(out[0].u @ X), 1.0)
    for m in out:
        assert isinstance(m, PlaneModel) and abs(np.linalg.norm(m.normal) - 1) < 1e-12


def test_a_scene_with_no_walls_keeps_the_hint_axes():
    fr = scene_frame([np.array([0, 0, 1.0])], [100.0], np.array([0, 0, 1.0]), 15.0)
    assert fr["n_walls"] == 0 and abs(fr["azimuth_deg"]) < 1e-9 and np.allclose(fr["up"], [0, 0, 1])
    fr0 = scene_frame([], [], np.array([0, 0, 1.0]), 15.0)
    assert fr0["up_from"] is None


@pytest.fixture()
def session(tmp_path):
    """A floor 1.5° off horizontal and a wall 2° off its axis, written as a session."""
    import open3d as o3d
    floor = _plane_pts(_rot([1, 0, 0], 1.5) @ np.array([0, 0, 1.0]), 0.0, (3, 2), 40000, 21)
    wall = _plane_pts(_rot([0, 0, 1], 2.0) @ np.array([1.0, 0, 0]), -3.0, (2, 1.2), 30000, 22)
    cloud = np.vstack([floor, wall])
    out = tmp_path / "output"
    out.mkdir()
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud))
    o3d.io.write_point_cloud(str(out / "cleaned_cloud.ply"), pcd)
    seg = {"instances": [{"id": 1, "instance_id": 1, "label": "floor", "globalIndices": list(range(len(floor)))},
                         {"id": 2, "instance_id": 2, "label": "wall",
                          "globalIndices": list(range(len(floor), len(floor) + len(wall)))}]}
    (out / "segmentation_result.json").write_text(json.dumps(seg))
    return tmp_path


def test_fit_scene_models_the_marked_objects_straight_and_the_unmarked_as_measured(session):
    from reconstruction.surface_fit.scene import fit_scene
    ov = {"consolidate_method": "none", "models": ["bspline"], "hole_audit": False,
          "texture_objects": False, "decompose": False}
    rep = fit_scene(session, overrides=ov, hole_audit=False, texture_objects=False, decompose=False)
    by = {r["instance_id"]: r for r in rep["fitted"]}
    assert by[1]["kind"] == "plane" and by[2]["kind"] == "plane", "marked → planes, not the bspline ladder"
    assert by[1]["manhattan"]["snapped"] and by[1]["manhattan"]["axis"] == "up"
    assert by[2]["manhattan"]["snapped"] and by[2]["manhattan"]["axis"] in ("X", "Z")
    # ONE wall: the frame's horizontal axis IS its direction — only its tilt from vertical moves
    assert 0.0 <= by[2]["manhattan"]["moved_deg"] < 0.5
    fr = rep["manhattan"]["frame"]
    assert fr["up_from_instance"] == 1 and 1.0 < fr["up_tilt_deg"] < 2.0
    up, X, Z = (np.asarray(fr[k]) for k in ("up", "X", "Z"))
    assert abs(up @ X) < 1e-12 and abs(up @ Z) < 1e-12 and abs(X @ Z) < 1e-12
    side = json.loads((session / "output" / "surface_fit" / "wall_2" / "manhattan.json").read_text())
    assert side["snapped"] and side["frame"]["n_walls"] == 1
    # the user unmarks the wall: it follows the configured ladder (bspline), the floor stays straight
    M.set_flag(session / "output", 2, False)
    rep2 = fit_scene(session, overrides=ov, hole_audit=False, texture_objects=False, decompose=False)
    by2 = {r["instance_id"]: r for r in rep2["fitted"]}
    assert by2[2]["kind"] == "bspline" and by2[2]["manhattan"] is None
    assert by2[1]["kind"] == "plane" and by2[1]["manhattan"]["snapped"]
    assert rep2["manhattan"]["frame"]["n_walls"] == 0


def test_the_keys_are_declared():
    from config import cfg
    from reconstruction.surface_fit.scene import scene_config_kwargs
    kw = scene_config_kwargs(cfg)
    assert kw["manhattan"] is True and kw["manhattan_snap_deg"] > 0 and 0 < kw["manhattan_min_inlier_frac"] < 1
    assert 0 < kw["manhattan_part_min_frac"] < 0.5
    assert kw["manhattan_residue"] == "declare" and kw["tolerance"] == "measured"     # USER 2026-10-16
    assert kw["tolerance_k"] > 0 and kw["noise_k_neighbours"] >= 4 and kw["noise_sample"] >= 1000


def test_the_frame_takes_the_sessions_up_from_the_floor_transform(tmp_path):
    """pccr's raw cloud is Y-up (baked orientation, floor_transform.npz = identity): the Manhattan
    up is Rᵀ·ŷ, never the fit's z-up default — which read the floor as a wall (2026-10-15)."""
    from reconstruction.surface_fit.scene import _session_up
    out = tmp_path / "output"
    out.mkdir()
    assert np.allclose(_session_up(out), [0, 0, 1]), "no transform: the fit's default"
    np.savez(out / "floor_transform.npz", s=1.0, R=np.eye(3), t=np.zeros(3))
    assert np.allclose(_session_up(out), [0, 1, 0])
    R = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)       # display y = raw z
    np.savez(out / "floor_transform.npz", s=1.0, R=R, t=np.zeros(3))
    assert np.allclose(_session_up(out), [0, 0, 1])


def test_a_cluttered_floor_is_still_a_straight_plane_and_a_blob_is_not(tmp_path):
    """pccr's floor: 87 % of its points within 4 cm of the plane, p95 88 mm (clutter) — Manhattan
    keeps the plane and reports the deviation; a blob (nothing within 4 cm) follows the ladder."""
    import open3d as o3d
    from reconstruction.surface_fit.scene import fit_scene
    rng = np.random.default_rng(9)
    floor = _plane_pts(np.array([0, 0, 1.0]), 0.0, (3, 2), 40000, 31)
    clutter = np.column_stack([rng.uniform(-3, 3, 8000), rng.uniform(-2, 2, 8000), rng.uniform(0.1, 0.4, 8000)])
    blob = rng.normal(0, 0.3, (6000, 3)) + np.array([2.0, 1.5, 1.0])
    cloud = np.vstack([floor, clutter, blob])
    out = tmp_path / "output"
    out.mkdir()
    o3d.io.write_point_cloud(str(out / "cleaned_cloud.ply"), o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud)))
    nf = len(floor) + len(clutter)
    seg = {"instances": [{"id": 1, "instance_id": 1, "label": "floor", "globalIndices": list(range(nf))},
                         {"id": 2, "instance_id": 2, "label": "thing", "globalIndices": list(range(nf, nf + len(blob)))}]}
    (out / "segmentation_result.json").write_text(json.dumps(seg))
    rep = fit_scene(tmp_path, overrides={"consolidate_method": "none", "models": ["plane"]},
                    hole_audit=False, texture_objects=False, decompose=False)
    by = {r["instance_id"]: r for r in rep["fitted"]}
    m = by[1]["manhattan"]
    assert by[1]["kind"] == "plane" and m["snapped"], "the floor stays a straight plane"
    # ONE segment holds the floor (83 % of the object); the clutter is the declared residue —
    # a volume at the floor's resolution seeds no plane (USER 2026-10-16: plane segments)
    assert m["facets"]["n_segments"] == 1 and 0.80 < m["facets"]["explained"] < 0.86, m["facets"]
    assert m["residue"]["declared"] and 0.14 < m["residue"]["frac"] < 0.20
    assert not m["partial"] and not m["residue_meshed"]
    assert not (out / "surface_fit" / "floor_1" / "unexplained_idx.npy").exists(), "a surface's clutter is declared, never meshed"
    assert 2 not in by and "2" in rep["manhattan"]["not_a_surface"]


def _desk_session(tmp_path):
    """pccr desk 351 (2026-10-15): a straight top with four tubular legs no plane holds."""
    import open3d as o3d
    from reconstruction.surface_fit.scene import fit_scene
    rng = np.random.default_rng(13)
    top = _plane_pts(np.array([0, 0, 1.0]), -0.75, (0.8, 0.4), 35000, 61)   # z = 0.75, tilted 0°
    legs = []
    for cx, cy in ((-0.7, -0.3), (0.7, -0.3), (-0.7, 0.3), (0.7, 0.3)):
        a = rng.uniform(0, 2 * np.pi, 4000); z = rng.uniform(0, 0.75, 4000)
        legs.append(np.column_stack([cx + 0.03 * np.cos(a), cy + 0.03 * np.sin(a), z]))
    legs = np.vstack(legs)
    cloud = np.vstack([top, legs])
    out = tmp_path / "output"
    out.mkdir()
    o3d.io.write_point_cloud(str(out / "cleaned_cloud.ply"), o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud)))
    seg = {"instances": [{"id": 1, "instance_id": 1, "label": "desk", "globalIndices": list(range(len(cloud)))}]}
    (out / "segmentation_result.json").write_text(json.dumps(seg))
    return out


def test_a_desk_top_is_a_straight_plane_and_its_legs_are_facets_never_cylinders(tmp_path):
    """USER 2026-10-16: "olvídate de los cilindros … puede estar facetado … sino va a forzar cilindros
    en donde no aplica" and "segmentos de plano, no único plano que describe todo". The top is the
    dominant segment, snapped to the frame (horizontal); each 6 cm leg is described by vertical plane
    SEGMENTS at the desk's own cell, kept as measured (never snapped, never a cylinder); nothing goes
    to a TSDF or Poisson. The tolerance is the resolution's, recorded."""
    from reconstruction.surface_fit.scene import fit_scene
    out = _desk_session(tmp_path)
    rep = fit_scene(tmp_path, overrides={"consolidate_method": "none", "models": ["bspline"]},
                    hole_audit=False, texture_objects=False, decompose=False)
    by = {r["instance_id"]: r for r in rep["fitted"]}
    m = by[1]["manhattan"]
    assert by[1]["kind"] == "plane" and m["snapped"] and m["axis"] == "up"
    fac = m["facets"]
    assert fac["explained"] > 0.95 and fac["residue"] < 0.05, fac
    side = json.loads((out / "surface_fit" / "desk_1" / "manhattan.json").read_text())
    segs = side["segments"]
    assert 0.60 < segs[0]["share"] < 0.75 and segs[0]["tilt_from_up_deg"] < 1.0            # the top
    legs = [q for q in segs[1:] if q["tilt_from_up_deg"] > 80.0]
    assert len(legs) >= 4 and sum(q["share"] for q in legs) > 0.2, segs                     # the legs, as facets
    assert "cylinders" not in m.get("residue", {}) and m["residue"]["declared"]
    assert not (out / "surface_fit" / "desk_1" / "unexplained_idx.npy").exists(), "never Poisson"
    parts = m["parts"]
    # the top is on the frame; a leg's facet may be too when it is nearly on an axis and the snap
    # moves its far edge less than the tolerance (harmless); the rest of the legs' facets are measured
    assert parts[0].get("snapped") and sum(1 for p in parts if p.get("snapped")) <= 2, parts
    assert sum(1 for p in parts if not p.get("snapped")) >= 4
    tol = m["tolerance"]
    assert tol["from"] == "measured" and abs(tol["tolerance_m"] - tol["cell_m"] / 2) < 1e-6, tol
    import trimesh
    mesh = trimesh.load(str(out / "surface_fit" / "desk_1" / "surface.glb"), force="mesh")
    V = np.asarray(mesh.vertices, float)
    assert V[:, 2].min() < 0.05 and V[:, 2].max() > 0.7, "the published mesh holds the top AND the legs"
    assert side["residue"]["declared"] and side["tolerance"]["from"] == "measured"


def test_the_objects_own_noise_is_measured_not_declared():
    from reconstruction.surface_fit.noise import local_thickness_sigma
    for sig in (0.003, 0.012):
        pts = _plane_pts(np.array([0, 0, 1.0]), 0.0, (2, 1), 30000, 5)
        pts[:, 2] = np.random.default_rng(1).normal(0, sig, len(pts))      # a plane with that noise
        nz = local_thickness_sigma(pts, k=24, n_max=5000)
        assert abs(nz["sigma_m"] - sig) < 0.25 * sig, (sig, nz)
        assert nz["n_sampled"] == 5000 and nz["k"] == 24
    assert local_thickness_sigma(np.zeros((3, 3)))["n_sampled"] == 0


def test_the_rest_that_is_neither_plane_nor_bar_gets_the_live_cloud_tsdf(tmp_path):
    """A marked object = a plate (plane) with a blob under it: under the SELECTABLE 2026-10-15 route
    (`manhattan_residue: bars_tsdf`) the blob is not a bar → the live-cloud TSDF, never Poisson
    ("y si no, tsdf sin poisson"). The default since 2026-10-16 declares it (the desk test)."""
    from test_hole_audit_torch import _session as _cam_session
    from reconstruction.surface_fit.scene import fit_scene
    out, xyz, plate = _cam_session(tmp_path)          # a camera, poses, a record grid, provenance
    rng = np.random.default_rng(17)
    blob = rng.normal(0, 0.08, (2500, 3)) + np.array([0.0, 0.0, 1.75])     # under the plate (z = 1.5)
    # append the blob to the cloud with provenance, re-write the PLY
    from test_p3_projection import PLY_DTYPE, _write_ply
    from segmentation.pipeline import _load_ply_origins
    o = _load_ply_origins(out / "cleaned_cloud.ply")
    xyz0, fg, pr, pc = o
    d = np.zeros(len(xyz0) + len(blob), PLY_DTYPE)
    d["x"] = np.concatenate([xyz0[:, 0], blob[:, 0]]); d["y"] = np.concatenate([xyz0[:, 1], blob[:, 1]])
    d["z"] = np.concatenate([xyz0[:, 2], blob[:, 2]])
    frames3 = np.asarray(sorted(set(int(x) for x in fg)))[:3]
    d["frame_global"] = np.concatenate([fg, frames3[np.arange(len(blob)) % len(frames3)]])   # seen from every keyframe
    d["pixel_row"] = np.concatenate([pr, np.zeros(len(blob), int)]); d["pixel_col"] = np.concatenate([pc, np.zeros(len(blob), int)])
    d["confidence"] = 0.9
    _write_ply(out / "cleaned_cloud.ply", d)
    n_wall = 5000
    idx = list(range(n_wall, n_wall + 1200)) + list(range(len(xyz0), len(xyz0) + len(blob)))
    (out / "segmentation_result.json").write_text(json.dumps(
        {"instances": [{"id": 1, "instance_id": 1, "label": "thing", "globalIndices": idx}]}))
    cfg = {"surface_fit": {}, "tsdf": {"object": {"depth_source": "live_cloud", "device": "cpu", "voxel_length": 0.02,
                                                   "sdf_trunc": 0.06, "depth_min": 0.1, "depth_trunc": 5.0, "fill_px": 1,
                                                   "weight_threshold": 1.0, "block_count": 5000}}}
    rep = fit_scene(tmp_path, config=cfg, overrides={"consolidate_method": "none", "models": ["plane"]},
                    hole_audit=False, texture_objects=False, decompose=False,
                    manhattan_min_inlier_frac=0.75, manhattan_part_min_frac=0.05,
                    manhattan_residue="bars_tsdf", tolerance="declared")      # the 2026-10-15 route, selectable
    by = {r["instance_id"]: r for r in rep["fitted"]}
    m = by[1]["manhattan"]
    assert m["snapped"] and m["partial"] and 0.4 < m["residue_frac"] < 0.8, m        # the blob seeds no plane
    assert m["residue"]["cylinders"] == [] and "vertices" in m["residue"]["tsdf"], m["residue"]
    assert m["residue"]["tsdf"]["n_points"] >= 0.5 * len(blob) and m["residue"]["tsdf"]["vertices"] > 50
    assert not (out / "surface_fit" / "thing_1" / "unexplained_idx.npy").exists()


def test_an_s_shaped_wall_becomes_frame_aligned_planes_with_exact_offsets(tmp_path):
    """USER 2026-10-15: a wall of three parallel runs (0, +0.30 m, 0) joined by two returns — one
    plane cannot explain it; the Manhattan decomposition gives five planes on the frame: the runs
    on one axis with their exact offsets (the first and third ONE plane), the returns on the other."""
    import open3d as o3d
    from reconstruction.surface_fit.scene import fit_scene
    rng = np.random.default_rng(41)

    def patch(n, x, y0, y1, z0, z1, normal):
        # points on a vertical patch: x fixed (normal x) or y fixed (normal y)
        if normal == "x":
            P = np.column_stack([np.full(n, x), rng.uniform(y0, y1, n), rng.uniform(z0, z1, n)])
            P[:, 0] += rng.normal(0, 0.003, n)
        else:
            P = np.column_stack([rng.uniform(y0, y1, n), np.full(n, x), rng.uniform(z0, z1, n)])
            P[:, 1] += rng.normal(0, 0.003, n)
        return P
    run1 = patch(20000, 0.0, 0.0, 2.0, 0.0, 2.5, "x")
    run2 = patch(20000, 0.3, 2.0, 4.0, 0.0, 2.5, "x")
    run3 = patch(20000, 0.0, 4.0, 6.0, 0.0, 2.5, "x")
    ret1 = patch(5000, 2.0, 0.0, 0.3, 0.0, 2.5, "y")     # the return at y = 2, x ∈ [0, 0.3]
    ret2 = patch(5000, 4.0, 0.0, 0.3, 0.0, 2.5, "y")
    wall = np.vstack([run1, run2, run3, ret1, ret2])
    floor = _plane_pts(np.array([0, 0, 1.0]), 0.0, (4, 4), 30000, 42) + np.array([1.0, 3.0, 0.0])
    cloud = np.vstack([floor, wall])
    out = tmp_path / "output"
    out.mkdir()
    o3d.io.write_point_cloud(str(out / "cleaned_cloud.ply"), o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud)))
    seg = {"instances": [{"id": 1, "instance_id": 1, "label": "floor", "globalIndices": list(range(len(floor)))},
                         {"id": 2, "instance_id": 2, "label": "wall",
                          "globalIndices": list(range(len(floor), len(floor) + len(wall)))}]}
    (out / "segmentation_result.json").write_text(json.dumps(seg))
    rep = fit_scene(tmp_path, overrides={"consolidate_method": "none", "models": ["plane"]},
                    hole_audit=False, texture_objects=False, decompose=False,
                    decompose_min_points=1500, decompose_min_frac=0.05, decompose_max_primitives=8)
    by = {r["instance_id"]: r for r in rep["fitted"]}
    w = by[2]["manhattan"]
    # the first and third runs lie in ONE plane (x = 0): two segments 2 m apart, the coplanar
    # merge keeps them one plane; the returns are their own segments; a few tiny facets at the
    # corners (under manhattan_part_min_frac) stay as measured, never on the frame
    big = [p for p in w["parts"] if p.get("snapped")]
    assert w["decomposed"] and len(big) == 5, w
    assert all(p["n_points"] < 0.05 * 70000 for p in w["parts"] if not p.get("snapped"))
    axes = sorted(p["axis"] for p in big)
    assert axes == ["X", "X", "X", "Z", "Z"], axes
    runs = [p for p in big if p["axis"] == "X"]
    assert sum(1 for p in runs if p.get("coplanar_group")) == 2, "the first and third runs: one plane"
    assert w["residue_frac"] < 0.05
    assert rep["manhattan"]["frame"]["up_from_instance"] == 1, "up = the floor, not the heavier wall"
    side = json.loads((out / "surface_fit" / "wall_2" / "manhattan.json").read_text())
    assert side["decomposed"] and sum(1 for p in side["parts"] if p.get("snapped")) == 5
    # the exported mesh spans the whole S: both offsets and the full length
    import trimesh
    mesh = trimesh.load(str(out / "surface_fit" / "wall_2" / "surface.glb"), force="mesh")
    V = np.asarray(mesh.vertices, float)
    assert V[:, 0].min() < 0.02 and V[:, 0].max() > 0.28 and V[:, 1].min() < 0.1 and V[:, 1].max() > 5.9
    xs = np.unique(np.round(V[:, 0], 2))
    assert 0.0 in xs and 0.3 in xs, f"the runs sit at exactly 0 and 0.30 m: {xs[:8]}"
