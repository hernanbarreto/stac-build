"""The complete synthetic run of the audited chain (2026-10-23): a rendered room, a walk with a
revisit, Omega's errors injected (drift, per-keyframe scale-and-shift in inverse depth, a floater
three keyframes agree on), then the REAL code paths — F5 (odometry priors, loops), F6 (bend judged
+ vote + owner fusion + free-space carving), the cloud assembled the way publish_cloud does (the
GPU cleaner and the epoch transaction are the two steps that cannot run here), SAM-like masks from
the rendered labels, the mask→cloud projection (member pixels) — and the numbers against the truth.

Three clouds are compared: (a) Omega as it comes (drifted poses, raw depth, every observation),
(b) F5 + bend + vote without the fusion (the layered cloud of every run before the audit),
(c) F5 + bend + vote + owner fusion + carving (the product). The projection runs on (b) and (c).

    python analysis/2026-10-23_precision_audit/e2e_synthetic.py [--stage f5|f6|project|figures]
Outputs: e2e_synthetic/{metrics.json, figures/*.png}; the heavy stages are cached in e2e_synthetic/cache.
Numbers for the eye, never a verdict (claude_stac.txt rule 3).
"""
import json
import os
import pickle
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SERVER = HERE.parents[1] / "server"
sys.path.insert(0, str(SERVER)); sys.path.insert(0, str(SERVER / "tests"))
OUT = HERE / "e2e_synthetic"
CACHE = OUT / "cache"
FIG = OUT / "figures"
for d in (CACHE, FIG):
    d.mkdir(parents=True, exist_ok=True)

# ── the world (camera-like world frame: x right, y down, z forward; metres) ──────────────────
W, H = 160, 120
FK = (130.0, 130.0, 79.5, 59.5)
FLOOR_Y, CEIL_Y, BACK_Z, LEFT_X, RIGHT_X = 1.3, -1.2, 5.0, -2.6, 2.6
BOX = np.array([[-0.5, 0.5], [0.5, 1.3], [2.6, 3.4]])          # a crate on the floor, in front of the wall
CAB = np.array([[0.9, 2.1], [0.4, 1.3], [3.6, 4.2]])            # a cabinet against the wall, to the right
FLOATER = {"centre": np.array([0.1, 0.55, 1.9]), "half": 0.15, "kfs": (3, 4, 5)}   # Omega's hallucination
L_FLOOR, L_WALL, L_BOX, L_CAB, L_FLOATER = 1, 2, 3, 4, 9
LABEL_NAME = {L_FLOOR: "floor", L_WALL: "wall", L_BOX: "box", L_CAB: "cabinet", L_FLOATER: "floater"}
N_KF = 14
FRAMES = [10 * (k + 1) for k in range(N_KF)]
SEED = 7


def rot_y(a):
    c, s = np.cos(a), np.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_x(a):
    c, s = np.cos(a), np.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def truth_poses():
    """Out along x looking at the far wall, back along x closer and a little lower, looking down —
    the return visit sees the crate and the cabinet again from the other side (the loop)."""
    c2w = []
    for k in range(N_KF):
        T = np.eye(4)
        if k <= 6:
            x, z, y = -1.0 + k / 3.0, 0.0, 0.0
            yaw, pitch = np.radians(-12 + 24 * k / 6), np.radians(2.0)
        else:
            j = k - 7
            x, z, y = 1.0 - j / 3.0, 0.5, 0.06
            yaw, pitch = np.radians(10 - 20 * j / 6), np.radians(7.0)
        T[:3, :3] = rot_y(yaw) @ rot_x(pitch)
        T[:3, 3] = [x, y, z]
        c2w.append(T)
    return np.array(c2w)


def drifted_poses(c2w, rng, rot_deg=0.25, trans_m=0.015):
    """Omega's poses: the truth with a random walk accumulated keyframe by keyframe (drift)."""
    out = c2w.copy()
    dR, dt = np.eye(3), np.zeros(3)
    for k in range(1, N_KF):
        a = np.radians(rng.normal(0, rot_deg, 3))
        dR = dR @ rot_x(a[0]) @ rot_y(a[1]) @ np.array([[np.cos(a[2]), -np.sin(a[2]), 0], [np.sin(a[2]), np.cos(a[2]), 0], [0, 0, 1]])
        dt = dt + rng.normal(0, trans_m, 3)
        out[k, :3, :3] = dR @ c2w[k, :3, :3]
        out[k, :3, 3] = dR @ c2w[k, :3, 3] + dt
    return out


def _aabb(C, D, box):
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (box[:, 0][None, :] - C[None, :]) / D
        t2 = (box[:, 1][None, :] - C[None, :]) / D
    tmin = np.max(np.minimum(t1, t2), 1); tmax = np.min(np.maximum(t1, t2), 1)
    hit = (tmax >= np.maximum(tmin, 1e-6))
    return np.where(hit, tmin, np.inf)


def render(c2w_k, k, with_floater):
    """Depth (camera z) and label per pixel by ray casting; the floater is a real surface for the
    keyframes Omega hallucinates it in."""
    fx, fy, cx, cy = FK
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    d = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], -1).reshape(-1, 3)
    R, C = c2w_k[:3, :3], c2w_k[:3, 3]
    D = d @ R.T
    with np.errstate(divide="ignore", invalid="ignore"):
        cands = {}
        s = (FLOOR_Y - C[1]) / D[:, 1]; cands[L_FLOOR] = np.where(s > 0, s, np.inf)
        s = (CEIL_Y - C[1]) / D[:, 1]; cands[-L_WALL] = np.where(s > 0, s, np.inf)           # the ceiling counts as wall
        s = (BACK_Z - C[2]) / D[:, 2]; cands[L_WALL] = np.where(s > 0, s, np.inf)
        s = (LEFT_X - C[0]) / D[:, 0]; cands[-L_WALL - 10] = np.where(s > 0, s, np.inf)
        s = (RIGHT_X - C[0]) / D[:, 0]; cands[-L_WALL - 20] = np.where(s > 0, s, np.inf)
    cands[L_BOX] = _aabb(C, D, BOX)
    cands[L_CAB] = _aabb(C, D, CAB)
    if with_floater:
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (FLOATER["centre"][2] - C[2]) / D[:, 2]
        P = C[None, :] + s[:, None] * D
        ok = (s > 0) & (np.abs(P[:, 0] - FLOATER["centre"][0]) <= FLOATER["half"]) & (np.abs(P[:, 1] - FLOATER["centre"][1]) <= FLOATER["half"])
        cands[L_FLOATER] = np.where(ok, s, np.inf)
    keys = list(cands)
    S = np.stack([cands[kk] for kk in keys], 1)
    j = np.argmin(S, 1); depth = S[np.arange(len(j)), j]
    lab = np.array([abs(kk) if kk > 0 else L_WALL for kk in keys])[j]
    depth = np.where(np.isfinite(depth), depth, 0.0)
    return depth.reshape(H, W), lab.reshape(H, W)


def omega_record(depth, label, rng, a, b, floater_depth_scale=1.0):
    """Omega's reading of the keyframe: 1/z' = (1/z − b)/a (scale-and-shift in inverse depth),
    0.3 % multiplicative noise, confidence 3..6."""
    z = depth.copy()
    m = z > 0
    inv = np.where(m, 1.0 / np.where(m, z, 1.0), 0.0)
    z_om = np.where(m, 1.0 / ((inv - b) / a), 0.0) * (1.0 + rng.normal(0, 0.003, z.shape))
    conf = rng.uniform(3.0, 6.0, z.shape).astype(np.float32)
    return z_om.astype(np.float32), conf


def surface_distance(P):
    """Distance of world points to each truth surface → (nearest label, distance)."""
    def aabb_dist(box):
        lo, hi = box[:, 0], box[:, 1]
        inside = np.all((P >= lo) & (P <= hi), 1)
        d_out = np.linalg.norm(np.maximum(lo - P, 0) + np.maximum(P - hi, 0), axis=1)
        d_in = np.min(np.minimum(P - lo, hi - P), 1)
        return np.where(inside, d_in, d_out)
    d = {L_FLOOR: np.abs(P[:, 1] - FLOOR_Y),
         L_WALL: np.minimum.reduce([np.abs(P[:, 2] - BACK_Z), np.abs(P[:, 1] - CEIL_Y), np.abs(P[:, 0] - LEFT_X), np.abs(P[:, 0] - RIGHT_X)]),
         L_BOX: aabb_dist(BOX), L_CAB: aabb_dist(CAB)}
    keys = list(d); S = np.stack([d[k] for k in keys], 1)
    j = np.argmin(S, 1)
    return np.array(keys)[j], S[np.arange(len(j)), j]


# ── stage 1: the scene, Omega's records, the tracks ───────────────────────────────────────────
def build_scene():
    rng = np.random.default_rng(SEED)
    c2w = truth_poses()
    c2w_om = drifted_poses(c2w, rng)
    depth_t, label_t, om_depth, om_conf, ab = {}, {}, {}, {}, {}
    for k, f in enumerate(FRAMES):
        d, lab = render(c2w[k], k, with_floater=(k in FLOATER["kfs"]))
        depth_t[f], label_t[f] = d, lab
        a, b = 1.0 + rng.normal(0, 0.04), rng.normal(0, 0.015)
        ab[f] = (a, b)
        om_depth[f], om_conf[f] = omega_record(d, lab, rng, a, b)
    # the tracks the way F4 makes them: per query keyframe q, points of q tracked over the window
    # [q−2, q+2] of its own visit and — the loop bridge — over [j−2, j+2] around the revisit keyframe
    # j = 13 − q that sees the same place from the other pass. window_id = the query keyframe,
    # is_loop = the track has observations on both visits (the pair window of F4)
    fx, fy, cx, cy = FK
    w2c = np.linalg.inv(c2w)
    track, frame, uv, window_id, is_loop, query_kf = [], [], [], [], [], []
    X_all = []
    tid = 0
    for q, fq in enumerate(FRAMES):
        d, lab = depth_t[fq], label_t[fq]
        vv, uu = np.nonzero((d > 0) & (lab != L_FLOATER))
        sel = rng.choice(len(vv), 80, replace=False)
        j = N_KF - 1 - q
        own = [k for k in range(max(0, q - 2), min(N_KF, q + 3)) if (k <= 6) == (q <= 6)]
        other = [k for k in range(max(0, j - 2), min(N_KF, j + 3)) if (k <= 6) != (q <= 6)]
        for v0, u0 in zip(vv[sel], uu[sel]):
            z0 = d[v0, u0]
            Pc = np.array([(u0 + rng.uniform(-0.5, 0.5) - cx) / fx * z0, (v0 + rng.uniform(-0.5, 0.5) - cy) / fy * z0, z0])
            Xw = c2w[q][:3, :3] @ Pc + c2w[q][:3, 3]
            obs = []
            for side, kfs in ((0, own), (1, other)):
                for k in kfs:
                    P = w2c[k][:3, :3] @ Xw + w2c[k][:3, 3]
                    z = P[2]
                    if z <= 0.3:
                        continue
                    u = fx * P[0] / z + cx; v = fy * P[1] / z + cy
                    if not (0 <= u <= W - 1 and 0 <= v <= H - 1):
                        continue
                    ds = depth_t[FRAMES[k]][int(round(v)), int(round(u))]
                    if ds > 0 and abs(ds - z) < 0.015 * z:
                        obs.append((side, k, u + rng.normal(0, 0.3), v + rng.normal(0, 0.3)))
            if sum(1 for o in obs if o[0] == 0) < 2:
                continue
            both = any(o[0] == 1 for o in obs)
            for side, k, u, v in obs:
                track.append(tid); frame.append(FRAMES[k]); uv.append([u, v]); window_id.append(q + 1)
                is_loop.append(both); query_kf.append(q)
            X_all.append(Xw); tid += 1
    track = np.array(track); frame = np.array(frame); uv = np.array(uv)
    window_id = np.array(window_id, np.int64); is_loop = np.array(is_loop, bool); query_kf = np.array(query_kf)
    X = np.array(X_all)
    # the held-out split per track
    ids = np.unique(track)
    held = set(ids[rng.random(len(ids)) < 0.2].tolist())
    split = np.array([1 if t in held else 0 for t in track], np.int8)
    visit = np.array([0 if f <= FRAMES[6] else 1 for f in frame])
    n_both = len({t for t, b in zip(track, is_loop) if b})
    return {"c2w": c2w, "c2w_om": c2w_om, "depth_t": depth_t, "label_t": label_t, "om_depth": om_depth,
            "om_conf": om_conf, "ab": ab, "X": X, "track": track, "frame": frame, "uv": uv, "split": split,
            "window_id": window_id, "is_loop": is_loop, "visit": visit, "query_kf": query_kf,
            "n_cross_visit_tracks": n_both}


def pose_errors(c2w_a, c2w_b):
    """Position and rotation error per keyframe (the first pose is the common gauge)."""
    dp = np.linalg.norm(c2w_a[:, :3, 3] - c2w_b[:, :3, 3], axis=1)
    dr = []
    for A, B in zip(c2w_a, c2w_b):
        Rrel = A[:3, :3].T @ B[:3, :3]
        dr.append(np.degrees(np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1, 1))))
    return dp, np.array(dr)


# ── stage 2: F5 ────────────────────────────────────────────────────────────────────────────────
def run_f5(S, bridge: bool, log):
    import test_precision_refine as T
    from precision import refine as R
    track, frame, uv, split = S["track"], S["frame"], S["uv"], S["split"]
    wid, lp = S["window_id"], S["is_loop"]
    if not bridge:
        # without the bridge a track lives in its query's visit only (the tracker loses it across the
        # revisit): the other visit's observations leave, the windows stay, nothing is a loop window
        q_visit = np.where(S["query_kf"] <= 6, 0, 1)
        keep = S["visit"] == q_visit
        track, frame, uv, split, wid = track[keep], frame[keep], uv[keep], split[keep], wid[keep]
        lp = np.zeros(len(track), bool)
        cnt = np.bincount(track)
        keep = cnt[track] >= 2
        track, frame, uv, split, wid, lp = track[keep], frame[keep], uv[keep], split[keep], wid[keep], lp[keep]
    params0 = [*FK, 0.0, 0.0, 0.0, 0.0]
    cfg = replace(T.CFG, permutations=300)
    t0 = time.time()
    core = R.refine_core(np.linalg.inv(S["c2w_om"]), params0, (W, H), track, frame, uv, split, FRAMES, 0.01, cfg,
                         T.SOLVER, error_factor=T.FAC, log=log, window_id=wid, is_loop=lp)
    best = core["best"]
    c2w_f5 = np.linalg.inv(best.w2c)
    rep = {"best": best.name, "rungs": {k: {kk: v[kk] for kk in ("taken", "heldout_median_px") if kk in v}
                                       for k, v in core["rungs"].items()},
           "odometry": core.get("odometry"), "loops": core.get("loops"), "scale_lock": core.get("scale_lock"),
           "rolling_shutter": {k: core["rolling_shutter"].get(k) for k in ("taken", "tau_frames", "reason")},
           "params": [float(x) for x in best.params_by_block[0]], "seconds": round(time.time() - t0, 1),
           "n_obs": int(len(track)), "n_tracks": int(len(np.unique(track)))}
    return c2w_f5, rep


# ── stage 3: F6 on the session the way F0–F5 leave it ─────────────────────────────────────────
def write_session(root: Path, S, c2w_pub, params):
    import cv2
    from precision.camera import CameraModel, grid_full_frame_resize, save_camera_json
    from precision.refine import REFINE_NAME, RESIDUALS_NAME
    from precision.tracks import TRACKS_NAME
    s = root / "sess"; out = s / "output"; rec = out / "omega_run" / "results_output"
    shutil.rmtree(s, ignore_errors=True)
    for d in (s / "frames", rec, out / "precision"):
        d.mkdir(parents=True)
    g = grid_full_frame_resize(W, H, W, H, "omega")
    cam = CameraModel(W, H, tuple(float(x) for x in params), "refine", 3, g, mask_grid=grid_full_frame_resize(W, H, W, H, "mask"))
    save_camera_json(out / "camera.json", cam, 5)
    (out / "geometry_epoch.json").write_text(json.dumps({"epoch": 5}))
    rng = np.random.default_rng(1)
    for k, f in enumerate(FRAMES):
        img = np.full((H, W, 3), 120, np.uint8)
        lab = S["label_t"][f]
        img[lab == L_BOX] = (60, 90, 200); img[lab == L_CAB] = (200, 120, 60); img[lab == L_FLOOR] = (150, 150, 150)
        img = np.clip(img.astype(int) + rng.integers(-10, 10, img.shape), 0, 255).astype(np.uint8)
        cv2.imwrite(str(s / "frames" / f"{f:06d}.jpg"), img)
        np.savez(rec / f"frame_{f}.npz", depth=S["om_depth"][f], conf=S["om_conf"][f], chunk=np.int64(0))
    np.savetxt(out / "camera_poses.txt", np.stack([T.ravel() for T in c2w_pub]))
    (out / "camera_frames.txt").write_text("\n".join(map(str, FRAMES)) + "\n")
    (out / "precision" / REFINE_NAME).write_text(json.dumps({"applied": True, "epoch_to": 5}))
    np.savez(out / "precision" / RESIDUALS_NAME, heldout_rms_px=np.full(20, 0.5, np.float32))
    np.savez(out / "precision" / TRACKS_NAME, obs_track=S["track"], obs_frame=S["frame"],
             obs_uv_native=S["uv"].astype(np.float32), track_query_id=np.arange(S["track"].max() + 1, dtype=np.int64),
             track_split=np.array([S["split"][np.flatnonzero(S["track"] == t)[0]] if (S["track"] == t).any() else 0
                                   for t in range(S["track"].max() + 1)], np.int8))
    (out / ".orientation_applied").write_text("")
    return s


def pcfg_for(fusion_on: bool):
    from precision.config import load_precision_config
    p = load_precision_config()
    return replace(p, mono_detail=replace(p.mono_detail, enabled=False),
                   bend=replace(p.bend, windows=(0, 1), neighbors=(-1, 1)),
                   fusion=replace(p.fusion, enabled=fusion_on, colmap_tier=False))


PLY_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"),
                      ("blue", "u1"), ("frame_global", "<i4"), ("pixel_row", "<u2"),
                      ("pixel_col", "<u2"), ("confidence", "<f4")])


def write_ply(path, data):
    names = {"red": "red", "green": "green", "blue": "blue"}
    types = {"<f4": "float", "|u1": "uchar", "<i4": "int", "<u2": "ushort"}
    head = ["ply", "format binary_little_endian 1.0", f"element vertex {len(data)}"]
    head += [f"property {types[data.dtype[n].str]} {names.get(n, n)}" for n in data.dtype.names]
    head += ["end_header"]
    with open(path, "wb") as f:
        f.write(("\n".join(head) + "\n").encode()); f.write(data.tobytes())


def assemble_cloud(frames, c2w, K, zmap_of, mask_of, conf_of):
    """The owners (or every voted pixel) → world points with their birth keyframe and pixel — the
    loop of publish_cloud before the cleaner."""
    vv, uu = np.mgrid[0:H, 0:W]
    rows = []
    for k, f in enumerate(frames):
        m = mask_of(f)
        r, c = vv[m], uu[m]; zz = zmap_of(f)[m].astype(np.float64)
        T = c2w[f] if isinstance(c2w, dict) else c2w[k]
        X = np.stack([(c - K[0, 2]) / K[0, 0] * zz, (r - K[1, 2]) / K[1, 1] * zz, zz], 1) @ T[:3, :3].T + T[:3, 3]
        d = np.zeros(len(r), PLY_DTYPE)
        d["x"], d["y"], d["z"] = X[:, 0], X[:, 1], X[:, 2]
        d["red"] = 110; d["green"] = 120; d["blue"] = 130
        d["frame_global"] = f; d["pixel_row"] = r; d["pixel_col"] = c; d["confidence"] = conf_of(f)[m]
        rows.append(d)
    return np.concatenate(rows)


def write_cloud_session(dst: Path, src_session: Path, data, fusion=None, frames=None, valid=None):
    """A session output for the projection: the common files of ``src_session``, this cloud, its
    origins v2 and (with a fusion) its rejected_points.npz."""
    import precision.provenance as PV
    from precision import owner_fusion as OF
    out = dst / "output"
    shutil.rmtree(dst, ignore_errors=True); out.mkdir(parents=True)
    so = src_session / "output"
    for n in ("camera.json", "camera_frames.txt", "camera_poses.txt", ".orientation_applied", "geometry_epoch.json"):
        shutil.copy(so / n, out / n)
    shutil.copytree(so / "omega_run", out / "omega_run")
    write_ply(out / "cleaned_cloud.ply", data)
    m = len(data)
    PV.write_origins(out / "origins.npz", {
        "frame_global": data["frame_global"], "pixel_row": data["pixel_row"], "pixel_col": data["pixel_col"],
        "pixel_u_und": data["pixel_col"], "pixel_v_und": data["pixel_row"],
        "n_consistent": np.clip(data["confidence"], 0, 255).astype(np.uint8), "ncc": np.zeros(m), "source": np.ones(m),
        "residual_rel": np.zeros(m), "content_flags": np.zeros(m), "geometry_epoch": np.full(m, 5), "camera_epoch": np.full(m, 3)},
        {"e2e_synthetic": True})
    if fusion is not None:
        OF.write_rejected(out / PV.REJECTED_NAME, frames, fusion, valid)
    return out


def write_masks(out: Path, S):
    from segmentation import mask_space
    from segmentation import pipeline as P
    objs = [(0, L_BOX, "crate"), (1, L_CAB, "cabinet"), (2, L_FLOOR, "floor"), (3, L_WALL, "wall")]
    masks = {mask_space.NPZ_KEY: mask_space.declaration(mask_space.SPACE_KEYFRAME),
             mask_space.KEYFRAMES_KEY: np.asarray(FRAMES, np.int32),
             "obj_ids": np.array([o[0] for o in objs], np.int32),
             "frames": np.arange(N_KF, dtype=np.int32), "scaled_res": np.array([H, W], np.int32)}
    for k, f in enumerate(FRAMES):
        lab = S["label_t"][f].copy()
        lab[lab == L_FLOATER] = L_BOX               # SAM sees the IMAGE: the crate, never Omega's floater
        for oid, L, _ in objs:
            masks[f"f{k}_o{oid}"] = (lab == L).astype(np.uint8)
    P.write_canonical_store(out / "seg_masks.npz", masks)
    (out / "segmentation.json").write_text(json.dumps({
        "version": "3.0", "prompt": ";".join(o[2] for o in objs), "prompts": [o[2] for o in objs],
        "resolution": {"scaled": [H, W], "original": [H, W]}, "mask_file": "seg_masks.npz",
        "instances": [{"id": oid, "instance_id": oid + 1, "label": name, "color": [oid * 40, 100, 200 - oid * 40]}
                      for oid, _, name in objs]}, indent=1))
    return {oid + 1: L for oid, L, _ in objs}


def project(out: Path):
    import potree_converter as PC
    from segmentation import pipeline as P
    from reconstruction.surface_fit import fit_torch, hole_audit
    hole_audit.use_device("cpu"); fit_torch.use_device("cpu")        # the synthetic tests' rule: the same code path, on the CPU

    def _convert(session_dir, force=False, ply_override=None, potree_dir_override=None):
        d = Path(session_dir) / "output" / "potree"; d.mkdir(parents=True, exist_ok=True)
        (d / "metadata.json").write_text("{}"); return True
    PC.convert_ply_to_potree = _convert
    res = P._match_and_save_result(out)
    ids = np.load(out / "instance_ids.npy")
    return res, ids


def truth_surface_samples(S):
    """The true surface as the keyframes could see it (every pixel of every keyframe, the floater
    excluded), on a 3 cm grid — what a complete cloud must cover."""
    fx, fy, cx, cy = FK
    vv, uu = np.mgrid[0:H, 0:W]
    P = []
    for k, f in enumerate(FRAMES):
        d, lab = S["depth_t"][f], S["label_t"][f]
        m = (d > 0) & (lab != L_FLOATER)
        zz = d[m]
        X = np.stack([(uu[m] - cx) / fx * zz, (vv[m] - cy) / fy * zz, zz], 1)
        P.append(X @ S["c2w"][k][:3, :3].T + S["c2w"][k][:3, 3])
    P = np.concatenate(P)
    cells = np.unique(np.floor(P / 0.03).astype(np.int64), axis=0)
    return (cells + 0.5) * 0.03


def views_of_cells(S, cells):
    """How many keyframes see each true-surface cell (projection + depth test, the floater absent)."""
    fx, fy, cx, cy = FK
    n = np.zeros(len(cells), int)
    for k, f in enumerate(FRAMES):
        w2c = np.linalg.inv(S["c2w"][k]); Pc = cells @ w2c[:3, :3].T + w2c[:3, 3]
        z = Pc[:, 2]; ok = z > 0.3
        u = fx * Pc[:, 0] / np.where(ok, z, 1) + cx; v = fy * Pc[:, 1] / np.where(ok, z, 1) + cy
        ok &= (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
        ui = np.clip(np.round(u).astype(int), 0, W - 1); vi = np.clip(np.round(v).astype(int), 0, H - 1)
        d = render(S["c2w"][k], k, with_floater=False)[0][vi, ui]
        n += (ok & (np.abs(d - z) < 0.03)).astype(int)
    return n


def coverage(data, cells, nview):
    """Share of the true-surface cells with a published point within r, for r = 3 / 5 / 8 cm (3 cm is
    below the far wall's pixel footprint: one point per surface element is sparser than the 3 cm
    grid, not emptier), over all cells, the cells two or more keyframes saw and the single-view cells
    (what fusion.min_evidence 2 removes); plus the published points' median neighbour spacing."""
    from scipy.spatial import cKDTree
    P = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float64)
    tree = cKDTree(P)
    out = {}
    for r in (0.03, 0.05, 0.08):
        d, _ = tree.query(cells, distance_upper_bound=r)
        cov = np.isfinite(d)
        out[f"r{int(r * 100)}cm"] = {"all": float(cov.mean()), "seen_by_2plus": float(cov[nview >= 2].mean()),
                                    "seen_by_1": float(cov[nview == 1].mean())}
    dd, _ = tree.query(P, k=2)
    out["nn_spacing_cm"] = {"median": float(np.median(dd[:, 1]) * 100), "p90": float(np.percentile(dd[:, 1], 90) * 100)}
    return out


def cloud_metrics(data, ids=None, inst_label=None):
    P = np.stack([data["x"], data["y"], data["z"]], 1).astype(np.float64)
    lab, dist = surface_distance(P)
    on = dist < 0.03
    rep = {"n_points": int(len(P)), "off_surface_points_gt_3cm": int((~on).sum()),
           "off_surface_points_gt_10cm": int((dist > 0.10).sum()),
           "wall_rms_cm": float(np.sqrt(np.mean(dist[(lab == L_WALL) & (dist < 0.3)] ** 2)) * 100),
           "floor_rms_cm": float(np.sqrt(np.mean(dist[(lab == L_FLOOR) & (dist < 0.3)] ** 2)) * 100),
           "box_rms_cm": float(np.sqrt(np.mean(dist[(lab == L_BOX) & (dist < 0.3)] ** 2)) * 100),
           "median_dist_cm": float(np.median(dist) * 100), "p95_dist_cm": float(np.percentile(dist, 95) * 100)}
    if ids is not None:
        per = {}
        for iid, L in inst_label.items():
            got = ids == iid
            truth = (lab == L) & on
            per[LABEL_NAME[L]] = {"labelled": int(got.sum()), "truth_on_surface": int(truth.sum()),
                                  "purity": float((got & truth).sum() / max(got.sum(), 1)),
                                  "recall": float((got & truth).sum() / max(truth.sum(), 1)),
                                  "labelled_off_surface": int((got & ~on).sum()),
                                  "labelled_on_other_surface": int((got & on & (lab != L)).sum())}
        rep["instances"] = per
        rep["unlabelled"] = int((ids == 0).sum())
    return rep, lab, dist


def log_to(path):
    fh = open(path, "a")

    def _log(*a):
        fh.write(" ".join(str(x) for x in a) + "\n"); fh.flush()
    return _log


# ── the run ────────────────────────────────────────────────────────────────────────────────────
def stage_f5():
    p = CACHE / "f5.pkl"
    if p.exists():
        return pickle.load(open(p, "rb"))
    S = build_scene()
    log = log_to(OUT / "f5.log")
    res = {"S": S}
    for name, bridge in (("bridge", True), ("no_bridge", False)):
        c2w_f5, rep = run_f5(S, bridge, log)
        res[name] = {"c2w": c2w_f5, "rep": rep}
    with open(p.with_suffix(".tmp"), "wb") as fh:
        pickle.dump(res, fh)
    os.replace(p.with_suffix(".tmp"), p)
    return res


def stage_f6(F):
    p = CACHE / "f6.pkl"
    if p.exists():
        return pickle.load(open(p, "rb"))
    from precision.depth_on_f5 import compute
    S = F["S"]
    sess = write_session(CACHE, S, F["bridge"]["c2w"], F["bridge"]["rep"]["params"])
    log = log_to(OUT / "f6.log")
    res = {}
    for name, on in (("new", True), ("old", False)):
        t0 = time.time()
        C = compute(sess, pcfg_for(on), log=log)
        rep = {"model": C.model, "window_rule": C.window_rule, "score_B": C.score_B, "tau": C.tau, "coverage": C.cover,
               "seconds": round(time.time() - t0, 1), "fusion": (C.fusion.report if C.fusion is not None else None),
               "per_frame_coef": {str(f): [float(x) for x in C.coefs[i]] for i, f in enumerate(C.frames)}}
        import types
        Cp = types.SimpleNamespace(**{k: getattr(C, k) for k in ("frames", "c2w", "K", "voted", "fusion", "model", "window_rule",
                                                                   "score_B", "tau", "cover", "coefs", "W", "H", "params")})
        res[name] = {"C": Cp, "rep": rep}
    res["session"] = sess
    with open(p.with_suffix(".tmp"), "wb") as fh:
        pickle.dump(res, fh)
    os.replace(p.with_suffix(".tmp"), p)
    return res


def stage_project(F, G):
    S = F["S"]; sess = G["session"]
    K = np.array([[FK[0], 0, FK[2]], [0, FK[1], FK[3]], [0, 0, 1]])
    Cn, Co = G["new"]["C"], G["old"]["C"]
    Kf = Cn.K
    # (a) Omega as it comes
    data_a = assemble_cloud(FRAMES, S["c2w_om"], K, lambda f: S["om_depth"][f], lambda f: S["om_depth"][f] > 0, lambda f: S["om_conf"][f])
    # (b) F5 + bend + vote, every voted pixel
    data_b = assemble_cloud(FRAMES, Co.c2w, Kf, lambda f: Co.voted[f][0], lambda f: Co.voted[f][0] > 0, lambda f: Co.voted[f][1].astype(np.float32))
    # (c) the product: the owners at the fused depth
    Fu = Cn.fusion
    data_c = assemble_cloud(FRAMES, Cn.c2w, Kf, lambda f: Fu.depth[f], lambda f: (Cn.voted[f][0] > 0) & Fu.owner[f] & (Fu.depth[f] > 0),
                            lambda f: Fu.members[f].astype(np.float32))
    out_b = write_cloud_session(CACHE / "proj_old", sess, data_b)
    out_c = write_cloud_session(CACHE / "proj_new", sess, data_c, fusion=Fu, frames=FRAMES, valid={f: Cn.voted[f][0] > 0 for f in FRAMES})
    inst = write_masks(out_b, S); write_masks(out_c, S)
    res_b, ids_b = project(out_b)
    res_c, ids_c = project(out_c)
    return {"a": data_a, "b": data_b, "c": data_c, "ids_b": ids_b, "ids_c": ids_c, "inst": inst,
            "res_b": {k: res_b.get(k) for k in ("absorbed", "decisions")}, "res_c": {k: res_c.get(k) for k in ("absorbed", "decisions")},
            "out_b": out_b, "out_c": out_c}


def figures(F, G, Pj, M):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    S = F["S"]
    # 1. trajectories
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    T0 = S["c2w"][:, :3, 3]
    for arr, c, lab, ls in ((S["c2w"], "k", "verdad", "-"), (S["c2w_om"], "tab:red", "Omega (deriva inyectada)", "--"),
                            (F["no_bridge"]["c2w"], "tab:orange", "F5 sin puente de lazos", "-"), (F["bridge"]["c2w"], "tab:green", "F5 con puente (tracks en ambas visitas)", "-")):
        E10 = T0 + 10.0 * (arr[:, :3, 3] - T0)                      # the errors drawn x10
        ax[0].plot(E10[:, 0], E10[:, 2], ls, c=c, marker="o", ms=3, label=lab)
    for box, c in ((BOX, "tab:blue"), (CAB, "tab:brown")):
        ax[0].add_patch(Rectangle((box[0, 0], box[2, 0]), box[0, 1] - box[0, 0], box[2, 1] - box[2, 0], fill=False, ec=c, lw=1.5))
    ax[0].axhline(BACK_Z, c="gray", lw=2); ax[0].set_xlabel("x [m]"); ax[0].set_ylabel("z [m]"); ax[0].set_aspect("equal")
    ax[0].set_title("Trayectorias (vista superior), errores de posición dibujados x10: ida (kf 0–6) y vuelta (kf 7–13)", fontsize=9); ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)
    ax[0].set_xlim(-2.9, 2.9); ax[0].set_ylim(-1.2, 5.3)
    for arr, c, lab in ((S["c2w_om"], "tab:red", "Omega"), (F["no_bridge"]["c2w"], "tab:orange", "F5 sin puente"), (F["bridge"]["c2w"], "tab:green", "F5 con puente")):
        dp, dr = pose_errors(S["c2w"], arr)
        ax[1].plot(range(N_KF), dp * 100, "-o", c=c, ms=3, label=f"{lab}: RMS {np.sqrt(np.mean(dp**2))*100:.2f} cm, rot {np.sqrt(np.mean(dr**2)):.3f}°")
    ax[1].set_xlabel("keyframe"); ax[1].set_ylabel("error de posición vs verdad [cm]"); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=7)
    rb = F["bridge"]["rep"]; ax[1].set_title(f"F5: peldaño {rb['best']}, priors de odometría; held-out {rb['rungs'].get(rb['best'], {}).get('heldout_median_px', float('nan')):.3f} px", fontsize=9)
    fig.suptitle("E2E sintético — poses", fontsize=11); fig.tight_layout(); fig.savefig(FIG / "e2e_1_poses.png", dpi=120); plt.close(fig)

    # 2. the three clouds, top view and side view
    fig, ax = plt.subplots(2, 3, figsize=(19, 9.5))
    for j, (key, title) in enumerate((("a", "(a) Omega tal cual: poses con deriva, profundidad cruda, todas las observaciones"),
                                      ("b", "(b) F5 + bend + voto, SIN fusión (la nube por capas de antes)"),
                                      ("c", "(c) F5 + bend + voto + fusión por propietario + carving + min_evidence 2 (producto)"))):
        d = Pj[key]; m = M[key]
        Pw = np.stack([d["x"], d["y"], d["z"]], 1)
        sel = np.random.default_rng(0).choice(len(Pw), min(len(Pw), 60000), replace=False)
        _, dist = surface_distance(Pw[sel])
        col = np.where(dist < 0.03, "tab:green", "tab:red")
        ax[0, j].scatter(Pw[sel, 0], Pw[sel, 2], s=0.4, c=col, alpha=0.5)
        for box, c in ((BOX, "tab:blue"), (CAB, "tab:brown")):
            ax[0, j].add_patch(Rectangle((box[0, 0], box[2, 0]), box[0, 1] - box[0, 0], box[2, 1] - box[2, 0], fill=False, ec=c, lw=1.2))
        ax[0, j].set_xlim(-2.8, 2.8); ax[0, j].set_ylim(1.2, 5.3); ax[0, j].set_xlabel("x [m]"); ax[0, j].set_ylabel("z [m]")
        cv = m['coverage_of_true_surface']
        ax[0, j].set_title(f"{title}\n{m['n_points']:,} pts; fuera de superficie >3 cm: {m['off_surface_points_gt_3cm']:,} (>10 cm: {m['off_surface_points_gt_10cm']:,}); "
                           f"pared RMS {m['wall_rms_cm']:.2f} cm, piso {m['floor_rms_cm']:.2f} cm\ncobertura a 5 cm: {cv['r5cm']['all']*100:.1f} % "
                           f"(celdas vistas por ≥2 kf: {cv['r5cm']['seen_by_2plus']*100:.1f} %, por 1 kf: {cv['r5cm']['seen_by_1']*100:.1f} %); "
                           f"espaciado mediano {cv['nn_spacing_cm']['median']:.1f} cm", fontsize=7)
        # side slice through the crate (|x| < 0.2): y vs z
        sl = np.abs(Pw[sel, 0]) < 0.2
        ax[1, j].scatter(Pw[sel][sl, 2], -Pw[sel][sl, 1], s=0.6, c=col[sl], alpha=0.6)
        ax[1, j].add_patch(Rectangle((BOX[2, 0], -BOX[1, 1]), BOX[2, 1] - BOX[2, 0], BOX[1, 1] - BOX[1, 0], fill=False, ec="tab:blue", lw=1.2))
        ax[1, j].axhline(-FLOOR_Y, c="gray", lw=1); ax[1, j].axvline(BACK_Z, c="gray", lw=2)
        ax[1, j].set_xlim(1.2, 5.3); ax[1, j].set_ylim(-1.5, 1.4); ax[1, j].set_xlabel("z [m]"); ax[1, j].set_ylabel("altura −y [m]")
        ax[1, j].set_title("corte |x| < 0.2 m (el cajón, el volador a z = 1.9, la pared, el piso)", fontsize=8)
    fig.suptitle("E2E sintético — las tres nubes (verde: a < 3 cm de una superficie verdadera; rojo: fuera)", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / "e2e_2_nubes.png", dpi=120); plt.close(fig)

    # 3. the mask projection on keyframe 4 (sees the floater) — truth labels, SAM masks, labelled points
    k = 4; f = FRAMES[k]
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    lab_img = S["label_t"][f]
    cmap = {L_FLOOR: (0.6, 0.6, 0.6), L_WALL: (0.85, 0.85, 0.85), L_BOX: (0.2, 0.4, 0.9), L_CAB: (0.8, 0.5, 0.2), L_FLOATER: (1.0, 0.1, 0.1)}
    rgb = np.zeros((H, W, 3))
    for L, c in cmap.items():
        rgb[lab_img == L] = c
    ax[0, 0].imshow(rgb); ax[0, 0].set_title(f"keyframe {k}: etiquetas verdaderas de la escena (rojo = el volador que Omega alucina)", fontsize=8)
    masks = np.load(Pj["out_c"] / "seg_masks.npz")
    mk = np.zeros((H, W, 3))
    for oid, c in ((0, (0.2, 0.4, 0.9)), (1, (0.8, 0.5, 0.2)), (2, (0.6, 0.6, 0.6)), (3, (0.85, 0.85, 0.85))):
        mk[masks[f"f{k}_o{oid}"] > 0] = c
    ax[0, 1].imshow(mk); ax[0, 1].set_title("las máscaras 'SAM' (de la imagen: el cajón cubre el volador)", fontsize=8)
    ax[0, 2].axis("off")
    inst = Pj["inst"]; name_of = {iid: LABEL_NAME[L] for iid, L in inst.items()}
    icol = {1: (0.2, 0.4, 0.9), 2: (0.8, 0.5, 0.2), 3: (0.6, 0.6, 0.6), 4: (0.85, 0.85, 0.85), 0: (0, 0, 0)}
    for j, (key, idk, title) in enumerate((("b", "ids_b", "(b) sin fusión: proyección con el píxel propio"), ("c", "ids_c", "(c) producto: proyección con los píxeles miembro"))):
        d = Pj[key]; ids = Pj[idk]
        Pw = np.stack([d["x"], d["y"], d["z"]], 1)
        w2c = np.linalg.inv(S["c2w"][k])
        Pc = Pw @ w2c[:3, :3].T + w2c[:3, 3]
        z = Pc[:, 2]; ok = z > 0.2
        u = FK[0] * Pc[:, 0] / np.where(ok, z, 1) + FK[2]; v = FK[1] * Pc[:, 1] / np.where(ok, z, 1) + FK[3]
        ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
        # only what this keyframe can see: within 3 cm of its truth depth, or in FRONT of it (a floater)
        vi = np.clip(np.round(v).astype(int), 0, H - 1); ui = np.clip(np.round(u).astype(int), 0, W - 1)
        dt = render(S["c2w"][k], k, with_floater=False)[0][vi, ui]        # the scene's depth: a floater in front shows
        vis = ok & (dt > 0) & (z < dt * 1.01)
        img = np.ones((H, W, 3))
        order = np.argsort(-z[vis])
        for idx in np.flatnonzero(vis)[order]:
            img[vi[idx]:vi[idx] + 2, ui[idx]:ui[idx] + 2] = icol.get(int(ids[idx]), (0, 0, 0))   # 2x2 splat
        ax[1, j].imshow(img)
        m = M[key]["instances"]["box"]
        ax[1, j].set_title(f"{title}\ncajón: pureza {m['purity']*100:.1f} %, recall {m['recall']*100:.1f} %, "
                           f"etiquetados fuera de superficie {m['labelled_off_surface']:,}, sobre otra superficie {m['labelled_on_other_surface']:,}", fontsize=7.5)
    # bar chart: purity / recall per instance for (b) and (c)
    names = ["box", "cabinet", "floor", "wall"]
    x = np.arange(len(names)); wd = 0.2
    for off, key, c, lab in ((-1.5 * wd, "b", "tab:red", "(b) pureza"), (-0.5 * wd, "c", "tab:green", "(c) pureza")):
        ax[1, 2].bar(x + off, [M[key]["instances"][n]["purity"] * 100 for n in names], wd, color=c, label=lab)
    for off, key, c, lab in ((0.5 * wd, "b", "salmon", "(b) recall"), (1.5 * wd, "c", "lightgreen", "(c) recall")):
        ax[1, 2].bar(x + off, [M[key]["instances"][n]["recall"] * 100 for n in names], wd, color=c, label=lab)
    ax[1, 2].set_xticks(x); ax[1, 2].set_xticklabels(names); ax[1, 2].set_ylim(0, 105); ax[1, 2].set_ylabel("%"); ax[1, 2].legend(fontsize=7)
    ax[1, 2].set_title("proyección de máscaras: pureza (puntos etiquetados que están sobre esa superficie) y recall", fontsize=8); ax[1, 2].grid(alpha=0.3, axis="y")
    mp = Pj["res_c"]["decisions"].get("member_pixels", {})
    fig.suptitle(f"E2E sintético — máscaras y proyección (píxeles miembro: {mp.get('resolved', 0):,} observaciones consumidas resueltas, {mp.get('lookups_per_point', 1):.2f} píxeles de consulta por punto)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "e2e_3_mascaras.png", dpi=120); plt.close(fig)

    # 4. what left the cloud, per keyframe and reason
    with np.load(Pj["out_c"] / "rejected_points.npz") as z:
        fg = z["frame_global"]; rs = z["reason"]
    fig, ax = plt.subplots(figsize=(13, 5))
    bottom = np.zeros(N_KF)
    for code, name, c in ((10, "fused (consumida: otro keyframe publica el elemento)", "tab:blue"), (9, "contradicted (otros keyframes ven espacio libre a través)", "tab:red"), (1, "unverified (menos de min_evidence vistas)", "tab:gray")):
        cnt = np.array([int(((fg == f) & (rs == code)).sum()) for f in FRAMES])
        ax.bar(range(N_KF), cnt, bottom=bottom, color=c, label=f"{name}: {cnt.sum():,}"); bottom += cnt
    own = np.array([int((Pj["c"]["frame_global"] == f).sum()) for f in FRAMES])
    ax.bar(range(N_KF), own, bottom=bottom, color="tab:green", alpha=0.6, label=f"publicadas (propietarias): {own.sum():,}")
    ax.set_xlabel("keyframe"); ax.set_ylabel("observaciones"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    oc = M["c"]
    ax.set_title(f"E2E sintético — el destino de cada observación (rejected_points.npz v5); kf 3–5 llevan el volador\n"
                 f"free_space_factor {oc['free_space_factor']:g}: contradicted sobre superficies REALES {oc['over_carved_true_surface_obs']:,} "
                 f"({oc['over_carved_share_of_all_obs']*100:.2f} % de las observaciones; con factor 1 eran 8,907 = 3.9 %); volador: {oc['floater_obs']['total']} obs → "
                 f"{oc['floater_obs']['consumed']} fusionadas, {oc['floater_obs']['contradicted']} contradichas, {oc['floater_obs']['published']} publicadas "
                 f"(3 kf lo ven, 2 ven a través)", fontsize=8.5)
    fig.tight_layout(); fig.savefig(FIG / "e2e_4_destino_observaciones.png", dpi=120); plt.close(fig)


def main(argv):
    stage = argv[1] if len(argv) > 1 and argv[1] == "--stage" and len(argv) > 2 else None
    F = stage_f5()
    if stage == "f5":
        return
    G = stage_f6(F)
    if stage == "f6":
        return
    Pj = stage_project(F, G)
    M = {}
    M["a"], _, _ = cloud_metrics(Pj["a"])
    M["b"], _, _ = cloud_metrics(Pj["b"], Pj["ids_b"], Pj["inst"])
    M["c"], _, _ = cloud_metrics(Pj["c"], Pj["ids_c"], Pj["inst"])
    S = F["S"]
    cells = truth_surface_samples(S)
    nview = views_of_cells(S, cells)
    M["truth_cells"] = {"n": int(len(cells)), "seen_by_0": int((nview == 0).sum()), "seen_by_1": int((nview == 1).sum()),
                        "seen_by_2plus": int((nview >= 2).sum())}
    for k in ("a", "b", "c"):
        M[k]["coverage_of_true_surface"] = coverage(Pj[k], cells, nview)
    # what the carving and the evidence rule took from REAL surfaces (over-carving), by truth label
    with np.load(Pj["out_c"] / "rejected_points.npz") as z:
        fg, pr, pc, rs = z["frame_global"], z["pixel_row"].astype(int), z["pixel_col"].astype(int), z["reason"]
    lab_r = np.array([S["label_t"][int(f)][r, c] for f, r, c in zip(fg, pr, pc)])
    n_obs = int(sum(int((S["depth_t"][f] > 0).sum()) for f in FRAMES))
    M["c"]["rejected_by_reason_and_truth"] = {name: {LABEL_NAME[L]: int(((rs == code) & (lab_r == L)).sum()) for L in (1, 2, 3, 4, 9)}
                                             for code, name in ((10, "fused"), (9, "contradicted"), (1, "unverified"))}
    M["c"]["over_carved_true_surface_obs"] = int(((rs == 9) & (lab_r != L_FLOATER)).sum())
    M["c"]["over_carved_share_of_all_obs"] = float(((rs == 9) & (lab_r != L_FLOATER)).sum() / n_obs)
    M["c"]["floater_obs"] = {"total": int(sum(int((S["label_t"][f] == L_FLOATER).sum()) for f in FRAMES)),
                             "consumed": int(((rs == 10) & (lab_r == L_FLOATER)).sum()), "contradicted": int(((rs == 9) & (lab_r == L_FLOATER)).sum()),
                             "published": int(M["c"]["off_surface_points_gt_10cm"])}
    M["c"]["free_space_factor"] = float(G["new"]["rep"]["fusion"]["free_space"].get("factor", 1.0))
    poses = {}
    for name, arr in (("omega", S["c2w_om"]), ("f5_bridge", F["bridge"]["c2w"]), ("f5_no_bridge", F["no_bridge"]["c2w"])):
        dp, dr = pose_errors(S["c2w"], arr)
        poses[name] = {"rms_cm": float(np.sqrt(np.mean(dp ** 2)) * 100), "max_cm": float(dp.max() * 100),
                       "rot_rms_deg": float(np.sqrt(np.mean(dr ** 2))), "revisit_rms_cm": float(np.sqrt(np.mean(dp[7:] ** 2)) * 100)}
    metrics = {"scene": {"keyframes": N_KF, "grid": [W, H], "tracks": int(len(np.unique(S["track"]))), "observations": int(len(S["track"])),
                         "cross_visit_tracks": S["n_cross_visit_tracks"], "omega_ab": {str(f): S["ab"][f] for f in FRAMES},
                         "floater_keyframes": list(FLOATER["kfs"])},
               "poses": poses, "f5": {k: F[k]["rep"] for k in ("bridge", "no_bridge")},
               "f6": {k: G[k]["rep"] for k in ("new", "old")}, "clouds": M,
               "projection": {"old": Pj["res_b"], "new": Pj["res_c"]}}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    figures(F, G, Pj, M)
    print(json.dumps({"poses": poses, "clouds": {k: {kk: v for kk, v in M[k].items() if kk != "instances"} for k in M},
                      "instances_new": M["c"].get("instances"), "instances_old": M["b"].get("instances")}, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv)
