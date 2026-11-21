"""THE NEW ORDER, step 4: F5 PER CUBE WITH ONE CAMERA (USER 2026-11-20: "Una sola cámara para toda la sesión" / "Ajuste
de cada cubo con esa cámara y las correspondencias").

A VERSION is (cube, keyframe): the keyframe as that cube holds it. Every version is a camera of its own, starting at
its pose in its cube; a keyframe's observations (F4's tracks) go to EACH of its versions; the landmarks are separate
per cube (a track seen in two cubes is two landmarks, one in each); the camera model is ONE for every version.
F5's own ladder solves it (precision.refine.refine_core: R0, R1, R2, each with its continuation and the held-out judge)
— nothing of F5 is changed: it is only given another list of cameras. The cubes are NOT tied to one another here: a
cube's cameras are held by F5's own position priors around where they start.

Reads: the test session (keyframes, F0's camera, F4's tracks, its cameras after the cube stage) and the kept cubes.
Writes ONE file with the camera, every version's pose and each cube's landmarks.

    python f5_per_cube.py <scan dir> <cubes dir> <gauge.json the priors' sigma is read from> <out npz>"""
import json
import sys
import time
from pathlib import Path

import numpy as np

SERVER = "/workspace/stac-build/server"
if SERVER not in sys.path:
    sys.path.insert(0, SERVER)
FRAME_STRIDE = 1_000_000                 # a version's frame id = cube x this + the keyframe's video frame


def proj_so3(M):
    U, _s, Vt = np.linalg.svd(M)
    return U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def load_cube_cameras(cdir: Path, n_cubes: int):
    """Each cube's keyframes and their cameras in the cube stage's world: the cube's own camera carried by the pose
    the cube graph gave the cube (world = s R x + t)."""
    cubes = []
    for c in range(n_cubes):
        o = np.load(cdir / f"chunk_{c}.npy", allow_pickle=True).item()
        p = o["_stac_cube_pose"]
        s, R, t = float(p["s"]), np.asarray(p["R"], np.float64), np.asarray(p["t"], np.float64)
        E = np.asarray(o["extrinsic"], np.float64)
        cubes.append({"frames": [int(x) for x in o["_stac_frames"]],
                      "R": np.einsum("ij,njk->nik", R, E[:, :3, :3]), "C": s * E[:, :3, 3] @ R.T + t})
        del o
    return cubes


def similarity(src: np.ndarray, dst: np.ndarray, Rs: np.ndarray, Rd: np.ndarray):
    """dst = s Q src + t: Q from the cameras' orientations, s and t from their centres."""
    Q = proj_so3(np.einsum("nij,nkj->ik", Rd, Rs))
    a, b = (src - src.mean(0)) @ Q.T, dst - dst.mean(0)
    s = float((a * b).sum() / (a * a).sum())
    return s, Q, dst.mean(0) - s * Q @ src.mean(0)


def main() -> int:
    scan, cdir, gauge_json, dst = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    out = scan / "output"
    from config import cfg as raw_cfg
    from precision import refine as R
    from precision.camera import load_camera_json, undistort_solver
    from precision.config import load_precision_config
    from precision.tracks import load_tracks_v2
    from reconstruction.loops.config import improvement_error_factor
    pcfg = load_precision_config()
    log = lambda m: print(m, flush=True)   # noqa: E731

    kf = [int(float(x)) for x in (out / "camera_frames.txt").read_text().split()]
    cam = load_camera_json(out / "camera.json")
    sess = np.loadtxt(out / "camera_poses.txt").reshape(-1, 4, 4)            # the session's cameras after the cube stage
    plan = json.loads((scan / "intake" / "cubes.json").read_text())
    cubes = load_cube_cameras(cdir, len(plan["cubes"]))
    for c, cb in enumerate(cubes):
        if cb["frames"] != [int(k) for k in plan["cubes"][c]["keyframes"]]:
            raise SystemExit(f"cube {c}: the kept cube's keyframes are not the plan's")

    # the cubes' world -> the session's frame (metres, upright): ONE similarity, fitted on every version
    Cv = np.concatenate([cb["C"] for cb in cubes]); Rv = np.concatenate([cb["R"] for cb in cubes])
    kidx = np.concatenate([np.asarray(cb["frames"]) for cb in cubes])
    s, Q, t = similarity(Cv, sess[kidx, :3, 3], Rv, sess[kidx, :3, :3])
    Cs, Rs = s * Cv @ Q.T + t, np.einsum("ij,njk->nik", Q, Rv)
    off = np.linalg.norm(Cs - sess[kidx, :3, 3], axis=1)
    log(f"[f5 per cube] {len(kidx)} versions of {len(kf)} keyframes in {len(cubes)} cubes; carried onto the session's "
        f"frame with one similarity (scale {s:.5f}): a version starts {np.median(off) * 100:.2f} cm (median), "
        f"{off.max() * 100:.1f} cm (most) from the session's camera of its keyframe")

    cube_of = np.concatenate([np.full(len(cb["frames"]), c) for c, cb in enumerate(cubes)])
    c2w0 = np.tile(np.eye(4), (len(kidx), 1, 1)); c2w0[:, :3, :3], c2w0[:, :3, 3] = Rs, Cs
    w2c0 = np.linalg.inv(c2w0)
    kf_frames = [int(c) * FRAME_STRIDE + kf[int(k)] for c, k in zip(cube_of, kidx)]

    tr = load_tracks_v2(scan)
    split = R.split_of_observations(tr["track_query_id"], tr["track_split"], tr["obs_track"])
    stride_t = int(tr["obs_track"].max()) + 1
    T, F, UV, SP = [], [], [], []
    for c, cb in enumerate(cubes):
        sel = np.isin(tr["obs_frame"], np.asarray([kf[k] for k in cb["frames"]]))
        T.append(tr["obs_track"][sel].astype(np.int64) + c * stride_t)
        F.append(tr["obs_frame"][sel].astype(np.int64) + c * FRAME_STRIDE)
        UV.append(np.asarray(tr["obs_uv_native"][sel], np.float64)); SP.append(split[sel])
        log(f"[f5 per cube]   cube {c}: {len(cb['frames'])} keyframes, {int(sel.sum()):,} observations")
    T, F, UV, SP = np.concatenate(T), np.concatenate(F), np.concatenate(UV), np.concatenate(SP)

    g = json.loads(gauge_json.read_text())
    sigma_rel = float(g["sigma_by_instrument"][g["applied_instrument"]])
    log(f"[f5 per cube] {len(T):,} observations of {len(np.unique(T)):,} landmarks (a track seen in two cubes is two); "
        f"the position priors' relative sigma {sigma_rel:.5f} (the one F2 measured for this video: {gauge_json})")

    # F5's own position priors (a relative sigma accumulated along the walk, the first keyframe carrying its first
    # step's) computed PER CUBE: each cube is a walk of its own, held at its own first keyframe. Left as one list they
    # would accumulate across the jump from a cube's last keyframe back to the next cube's first.
    walk_sigmas = R.prior_sigmas

    def per_cube_sigmas(centres, rel):
        sig = np.empty(len(centres))
        for c in range(len(cubes)):
            sig[cube_of == c] = walk_sigmas(centres[cube_of == c], rel)
        return sig
    R.prior_sigmas = per_cube_sigmas
    sig0 = per_cube_sigmas(Cs, sigma_rel)
    log("[f5 per cube] position priors per cube (first / last keyframe of each, cm): "
        + ", ".join(f"cube {c}: {sig0[cube_of == c][0] * 100:.2f} / {sig0[cube_of == c][-1] * 100:.2f}"
                    for c in range(len(cubes))))

    R.use_device("cuda")
    t0 = time.time()
    core = R.refine_core(w2c0, list(cam.params), (cam.width, cam.height), T, F, UV, SP, kf_frames, sigma_rel,
                         pcfg.refine, undistort_solver(pcfg.camera),
                         error_factor=float(improvement_error_factor(dict(raw_cfg))), log=log,
                         range_m=float(pcfg.view_range_m))
    best = core["best"]
    log(f"[f5 per cube] the ladder took {time.time() - t0:.0f} s; the rung kept: {best.name}")
    params = [float(v) for v in best.params_by_block[0]]
    log(f"[f5 per cube] ONE CAMERA: fx {params[0]:.2f} fy {params[1]:.2f} cx {params[2]:.2f} cy {params[3]:.2f} "
        f"k1 {params[4]:+.4f} k2 {params[5]:+.4f} p1 {params[6]:+.5f} p2 {params[7]:+.5f}   (F0 started at fx "
        f"{cam.params[0]:.2f} fy {cam.params[1]:.2f})")
    for name, rep in core["rungs"].items():
        log(f"[f5 per cube]   {name}: held-out median {rep.get('heldout_median_px')} px, taken {rep.get('taken')}")
    c2w = np.linalg.inv(best.w2c)
    moved = np.linalg.norm(c2w[:, :3, 3] - Cs, axis=1)
    log(f"[f5 per cube] the versions moved {np.median(moved) * 100:.2f} cm (median), {moved.max() * 100:.1f} cm (most)")
    # the two versions of the same keyframe, before / after (no tie yet: each cube is its own)
    by_k = {}
    for v, k in enumerate(kidx):
        by_k.setdefault(int(k), []).append(v)
    pairs = [(a, b) for vs in by_k.values() for i, a in enumerate(vs) for b in vs[i + 1:]]
    d0 = np.array([np.linalg.norm(Cs[a] - Cs[b]) for a, b in pairs]); d1 = np.array([np.linalg.norm(c2w[a, :3, 3] - c2w[b, :3, 3]) for a, b in pairs])
    log(f"[f5 per cube] the two versions of the same keyframe ({len(pairs)} pairs) are apart: before {np.median(d0) * 100:.2f} cm "
        f"(median) / {np.percentile(d0, 90) * 100:.2f} (worst tenth); after, each cube adjusted on its own, "
        f"{np.median(d1) * 100:.2f} / {np.percentile(d1, 90) * 100:.2f} cm — the tie of step 5 is what brings them together")
    ids = np.fromiter(best.X.keys(), np.int64, len(best.X))
    X = np.asarray([best.X[int(i)] for i in ids], np.float64)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, params=np.asarray(params), wh=np.asarray([cam.width, cam.height]), rung=best.name,
             c2w=c2w, c2w_start=c2w0, cube_of=cube_of, kf_index=kidx, kf_frames=np.asarray(kf),
             landmark_cube=ids // stride_t, landmark_track=ids % stride_t, landmark_xyz=X,
             cubes_to_session=np.concatenate([[s], Q.ravel(), t]),
             heldout_px=json.dumps({n: r.get("heldout_median_px") for n, r in core["rungs"].items()}))
    log(f"[f5 per cube] written: {dst} ({len(ids):,} landmarks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
