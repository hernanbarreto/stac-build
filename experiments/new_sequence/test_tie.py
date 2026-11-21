"""The tie's arithmetic, checked on a made-up scene with a known answer (CPU, seconds):
  1. its analytic gradient equals the numeric gradient of its own cost;
  2. three cubes that are one scene seen in three frames (known scale, turn and shift each) come back onto the first.

    python test_tie.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tie_cubes import SERVER, Tie, rot  # noqa: E402,F401


def scene(noise_px=0.0):
    from precision.refine import project_obs
    rng = np.random.default_rng(7)
    params = np.array([800.0, 790.0, 512.0, 288.0, 0.02, -0.01, 0.001, -0.0005])
    P = rng.uniform([-2, -1, 4], [2, 1, 8], (3000, 3))
    true = [(1.0, np.eye(3), np.zeros(3)), (1.03, rot([0.02, -0.03, 0.01]), np.array([0.10, -0.05, 0.08])),
            (0.96, rot([-0.01, 0.02, 0.03]), np.array([-0.07, 0.04, -0.12]))]
    cams = []
    for _ in range(5):
        c2w = np.eye(4); c2w[:3, :3] = rot(rng.normal(0, 0.05, 3)); c2w[:3, 3] = rng.normal(0, 0.3, 3)
        cams.append(c2w)

    def in_cube(c, X):
        s, R, t = true[c]
        return (X - t) @ R / s                                  # R' (X - t) / s

    groups = []
    for a in range(3):
        for b in range(3):
            if a == b:
                continue
            X, W, uv = [], [], []
            for c2w in cams:
                w2c_world = np.linalg.inv(c2w)
                obs = project_obs(P, np.broadcast_to(w2c_world, (len(P), 4, 4)), params) + rng.normal(0, noise_px, (len(P), 2))
                cb = np.eye(4); cb[:3, :3] = true[b][1].T @ c2w[:3, :3]; cb[:3, 3] = in_cube(b, c2w[:3, 3][None])[0]
                X.append(in_cube(a, P)); W.append(np.broadcast_to(np.linalg.inv(cb), (len(P), 4, 4))); uv.append(obs)
            groups.append({"a": a, "b": b, "X": np.concatenate(X), "W": np.concatenate(W), "uv": np.concatenate(uv)})
    return groups, params, true


def main() -> int:
    groups, params, true = scene()
    st0 = (np.ones(3), np.tile(np.eye(3), (3, 1, 1)), np.zeros((3, 3)))

    # 1. the gradient (Huber wide open: plain least squares)
    tie = Tie(groups, params, 1e9, 3)
    _H, b = tie.normal(st0)
    num = np.empty_like(b)
    for i in range(len(b)):
        d = np.zeros(len(b)); d[i] = 1e-6
        num[i] = (tie.cost(tie.moved(st0, d)) - tie.cost(tie.moved(st0, -d))) / 2e-6
    rel = np.abs(b - num).max() / np.abs(num).max()
    print(f"gradient: analytic against numeric, largest difference {rel:.2e} of the largest entry")
    assert rel < 1e-6, rel

    # 2. the known answer, from the identity
    st, cost = Tie(groups, params, 1.0, 3).solve(st0, lambda m: None)
    s, R, t = st
    worst = 0.0
    for c in (1, 2):
        ang = np.degrees(np.arccos(np.clip((np.trace(R[c] @ true[c][1].T) - 1) / 2, -1, 1)))
        print(f"cube {c}: scale {s[c]:.9f} (true {true[c][0]}), turn off by {ang:.2e} deg, shift off by "
              f"{np.linalg.norm(t[c] - true[c][2]) * 1000:.2e} mm; final cost {cost:.3e}")
        worst = max(worst, abs(s[c] - true[c][0]), np.radians(ang), np.linalg.norm(t[c] - true[c][2]))
    assert worst < 1e-7, worst
    print("the tie's arithmetic: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
