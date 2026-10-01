"""The reference human of the object library (USER 2026-10-01: "un modelo de un humano de referencia
sin textura, con la función add object"): a 1.75 m mannequin built from primitives — no external
model, no licence — standing on y = 0 (Y up, facing +Z), one flat grey material. Written to
server/collection/, where /api/objects/library lists it.

    python server/tools/make_reference_human.py [--height 1.75]
"""
import argparse
from pathlib import Path

import numpy as np
import trimesh

GREY = [200, 200, 200, 255]


def _capsule(a, b, r):
    a, b = np.asarray(a, float), np.asarray(b, float)
    h = float(np.linalg.norm(b - a))
    m = trimesh.creation.capsule(height=h, radius=r, count=[24, 24])
    z = np.array([0.0, 0.0, 1.0]); d = (b - a) / h
    m.apply_transform(trimesh.geometry.align_vectors(z, d))
    m.apply_translation((a + b) / 2)
    return m


def _ellipsoid(c, rx, ry, rz):
    m = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    m.apply_scale([rx, ry, rz]); m.apply_translation(c)
    return m


def build(height: float = 1.75) -> trimesh.Trimesh:
    s = height / 1.75                       # proportions of a 1.75 m adult, scaled
    P = lambda x, y, z: np.array([x, y, z]) * s
    parts = [
        _ellipsoid(P(0, 1.635, 0), 0.085 * s, 0.11 * s, 0.10 * s),                 # head
        _capsule(P(0, 1.47, 0), P(0, 1.53, 0), 0.05 * s),                         # neck
        _ellipsoid(P(0, 1.27, 0), 0.19 * s, 0.24 * s, 0.11 * s),                  # chest
        _ellipsoid(P(0, 1.00, 0), 0.17 * s, 0.13 * s, 0.11 * s),                  # pelvis
    ]
    for side in (-1, 1):
        parts += [
            _capsule(P(0.20 * side, 1.43, 0), P(0.25 * side, 1.12, 0), 0.045 * s),   # upper arm
            _capsule(P(0.25 * side, 1.12, 0), P(0.27 * side, 0.86, 0.02), 0.038 * s),  # forearm
            _ellipsoid(P(0.275 * side, 0.79, 0.025), 0.03 * s, 0.07 * s, 0.045 * s),  # hand
            _capsule(P(0.10 * side, 0.93, 0), P(0.11 * side, 0.50, 0), 0.07 * s),      # thigh
            _capsule(P(0.11 * side, 0.50, 0), P(0.11 * side, 0.10, -0.01), 0.05 * s),  # shin
            _ellipsoid(P(0.11 * side, 0.04, 0.06), 0.05 * s, 0.04 * s, 0.13 * s),     # foot
        ]
    m = trimesh.util.concatenate(parts)
    m.apply_scale(height / float(m.extents[1]))               # exactly `height` tall
    m.apply_translation([0.0, -m.bounds[0][1], 0.0])          # the soles on y = 0
    m.visual = trimesh.visual.ColorVisuals(m, vertex_colors=np.tile(GREY, (len(m.vertices), 1)))
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--height", type=float, default=1.75)
    a = ap.parse_args()
    out = Path(__file__).resolve().parents[1] / "collection" / f"reference_human_{a.height:.2f}m.glb"
    out.parent.mkdir(exist_ok=True)
    m = build(a.height)
    m.export(out)
    print(f"{out}  {len(m.faces):,} faces, height {m.extents[1]:.3f} m, footprint "
          f"{m.extents[0]:.2f} x {m.extents[2]:.2f} m")


if __name__ == "__main__":
    main()
