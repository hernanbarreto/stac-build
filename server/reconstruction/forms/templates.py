"""Parametric templates of the catalogue's forms, in torch (USER 2026-10-16).

Two families carry every fitted form:

* EXTRUSION — a 2-D section swept along an axis over a length: a panel (an outline with no
  thickness), a box (a rectangle), a cylinder (a disc, a ring when hollow), a profile (I, L, C,
  T, rect, rect_hollow, round tube, rail). The section's signed distance is exact (rectangles,
  discs, their unions and differences), the extrusion the classic one.
* REVOLUTION — a radius profile about an axis: a cone (two stations), a surface of revolution
  (K stations), and the sphere apart (exact).

Every template exposes, differentiably in its parameters: ``distance(P)`` (metres, unsigned, to
the surface — what the points are fitted with), ``samples()`` (points on the surface — what the
silhouettes are rendered from) and, in numpy, ``mesh()`` (the GLB) and ``dims()`` (the
dimensions in metres). The pose follows the VLM's words: "vertical" fixes the axis EXACTLY to the
scene's up, "horizontal" keeps it in the horizontal plane, "axis_aligned" snaps the section to the
scene's horizontal axes; "inclined"/"unknown" leave it free. Positive sizes are ``exp`` of their
parameters — no clamp ever touches them. The frame is the raw cloud's; nothing here reads a mask
or a camera.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .catalog import FormAnswer

_EPS = 1e-14   # inside the square roots: a 1e-7 m floor, gradients finite at the surface
# the robust extent of a coordinate: its (q, 1 − q) quantiles. DECLARED bias of the cloud-only
# stage: a uniformly filled extent reads short by q per side, a measured end face reads long by
# the noise tail (~2.6 σ at q = 0.5 %); the silhouette corrector (the images see where the object
# ends) refines every extent afterwards
EXTENT_Q = 0.005
_KEEP_TOL = 1e-6   # a sample on the section's boundary (sdf = the 1e-7 floor) belongs to the surface
SECTIONS = ("rect", "circle", "ring", "rect_hollow", "I", "L", "C", "T", "arch")


# ── the scene frame ────────────────────────────────────────────────────────────

class Frame:
    """The scene's orthonormal frame (up, X, Z), numpy unit vectors in the cloud's coordinates."""

    def __init__(self, up: np.ndarray, X: Optional[np.ndarray] = None):
        up = np.asarray(up, np.float64)
        up = up / max(float(np.linalg.norm(up)), _EPS)
        if X is None:
            X = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
        X = np.asarray(X, np.float64)
        X = X - up * float(X @ up)
        X = X / max(float(np.linalg.norm(X)), _EPS)
        self.up, self.X, self.Z = up, X, np.cross(up, X)

    def to_dict(self) -> dict:
        return {"up": self.up.tolist(), "X": self.X.tolist(), "Z": self.Z.tolist()}


def _t(a, dev, dtype=torch.float64) -> torch.Tensor:
    return torch.as_tensor(np.asarray(a, np.float64), dtype=dtype, device=dev)


# ── the pose: axis + roll under the VLM's words ────────────────────────────────

class Pose:
    """centre c, axis a, section basis (u, v) — parametrised by the orientation words.

    axis_mode: 'fixed' (a = the given vector, exactly), 'horizontal' (a = cos ψ·X + sin ψ·Z),
    'free' (two angles about the frame). roll_mode: 'free' (an angle about the axis) or
    'snapped' (the section's u follows the frame: u = X for a vertical axis, u = up for a
    horizontal one — the Manhattan alignment, exact)."""

    def __init__(self, frame: Frame, axis_mode: str, roll_mode: str, dev,
                 c: np.ndarray, axis0: np.ndarray, roll0: float = 0.0, psi0: float = 0.0):
        self.frame, self.axis_mode, self.roll_mode, self.dev = frame, axis_mode, roll_mode, dev
        self.c = torch.nn.Parameter(_t(c, dev))
        self._up, self._X, self._Z = _t(frame.up, dev), _t(frame.X, dev), _t(frame.Z, dev)
        self._fixed = _t(axis0 / max(float(np.linalg.norm(axis0)), _EPS), dev)
        if axis_mode == "free":
            # the axis as two angles about the frame: elevation from up, azimuth in (X, Z)
            a = np.asarray(axis0, np.float64) / max(float(np.linalg.norm(axis0)), _EPS)
            el = math.acos(max(-1.0, min(1.0, float(a @ frame.up))))
            az = math.atan2(float(a @ frame.Z), float(a @ frame.X))
            self.ang = torch.nn.Parameter(_t([el, az], dev))
        elif axis_mode == "horizontal":
            self.ang = torch.nn.Parameter(_t([psi0], dev))
        elif axis_mode == "fixed":
            self.ang = None
        else:
            raise ValueError(f"axis_mode {axis_mode!r}")
        self.roll = torch.nn.Parameter(_t([roll0], dev)) if roll_mode == "free" else None
        if roll_mode not in ("free", "snapped"):
            raise ValueError(f"roll_mode {roll_mode!r}")

    def parameters(self) -> List[torch.Tensor]:
        return [p for p in (self.c, self.ang, self.roll) if p is not None]

    def axis(self) -> torch.Tensor:
        if self.axis_mode == "fixed":
            return self._fixed
        if self.axis_mode == "horizontal":
            return torch.cos(self.ang[0]) * self._X + torch.sin(self.ang[0]) * self._Z
        el, az = self.ang[0], self.ang[1]
        return (torch.cos(el) * self._up
                + torch.sin(el) * (torch.cos(az) * self._X + torch.sin(az) * self._Z))

    def basis(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(u, v, a): an orthonormal right-handed basis, u·v spanning the section."""
        a = self.axis()
        if self.axis_mode == "fixed" and abs(float(self._fixed @ self._up)) > 0.999:
            u0 = self._X
        elif self.axis_mode == "horizontal":
            u0 = self._up
        else:
            # the frame direction least aligned with the axis
            cands = [self._X, self._up, self._Z]
            dots = [abs(float(a.detach() @ q)) for q in cands]
            u0 = cands[int(np.argmin(dots))]
        u = u0 - a * (u0 @ a)
        u = u / torch.clamp(torch.linalg.norm(u), min=_EPS)
        v = torch.linalg.cross(a, u)
        if self.roll is not None:
            cr, sr = torch.cos(self.roll[0]), torch.sin(self.roll[0])
            u, v = cr * u + sr * v, -sr * u + cr * v
        return u, v, a

    def local(self, P: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u, v, a = self.basis()
        q = P - self.c
        return q @ u, q @ v, q @ a

    def world(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        u, v, a = self.basis()
        return self.c + x[:, None] * u + y[:, None] * v + z[:, None] * a

    def matrix(self) -> np.ndarray:
        u, v, a = [t.detach().cpu().numpy() for t in self.basis()]
        M = np.eye(4)
        M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = u, v, a, self.c.detach().cpu().numpy()
        return M


# ── 2-D sections: signed distance, boundary polyline, shapely polygon ──────────

def _sd_rect(x, y, bx, by):
    qx, qy = torch.abs(x) - bx, torch.abs(y) - by
    outside = torch.sqrt(torch.clamp(qx, min=0.0) ** 2 + torch.clamp(qy, min=0.0) ** 2 + _EPS)
    return outside + torch.clamp(torch.maximum(qx, qy), max=0.0)


def _sd_circle(x, y, r):
    return torch.sqrt(x * x + y * y + _EPS) - r


class Section:
    """A 2-D section: kind + positive size parameters (log-parametrised)."""

    def __init__(self, kind: str, sizes: Dict[str, float], dev):
        if kind not in SECTIONS:
            raise ValueError(f"section {kind!r}")
        self.kind, self.dev = kind, dev
        self.names = list(sizes)
        self.logs = torch.nn.Parameter(_t([math.log(max(float(sizes[k]), 1e-4)) for k in self.names], dev))

    def parameters(self) -> List[torch.Tensor]:
        return [self.logs]

    def size(self) -> Dict[str, torch.Tensor]:
        e = torch.exp(self.logs)
        return {k: e[i] for i, k in enumerate(self.names)}

    def sizes_np(self) -> Dict[str, float]:
        return {k: float(v) for k, v in zip(self.names, torch.exp(self.logs).detach().cpu().numpy())}

    # signed distance of (x, y) to the section
    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        s = self.size()
        k = self.kind
        if k == "rect":
            return _sd_rect(x, y, s["bx"], s["by"])
        if k == "circle":
            return _sd_circle(x, y, s["r"])
        if k == "ring":
            return torch.maximum(_sd_circle(x, y, s["r"]), -_sd_circle(x, y, s["r"] - s["t"]))
        if k == "rect_hollow":
            return torch.maximum(_sd_rect(x, y, s["bx"], s["by"]),
                                 -_sd_rect(x, y, s["bx"] - s["t"], s["by"] - s["t"]))
        if k == "I":
            b, h, tf, tw = s["b"], s["h"], s["tf"], s["tw"]
            top = _sd_rect(x, y - (h / 2 - tf / 2), b / 2, tf / 2)
            bot = _sd_rect(x, y + (h / 2 - tf / 2), b / 2, tf / 2)
            web = _sd_rect(x, y, tw / 2, h / 2 - tf)
            return torch.minimum(torch.minimum(top, bot), web)
        if k == "T":
            b, h, tf, tw = s["b"], s["h"], s["tf"], s["tw"]
            top = _sd_rect(x, y - (h / 2 - tf / 2), b / 2, tf / 2)
            web = _sd_rect(x, y + tf / 2, tw / 2, h / 2 - tf / 2)
            return torch.minimum(top, web)
        if k == "C":
            b, h, tf, tw = s["b"], s["h"], s["tf"], s["tw"]
            web = _sd_rect(x + (b / 2 - tw / 2), y, tw / 2, h / 2)
            top = _sd_rect(x, y - (h / 2 - tf / 2), b / 2, tf / 2)
            bot = _sd_rect(x, y + (h / 2 - tf / 2), b / 2, tf / 2)
            return torch.minimum(torch.minimum(top, bot), web)
        if k == "L":
            b, h, t = s["b"], s["h"], s["t"]
            vert = _sd_rect(x + (b / 2 - t / 2), y, t / 2, h / 2)
            horz = _sd_rect(x, y + (h / 2 - t / 2), b / 2, t / 2)
            return torch.minimum(vert, horz)
        if k == "arch":
            b, H = s["b"], s["H"]
            rc = b / 2
            rect = _sd_rect(x, y + rc / 2, b / 2, (H - rc) / 2)
            cap = _sd_circle(x, y - (H / 2 - rc), rc)
            return torch.minimum(rect, cap)
        raise ValueError(k)

    # the boundary as closed polylines (lists of (x, y) tensors), differentiable
    def boundary(self, m: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        s = self.size()
        k = self.kind
        dev = self.dev

        def poly(pts: Sequence[Tuple[torch.Tensor, torch.Tensor]]):
            xs, ys = [], []
            n = len(pts)
            per = max(2, m // n)
            tt = torch.linspace(0.0, 1.0, per + 1, dtype=torch.float64, device=dev)[:-1]
            for i in range(n):
                x0, y0 = pts[i]
                x1, y1 = pts[(i + 1) % n]
                xs.append(x0 + (x1 - x0) * tt)
                ys.append(y0 + (y1 - y0) * tt)
            return torch.cat(xs), torch.cat(ys)

        def circ(r, cx=0.0, cy=0.0, n=None):
            n = n or m
            ang = torch.linspace(0.0, 2 * math.pi, n + 1, dtype=torch.float64, device=dev)[:-1]
            return cx + r * torch.cos(ang), cy + r * torch.sin(ang)

        z = torch.zeros((), dtype=torch.float64, device=dev)
        if k == "rect":
            bx, by = s["bx"], s["by"]
            return [poly([(bx, by), (-bx, by), (-bx, -by), (bx, -by)])]
        if k == "circle":
            return [circ(s["r"])]
        if k == "ring":
            return [circ(s["r"]), circ(s["r"] - s["t"], n=max(8, m // 2))]
        if k == "rect_hollow":
            bx, by, t = s["bx"], s["by"], s["t"]
            return [poly([(bx, by), (-bx, by), (-bx, -by), (bx, -by)]),
                    poly([(bx - t, by - t), (-(bx - t), by - t), (-(bx - t), -(by - t)), (bx - t, -(by - t))])]
        if k == "I":
            b, h, tf, tw = s["b"], s["h"], s["tf"], s["tw"]
            return [poly([(b / 2, h / 2), (-b / 2, h / 2), (-b / 2, h / 2 - tf), (-tw / 2, h / 2 - tf),
                          (-tw / 2, -h / 2 + tf), (-b / 2, -h / 2 + tf), (-b / 2, -h / 2), (b / 2, -h / 2),
                          (b / 2, -h / 2 + tf), (tw / 2, -h / 2 + tf), (tw / 2, h / 2 - tf), (b / 2, h / 2 - tf)])]
        if k == "T":
            b, h, tf, tw = s["b"], s["h"], s["tf"], s["tw"]
            return [poly([(b / 2, h / 2), (-b / 2, h / 2), (-b / 2, h / 2 - tf), (-tw / 2, h / 2 - tf),
                          (-tw / 2, -h / 2), (tw / 2, -h / 2), (tw / 2, h / 2 - tf), (b / 2, h / 2 - tf)])]
        if k == "C":
            b, h, tf, tw = s["b"], s["h"], s["tf"], s["tw"]
            return [poly([(b / 2, h / 2), (-b / 2, h / 2), (-b / 2, -h / 2), (b / 2, -h / 2),
                          (b / 2, -h / 2 + tf), (-b / 2 + tw, -h / 2 + tf), (-b / 2 + tw, h / 2 - tf),
                          (b / 2, h / 2 - tf)])]
        if k == "L":
            b, h, t = s["b"], s["h"], s["t"]
            return [poly([(-b / 2, h / 2), (-b / 2, -h / 2), (b / 2, -h / 2), (b / 2, -h / 2 + t),
                          (-b / 2 + t, -h / 2 + t), (-b / 2 + t, h / 2)])]
        if k == "arch":
            b, H = s["b"], s["H"]
            rc = b / 2
            yc = H / 2 - rc
            n_arc = max(8, m // 2)
            ang = torch.linspace(0.0, math.pi, n_arc, dtype=torch.float64, device=dev)
            ax, ay = rc * torch.cos(ang), yc + rc * torch.sin(ang)
            # from the right top of the rectangle, over the arc, down the left side, along the base
            xs = torch.cat([ax, torch.stack([-b / 2, -b / 2, b / 2, b / 2]) + z])
            ys = torch.cat([ay, torch.stack([yc, -H / 2, -H / 2, yc]) + z])
            return [(xs, ys)]
        raise ValueError(k)

    def polygon(self):
        """The section as a shapely polygon (holes for ring / rect_hollow) for the mesh."""
        from shapely.geometry import Polygon
        rings = []
        with torch.no_grad():
            for xs, ys in self.boundary(96):
                rings.append(list(zip(xs.detach().cpu().numpy().tolist(), ys.detach().cpu().numpy().tolist())))
        if len(rings) == 1:
            return Polygon(rings[0])
        outer = max(rings, key=lambda r: Polygon(r).area)
        holes = [r for r in rings if r is not outer]
        return Polygon(outer, holes=holes)

    def bbox(self) -> Tuple[float, float]:
        """Half-extents of the section's bounding box (numpy)."""
        s = self.sizes_np()
        k = self.kind
        if k == "rect" or k == "rect_hollow":
            return s["bx"], s["by"]
        if k in ("circle", "ring"):
            return s["r"], s["r"]
        if k in ("I", "T", "C"):
            return s["b"] / 2, s["h"] / 2
        if k == "L":
            return s["b"] / 2, s["h"] / 2
        if k == "arch":
            return s["b"] / 2, s["H"] / 2
        raise ValueError(k)


# ── the templates ──────────────────────────────────────────────────────────────

class Template:
    name = "template"
    form = ""

    def parameters(self) -> List[torch.Tensor]:
        raise NotImplementedError

    def distance(self, P: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def samples(self, n: int) -> torch.Tensor:
        raise NotImplementedError

    def mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def dims(self) -> Dict[str, float]:
        raise NotImplementedError

    def record(self) -> dict:
        return {"template": self.name, "form": self.form, "dims_m": self.dims(),
                "pose": self.pose.matrix().tolist() if hasattr(self, "pose") else None}

    def measure_extents(self, P: torch.Tensor) -> None:
        """The extents the points cannot push at all — a plate's OUTLINE: every point inside it
        measures |z| whatever its size — are MEASURED from the points in the current pose
        (robust percentiles). A solid's faces are pulled by their points through the distance
        itself (a face converges to its points' median, unbiased); only where an end has no
        points does the optimiser slide along a plateau — ``bound_extents`` stops that. The
        silhouette corrector, which sees where the object ends, frees everything again."""
        return None

    def bound_extents(self, P: torch.Tensor, sigma: float) -> None:
        """No surface beyond the last measured point: an extent the optimiser carried past the
        points' extreme (momentum into the plateau — an I-beam grew 4.0 → 4.3 m) is brought
        back to that extreme plus σ. Default: nothing to bound."""
        return None

    def state(self) -> List[np.ndarray]:
        return [p.detach().cpu().numpy().copy() for p in self.parameters()]

    def load_state(self, st: List[np.ndarray]) -> None:
        with torch.no_grad():
            for p, v in zip(self.parameters(), st):
                p.copy_(torch.as_tensor(v, dtype=p.dtype, device=p.device))


def _grid(n: int, dev) -> Tuple[torch.Tensor, torch.Tensor]:
    k = max(2, int(math.sqrt(n)))
    g = torch.linspace(-1.0, 1.0, k, dtype=torch.float64, device=dev)
    gx, gy = torch.meshgrid(g, g, indexing="ij")
    return gx.reshape(-1), gy.reshape(-1)


class Extrusion(Template):
    """A section swept along the axis over 2·half_length (0 for a panel: a flat plate).

    ``open``: the surface is the lateral band only (a hollow tube seen from outside, a plate
    with no thickness): the caps are not surfaces."""
    name = "extrusion"

    def __init__(self, form: str, section: Section, pose: Pose, half_length: Optional[float],
                 open_: bool, dev):
        self.form, self.section, self.pose, self.dev, self.open = form, section, pose, dev, bool(open_)
        self.flat = half_length is None
        self.log_h = (None if self.flat
                      else torch.nn.Parameter(_t([math.log(max(float(half_length), 1e-4))], dev)))

    def parameters(self):
        return self.pose.parameters() + self.section.parameters() + ([] if self.flat else [self.log_h])

    def half_length(self) -> torch.Tensor:
        return (torch.zeros((), dtype=torch.float64, device=self.dev) if self.flat
                else torch.exp(self.log_h[0]))

    def measure_extents(self, P: torch.Tensor) -> None:
        if not self.flat:
            return None          # a solid's faces are fitted by the distance itself
        with torch.no_grad():
            x, y, z = self.pose.local(P)
            q = torch.tensor([EXTENT_Q, 1.0 - EXTENT_Q], dtype=torch.float64, device=self.dev)
            sec = self.section
            if sec.kind in ("rect", "circle", "arch"):
                x0, x1 = torch.quantile(x, q)
                y0, y1 = torch.quantile(y, q)
                u, v, _a = self.pose.basis()
                self.pose.c += u * ((x0 + x1) / 2) + v * ((y0 + y1) / 2)
                ex, ey = torch.clamp(x1 - x0, min=1e-3), torch.clamp(y1 - y0, min=1e-3)
                if sec.kind == "rect":
                    sec.logs[sec.names.index("bx")] = torch.log(ex / 2)
                    sec.logs[sec.names.index("by")] = torch.log(ey / 2)
                elif sec.kind == "circle":
                    sec.logs[sec.names.index("r")] = torch.log(torch.maximum(ex, ey) / 2)
                elif sec.kind == "arch":
                    sec.logs[sec.names.index("b")] = torch.log(ex)
                    sec.logs[sec.names.index("H")] = torch.log(torch.maximum(ey, ex * 0.75))

    def bound_extents(self, P: torch.Tensor, sigma: float) -> None:
        """Each END of the solid is brought back to the last measured point (+ σ) on its side,
        the centre re-placed between the ends — the plateau drift moves the centre and one end
        together (a plate grew 1.2 → 1.4 m along its own length)."""
        if self.flat:
            return None
        with torch.no_grad():
            x, y, z = self.pose.local(P)
            s = float(sigma)
            u, v, a = self.pose.basis()

            def clamp_ends(vals: torch.Tensor, half: float):
                lo, hi = -half, half
                lo2 = max(lo, float(vals.min()) - s)
                hi2 = min(hi, float(vals.max()) + s)
                if hi2 <= lo2:
                    return None, None
                return (lo2 + hi2) / 2.0, max((hi2 - lo2) / 2.0, 1e-4)

            mid, half = clamp_ends(z, float(torch.exp(self.log_h[0])))
            if mid is not None:
                self.pose.c += a * mid
                self.log_h[0] = math.log(half)
            sec = self.section
            if sec.kind in ("rect", "rect_hollow"):
                for nm, vals, axis in (("bx", x, u), ("by", y, v)):
                    i = sec.names.index(nm)
                    mid, half = clamp_ends(vals, float(torch.exp(sec.logs[i])))
                    if mid is not None:
                        self.pose.c += axis * mid
                        sec.logs[i] = math.log(half)

    def distance(self, P: torch.Tensor) -> torch.Tensor:
        x, y, z = self.pose.local(P)
        d2 = self.section.sdf(x, y)
        h = self.half_length()
        if self.flat:
            # a plate: |z| over the outline, the outline's edge distance outside it
            return torch.sqrt(torch.clamp(d2, min=0.0) ** 2 + z * z + _EPS)
        dz = torch.abs(z) - h
        if self.open:
            return torch.sqrt(d2 * d2 + torch.clamp(dz, min=0.0) ** 2 + _EPS)
        inside = torch.clamp(torch.maximum(d2, dz), max=0.0)
        outside = torch.sqrt(torch.clamp(d2, min=0.0) ** 2 + torch.clamp(dz, min=0.0) ** 2 + _EPS)
        return torch.abs(inside + outside)

    def _layout(self, n: int) -> Tuple[int, int, int]:
        """(ring points, stations, cap grid side) for ~n samples at ONE spacing over the whole
        surface (lateral area + caps): a long thin bar gets many stations and few ring points,
        a plate the other way round — the silhouette term measures the form, never the gaps
        between rings (a 1024-sample column had its rings 11 cm apart)."""
        with torch.no_grad():
            rings = self.section.boundary(64)
            per = 0.0
            for xs, ys in rings:
                x = xs.detach().cpu().numpy(); y = ys.detach().cpu().numpy()
                per += float(np.sum(np.hypot(np.diff(np.r_[x, x[0]]), np.diff(np.r_[y, y[0]]))))
            L = 2.0 * float(self.half_length().detach())
            bx, by = self.section.bbox()
            cap_area = 0.0 if self.open else 2.0 * float(self.section.polygon().area)
            total = max(per * L + cap_area, 1e-9)
            d = math.sqrt(total / max(int(n), 16))
            m_ring = int(max(16, round(per / d)))
            s = int(max(2, round(L / d)))
            k_cap = int(max(2, round(2.0 * max(bx, by) / d))) if cap_area > 0 else 0
        return m_ring, s, k_cap

    def samples(self, n: int) -> torch.Tensor:
        h = self.half_length()
        dev = self.dev
        if self.flat:
            gx, gy = _grid(n, dev)
            bx, by = self.section.bbox()
            x, y = gx * bx, gy * by
            keep = self.section.sdf(x, y) <= _KEEP_TOL
            z = torch.zeros_like(x)
            return self.pose.world(x[keep], y[keep], z[keep])
        m_ring, s, k_cap = self._layout(n)
        rings = self.section.boundary(m_ring)
        bx = torch.cat([r[0] for r in rings])
        by = torch.cat([r[1] for r in rings])
        m = len(bx)
        zs = torch.linspace(-1.0, 1.0, s, dtype=torch.float64, device=dev) * h
        X = bx.repeat(s)
        Y = by.repeat(s)
        Z = zs.repeat_interleave(m)
        if not self.open and k_cap >= 2:
            g = torch.linspace(-1.0, 1.0, k_cap, dtype=torch.float64, device=dev)
            gx, gy = torch.meshgrid(g, g, indexing="ij")
            gx, gy = gx.reshape(-1), gy.reshape(-1)
            cbx, cby = self.section.bbox()
            cx, cy = gx * cbx, gy * cby
            keep = self.section.sdf(cx, cy) <= _KEEP_TOL
            cx, cy = cx[keep], cy[keep]
            X = torch.cat([X, cx, cx])
            Y = torch.cat([Y, cy, cy])
            Z = torch.cat([Z, torch.full_like(cx, 1.0) * h, torch.full_like(cx, -1.0) * h])
        return self.pose.world(X, Y, Z)

    def mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        import trimesh
        poly = self.section.polygon()
        M = self.pose.matrix()
        if self.flat:
            V2, F = trimesh.creation.triangulate_polygon(poly, engine="earcut")
            V = np.column_stack([V2, np.zeros(len(V2))])
        else:
            h = float(self.half_length().detach())
            m = trimesh.creation.extrude_polygon(poly, 2.0 * h, engine="earcut")
            V, F = np.asarray(m.vertices, np.float64).copy(), np.asarray(m.faces, np.int64).copy()
            V[:, 2] -= h
        Vw = V @ M[:3, :3].T + M[:3, 3]
        return Vw, np.asarray(F, np.int64)

    def dims(self) -> Dict[str, float]:
        s = self.section.sizes_np()
        out = {"section": self.section.kind}
        k = self.section.kind
        if k in ("rect", "rect_hollow"):
            out.update(width_m=2 * s["bx"], height_m=2 * s["by"])
            if k == "rect_hollow":
                out["wall_m"] = s["t"]
        elif k in ("circle", "ring"):
            out.update(radius_m=s["r"], diameter_m=2 * s["r"])
            if k == "ring":
                out["wall_m"] = s["t"]
        elif k in ("I", "T", "C"):
            out.update(flange_width_m=s["b"], depth_m=s["h"], flange_m=s["tf"], web_m=s["tw"])
        elif k == "L":
            out.update(leg_a_m=s["b"], leg_b_m=s["h"], thickness_m=s["t"])
        elif k == "arch":
            out.update(width_m=s["b"], height_m=s["H"])
        out["length_m"] = 0.0 if self.flat else 2.0 * float(self.half_length().detach())
        # the extents along the SCENE's axes (up, X, Z) — what a BIM reads, whatever the
        # template's own axis is (a box's extrusion axis is arbitrary)
        with torch.no_grad():
            u, v, a = [t.detach().cpu().numpy() for t in self.pose.basis()]
        bx, by = self.section.bbox()
        half = np.array([bx, by, 0.0 if self.flat else float(self.half_length().detach())])
        fr = self.pose.frame
        for name, ax in (("extent_up_m", fr.up), ("extent_X_m", fr.X), ("extent_Z_m", fr.Z)):
            out[name] = 2.0 * float(abs(u @ ax) * half[0] + abs(v @ ax) * half[1] + abs(a @ ax) * half[2])
        return {k2: (float(v) if not isinstance(v, str) else v) for k2, v in out.items()}


class Revolution(Template):
    """A radius profile r(z) about the axis over 2·half_length: K stations, linear between.
    A cone is K = 2; a cylinder of revolution K = 2 with one radius."""
    name = "revolution"

    def __init__(self, form: str, pose: Pose, radii: Sequence[float], half_length: float, dev,
                 shared_radius: bool = False):
        self.form, self.pose, self.dev = form, pose, dev
        self.shared = bool(shared_radius)
        r = [max(float(x), 1e-4) for x in radii]
        self.log_r = torch.nn.Parameter(_t([math.log(r[0])] if self.shared else [math.log(x) for x in r], dev))
        self.K = 2 if self.shared else len(r)
        self.log_h = torch.nn.Parameter(_t([math.log(max(float(half_length), 1e-4))], dev))

    def parameters(self):
        return self.pose.parameters() + [self.log_r, self.log_h]

    def radii(self) -> torch.Tensor:
        r = torch.exp(self.log_r)
        return r.repeat(2) if self.shared else r

    def half_length(self) -> torch.Tensor:
        return torch.exp(self.log_h[0])

    def _stations(self) -> torch.Tensor:
        return torch.linspace(-1.0, 1.0, self.K, dtype=torch.float64, device=self.dev) * self.half_length()

    def bound_extents(self, P: torch.Tensor, sigma: float) -> None:
        with torch.no_grad():
            _x, _y, z = self.pose.local(P)
            s = float(sigma)
            half = float(torch.exp(self.log_h[0]))
            lo2 = max(-half, float(z.min()) - s)
            hi2 = min(half, float(z.max()) + s)
            if hi2 > lo2:
                self.pose.c += self.pose.basis()[2] * ((lo2 + hi2) / 2.0)
                self.log_h[0] = math.log(max((hi2 - lo2) / 2.0, 1e-4))

    def distance(self, P: torch.Tensor) -> torch.Tensor:
        x, y, z = self.pose.local(P)
        rho = torch.sqrt(x * x + y * y + _EPS)
        r, zs = self.radii(), self._stations()
        # boundary of the solid in the (rho, z) half-plane: the profile segments + the two caps
        segs = [((r[i], zs[i]), (r[i + 1], zs[i + 1])) for i in range(self.K - 1)]
        zero = torch.zeros((), dtype=torch.float64, device=self.dev)
        segs.append(((zero, zs[0]), (r[0], zs[0])))
        segs.append(((zero, zs[-1]), (r[-1], zs[-1])))
        best = None
        for (ax, az), (bx, bz) in segs:
            dx, dz = bx - ax, bz - az
            L2 = dx * dx + dz * dz + _EPS
            tt = torch.clamp(((rho - ax) * dx + (z - az) * dz) / L2, 0.0, 1.0)
            d = torch.sqrt((rho - (ax + tt * dx)) ** 2 + (z - (az + tt * dz)) ** 2 + _EPS)
            best = d if best is None else torch.minimum(best, d)
        return best

    def samples(self, n: int) -> torch.Tensor:
        dev = self.dev
        r, zs = self.radii(), self._stations()
        with torch.no_grad():
            rn = r.detach().cpu().numpy()
            L = 2.0 * float(self.half_length().detach())
            lateral = float(np.mean(2 * math.pi * rn)) * L
            caps = math.pi * (float(rn[0]) ** 2 + float(rn[-1]) ** 2)
            d = math.sqrt(max(lateral + caps, 1e-9) / max(int(n), 16))
            S = int(max(8, round(L / d)))
            A = int(max(8, round(float(np.mean(2 * math.pi * rn)) / d)))
        tt = torch.linspace(0.0, 1.0, S, dtype=torch.float64, device=dev)
        # linear interpolation of the profile along the stations
        pos = tt * (self.K - 1)
        i0 = torch.clamp(pos.floor().to(torch.int64), 0, self.K - 2)
        w = pos - i0.to(torch.float64)
        rz = r[i0] * (1 - w) + r[i0 + 1] * w
        zz = zs[i0] * (1 - w) + zs[i0 + 1] * w
        ang = torch.linspace(0.0, 2 * math.pi, A + 1, dtype=torch.float64, device=dev)[:-1]
        X = (rz[:, None] * torch.cos(ang)[None, :]).reshape(-1)
        Y = (rz[:, None] * torch.sin(ang)[None, :]).reshape(-1)
        Z = zz[:, None].expand(S, A).reshape(-1)
        # the caps, at the same spacing
        k_cap = int(max(2, round(2.0 * float(rn.max()) / d)))
        g = torch.linspace(-1.0, 1.0, k_cap, dtype=torch.float64, device=dev)
        gx, gy = torch.meshgrid(g, g, indexing="ij")
        gx, gy = gx.reshape(-1), gy.reshape(-1)
        keep = gx * gx + gy * gy <= 1.0
        gx, gy = gx[keep], gy[keep]
        X = torch.cat([X, gx * r[0], gx * r[-1]])
        Y = torch.cat([Y, gy * r[0], gy * r[-1]])
        Z = torch.cat([Z, torch.full_like(gx, 1.0) * zs[0], torch.full_like(gx, 1.0) * zs[-1]])
        return self.pose.world(X, Y, Z)

    def mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        import trimesh
        r = self.radii().detach().cpu().numpy()
        zs = self._stations().detach().cpu().numpy()
        line = [(0.0, float(zs[0]))] + [(float(r[i]), float(zs[i])) for i in range(self.K)] + [(0.0, float(zs[-1]))]
        m = trimesh.creation.revolve(np.asarray(line, np.float64), sections=64)
        V = np.asarray(m.vertices, np.float64).copy()
        F = np.asarray(m.faces, np.int64).copy()
        # trimesh revolves the 2-D line about its y axis and puts that axis on 3-D z — the
        # template's local axis: (x, y) of the line = (radius, axial) → (x, y, z) local as is
        M = self.pose.matrix()
        return V @ M[:3, :3].T + M[:3, 3], F

    def dims(self) -> Dict[str, float]:
        r = self.radii().detach().cpu().numpy()
        return {"radius_base_m": float(r[0]), "radius_top_m": float(r[-1]),
                "radius_max_m": float(r.max()), "length_m": 2.0 * float(self.half_length().detach()),
                "stations": int(self.K)}


class Sphere(Template):
    name = "sphere"
    form = "sphere"

    def __init__(self, c: np.ndarray, radius: float, dev):
        self.dev = dev
        self.c = torch.nn.Parameter(_t(c, dev))
        self.log_r = torch.nn.Parameter(_t([math.log(max(float(radius), 1e-4))], dev))

    def parameters(self):
        return [self.c, self.log_r]

    def radius(self) -> torch.Tensor:
        return torch.exp(self.log_r[0])

    def distance(self, P: torch.Tensor) -> torch.Tensor:
        return torch.abs(torch.sqrt(((P - self.c) ** 2).sum(1) + _EPS) - self.radius())

    def samples(self, n: int) -> torch.Tensor:
        i = torch.arange(n, dtype=torch.float64, device=self.dev) + 0.5
        phi = torch.acos(1 - 2 * i / n)
        th = math.pi * (1 + 5 ** 0.5) * i
        r = self.radius()
        return self.c + r * torch.stack([torch.cos(th) * torch.sin(phi), torch.sin(th) * torch.sin(phi),
                                         torch.cos(phi)], 1)

    def mesh(self):
        import trimesh
        m = trimesh.creation.icosphere(subdivisions=3, radius=float(self.radius().detach()))
        return np.asarray(m.vertices, np.float64) + self.c.detach().cpu().numpy(), np.asarray(m.faces, np.int64)

    def dims(self):
        r = float(self.radius().detach())
        return {"radius_m": r, "diameter_m": 2 * r}

    def record(self) -> dict:
        return {"template": self.name, "form": self.form, "dims_m": self.dims(),
                "centre": self.c.detach().cpu().numpy().tolist()}


# ── building a template from the VLM's answer and the object's points ──────────

def _pca(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = np.median(P, axis=0)
    q = P - c
    _, sv, vt = np.linalg.svd(q[:: max(1, len(q) // 20000)], full_matrices=False)
    return c, vt, sv


def _extent(q: np.ndarray, axis: np.ndarray) -> Tuple[float, float]:
    s = q @ axis
    return float(np.percentile(s, 1)), float(np.percentile(s, 99))


def _axis_for(ans: FormAnswer, frame: Frame, vt: np.ndarray, panel: bool) -> Tuple[str, np.ndarray, float]:
    """(axis_mode, axis0, psi0) from the orientation words. For a panel the axis is the plate's
    NORMAL (the least-variance direction); for the others the length (the most variance)."""
    up = frame.up
    principal = vt[2] if panel else vt[0]
    o = ans.orientation
    if panel:
        if o == "horizontal":
            return "fixed", up.copy(), 0.0
        if o == "vertical":
            n = principal - up * float(principal @ up)
            if np.linalg.norm(n) < 1e-6:
                n = frame.X
            psi = math.atan2(float(n @ frame.Z), float(n @ frame.X))
            if ans.axis_aligned:
                psi = round(psi / (math.pi / 2)) * (math.pi / 2)
            return "horizontal", n, psi
        return "free", principal, 0.0
    if o == "vertical":
        return "fixed", up.copy(), 0.0
    if o == "horizontal":
        a = principal - up * float(principal @ up)
        if np.linalg.norm(a) < 1e-6:
            a = frame.X
        psi = math.atan2(float(a @ frame.Z), float(a @ frame.X))
        if ans.axis_aligned:
            psi = round(psi / (math.pi / 2)) * (math.pi / 2)
        return "horizontal", a, psi
    return "free", principal, 0.0


def _section_for(ans: FormAnswer, kind_hint: Optional[str], ex_u: float, ex_v: float, radial: float,
                 dev) -> Section:
    """The section of an extruded form, sized from the object's extents (the fit refines)."""
    form = kind_hint or ans.form
    hollow = bool(ans.hollow)
    if form == "panel":
        o = ans.outline
        if o == "circle":
            return Section("circle", {"r": max(ex_u, ex_v) / 2}, dev)
        if o == "arch":
            return Section("arch", {"b": ex_u, "H": max(ex_v, ex_u * 0.75)}, dev)
        return Section("rect", {"bx": ex_u / 2, "by": ex_v / 2}, dev)
    if form == "box":
        return Section("rect", {"bx": ex_u / 2, "by": ex_v / 2}, dev)
    if form == "cylinder":
        r = max(radial, 1e-3)
        return Section("ring", {"r": r, "t": r * 0.2}, dev) if hollow else Section("circle", {"r": r}, dev)
    if form == "profile":
        p = ans.profile
        b, h = max(ex_u, 1e-3), max(ex_v, 1e-3)
        if p in ("I", "T", "C"):
            return Section(p, {"b": b, "h": h, "tf": h * 0.1, "tw": b * 0.1}, dev)
        if p == "L":
            return Section("L", {"b": b, "h": h, "t": min(b, h) * 0.12}, dev)
        if p == "rect_hollow":
            return Section("rect_hollow", {"bx": b / 2, "by": h / 2, "t": min(b, h) * 0.1}, dev)
        if p == "round_tube":
            r = max(radial, 1e-3)
            return Section("ring", {"r": r, "t": r * 0.2}, dev)
        if p == "rail":
            return Section("I", {"b": b, "h": h, "tf": h * 0.2, "tw": b * 0.3}, dev)
        return Section("rect", {"bx": b / 2, "by": h / 2}, dev)
    raise ValueError(f"no section for form {form!r}")


def build_template(ans: FormAnswer, P: np.ndarray, frame: Frame, dev, *,
                   form: Optional[str] = None, revolution_stations: int = 8) -> Template:
    """The template of ``ans.form`` (or ``form``) initialised on the object's points ``P``
    (the cloud's frame): centre, axis by the orientation words, sizes from the extents.
    Raises ValueError for 'composite' / 'freeform' (no single template)."""
    form = form or ans.form
    P = np.asarray(P, np.float64)
    if len(P) < 10:
        raise ValueError("fewer than 10 points")
    c, vt, _sv = _pca(P)
    q = P - c
    if form == "sphere":
        r = float(np.median(np.linalg.norm(q, axis=1)))
        return Sphere(c, r, dev)
    if form in ("cone", "revolution"):
        mode, a0, psi0 = _axis_for(ans, frame, vt, panel=False)
        pose = Pose(frame, mode, "snapped", dev, c, a0, psi0=psi0)
        with torch.no_grad():
            Pt = _t(P, dev)
            x, y, z = pose.local(Pt)
            rho = torch.sqrt(x * x + y * y).cpu().numpy()
            zz = z.cpu().numpy()
        z0, z1 = float(np.percentile(zz, 0.5)), float(np.percentile(zz, 99.5))
        K = 2 if form == "cone" else max(3, int(revolution_stations))
        edges = np.linspace(z0, z1, K + 1)
        radii = []
        for i in range(K):
            sel = (zz >= edges[i]) & (zz <= edges[i + 1])
            radii.append(float(np.median(rho[sel])) if sel.sum() >= 5 else float(np.median(rho)))
        half = (z1 - z0) / 2
        with torch.no_grad():
            u, v, a_now = pose.basis()
            pose.c += (a_now * ((z0 + z1) / 2) + u * float(np.median(x.cpu().numpy()))
                       + v * float(np.median(y.cpu().numpy())))
        return Revolution(form, pose, radii, half, dev)
    if form in ("panel", "box", "cylinder", "profile"):
        panel = form == "panel"
        mode, a0, psi0 = _axis_for(ans, frame, vt, panel=panel)
        roll_mode = "snapped" if (ans.axis_aligned or form == "cylinder" or mode == "fixed" and panel) else "free"
        if form == "cylinder":
            roll_mode = "snapped"
        pose = Pose(frame, mode, roll_mode, dev, c, a0, psi0=psi0)
        with torch.no_grad():
            Pt = _t(P, dev)
            x, y, z = pose.local(Pt)
            xn, yn, zn = x.cpu().numpy(), y.cpu().numpy(), z.cpu().numpy()
        ex_u = float(np.percentile(xn, 99.5) - np.percentile(xn, 0.5))
        ex_v = float(np.percentile(yn, 99.5) - np.percentile(yn, 0.5))
        radial = float(np.median(np.sqrt(xn * xn + yn * yn)))
        if roll_mode == "free" and not panel:
            # start the roll on the section's own principal direction
            uv = np.column_stack([xn, yn]) - np.array([np.median(xn), np.median(yn)])
            _, _, vt2 = np.linalg.svd(uv[:: max(1, len(uv) // 20000)], full_matrices=False)
            with torch.no_grad():
                pose.roll[0] = math.atan2(float(vt2[0][1]), float(vt2[0][0]))
                x, y, z = pose.local(Pt)
                xn, yn = x.cpu().numpy(), y.cpu().numpy()
            ex_u = float(np.percentile(xn, 99.5) - np.percentile(xn, 0.5))
            ex_v = float(np.percentile(yn, 99.5) - np.percentile(yn, 0.5))
        sec = _section_for(ans, form, max(ex_u, 1e-3), max(ex_v, 1e-3), radial, dev)
        # centre the section on the points
        with torch.no_grad():
            u, v, a = pose.basis()
            pose.c += u * float(np.median(xn)) + v * float(np.median(yn))
            if not panel:
                z0, z1 = float(np.percentile(zn, 0.5)), float(np.percentile(zn, 99.5))
                pose.c += a * ((z0 + z1) / 2)
            else:
                pose.c += a * float(np.median(zn))
        if panel:
            return Extrusion(form, sec, pose, None, True, dev)
        half = (float(np.percentile(zn, 99.5)) - float(np.percentile(zn, 0.5))) / 2
        open_ = bool(ans.hollow) and form == "cylinder"
        return Extrusion(form, sec, pose, max(half, 1e-3), open_, dev)
    raise ValueError(f"no single template for form {form!r}")
