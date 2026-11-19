"""A scan's mesh as it is exported: the published binary glTF IN THE DISPLAY FRAME.

USER 2026-11-19: "cuando se aplica ajuste de piso debe alcanzar a las mallas tambien, total es una
rotacion". A mesh on disk is in the frame of the cloud on disk; a floor adjustment made after it
(the floor-level tool, the alignment gizmo: ``floor_transform.npz``) is applied by the viewer to
both — the meshes hang from the cloud's group. An exported file has no viewer to do that, so the
exported GLB is the published one with the adjustment IN ITS VERTICES: every position moved by it,
every normal turned by its rotation. Everything else — indices, texture coordinates, textures —
is copied byte for byte, and the published mesh on disk is not touched.

(Why not a transform on a root node, which would leave the vertices' bytes alone: the published
meshes are written by trimesh with a root node named "world", which trimesh takes for the base of
the scene when it reads a file — a node placed above it is ignored by that reader. Measured
2026-11-19. Vertices are right in every reader.)
"""
from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

GLB_MAGIC = 0x46546C67            # "glTF"
JSON_CHUNK = 0x4E4F534A           # "JSON"
BIN_CHUNK = 0x004E4942            # "BIN\0"
FLOAT = 5126
BLOCK = 4_000_000                 # vertices a pass (48 MB of positions)
_SLACK = 512                      # bytes the document may grow by, per position accessor (its new min / max)


class MeshExportError(ValueError):
    """The mesh cannot be exported as asked — the message says why."""


def is_identity(A: np.ndarray, t: np.ndarray) -> bool:
    return bool(np.array_equal(np.asarray(A), np.eye(3)) and not np.asarray(t).any())


def _read_head(f, src: Path) -> Tuple[dict, int, int]:
    """(the glTF document, the JSON chunk's length, the file's declared length); the file is left at
    the first byte after the JSON chunk."""
    head = f.read(12)
    if len(head) != 12:
        raise MeshExportError(f"{src}: not a binary glTF (cut short)")
    magic, version, total = struct.unpack("<III", head)
    if magic != GLB_MAGIC or version != 2:
        raise MeshExportError(f"{src}: not a binary glTF 2")
    clen, ctype = struct.unpack("<II", f.read(8))
    if ctype != JSON_CHUNK:
        raise MeshExportError(f"{src}: its first chunk is not the glTF document")
    return json.loads(f.read(clen)), clen, total


def _vector_accessors(doc: dict, src: Path) -> Dict[int, str]:
    """{accessor index: POSITION | NORMAL | TANGENT} over every primitive — the data a rigid motion
    changes. Refuses what this writer does not move correctly rather than write a wrong mesh."""
    if doc.get("extensionsRequired"):
        raise MeshExportError(f"{src}: requires {doc['extensionsRequired']} — its vertices are not plain accessors")
    for n in doc.get("nodes", []):
        if any(k in n for k in ("matrix", "translation", "rotation", "scale")):
            raise MeshExportError(f"{src}: node '{n.get('name')}' carries a transform of its own — the floor "
                                  f"adjustment is written into the vertices of a mesh whose nodes carry none")
    roles: Dict[int, str] = {}
    for mesh in doc.get("meshes", []):
        for prim in mesh.get("primitives", []):
            if prim.get("targets"):
                raise MeshExportError(f"{src}: a primitive with morph targets")
            for role in ("POSITION", "NORMAL", "TANGENT"):
                if role in prim.get("attributes", {}):
                    index = int(prim["attributes"][role])
                    if roles.setdefault(index, role) != role:
                        raise MeshExportError(f"{src}: accessor {index} is used as {roles[index]} and as {role}")
    if not any(r == "POSITION" for r in roles.values()):
        raise MeshExportError(f"{src}: no vertex positions")
    return roles


def bake_transform(src: Path | str, dst: Path | str, A: np.ndarray, t: np.ndarray) -> Dict[str, int]:
    """Write ``dst``: the binary glTF ``src`` moved by  display = A · stored + t  (A = s·R) — every
    POSITION by it, every NORMAL and TANGENT by R. Returns {"vertices": moved positions, "accessors": n}."""
    src, dst = Path(src), Path(dst)
    A = np.asarray(A, np.float64).reshape(3, 3)
    t = np.asarray(t, np.float64).reshape(3)
    scale = float(np.cbrt(np.linalg.det(A)))
    if not (scale > 0.0) or not np.allclose(A @ A.T, scale * scale * np.eye(3), atol=1e-9 * max(1.0, scale * scale)):
        raise MeshExportError(f"{src}: the floor adjustment is not a rotation with a uniform scale — it is not written into a mesh")
    R = A / scale
    with open(src, "rb") as f:
        doc, clen, total = _read_head(f, src)
        roles = _vector_accessors(doc, src)
        bin_head = f.read(8)
        if len(bin_head) != 8 or struct.unpack("<II", bin_head)[1] != BIN_CHUNK:
            raise MeshExportError(f"{src}: no binary chunk after its document")
        if src.stat().st_size != total:
            raise MeshExportError(f"{src}: is {src.stat().st_size} bytes and declares {total} — a cut file is not exported")
        accessors, views = doc["accessors"], doc["bufferViews"]
        plan: List[Tuple[int, str, int, int, int]] = []          # (accessor, role, offset in the binary chunk, count, components)
        for index, role in sorted(roles.items()):
            a = accessors[index]
            comps = 4 if role == "TANGENT" else 3
            if a.get("componentType") != FLOAT or a.get("type") != ("VEC4" if comps == 4 else "VEC3") or "sparse" in a:
                raise MeshExportError(f"{src}: its {role} accessor {index} is not plain 32-bit floats")
            v = views[a["bufferView"]]
            if v.get("buffer", 0) != 0 or v.get("byteStride") not in (None, 4 * comps):
                raise MeshExportError(f"{src}: its {role} accessor {index} is interleaved or in another buffer")
            plan.append((index, role, int(v.get("byteOffset", 0)) + int(a.get("byteOffset", 0)), int(a["count"]), comps))
        # the document may grow (the positions' new bounds): its chunk is written with room, padded with spaces
        reserved = clen + _SLACK * sum(1 for p in plan if p[1] == "POSITION")
        reserved += -reserved % 4
        base = 12 + 8 + reserved + 8                              # where the binary chunk's data begins in dst
        with open(dst, "wb") as g:
            g.write(struct.pack("<III", GLB_MAGIC, 2, total - clen + reserved))
            g.write(struct.pack("<II", reserved, JSON_CHUNK))
            g.write(b" " * reserved)
            g.write(bin_head)
            shutil.copyfileobj(f, g, 1 << 24)                     # the binary chunk, as it is
    moved = 0
    with open(dst, "r+b") as g:
        for index, role, offset, count, comps in plan:
            lo, hi = np.full(3, np.inf, np.float32), np.full(3, -np.inf, np.float32)
            for start in range(0, count, BLOCK):
                n = min(BLOCK, count - start)
                g.seek(base + offset + start * 4 * comps)
                block = np.frombuffer(g.read(n * 4 * comps), "<f4").reshape(n, comps)
                if len(block) != n:
                    raise MeshExportError(f"{src}: its {role} accessor {index} runs past the end of the file")
                out = block.astype(np.float64)
                if role == "POSITION":
                    out[:, :3] = out[:, :3] @ A.T + t
                else:
                    out[:, :3] = out[:, :3] @ R.T                 # (a tangent's fourth number, its handedness, stays)
                out32 = out.astype("<f4")
                if role == "POSITION":
                    lo, hi = np.minimum(lo, out32.min(axis=0)), np.maximum(hi, out32.max(axis=0))
                g.seek(base + offset + start * 4 * comps)
                g.write(out32.tobytes())
            if role == "POSITION":
                moved += count
                if count:
                    accessors[index]["min"] = [float(x) for x in lo]
                    accessors[index]["max"] = [float(x) for x in hi]
        body = json.dumps(doc, separators=(",", ":")).encode("utf-8")
        if len(body) > reserved:
            raise MeshExportError(f"{src}: its document grew by {len(body) - clen} bytes, more than the room left for it")
        g.seek(20)
        g.write(body + b" " * (reserved - len(body)))
    return {"vertices": moved, "accessors": len(plan)}
