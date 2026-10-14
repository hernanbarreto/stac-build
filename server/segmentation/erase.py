# STAC-BUILD: Eraser tool core (USER 2026-08-29)
"""One erase gesture, one truth.

The user brushes a sphere over the scene (cloud OR mesh — same gesture): the
instance points inside leave their segment (become unsegmented) AND their
pixels are cleared from the instance's SAM3 mask in seg_masks.npz — the mask
stays the single source of truth, so any future re-match preserves the
deletion. Meshes derive from points: the caller (main.py) debounces a
per-instance re-fit for touched instances that have a published mesh, and
OBBs are recomputed from MESH vertices once a fresh mesh exists (user rule:
"una vez que haya malla, el bbox se obtiene de la malla"), from the remaining
points meanwhile.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _load_floor_transform(output_dir: Path):
    """The DISPLAY frame the zones were drawn in — the viewer's: floor_transform.npz applied
    only under the seal of the geometry on disk (docs/plan_determinismo.md point 158,
    `segmentation.floor_frame`: the one rule for the viewer, the tools and the stages), else
    identity, declared. A zone drawn on a cloud shown in its own frame must be tested in
    that frame, or the brush lands beside the points it covered."""
    from segmentation.floor_frame import sealed_floor_transform
    srt = sealed_floor_transform(Path(output_dir), tag="Erase")
    if srt is None:
        return 1.0, np.eye(3), np.zeros(3)
    return srt


def _display_sphere_to_raw(output_dir: Path, center, radius: float):
    s, R, t = _load_floor_transform(output_dir)
    c = np.asarray(center, dtype=np.float64)
    c_raw = R.T @ ((c - t) / s)
    return c_raw, float(radius) / s


def raw_to_display(output_dir: Path, pts: np.ndarray) -> np.ndarray:
    s, R, t = _load_floor_transform(output_dir)
    return s * (np.asarray(pts, np.float64) @ R.T) + t


def _atomic_savez(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    # the tmp name MUST end in ".npz": np.savez appends ".npz" to any other
    # name, so the replace() moved the EMPTY mkstemp file over seg_masks.npz
    # and truncated it to 0 bytes (bug 2026-08-29, recovered from the stray
    # data file it left behind)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp.npz")
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
        if Path(path).name == "seg_masks.npz":
            # the frame-space cache is keyed by the store's mtime, but an
            # in-process reader that already resolved it must not keep a
            # verdict measured on the previous contents
            from segmentation import mask_space
            mask_space.invalidate(Path(path).parent)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


#: the segmentation.json key that counts the interactive edits of a sealed store, per tool
STORE_EDITS_KEY = "edits"


def reseal_store(output_dir: Path, *, by: str, log=print) -> dict:
    """Re-seal the SAM3 store after an interactive edit rewrote ``seg_masks.npz`` (the erase
    / undo of `segmentation.erase`, main.py's delete and mask paint), so the projection and
    the certification ACCEPT it again: they refuse a store whose seal names other mask bytes
    than the ones on disk (`segmentation.pipeline.check_store_seal`, point 83 — a seal that
    lies), and an edit used to leave exactly that behind.

    The seal is the stamp `finalize_sam3_store` wrote (`sam3_store_seal`): its ``inputs``
    now name the bytes on disk; its ``code`` and ``config`` digests — the SAM3 code, the
    prompts, the keyframes, the model and its thresholds that DREW the masks — are carried
    over unchanged, because an edit changes none of them and recomputing them here would
    hash the checkpoint and build the wrapper on every brush stroke in the server process;
    the digest is recomputed as `repro.stamp` computes it. The edit itself is RECORDED in
    segmentation.json (``edits``: a count per tool, no clock), so the store's provenance says
    it is the stage's masks plus human edits. A store that carries NO seal (written before
    2026-10-08, or the interactive manager's upsert, which drops it by construction) is left
    as it is — a store with no seal is declared by its readers, not refused — and nothing is
    invented for it. Returns {resealed, reason, sha256}."""
    from atomic_io import atomic_write_json
    from repro import sha256_file, sha256_json
    output_dir = Path(output_dir)
    seg_json = output_dir / "segmentation.json"
    npz = output_dir / "seg_masks.npz"
    if not seg_json.is_file() or not npz.is_file():
        why = "no segmentation.json / seg_masks.npz to seal"
        log(f"[Erase] DECLARED: store not re-sealed after {by} — {why}")
        return {"resealed": False, "reason": why, "sha256": None}
    try:
        doc = json.loads(seg_json.read_text())
    except ValueError as e:
        why = f"segmentation.json is unreadable ({e})"
        log(f"[Erase] DECLARED: store not re-sealed after {by} — {why}")
        return {"resealed": False, "reason": why, "sha256": None}
    old = doc.get("stamp") if isinstance(doc, dict) else None
    if not isinstance(old, dict) or "sha256" not in old or not isinstance(old.get("inputs"), dict):
        why = ("the store carries no seal (written before 2026-10-08, or edited by the "
               "interactive manager) — left unsealed, as its readers declare it")
        log(f"[Erase] DECLARED: store not re-sealed after {by} — {why}")
        return {"resealed": False, "reason": why, "sha256": None}
    have = sha256_file(npz)
    body = {"stamp_version": old.get("stamp_version"),
            "inputs": {**old["inputs"], "seg_masks.npz": have},
            "code": dict(old.get("code") or {}),
            "config": dict(old.get("config") or {})}
    body["sha256"] = sha256_json(body)
    edits = dict(doc.get(STORE_EDITS_KEY) or {})
    edits[str(by)] = int(edits.get(str(by), 0)) + 1
    doc["stamp"] = body
    doc[STORE_EDITS_KEY] = edits
    atomic_write_json(seg_json, doc, indent=1, sort_keys=True)
    log(f"[Erase] store re-sealed after {by}: seg_masks.npz {have[:12]} (edits so far: "
        + ", ".join(f"{k} x{v}" for k, v in sorted(edits.items())) + ")")
    return {"resealed": True, "reason": "sealed to the mask bytes on disk", "sha256": have}


def reseal_floor(output_dir: Path, *, by: str, held_before: Tuple[bool, str], log=print) -> dict:
    """Carry the floor seal to the cloud an interactive edit just REWROTE — the delete mode of
    the brush (``conf_below`` over the unsegmented points: they leave ``cleaned_cloud.ply``
    physically) — so the stages and the viewer keep applying ``floor_transform.npz``. The seal
    in floor_level.json names the cloud's sha256 (docs/plan_determinismo.md point 158,
    `segmentation.pipeline._floor_seal_status`); the edit changed the bytes — points left, none
    moved — not the frame the transform levels, exactly as `reseal_store` carries the SAM3
    store's seal over an edit. Without this the next projection (the Segmentation Manager's
    close, the certification) FAILS naming the old cloud, and the viewer, the ``potree_ready``
    of the erase and the erase tools all drop the transform: the cloud shown unlevelled, the
    next zone tested in identity while the previous one was tested in the levelled frame.

    ``held_before`` is the verdict `segmentation.floor_frame.floor_seal_verdict` gave BEFORE
    the rewrite (measured by the caller, memoised: the zones were tested under it). The seal is
    carried over ONLY when it held for the bytes that were edited. One that did not hold —
    another cloud's, another reconstruction's, nobody's, levelled before the seal existed — is
    left exactly as it is, declared: sealing it to the new bytes would turn a transform the
    viewer was NOT applying into one it applies, and the zones were drawn in the frame the
    viewer showed. ``reconstruction_id`` and ``floor_transform_sha256`` are unchanged (neither
    the reconstruction nor the npz moved); the edit is recorded under ``edits`` (a count per
    tool, no clock) like the store's. With no npz beside the cloud nothing applies a transform
    to it and there is nothing to carry (silent). Returns {resealed, reason, cloud_sha256}."""
    from atomic_io import atomic_write_json
    from repro import sha256_file
    output_dir = Path(output_dir)
    fl = output_dir / "floor_level.json"
    npz = output_dir / "floor_transform.npz"
    cloud = output_dir / "cleaned_cloud.ply"
    if not npz.is_file() or not fl.is_file():
        return {"resealed": False, "cloud_sha256": None,
                "reason": "no floor_transform.npz / floor_level.json beside the cloud — nothing "
                          "applies a transform to it, nothing to carry"}
    ok_before, why_before = held_before
    if not ok_before:
        why = (f"the seal did not hold before the edit ({why_before}) — left as it is: the viewer "
               f"showed the cloud in its own frame and the zones were tested there")
        log(f"[Erase] DECLARED: floor seal not carried after {by} — {why}")
        return {"resealed": False, "reason": why, "cloud_sha256": None}
    try:
        doc = json.loads(fl.read_text())
    except ValueError as e:
        why = f"floor_level.json is unreadable ({e})"
        log(f"[Erase] DECLARED: floor seal not carried after {by} — {why}")
        return {"resealed": False, "reason": why, "cloud_sha256": None}
    st = doc.get("stamp") if isinstance(doc, dict) else None
    if not isinstance(st, dict) or "reconstruction_id" not in st or "cloud_sha256" not in st:
        why = "floor_level.json carries no seal — left as it is, as its readers declare it"
        log(f"[Erase] DECLARED: floor seal not carried after {by} — {why}")
        return {"resealed": False, "reason": why, "cloud_sha256": None}
    if not cloud.is_file():
        why = "no cleaned_cloud.ply to seal to"
        log(f"[Erase] DECLARED: floor seal not carried after {by} — {why}")
        return {"resealed": False, "reason": why, "cloud_sha256": None}
    have = sha256_file(cloud)
    doc["stamp"] = {**st, "cloud_sha256": have}
    edits = dict(doc.get(STORE_EDITS_KEY) or {})
    edits[str(by)] = int(edits.get(str(by), 0)) + 1
    doc[STORE_EDITS_KEY] = edits
    atomic_write_json(fl, doc, indent=2, sort_keys=True)
    log(f"[Erase] floor seal carried to the edited cloud after {by}: cleaned_cloud.ply {have[:12]} "
        f"(reconstruction and floor_transform.npz unchanged; edits so far: "
        + ", ".join(f"{k} x{v}" for k, v in sorted(edits.items())) + ")")
    return {"resealed": True, "reason": "sealed to the cloud bytes on disk", "cloud_sha256": have}


def _atomic_append_npz(path: Path, arrays: Dict[str, np.ndarray], *, keep) -> None:
    """Write ``arrays`` into the npz at ``path`` WITHOUT touching the entries it
    keeps: the entries named in ``keep`` (and not in ``arrays``) are carried
    over AS STORED — never decompressed, never recompressed —, ``arrays``
    replace or add theirs, every other entry is dropped. The same logical
    content ``_atomic_savez({**kept, **arrays})`` would write, at the cost of the
    new entries only. Atomic like ``_atomic_savez`` (a copy is edited, then
    replaced).

    A dropped / replaced entry leaves the central directory, which is what a
    zip reader goes by; its old bytes stay in the file as unreferenced space
    until the next full rewrite (``_atomic_savez``) compacts it."""
    import shutil
    import zipfile
    fd, tmp = tempfile.mkstemp(dir=str(Path(path).parent), suffix=".tmp.npz")
    os.close(fd)
    try:
        shutil.copyfile(path, tmp)
        carry = {f"{k}.npy" for k in keep} - {f"{k}.npy" for k in arrays}
        with zipfile.ZipFile(tmp, mode="a", compression=zipfile.ZIP_DEFLATED,
                             allowZip64=True) as zf:
            zf.filelist = [i for i in zf.filelist if i.filename in carry]
            zf.NameToInfo = {i.filename: i for i in zf.filelist}
            for key, val in arrays.items():
                # how np.savez_compressed writes an entry (no pickled arrays)
                with zf.open(f"{key}.npy", "w", force_zip64=True) as f:
                    np.lib.format.write_array(f, np.asanyarray(val), allow_pickle=False)
        os.replace(tmp, path)
        if Path(path).name == "seg_masks.npz":
            from segmentation import mask_space
            mask_space.invalidate(Path(path).parent)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _cube_hit(disp_pts: np.ndarray, c: np.ndarray, r: float,
              yaw: float = 0.0) -> np.ndarray:
    """Points inside a cube of half-side r centred at c, rotated ``yaw``
    radians about the vertical (Y) axis of the display frame — same
    convention as three.js ``rotation.y`` (user 2026-08-30: the cube must
    rotate to adapt to diagonal walls)."""
    d = np.asarray(disp_pts, np.float64) - c
    if abs(yaw) > 1e-9:
        ca, sa = np.cos(yaw), np.sin(yaw)
        x = ca * d[:, 0] - sa * d[:, 2]
        z = sa * d[:, 0] + ca * d[:, 2]
        d = np.column_stack([x, d[:, 1], z])
    return (np.abs(d) <= r).all(axis=1)


def _rewrite_ply_keep(ply_path: Path, keep: np.ndarray) -> bool:
    """Rewrite a binary-little-endian PLY keeping only ``keep`` rows —
    physical point deletion (user 2026-08-31: low-confidence unsegmented
    points must actually LEAVE the cloud). Header preserved verbatim except
    the vertex count. Atomic (tmp + replace). Returns False (untouched file)
    when any property type is unknown — never corrupt the cloud."""
    _ply_type = {
        'float': '<f4', 'float32': '<f4', 'double': '<f8', 'float64': '<f8',
        'uchar': 'u1', 'uint8': 'u1', 'char': 'i1', 'int8': 'i1',
        'ushort': '<u2', 'uint16': '<u2', 'short': '<i2', 'int16': '<i2',
        'uint': '<u4', 'uint32': '<u4', 'int': '<i4', 'int32': '<i4',
    }
    try:
        header: List[bytes] = []
        props = []
        n_pts = 0
        with open(ply_path, 'rb') as f:
            while True:
                line = f.readline()
                header.append(line)
                s = line.decode('ascii', 'ignore').strip()
                if s.startswith('element vertex'):
                    n_pts = int(s.split()[-1])
                elif s.startswith('property'):
                    parts = s.split()
                    if len(parts) < 3 or parts[1] not in _ply_type:
                        print(f"[Erase] delete aborted: unsupported PLY "
                              f"property '{s}'")
                        return False
                    props.append((parts[2], _ply_type[parts[1]]))
                elif s == 'end_header':
                    break
            if n_pts != len(keep):
                print(f"[Erase] delete aborted: keep mask {len(keep)} vs "
                      f"{n_pts} pts")
                return False
            data = np.frombuffer(f.read(), dtype=np.dtype(props), count=n_pts)
        kept = data[keep]
        fd, tmp = tempfile.mkstemp(dir=str(ply_path.parent), suffix=".ply")
        os.close(fd)
        with open(tmp, 'wb') as f:
            for line in header:
                s = line.decode('ascii', 'ignore').strip()
                if s.startswith('element vertex'):
                    f.write(f"element vertex {len(kept)}\n".encode('ascii'))
                else:
                    f.write(line)
            f.write(kept.tobytes())
        os.replace(tmp, ply_path)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[Erase] PLY rewrite failed ({e}) — cloud untouched")
        return False


def _load_confidence_norm(ply_path: Path) -> Optional[np.ndarray]:
    """Per-point confidence from the binary PLY, normalized to [0,1] with the
    array's own min/max — the SAME normalization the viewer applies (Potree
    metadata min/max come from these values), so a UI threshold means the
    same thing here. None when the cloud has no confidence field."""
    _ply_type = {
        'float': '<f4', 'float32': '<f4', 'double': '<f8', 'float64': '<f8',
        'uchar': 'u1', 'uint8': 'u1', 'char': 'i1', 'int8': 'i1',
        'ushort': '<u2', 'uint16': '<u2', 'short': '<i2', 'int16': '<i2',
        'uint': '<u4', 'uint32': '<u4', 'int': '<i4', 'int32': '<i4',
    }
    try:
        with open(ply_path, 'rb') as f:
            n_pts = 0
            props = []
            while True:
                line = f.readline().decode('ascii').strip()
                if line.startswith('element vertex'):
                    n_pts = int(line.split()[-1])
                elif line.startswith('property') and n_pts > 0:
                    parts = line.split()
                    if len(parts) >= 3 and parts[1] in _ply_type:
                        props.append((parts[2], _ply_type[parts[1]]))
                elif line == 'end_header':
                    break
            if n_pts == 0 or 'confidence' not in {p[0] for p in props}:
                return None
            data = np.frombuffer(f.read(), dtype=np.dtype(props), count=n_pts)
            c = np.asarray(data['confidence'], dtype=np.float64)
            lo, hi = float(c.min()), float(c.max())
            return ((c - lo) / max(hi - lo, 1e-9)).astype(np.float32)
    except Exception as e:  # noqa: BLE001
        print(f"[Erase] confidence read failed ({e})")
        return None


def _mask_obj_by_iid(output_dir: Path) -> Dict[int, int]:
    """instance_id → seg_masks obj id (mesh_export convention: via
    segmentation.json, label fallback)."""
    out: Dict[int, int] = {}
    seg_json = output_dir / "segmentation.json"
    if not seg_json.exists():
        return out
    try:
        # the FUSED view (plan point 100): the raw store is immutable, the fusion is applied by
        # every reader from fusion_map.json
        from segmentation.fuse_parent import fused_parent
        for e in (fused_parent(output_dir).get("instances") or []):
            if e.get("id") is not None and e.get("instance_id") is not None:
                out[int(e["instance_id"])] = int(e["id"])
    except Exception:  # noqa: BLE001
        pass
    return out


def _mask_oids_by_iid(output_dir: Path) -> Dict[int, List[int]]:
    """instance_id → EVERY seg_masks obj id filed under it.

    More than one means SAM3 segmented several objects and they were fused into
    one instance downstream. That is SAM3's own evidence that they are
    different things, and the only ground on which an instance may be separated
    (USER 2026-09-15: "no hay que desconfiar tanto de SAM3").
    """
    out: Dict[int, List[int]] = {}
    seg_json = output_dir / "segmentation.json"
    if not seg_json.exists():
        return out
    try:
        from segmentation.fuse_parent import fused_parent, oids_of
        for e in (fused_parent(output_dir).get("instances") or []):
            if e.get("id") is not None and e.get("instance_id") is not None:
                # after the fusion the survivor's PARTS are those other mask
                # ids: reading only `id` would make every instance single-mask
                # and the split path — the only evidence the user accepts for
                # separating an instance — would go dead
                out.setdefault(int(e["instance_id"]), []).extend(oids_of(e))
    except Exception:  # noqa: BLE001
        pass
    return {k: sorted(set(v)) for k, v in out.items()}


def published_mesh_path(output_dir: Path, label: str, iid: int) -> Optional[Path]:
    from segmentation.tsdf_export import _safe_label
    safe = _safe_label(label or "segment", int(iid))
    p = Path(output_dir) / "tsdf" / safe / f"{safe}.glb"
    return p if p.exists() else None


def _write_classification(output_dir: Path, instances: List[dict],
                          n_points: int) -> np.ndarray:
    """Rebuild classification.npy — kept here as the name five call sites use.

    The body moved to `segmentation.republish.write_classification` so the
    per-point class byte has exactly ONE writer: a correction epoch has to
    write it too, from a staged directory, and two implementations of the same
    array drift apart (they already disagreed on the key — `id` here,
    `instance_id` there — equal only by the result file's convention).
    """
    from segmentation.republish import write_classification
    return write_classification(output_dir, instances, n_points)


def verify_octree_classification(output_dir: Path,
                                 sample: int = 5000) -> Optional[dict]:
    """Sample the ROOT node of the freshly-built Potree octree and check its
    per-point classification byte against classification.npy (matched by
    position on the conversion-source cloud). Closes the trace loop of a
    brush transaction: files → octree → what the viewer culls by
    (user 2026-08-31: 'cada transacción trazada y controlada').
    Returns {checked, matched, agreement} or None when unverifiable."""
    import struct
    output_dir = Path(output_dir)
    potree = output_dir / "potree" \
        if (output_dir / "potree").exists() else output_dir.parent / "potree"
    meta_p = potree / "metadata.json"
    cls_p = output_dir / "classification.npy"
    src = output_dir / "cleaned_cloud.ply"          # the ONE cloud of the live epoch (plan point 120)
    if not (meta_p.exists() and cls_p.exists() and src.exists()):
        return None
    try:
        meta = json.loads(meta_p.read_text())
        scale, offs = meta["scale"], meta["offset"]
        bpp = cls_off = 0
        cls_found = False
        for a in meta["attributes"]:
            if a["name"] == "classification":
                cls_off = bpp
                cls_found = True
            bpp += int(a["size"])
        if not cls_found or bpp <= 0:
            return None
        t, cm, npts, boff, bsz = struct.unpack(
            "<BBIQQ", (potree / "hierarchy.bin").read_bytes()[:22])
        with open(potree / "octree.bin", "rb") as f:
            f.seek(boff)
            buf = f.read(min(bsz, npts * bpp))
        n = min(int(npts), int(sample), len(buf) // bpp)
        if n < 100:
            return None
        rec = np.frombuffer(buf[:n * bpp], dtype=np.uint8).reshape(n, bpp)
        ixyz = rec[:, :12].copy().view("<i4").reshape(n, 3).astype(np.float64)
        pts = ixyz * np.asarray(scale) + np.asarray(offs)
        cls_oct = rec[:, cls_off].astype(np.int64)

        from segmentation.pipeline import _load_ply_origins
        origins = _load_ply_origins(src)
        if not origins:
            return None
        cloud = origins[0]
        gt = np.load(cls_p)
        if len(gt) != len(cloud):
            return {"checked": 0, "matched": 0, "agreement": 0.0,
                    "error": "classification/cloud size mismatch"}
        from scipy.spatial import cKDTree
        d, idx = cKDTree(cloud).query(pts, k=1)
        ok = d < 0.002    # 1 mm quantization → 2 mm tolerance
        if not ok.any():
            return {"checked": int(n), "matched": 0, "agreement": 0.0,
                    "error": "no positional matches — octree/cloud frames differ"}
        agree = float((cls_oct[ok] == gt[idx[ok]]).mean())
        return {"checked": int(ok.sum()),
                "matched": int((cls_oct[ok] == gt[idx[ok]]).sum()),
                "agreement": round(agree, 4)}
    except Exception as e:  # noqa: BLE001
        print(f"[Erase] octree verification failed (non-fatal): {e}")
        return None


def erase_sphere(output_dir: Path, center_display, radius: float) -> dict:
    """Single-sphere compatibility wrapper over erase_spheres()."""
    return erase_spheres(output_dir,
                         [{"center": center_display, "radius": radius}])


_NEW_SEGMENT_COLORS = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
                       "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe"]


def erase_spheres(output_dir: Path, spheres: List[dict],
                  target_iid: Optional[int] = None,
                  new_label: Optional[str] = None,
                  only_iids: Optional[List[int]] = None,
                  include_unsegmented: bool = True,
                  conf_below: Optional[float] = None) -> dict:
    """Apply ONE commit over the marked spheres (user 2026-08-29: mark zones
    first, then a single button applies everything — one mask edit, one OBB
    recompute, one octree rebuild).

    ``target_iid`` None → DELETE: points leave their segment (unsegmented).
    ``target_iid`` set → REASSIGN: every point inside the zones — owned by
    any other instance OR unsegmented — moves INTO the target segment (user:
    "no hay que borrarlos, son de otro segmento"). Mask pixels move with
    them (cleared at the source, set on the target) so the decision survives
    any re-match.
    ``new_label`` set → CREATE: a brand-new segment with that label is created
    from the zones and becomes the reassign target (user 2026-08-29: "se
    pueden asignar a un segmento existente o crear uno nuevo").
    ``only_iids`` (SAFETY, user 2026-08-30: "el selector está pasando por
    todos aunque no se vean — eso es peligroso"): when given, only these
    instance_ids can LOSE points — the UI sends the currently VISIBLE
    segments, so hidden ones are untouchable. The reassign target is exempt
    (chosen explicitly).
    ``conf_below`` (user 2026-08-31: brush confidence filter): points with
    normalized confidence BELOW this value are selected too — with no zones
    it selects globally. Delete-mode use: low-confidence points of the
    visible segments move to unsegmented. Same safety, ledger and undo.

    Returns {"touched": {iid: n_removed}, "total_removed": n, "reassigned": n,
    "mesh_instances": [...], "undo": {...}}."""
    from segmentation.pipeline import _load_ply_origins, _compute_obb

    output_dir = Path(output_dir)
    ply = output_dir / "cleaned_cloud.ply"
    res_path = output_dir / "segmentation_result.json"
    if not ply.exists() or not res_path.exists():
        raise FileNotFoundError("session has no cleaned cloud / segmentation")

    origins = _load_ply_origins(ply)
    if origins is None:
        raise RuntimeError("cloud has no origin fields (cannot edit masks)")
    xyz, fg, pr, pc = origins
    N = len(xyz)
    conf_norm = None
    if conf_below is not None:
        conf_norm = _load_confidence_norm(ply)
        if conf_norm is None or len(conf_norm) != N:
            raise ValueError("cloud has no usable confidence field")
        conf_below = float(conf_below)
    # zones live in the DISPLAY frame: a cube brush is axis-aligned in the
    # LEVELED frame the user sees (user 2026-08-30: sphere OR cube per zone),
    # which is NOT axis-aligned in raw — so all hit tests run on display pts
    s_ft, R_ft, t_ft = _load_floor_transform(output_dir)

    def _to_disp(p):
        return s_ft * (np.asarray(p, np.float64) @ R_ft.T) + t_ft

    zones = []
    for sp in spheres:
        kind = str(sp.get("shape") or "sphere").lower()
        if kind == "box":
            # GIZMO BOX (user 2026-08-30: "debe ser una caja, punto"): the
            # mesh is a 2x2x2 cube, so scale/rotation/position all live in the
            # matrix and the local test is simply |xyz| <= 1
            M = np.asarray(sp["matrix"], np.float64).reshape(4, 4, order="F")
            zones.append({"kind": "box", "inv": np.linalg.inv(M)})
        elif kind == "prism":
            # LASSO PRISM (user 2026-08-30): polygon (local XY) extruded along
            # local +Z by ``depth``; ``matrix`` maps local→display (three.js
            # toArray = column-major), gizmo edits (move/rotate/stretch) baked in
            M = np.asarray(sp["matrix"], np.float64).reshape(4, 4, order="F")
            zones.append({"kind": "prism", "inv": np.linalg.inv(M),
                          "poly": np.asarray(sp["polygon"], np.float64),
                          "depth": float(sp.get("depth") or 0.0)})
        else:
            zones.append({"kind": kind,
                          "c": np.asarray(sp["center"], np.float64),
                          "r": float(sp["radius"]),
                          "yaw": float(np.radians(float(sp.get("yaw_deg") or 0.0)))})
    if not zones and conf_below is None:
        return {"touched": {}, "total_removed": 0, "mesh_instances": [],
                "undo": None}

    def _zone_hit(disp_pts: np.ndarray) -> np.ndarray:
        h = np.zeros(len(disp_pts), dtype=bool)
        for z in zones:
            if z["kind"] == "box":
                ph = np.hstack([disp_pts, np.ones((len(disp_pts), 1))])
                local = (z["inv"] @ ph.T).T
                h |= (np.abs(local[:, :3]) <= 1.0).all(axis=1)
            elif z["kind"] == "prism":
                ph = np.hstack([disp_pts, np.ones((len(disp_pts), 1))])
                local = (z["inv"] @ ph.T).T
                zin = (local[:, 2] >= 0.0) & (local[:, 2] <= z["depth"])
                if zin.any():
                    from matplotlib.path import Path as _MplPath
                    inpoly = np.zeros(len(disp_pts), dtype=bool)
                    inpoly[zin] = _MplPath(z["poly"]).contains_points(
                        local[zin, :2])
                    h |= inpoly
            elif z["kind"] == "cube":
                h |= _cube_hit(disp_pts, z["c"], z["r"], z["yaw"])
            else:
                h |= ((disp_pts - z["c"]) ** 2).sum(axis=1) <= z["r"] * z["r"]
        return h

    result = json.loads(res_path.read_text())
    instances = result.get("instances") or []
    oid_map = _mask_obj_by_iid(output_dir)

    masks_path = output_dir / "seg_masks.npz"
    masks: Dict[str, np.ndarray] = {}
    if masks_path.exists():
        try:
            z = np.load(masks_path, allow_pickle=True)
            masks = {k: z[k] for k in z.files}
        except Exception as e:  # noqa: BLE001 — corrupt/empty npz: erase still
            # works on the point assignment; only the mask edit is skipped
            print(f"[Erase] seg_masks.npz unreadable ({e}) — erasing points "
                  "only (masks not edited)")
            masks = {}
    # The cloud stamps every point with the REAL video frame number while the
    # masks are keyed by KEYFRAME POSITION — the two spaces only ever coincided
    # by accident. Same translation the matcher uses, or the pixels of an erased
    # point are cleared in the wrong frame's mask (or in none at all).
    cloud_to_mask: Dict[int, int] = {}
    if masks:
        try:
            from segmentation.pipeline import _mask_frame_lookup
            cloud_to_mask = _mask_frame_lookup(
                output_dir, masks.get("frames", []), np.unique(fg).tolist())
        except Exception as e:  # noqa: BLE001 — mask edit is best-effort
            print(f"[Erase] mask frame lookup unavailable ({e}) — masks keyed as-is")
    orig_h = float(pr.max() + 1)
    orig_w = float(pc.max() + 1)

    # user 2026-08-31: when the Unsegmented toggle is ON, its low-confidence
    # points have no "more unsegmented" to fall to — they are DELETED from
    # the cloud itself (irreversible; the ledger says so). Computed on the
    # PRE-commit ownership so points moved to unsegmented by THIS commit are
    # not swept along.
    pending_delete = None
    if conf_norm is not None and target_iid is None and include_unsegmented:
        _assigned_any = np.zeros(N, dtype=bool)
        for _inst in instances:
            _g = np.asarray(_inst.get("globalIndices") or [], dtype=np.int64)
            _assigned_any[_g[(_g >= 0) & (_g < N)]] = True
        pending_delete = np.nonzero((~_assigned_any)
                                    & (conf_norm < conf_below))[0]

    touched: Dict[int, int] = {}
    mesh_instances: List[int] = []
    undo_indices: Dict[str, List[int]] = {}
    undo_pixels: List[Tuple[str, List[int], List[int]]] = []

    created_iid = None
    if new_label:
        # CREATE a new segment and use it as the reassign target. It needs an
        # identity in BOTH spaces: instance_id (result/viewer) and mask obj id
        # (seg_masks/segmentation.json) so re-matching keeps it alive.
        new_iid = 1 + max(
            [int(i.get("instance_id", i.get("id", 0))) for i in instances],
            default=0)
        seg_json = output_dir / "segmentation.json"
        meta = {}
        try:
            meta = json.loads(seg_json.read_text()) if seg_json.exists() else {}
        except Exception:  # noqa: BLE001
            meta = {}
        meta_insts = meta.get("instances") or []
        from segmentation.fuse_parent import high_water
        _hi_id, _hi_iid = high_water(output_dir)
        new_oid = 1 + max([int(e.get("id", 0)) for e in meta_insts] + [_hi_id],
                          default=0)
        # both ids come from the SAME high-water mark now: they used to be
        # drawn from two different files holding two different id sets
        new_iid = max(new_iid, _hi_iid + 1)
        color = _NEW_SEGMENT_COLORS[(new_iid - 1) % len(_NEW_SEGMENT_COLORS)]
        # result-file convention is id == instance_id (1-based); the mask obj
        # id (new_oid) lives ONLY in segmentation.json. Writing new_oid here
        # made every downstream .get("id")-first consumer (classification,
        # exports) treat the new segment as the PREVIOUS one.
        instances.append({
            "id": int(new_iid), "label": str(new_label),
            "instance_id": int(new_iid), "color": color,
            "total_points": 0, "globalIndices": [],
        })
        meta_insts.append({"id": int(new_oid), "instance_id": int(new_iid),
                           "label": str(new_label), "color": color})
        meta["instances"] = meta_insts
        try:
            seg_json.write_text(json.dumps(meta, indent=2))
        except Exception as e:  # noqa: BLE001
            print(f"[Erase] segmentation.json update failed: {e}")
        if masks and "obj_ids" in masks:
            masks["obj_ids"] = np.append(
                np.asarray(masks["obj_ids"], dtype=np.int32),
                np.int32(new_oid))
        oid_map[int(new_iid)] = int(new_oid)
        target_iid = int(new_iid)
        created_iid = int(new_iid)
        print(f"[Erase] created new segment '{new_label}' "
              f"(instance {new_iid}, mask obj {new_oid})")

    target_inst = None
    assigned_before = None
    moved_parts: List[np.ndarray] = []
    if target_iid is not None:
        for inst in instances:
            if int(inst.get("instance_id", inst.get("id"))) == int(target_iid):
                target_inst = inst
                break
        if target_inst is None:
            raise ValueError(f"reassign target instance {target_iid} not found")
        assigned_before = np.zeros(N, dtype=bool)
        for inst in instances:
            gi0 = np.asarray(inst.get("globalIndices") or [], dtype=np.int64)
            assigned_before[gi0[(gi0 >= 0) & (gi0 < N)]] = True

    only_set = set(int(i) for i in only_iids) if only_iids is not None else None
    # Transaction ledger (user 2026-08-31: every brush transaction traced and
    # controlled): per-source breakdown, silently-protected hidden points,
    # conservation balance, post-write file verification.
    labels_by_iid = {
        int(i.get("instance_id", i.get("id"))): str(i.get("label", "segment"))
        for i in instances}
    assigned_points_before = sum(
        len(i.get("globalIndices") or []) for i in instances)
    protected_hidden: Dict[int, int] = {}
    for inst in instances:
        if target_inst is not None and inst is target_inst:
            continue   # the target only GAINS points in a reassign
        iid = int(inst.get("instance_id", inst.get("id")))
        gi = np.asarray(inst.get("globalIndices") or [], dtype=np.int64)
        gi = gi[(gi >= 0) & (gi < N)]
        if not len(gi):
            continue
        hit = _zone_hit(_to_disp(xyz[gi]))
        if conf_norm is not None:
            hit |= conf_norm[gi] < conf_below
        n_hit = int(hit.sum())
        if not n_hit:
            continue
        if only_set is not None and iid not in only_set:
            # hidden in the viewer → untouchable (safety), but never silent:
            # the ledger reports what stayed put inside the zones
            protected_hidden[iid] = n_hit
            continue
        removed = gi[hit]
        keep = gi[~hit]
        if target_inst is not None:
            moved_parts.append(removed)
        inst["globalIndices"] = keep.tolist()
        inst["total_points"] = int(len(keep))
        touched[iid] = n_hit
        undo_indices[str(iid)] = removed.tolist()
        # immediate OBB from the remaining points (display frame); once the
        # mesh regenerates, the caller replaces it with the mesh-derived OBB
        if len(keep) >= 4:
            inst["obb"] = _compute_obb(raw_to_display(output_dir, xyz[keep]))
        if published_mesh_path(output_dir, inst.get("label", ""), iid):
            mesh_instances.append(iid)
        # clear the removed points' pixels in this instance's masks — the
        # mask is the truth; re-matching must not resurrect the deletion
        oid = oid_map.get(iid)
        if oid is None or not masks:
            continue
        for f in np.unique(fg[removed]):
            mf = cloud_to_mask.get(int(f))           # exact (plan point 121): no mask position for this frame → nothing to clear
            if mf is None:
                continue
            key = f"f{mf}_o{int(oid)}"
            m = masks.get(key)
            if m is None:
                continue
            sel = removed[fg[removed] == f]
            mh, mw = m.shape[:2]
            rr = np.clip((pr[sel] * (mh / orig_h)).astype(np.int64), 0, mh - 1)
            cc = np.clip((pc[sel] * (mw / orig_w)).astype(np.int64), 0, mw - 1)
            on = m[rr, cc] > 0
            if not on.any():
                continue
            undo_pixels.append((key, rr[on].tolist(), cc[on].tolist()))
            m = m.copy()
            m[rr[on], cc[on]] = 0
            masks[key] = m

    # ── REASSIGN: the target gains every point in the zones — the ones taken
    # from other instances plus the previously-unsegmented ones
    n_reassigned = 0
    n_unseg_taken = 0
    target_pixels: List[Tuple[str, List[int], List[int]]] = []
    if target_inst is not None:
        # unsegmented capture honors the panel toggle (user 2026-08-30: los
        # unsegmented solo son seleccionables cuando Unsegmented está visible)
        un_idx = np.nonzero(~assigned_before)[0] if include_unsegmented \
            else np.empty(0, dtype=np.int64)
        if len(un_idx):
            hit_u = _zone_hit(_to_disp(xyz[un_idx]))
            if hit_u.any():
                moved_parts.append(un_idx[hit_u])
                n_unseg_taken = int(hit_u.sum())
        if moved_parts:
            added = np.unique(np.concatenate(moved_parts))
            tgi = np.asarray(target_inst.get("globalIndices") or [],
                             dtype=np.int64)
            merged = np.union1d(tgi, added)
            target_inst["globalIndices"] = merged.tolist()
            target_inst["total_points"] = int(len(merged))
            n_reassigned = int(len(added))
            if len(merged) >= 4:
                target_inst["obb"] = _compute_obb(
                    raw_to_display(output_dir, xyz[merged]))
            t_id = int(target_inst.get("instance_id", target_inst.get("id")))
            if published_mesh_path(output_dir,
                                   target_inst.get("label", ""), t_id):
                mesh_instances.append(t_id)
            # paint the moved points ONTO the target's masks so a re-match
            # keeps the reassignment (create the frame's mask if absent)
            toid = oid_map.get(t_id)
            if toid is not None and masks:
                try:
                    sr = masks.get("scaled_res")
                    mh0, mw0 = (int(sr[0]), int(sr[1])) if sr is not None \
                        else next(m.shape[:2] for k, m in masks.items()
                                  if k.startswith("f"))
                    for f in np.unique(fg[added]):
                        mf = cloud_to_mask.get(int(f))   # exact (plan point 121)
                        if mf is None:
                            continue
                        key = f"f{mf}_o{int(toid)}"
                        m = masks.get(key)
                        if m is None:
                            m = np.zeros((mh0, mw0), dtype=np.uint8)
                        sel = added[fg[added] == f]
                        mh, mw = m.shape[:2]
                        rr = np.clip((pr[sel] * (mh / orig_h)).astype(np.int64),
                                     0, mh - 1)
                        cc = np.clip((pc[sel] * (mw / orig_w)).astype(np.int64),
                                     0, mw - 1)
                        off = m[rr, cc] == 0
                        if not off.any():
                            continue
                        target_pixels.append((key, rr[off].tolist(),
                                              cc[off].tolist()))
                        m = m.copy()
                        m[rr[off], cc[off]] = 1
                        masks[key] = m
                except Exception as e:  # noqa: BLE001
                    print(f"[Erase] target mask paint failed (non-fatal): {e}")

    if not touched and not n_reassigned \
            and not (pending_delete is not None and len(pending_delete)):
        if created_iid is not None:
            # nothing landed in the new segment — roll its registration back
            try:
                seg_json = output_dir / "segmentation.json"
                meta = json.loads(seg_json.read_text())
                meta["instances"] = [e for e in (meta.get("instances") or [])
                                     if int(e.get("instance_id", -1)) != created_iid]
                seg_json.write_text(json.dumps(meta, indent=2))
            except Exception:  # noqa: BLE001
                pass
        return {"touched": {}, "total_removed": 0, "reassigned": 0,
                "mesh_instances": [], "undo": None,
                "ledger": {
                    "mode": "reassign" if target_iid is not None else "delete",
                    "target": ({"instance_id": int(target_iid),
                                "label": labels_by_iid.get(int(target_iid), "?")}
                               if target_iid is not None else None),
                    "moved_from": {}, "unsegmented_taken": 0,
                    "total_moved": 0,
                    "protected_hidden": {
                        f"{labels_by_iid.get(i, 'segment')}_{i}": n
                        for i, n in protected_hidden.items()},
                    "balance": None, "files_verified": None,
                }}

    if (undo_pixels or target_pixels) and masks_path.exists():
        _atomic_savez(masks_path, masks)
        # the seal names the bytes just written, or the projection refuses the store (point 83)
        reseal_store(output_dir, by="erase")

    # ── PHYSICAL DELETION of low-confidence unsegmented points ──
    n_deleted = 0
    if pending_delete is not None and len(pending_delete):
        # the floor seal's verdict on the cloud ABOUT TO BE rewritten (memoised — the zones
        # were tested under it): carried to the new bytes by `reseal_floor` only when it
        # held, or the next projection refuses the npz and the viewer drops the frame
        from segmentation.floor_frame import floor_seal_verdict
        floor_held = (floor_seal_verdict(output_dir)
                      if (output_dir / "floor_transform.npz").is_file()
                      else (False, "no floor_transform.npz"))
        keep = np.ones(N, dtype=bool)
        keep[pending_delete] = False
        ok_del = _rewrite_ply_keep(output_dir / "cleaned_cloud.ply", keep)
        if ok_del:
            # the seal names the cloud's bytes — the edit changed them, not the frame
            reseal_floor(output_dir, by="erase_delete", held_before=floor_held)
            # every stored index shifts — remap all instances onto the new
            # cloud (deleted points were unsegmented, so instances only shift)
            new_idx = np.full(N, -1, dtype=np.int64)
            new_idx[keep] = np.arange(int(keep.sum()), dtype=np.int64)
            for inst in instances:
                _g = np.asarray(inst.get("globalIndices") or [],
                                dtype=np.int64)
                _g = new_idx[_g[(_g >= 0) & (_g < N)]]
                _g = _g[_g >= 0]
                inst["globalIndices"] = _g.tolist()
                inst["total_points"] = int(len(_g))
            n_deleted = int(len(pending_delete))
            N = int(keep.sum())
            result["total_points"] = N
            xyz = xyz[keep]
            # index remap invalidates this commit's undo — it is irreversible
            undo_indices.clear()
            undo_pixels.clear()
            print(f"[Erase] 🗑 {n_deleted:,} low-confidence unsegmented "
                  f"point(s) permanently removed from the cloud "
                  f"({N:,} remain)")

    result["segmented_points"] = sum(
        int(i.get("total_points") or 0) for i in instances)
    result["coverage"] = round(
        result["segmented_points"] / max(1, int(result.get("total_points") or N)), 4)
    res_path.write_text(json.dumps(result))
    classification = _write_classification(output_dir, instances, N)

    # ── FILE VERIFICATION: what we just wrote must balance exactly —
    # classification.npy per-class counts == each instance's index count, and
    # points are conserved (assigned_after − assigned_before == what left /
    # entered unsegmented). Any mismatch is reported, never swallowed.
    counts = np.bincount(classification, minlength=256)
    # THE BYTE IS NOT THE INSTANCE ID. `write_classification` stores the ENCODED
    # class (`class_map.json` → `class_of`): identity while every id fits in a
    # byte, a compact 1..N index when they do not. This check read `counts[iid]`
    # with the RAW id and therefore compared two different things on every
    # session with ids above 255 — pccr 2026-09-23, ids up to 402: 88 loud
    # "VERIFY FAILED" lines over data that balanced exactly (0 mismatches
    # through the map, and the octree check agreed at 99.98 %). It also SKIPPED
    # every id above 255, so the 48 instances most likely to be mis-encoded were
    # the ones never verified. A check that cries wolf is worse than none: it
    # teaches the reader to scroll past the real one.
    # ONE reader of the map, the same the viewer, the propagation resume and the
    # /segmentation endpoint use — `class_of` returns {} for a session written
    # before the map existed, which means identity, exactly as those do.
    from segmentation.republish import class_of as _class_of
    _code = _class_of(output_dir)
    files_verified = True
    for inst in instances:
        iid = int(inst.get("instance_id", inst.get("id")))
        cls_code = _code.get(iid, iid)
        if cls_code > 255:
            files_verified = False
            print(f"[Erase] ⚠ VERIFY FAILED: instance {iid} has no class byte "
                  f"(map says {cls_code}) — the octree cannot paint it")
            continue
        if int(counts[cls_code]) != len(inst.get("globalIndices") or []):
            files_verified = False
            print(f"[Erase] ⚠ VERIFY FAILED: instance {iid} (class byte "
                  f"{cls_code}) has {int(counts[cls_code])} pts in "
                  f"classification.npy vs "
                  f"{len(inst.get('globalIndices') or [])} in result json")
    assigned_points_after = sum(
        len(i.get("globalIndices") or []) for i in instances)
    # EXCLUSIVITY INVARIANT (user 2026-08-31): every point owned by exactly
    # one segment — verified on EVERY transaction, mismatches never silent.
    _all_gi = [np.asarray(i.get("globalIndices") or [], dtype=np.int64)
               for i in instances if i.get("globalIndices")]
    _cat = np.concatenate(_all_gi) if _all_gi else np.empty(0, dtype=np.int64)
    overlap_points = int(len(_cat) - len(np.unique(_cat)))
    if overlap_points:
        print(f"[Erase] ⚠ EXCLUSIVITY VIOLATION: {overlap_points:,} point(s) "
              f"owned by more than one segment")
    if target_inst is not None:
        expected_delta = n_unseg_taken          # net gain = unsegmented drawn in
    else:
        expected_delta = -int(sum(touched.values()))   # deletions leave the set
    balance = {
        "cloud_points": int(N),
        "assigned_before": int(assigned_points_before),
        "assigned_after": int(assigned_points_after),
        "expected_delta": int(expected_delta),
        "consistent": (assigned_points_after - assigned_points_before
                       == expected_delta),
    }
    if not balance["consistent"]:
        print(f"[Erase] ⚠ BALANCE MISMATCH: {balance}")

    t_id_led = (int(target_inst.get("instance_id", target_inst.get("id")))
                if target_inst is not None else None)
    ledger = {
        "mode": "reassign" if target_inst is not None else "delete",
        "target": ({"instance_id": t_id_led,
                    "label": str(target_inst.get("label", "segment"))}
                   if target_inst is not None else None),
        "moved_from": {f"{labels_by_iid.get(i, 'segment')}_{i}": int(n)
                       for i, n in touched.items()},
        "unsegmented_taken": int(n_unseg_taken),
        "total_moved": int(n_reassigned) if target_inst is not None
                       else int(sum(touched.values())),
        "protected_hidden": {f"{labels_by_iid.get(i, 'segment')}_{i}": int(n)
                             for i, n in protected_hidden.items()},
        "balance": balance,
        "files_verified": files_verified,
        "exclusive": overlap_points == 0,
        "overlap_points": overlap_points,
        "conf_below": conf_below,
        "deleted_points": n_deleted,
        "irreversible": n_deleted > 0,
    }
    print(f"[Erase] ledger: {json.dumps(ledger)}")

    try:
        from segmentation.pipeline import rebuild_instance_store
        rebuild_instance_store(output_dir)
    except Exception as e:  # noqa: BLE001
        print(f"[Erase] store rebuild failed (non-fatal): {e}")

    if n_deleted:
        # physical deletion shifted every index — this commit cannot be undone
        undo = None
        return {"touched": {int(k): v for k, v in touched.items()},
                "total_removed": int(sum(touched.values())),
                "reassigned": int(n_reassigned),
                "mesh_instances": sorted(set(mesh_instances)),
                "undo": None,
                "ledger": ledger}

    undo = {"indices": undo_indices, "pixels": undo_pixels}
    if target_inst is not None and n_reassigned:
        undo["target"] = int(target_inst.get("instance_id",
                                             target_inst.get("id")))
        undo["target_added"] = np.unique(
            np.concatenate(moved_parts)).tolist() if moved_parts else []
        undo["target_pixels"] = target_pixels
        if created_iid is not None:
            undo["created"] = created_iid
            undo["created_oid"] = oid_map.get(created_iid)
    return {"touched": {int(k): v for k, v in touched.items()},
            "total_removed": int(sum(touched.values())),
            "reassigned": int(n_reassigned),
            "mesh_instances": sorted(set(mesh_instances)),
            "undo": undo,
            "ledger": ledger}


MERGE_UNDO_DIR = "_merge_undo"


def _stat_sig(path: Path) -> Optional[list]:
    try:
        st = Path(path).stat()
    except OSError:
        return None
    return [int(st.st_ino), int(st.st_size), int(st.st_mtime_ns)]


def _points_fingerprint(instances: List[dict]) -> str:
    """Which instances exist and how many points each owns — what a re-projection changes and
    an OBB refresh does not."""
    import hashlib
    rows = sorted((int(i.get("instance_id", i.get("id"))), int(i.get("total_points") or 0))
                  for i in instances)
    return hashlib.sha256(json.dumps(rows).encode("utf-8")).hexdigest()


def _mesh_folders(output_dir: Path, label: str, iid: int) -> List[Path]:
    """Every published per-object mesh folder of one instance: tsdf/<safe>, its Poisson twin
    tsdf/<safe>_poisson, and the ShapeR shape/<safe>."""
    from segmentation.tsdf_export import _safe_label
    safe = _safe_label(label or "segment", int(iid))
    out = []
    for sub in ("tsdf", "shape"):
        d = Path(output_dir) / sub
        if not d.is_dir():
            continue
        for f in d.iterdir():
            if f.is_dir() and (f.name == safe or f.name.startswith(safe + "_")):
                out.append(f)
    return out


def prune_merge_snapshots(output_dir: Path, keep: List[str]) -> int:
    """Delete every merge snapshot no live undo entry refers to (the undo stack lives in the
    server process: an entry trimmed off it, cleared by a physical deletion or lost to a
    restart leaves its snapshot behind — CLAUDE.md storage rule 2026-10-05)."""
    import shutil
    root = Path(output_dir) / MERGE_UNDO_DIR
    if not root.is_dir():
        return 0
    keep_set = {str(Path(k).resolve()) for k in keep or []}
    n = 0
    for d in root.iterdir():
        if d.is_dir() and str(d.resolve()) not in keep_set:
            shutil.rmtree(d, ignore_errors=True)
            n += 1
    return n


def merge_instances(output_dir: Path, target_iid: int,
                    source_iids: List[int], keep_snapshots: Optional[List[str]] = None) -> dict:
    """Fuse whole segments into one (USER 2026-10-09: *"3 segmentos son lo mismo
    fusionarlos"*): every point of every source joins ``target_iid``, every mask
    of every source is OR-ed into the target's masks in the same keyframe, and
    the sources stop existing (result, segmentation.json, seg_masks.npz, their
    published meshes). The target keeps its label, ids and colour. A human edit
    of the raw store, like the brush reassign: recorded in the seal's ``edits``
    (``merge``), so a re-projection keeps it. Only segments that HAVE a SAM3
    masklet can be merged: one the projection created (a split child) has no mask
    of its own, and a merge that does not move masks would be undone by the next
    projection — refused, with the reason.

    The raw mask store, the fusion map and the sources' mesh folders go into a
    snapshot (output/_merge_undo/<k>) so one undo restores them; snapshots no live
    undo entry refers to (``keep_snapshots``) are pruned first. Returns
    {merged, target, moved_points, mesh_instances, undo, ledger}."""
    import shutil
    from segmentation.pipeline import _load_ply_origins, _compute_obb
    from segmentation.fuse_parent import (MAP, FusionStale, load_map, high_water,
                                          parent_identity, _canonical_bytes,
                                          _atomic_bytes)

    output_dir = Path(output_dir)
    target_iid = int(target_iid)
    sources = sorted({int(s) for s in source_iids} - {target_iid})
    if not sources:
        raise ValueError("merge needs at least one source segment besides the target")
    res_path = output_dir / "segmentation_result.json"
    ply = output_dir / "cleaned_cloud.ply"
    if not ply.exists() or not res_path.exists():
        raise FileNotFoundError("session has no cleaned cloud / segmentation")
    result = json.loads(res_path.read_text())
    instances = result.get("instances") or []
    by_iid = {int(i.get("instance_id", i.get("id"))): i for i in instances}
    missing = [s for s in [target_iid] + sources if s not in by_iid]
    if missing:
        raise ValueError(f"segment(s) {missing} not found")

    # mask ids: the FUSED view (a survivor owns its parts' masks too); a stale
    # or absent fusion map falls back to the raw store's own ids
    oids = _mask_oids_by_iid(output_dir)
    seg_json = output_dir / "segmentation.json"
    meta = json.loads(seg_json.read_text()) if seg_json.exists() else {}
    if not oids:
        for e in meta.get("instances") or []:
            if e.get("id") is not None and e.get("instance_id") is not None:
                oids.setdefault(int(e["instance_id"]), []).append(int(e["id"]))
    t_oids = oids.get(target_iid) or []
    t_oid = _mask_obj_by_iid(output_dir).get(target_iid, t_oids[0] if t_oids else None)
    def _name(i):
        return f"{by_iid[i].get('label', 'segment')} {i}"
    no_mask = [i for i in [target_iid] + sources
               if (t_oid is None if i == target_iid else not oids.get(i))]
    if no_mask:
        raise ValueError("no SAM3 masklet of its own (the projection created it): "
                         + ", ".join(_name(i) for i in no_mask)
                         + " — a merge that cannot move masks would be undone by the next "
                           "projection; merge the segment it was split from instead")
    src_oids = sorted({o for s in sources for o in (oids.get(s) or [])} - set(t_oids))
    origins = _load_ply_origins(ply)
    if origins is None:
        raise RuntimeError("cloud has no origin fields")
    xyz = origins[0]
    N = len(xyz)

    # snapshot for the undo: the raw store + the fusion map, byte for byte
    prune_merge_snapshots(output_dir, keep_snapshots or [])
    snap_root = output_dir / MERGE_UNDO_DIR
    snap_root.mkdir(exist_ok=True)
    k = 1 + max([int(p.name) for p in snap_root.iterdir() if p.name.isdigit()],
                default=0)
    snap = snap_root / str(k)
    snap.mkdir()
    for name in ("segmentation.json", "seg_masks.npz", MAP):
        if (output_dir / name).exists():
            shutil.copy2(output_dir / name, snap / name)
    # the sources' published meshes leave with them (else they stay in the scene as ghosts
    # of objects that no longer exist) — into the snapshot, so the undo brings them back
    moved_meshes = []
    for s_ in sources:
        for f in _mesh_folders(output_dir, by_iid[s_].get("label", ""), s_):
            dest = snap / "meshes" / f.parent.name / f.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(dest))
            moved_meshes.append([f"{f.parent.name}/{f.name}", str(dest)])

    map_ok = False
    try:
        map_ok = load_map(output_dir) is not None
    except FusionStale as e:
        print(f"[Merge] DECLARED: fusion map stale before the merge ({e}) — left as it is")

    # ── masks: OR every source mask into the target's, then drop the source ids
    masks_path = output_dir / "seg_masks.npz"
    n_mask_frames = 0
    if masks_path.exists() and src_oids:
        z = np.load(masks_path, allow_pickle=True)
        masks = {kk: z[kk] for kk in z.files}
        src_set = set(src_oids)
        for kk in [kk for kk in list(masks) if kk.startswith("f")]:
            m_ = re.match(r"^f(\d+)_o(\d+)$", kk)
            if not m_ or int(m_.group(2)) not in src_set:
                continue
            tk = f"f{m_.group(1)}_o{t_oid}"
            src = masks.pop(kk)
            if tk in masks and masks[tk].shape == src.shape:
                masks[tk] = ((masks[tk] > 0) | (src > 0)).astype(masks[tk].dtype)
            else:
                masks[tk] = (src > 0).astype(np.uint8)
            n_mask_frames += 1
        if "obj_ids" in masks:
            arr = np.asarray(masks["obj_ids"], dtype=np.int32)
            masks["obj_ids"] = arr[~np.isin(arr, np.asarray(src_oids, np.int32))]
        _atomic_savez(masks_path, masks)
    if meta.get("instances") is not None and src_oids:
        # the merged ids RETIRE: the high-water marks keep them from being handed out again
        # (a new brush segment or a SAM3 round would otherwise reuse a merged source's ids)
        hi_id, hi_iid = high_water(output_dir)
        meta["id_high_water"] = max(int(meta.get("id_high_water") or 0), int(hi_id))
        meta["instance_id_high_water"] = max(int(meta.get("instance_id_high_water") or 0),
                                             int(hi_iid))
        meta["instances"] = [e for e in meta["instances"]
                             if int(e.get("id", -1)) not in set(src_oids)]
        seg_json.write_text(json.dumps(meta, indent=2))
    if masks_path.exists() and seg_json.exists():
        reseal_store(output_dir, by="merge")

    # the fusion map was tied to the old store: drop the sources' records and
    # re-tie it to the bytes now on disk (only when it held before the edit)
    if map_ok and (output_dir / MAP).exists():
        doc = json.loads((output_dir / MAP).read_text())
        mp = {kk: v for kk, v in (doc.get("map") or {}).items()
              if int(kk) not in set(sources)}
        for v in mp.values():
            v["parts"] = [p for p in v.get("parts") or []
                          if int(p.get("id", -1)) not in set(src_oids)]
        mp = {kk: v for kk, v in mp.items() if v["parts"]}
        if mp:
            doc["map"] = mp
            doc["applied"] = sum(len(v["parts"]) for v in mp.values())
            doc["parent"] = parent_identity(output_dir)
            n_parent = len(meta.get("instances") or [])
            doc["objects"] = n_parent - doc["applied"]
            _atomic_bytes(output_dir / MAP, _canonical_bytes(doc))
        else:
            (output_dir / MAP).unlink()

    # ── points: the target absorbs every source; the sources leave the result
    tgt = by_iid[target_iid]
    parts = [np.asarray(tgt.get("globalIndices") or [], dtype=np.int64)]
    removed_entries = []
    moved = 0
    for s in sources:
        gi = np.asarray(by_iid[s].get("globalIndices") or [], dtype=np.int64)
        gi = gi[(gi >= 0) & (gi < N)]
        parts.append(gi)
        moved += int(len(gi))
        removed_entries.append(by_iid[s])
    merged = np.unique(np.concatenate(parts))
    tgt["globalIndices"] = merged.tolist()
    tgt["total_points"] = int(len(merged))
    if len(merged) >= 4:
        tgt["obb"] = _compute_obb(raw_to_display(output_dir, xyz[merged]))
    result["instances"] = [i for i in instances
                           if int(i.get("instance_id", i.get("id"))) not in set(sources)]
    result["segmented_points"] = sum(
        int(i.get("total_points") or 0) for i in result["instances"])
    res_path.write_text(json.dumps(result))
    _write_classification(output_dir, result["instances"], N)
    try:
        from segmentation.pipeline import rebuild_instance_store
        rebuild_instance_store(output_dir)
    except Exception as e:  # noqa: BLE001
        print(f"[Merge] store rebuild failed (non-fatal): {e}")
    # the target's own mesh now covers only its old part: refit it (its OBB then comes from
    # the mesh, the project's rule once a mesh exists)
    mesh_instances = ([target_iid] if published_mesh_path(output_dir, tgt.get("label", ""),
                                                          target_iid) else [])
    ledger = {"mode": "merge",
              "target": {"instance_id": target_iid, "label": str(tgt.get("label", ""))},
              "merged": {f"{e.get('label', 'segment')}_{int(e.get('instance_id', e.get('id')))}":
                         int(e.get("total_points") or 0) for e in removed_entries},
              "moved_points": moved, "mask_frames_moved": n_mask_frames,
              "mask_ids_removed": src_oids,
              "meshes_retired": [m[0] for m in moved_meshes]}
    print(f"[Merge] ledger: {json.dumps(ledger)}")
    # what the undo must find to be allowed: the store, the cloud and the instances exactly
    # as this merge left them (a re-projection, an Autosegment or an epoch select in between
    # makes the snapshot a different session's — the undo is refused, never applied blind)
    undo = {"kind": "merge", "snapshot": str(snap), "target": target_iid,
            "target_before": np.asarray(parts[0]).tolist(), "removed": removed_entries,
            "meshes": moved_meshes,
            "after": {"segmentation.json": _stat_sig(seg_json),
                      "seg_masks.npz": _stat_sig(masks_path),
                      "cleaned_cloud.ply": _stat_sig(ply),
                      "points": _points_fingerprint(result["instances"])}}
    return {"merged": sources, "target": target_iid, "moved_points": moved,
            "mesh_instances": mesh_instances, "undo": undo, "ledger": ledger}


def _undo_merge(output_dir: Path, undo: dict) -> dict:
    """Put the raw store + fusion map + the sources' meshes back byte for byte and the sources
    back into the result, with the target's own points — only when the session is still the
    one the merge left."""
    import shutil
    from segmentation.pipeline import _load_ply_origins, _compute_obb
    from segmentation.fuse_parent import MAP
    snap = Path(undo["snapshot"])
    res_path = output_dir / "segmentation_result.json"
    result = json.loads(res_path.read_text())
    after = undo.get("after") or {}
    now = {"segmentation.json": _stat_sig(output_dir / "segmentation.json"),
           "seg_masks.npz": _stat_sig(output_dir / "seg_masks.npz"),
           "cleaned_cloud.ply": _stat_sig(output_dir / "cleaned_cloud.ply"),
           "points": _points_fingerprint(result.get("instances") or [])}
    changed = [k for k in after if after[k] != now.get(k)]
    if changed or not snap.is_dir():
        shutil.rmtree(snap, ignore_errors=True)
        why = (f"the session changed since the merge ({', '.join(changed)})" if changed
               else "its snapshot is gone")
        print(f"[Merge] undo REFUSED: {why} — nothing restored")
        return {"restored": 0, "detail": f"merge not undone: {why}"}
    for name in ("segmentation.json", "seg_masks.npz", MAP):
        if (snap / name).exists():
            shutil.copy2(snap / name, output_dir / name)
        elif name == MAP and (output_dir / name).exists():
            (output_dir / name).unlink()
    for rel, src in undo.get("meshes") or []:
        dest = output_dir / rel
        if Path(src).exists() and not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(src, str(dest))
    origins = _load_ply_origins(output_dir / "cleaned_cloud.ply")
    xyz = origins[0] if origins else None
    tid = int(undo["target"])
    for inst in result.get("instances") or []:
        if int(inst.get("instance_id", inst.get("id"))) == tid:
            gi = np.asarray(undo.get("target_before") or [], dtype=np.int64)
            inst["globalIndices"] = gi.tolist()
            inst["total_points"] = int(len(gi))
            if xyz is not None and len(gi) >= 4:
                inst["obb"] = _compute_obb(raw_to_display(output_dir, xyz[gi]))
    result["instances"] = (result.get("instances") or []) + list(undo.get("removed") or [])
    result["instances"].sort(key=lambda i: int(i.get("instance_id", i.get("id"))))
    result["segmented_points"] = sum(
        int(i.get("total_points") or 0) for i in result["instances"])
    res_path.write_text(json.dumps(result))
    if xyz is not None:
        _write_classification(output_dir, result["instances"], len(xyz))
    try:
        from segmentation.pipeline import rebuild_instance_store
        rebuild_instance_store(output_dir)
    except Exception as e:  # noqa: BLE001
        print(f"[Merge] store rebuild failed (non-fatal): {e}")
    shutil.rmtree(snap, ignore_errors=True)
    n = sum(int(e.get("total_points") or 0) for e in undo.get("removed") or [])
    print(f"[Merge] undone: {len(undo.get('removed') or [])} segment(s) restored")
    return {"restored": n}


def undo_erase(output_dir: Path, undo: dict) -> dict:
    """Restore one erase stroke (indices back into their instances, mask
    pixels back on). Returns {restored: n}."""
    from segmentation.pipeline import _load_ply_origins, _compute_obb

    output_dir = Path(output_dir)
    if undo.get("kind") == "merge":
        return _undo_merge(output_dir, undo)
    res_path = output_dir / "segmentation_result.json"
    result = json.loads(res_path.read_text())
    by_iid = {int(i.get("instance_id", i.get("id"))): i
              for i in (result.get("instances") or [])}
    origins = _load_ply_origins(output_dir / "cleaned_cloud.ply")
    xyz = origins[0] if origins else None

    restored = 0
    for iid_s, idxs in (undo.get("indices") or {}).items():
        inst = by_iid.get(int(iid_s))
        if inst is None:
            continue
        gi = np.asarray(inst.get("globalIndices") or [], dtype=np.int64)
        merged = np.union1d(gi, np.asarray(idxs, dtype=np.int64))
        inst["globalIndices"] = merged.tolist()
        inst["total_points"] = int(len(merged))
        restored += len(idxs)
        if xyz is not None and len(merged) >= 4:
            inst["obb"] = _compute_obb(raw_to_display(output_dir, xyz[merged]))

    # reassign rollback: the target gives the gained points back; a CREATED
    # segment is removed entirely (result + segmentation.json + its masks)
    t_iid = undo.get("target")
    created = undo.get("created")
    if created is not None:
        result["instances"] = [
            i for i in (result.get("instances") or [])
            if int(i.get("instance_id", i.get("id"))) != int(created)]
        try:
            seg_json = output_dir / "segmentation.json"
            meta = json.loads(seg_json.read_text())
            meta["instances"] = [e for e in (meta.get("instances") or [])
                                 if int(e.get("instance_id", -1)) != int(created)]
            seg_json.write_text(json.dumps(meta, indent=2))
        except Exception as e:  # noqa: BLE001
            print(f"[Erase] undo: segmentation.json rollback failed: {e}")
    elif t_iid is not None:
        inst = by_iid.get(int(t_iid))
        added = np.asarray(undo.get("target_added") or [], dtype=np.int64)
        if inst is not None and len(added):
            gi = np.asarray(inst.get("globalIndices") or [], dtype=np.int64)
            keep = np.setdiff1d(gi, added)
            inst["globalIndices"] = keep.tolist()
            inst["total_points"] = int(len(keep))
            if xyz is not None and len(keep) >= 4:
                inst["obb"] = _compute_obb(
                    raw_to_display(output_dir, xyz[keep]))

    masks_path = output_dir / "seg_masks.npz"
    if (undo.get("pixels") or undo.get("target_pixels")) and masks_path.exists():
        try:
            z = np.load(masks_path, allow_pickle=True)
            masks = {k: z[k] for k in z.files}
        except Exception as e:  # noqa: BLE001
            print(f"[Erase] undo: seg_masks.npz unreadable ({e}) — "
                  "points restored, masks untouched")
            masks = None
        if masks is not None:
            for key, rr, cc in (undo.get("pixels") or []):
                m = masks.get(key)
                if m is None:
                    continue
                m = m.copy()
                m[np.asarray(rr, np.int64), np.asarray(cc, np.int64)] = 1
                masks[key] = m
            for key, rr, cc in (undo.get("target_pixels") or []):
                m = masks.get(key)
                if m is None:
                    continue
                m = m.copy()
                m[np.asarray(rr, np.int64), np.asarray(cc, np.int64)] = 0
                masks[key] = m
            c_oid = undo.get("created_oid")
            if undo.get("created") is not None and c_oid is not None:
                for key in [k for k in masks
                            if k.endswith(f"_o{int(c_oid)}")]:
                    del masks[key]
                if "obj_ids" in masks:
                    arr = np.asarray(masks["obj_ids"], dtype=np.int32)
                    masks["obj_ids"] = arr[arr != np.int32(c_oid)]
            _atomic_savez(masks_path, masks)
            # the seal names the bytes just written, or the projection refuses the store
            reseal_store(output_dir, by="undo_erase")

    result["segmented_points"] = sum(
        int(i.get("total_points") or 0) for i in (result.get("instances") or []))
    result["coverage"] = round(
        result["segmented_points"] / max(1, int(result.get("total_points") or 1)), 4)
    res_path.write_text(json.dumps(result))
    if xyz is not None:
        _write_classification(output_dir, result.get("instances") or [],
                              len(xyz))
    try:
        from segmentation.pipeline import rebuild_instance_store
        rebuild_instance_store(output_dir)
    except Exception as e:  # noqa: BLE001
        print(f"[Erase] store rebuild failed (non-fatal): {e}")
    return {"restored": restored}


def crop_glb_sphere(glb_path: Path, output_dir: Path, center_display,
                    radius: float, shape: str = "sphere",
                    yaw_deg: float = 0.0) -> bool:
    """Best-effort INSTANT visual crop: drop the published GLB's faces whose
    vertices fall inside the erase zone (sphere or display-axis-aligned cube).
    The definitive mesh comes from the debounced re-fit; this only keeps the
    screen honest meanwhile. Returns False (file untouched) on any trouble."""
    try:
        import trimesh
        s_ft, R_ft, t_ft = _load_floor_transform(output_dir)
        c = np.asarray(center_display, np.float64)
        r = float(radius)
        scene = trimesh.load(str(glb_path), force="scene")
        changed = False
        for name, geom in list(scene.geometry.items()):
            v = s_ft * (np.asarray(geom.vertices) @ R_ft.T) + t_ft
            if str(shape).lower() == "cube":
                inside = _cube_hit(v, c, r, float(np.radians(yaw_deg)))
            else:
                inside = ((v - c) ** 2).sum(axis=1) <= r * r
            if not inside.any():
                continue
            faces = np.asarray(geom.faces)
            keepf = ~inside[faces].any(axis=1)
            if keepf.all():
                continue
            geom.update_faces(keepf)
            geom.remove_unreferenced_vertices()
            changed = True
        if changed:
            scene.export(str(glb_path))
        return changed
    except Exception as e:  # noqa: BLE001
        print(f"[Erase] instant GLB crop skipped ({glb_path.name}): {e}")
        return False


def obb_from_mesh(glb_path: Path, output_dir: Path) -> Optional[dict]:
    """OBB from the published mesh's vertices, in DISPLAY frame (user rule:
    once a mesh exists, the bbox comes from the mesh — flatter, tighter)."""
    try:
        import trimesh
        from segmentation.pipeline import _compute_obb
        scene = trimesh.load(str(glb_path), force="scene")
        vs = [np.asarray(g.vertices) for g in scene.geometry.values()
              if len(getattr(g, "vertices", []))]
        if not vs:
            return None
        v = np.concatenate(vs)
        if len(v) < 4:
            return None
        return _compute_obb(raw_to_display(output_dir, v))
    except Exception as e:  # noqa: BLE001
        print(f"[Erase] mesh OBB failed ({glb_path.name}): {e}")
        return None
