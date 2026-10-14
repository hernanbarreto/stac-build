"""The names the USER gives segments (USER 2026-10-09: *"si le edito el nombre a un segmento,
¿eso se guarda también en los datos del chat?"*).

A rename used to rewrite the SAM3 store (segmentation.json) and the projection result: the raw
store's bytes changed, so the fusion map and the projection stamp went stale and the
certification refused to run; a re-projection gave the new name to ONE masklet and its
same-label fragments came back as a second object with the old name; a SAM3 round, a manager
re-propagation or an epoch select brought the old name back; the chat (scene_r.db) never saw it.

NOW the user's name lives HERE, ``output/label_overrides.json`` — never in the SAM3 store, never
an input of the projection or fusion stamps — and is APPLIED on top of what the segmentation
named: at the end of every projection (after the fusion: the object's fragments still fuse under
their SAM3 name), after an epoch select, before the descriptions, and by the rename itself. The
applied instance carries ``label`` (the user's), ``label_sam3`` (what the segmentation called it)
and ``label_origin: human_validated`` (provenance rule).

IDENTITY. An override names an instance by its ``instance_id``, the SAM3 name it had when the
user renamed it, and ``ANCHORS`` points of it (display frame). Where geometry is at hand (the
projection, a session with its chat store) the id is VERIFIED: among the instances with that SAM3
name near the anchors, the id's instance must be the one closest to them — a new segmentation
that numbered two chairs the other way round, or a split that shifted its children's ids, puts
the name on the chair the user named, not on whoever got its number. An override whose id no
longer names it is found again by geometry (the instance closest to its anchors, median distance
within ``REIDENTIFY_MAX_M``) and re-keyed; the anchors are refreshed on every confirmed match, so
they follow the epochs' corrections. Overrides are resolved TOGETHER (direct matches first, then
the searches; no two overrides on one instance; nothing overwritten). One nothing matches is kept
and declared — never applied to an object it cannot show is the one the user named. An instance
the user had named whose override is gone (renamed back to its segmentation name) gets its
segmentation name back. The file is read-modify-written under its own lock.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

FILE = "label_overrides.json"
VERSION = 1
#: points of the object kept to find it again after a new segmentation
ANCHORS = 300
#: the farthest (median anchor distance) an object can be from where the user named it and still
#: be found again by geometry alone (an epoch's correction moves points by cm to dm)
REIDENTIFY_MAX_M = 0.50
#: how far around the anchors same-name candidates are compared
NEAR_M = 2.0
HUMAN = "human_validated"
VLM = "vlm_proposed"


def _path(output_dir) -> Path:
    return Path(output_dir) / FILE


def load(output_dir) -> dict:
    """{"version", "labels": {"<instance_id>": {label, sam3_label, provenance, anchors}}}"""
    p = _path(output_dir)
    if not p.is_file():
        return {"version": VERSION, "labels": {}}
    try:
        doc = json.loads(p.read_text())
    except (OSError, ValueError):
        return {"version": VERSION, "labels": {}}
    if not isinstance(doc, dict) or not isinstance(doc.get("labels"), dict):
        return {"version": VERSION, "labels": {}}
    return doc


def _save(output_dir, doc: dict) -> None:
    from atomic_io import atomic_write_json
    if not doc.get("labels") and not doc.get("pending"):
        _path(output_dir).unlink(missing_ok=True)
        return
    atomic_write_json(_path(output_dir), doc, indent=1, sort_keys=True)


class _locked:
    """The overrides file's own lock (fcntl, across processes): the server's record() and a
    worker's apply() never interleave a read-modify-write."""

    def __init__(self, output_dir):
        self.p = Path(output_dir) / ".label_overrides.lock"

    def __enter__(self):
        import fcntl
        import os
        self.p.parent.mkdir(parents=True, exist_ok=True)
        self.fh = os.open(str(self.p), os.O_RDWR | os.O_CREAT, 0o666)
        fcntl.flock(self.fh, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        import fcntl
        import os
        fcntl.flock(self.fh, fcntl.LOCK_UN)
        os.close(self.fh)
        return False


def sam3_label_of(inst: dict) -> Optional[str]:
    """What the segmentation called this instance (before any user name)."""
    if inst.get("label_sam3"):
        return str(inst["label_sam3"])
    if inst.get("label_origin") == HUMAN:
        return None
    lab = inst.get("label")
    return str(lab) if lab is not None else None


def anchors_from(points: Optional[np.ndarray]) -> Optional[list]:
    """ANCHORS points of an object, a fixed stride over its points (deterministic)."""
    if points is None:
        return None
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    if not len(pts):
        return None
    step = max(1, len(pts) // ANCHORS)
    return np.round(pts[::step][:ANCHORS], 4).tolist()


def record(output_dir, instance_id: int, label: str, *, sam3_label: Optional[str],
           anchors: Optional[list]) -> Optional[dict]:
    """Keep the user's name for ``instance_id``; returns the override (None when the name given
    is the segmentation's own — the override is then removed)."""
    with _locked(output_dir):
        doc = load(output_dir)
        key = str(int(instance_id))
        rec = dict(doc["labels"].get(key) or {})
        rec["sam3_label"] = rec.get("sam3_label") or sam3_label
        rec["label"] = str(label)
        rec["provenance"] = HUMAN
        if anchors:
            rec["anchors"] = anchors
        if rec["sam3_label"] is not None and rec["label"] == rec["sam3_label"]:
            doc["labels"].pop(key, None)
            rec = None
        else:
            doc["labels"][key] = rec
        doc["version"] = VERSION
        _save(output_dir, doc)
        return rec


def current_label(output_dir, instance_id: int) -> Optional[str]:
    """The user's name for ``instance_id`` when one is recorded."""
    rec = (load(output_dir).get("labels") or {}).get(str(int(instance_id)))
    return rec.get("label") if rec else None


def _iid(inst: dict) -> int:
    return int(inst.get("instance_id", inst.get("id")))


def _median_dist(anchors: np.ndarray, pts: Optional[np.ndarray]) -> float:
    if pts is None:
        return float("inf")
    pts = np.asarray(pts, float).reshape(-1, 3)
    if not len(pts):
        return float("inf")
    if len(pts) > 20_000:
        pts = pts[:: len(pts) // 20_000 + 1]
    from scipy.spatial import cKDTree
    d, _ = cKDTree(pts).query(anchors, k=1)
    return float(np.median(d))


def _near(inst: dict, anchors: np.ndarray) -> bool:
    obb = inst.get("obb") or {}
    if not obb.get("center") or not obb.get("half_extents"):
        return True                      # no box to rule it out
    c = np.asarray(obb["center"], float)
    h = np.asarray(obb["half_extents"], float)
    R = np.asarray(obb.get("rotation", np.eye(3)), float)
    local = (anchors - c) @ R
    return bool((np.abs(local) <= h + NEAR_M).all(axis=1).any())


def _closest(anchors: np.ndarray, cands: List[dict],
             points_of: Callable[[dict], Optional[np.ndarray]]):
    best, best_d = None, float("inf")
    for inst in cands:
        if not _near(inst, anchors):
            continue
        d = _median_dist(anchors, points_of(inst))
        if d < best_d:
            best, best_d = inst, d
    return best, best_d


def apply(output_dir, instances: List[dict], *,
          points_of: Optional[Callable[[dict], Optional[np.ndarray]]] = None,
          log: Callable[[str], object] = print) -> Dict[int, Optional[str]]:
    """Put the user's names on ``instances`` (in place) and give back the segmentation's name to
    an instance whose override is gone. Returns {instance_id: user name} for the names applied
    and {instance_id: None} for the names given back. ``points_of`` (an instance → its
    display-frame points) verifies the ids and finds renumbered objects again."""
    changed: Dict[int, Optional[str]] = {}
    with _locked(output_dir):
        doc = load(output_dir)
        recs = [(k, r) for k, r in sorted((doc.get("labels") or {}).items(), key=lambda kv: int(kv[0]))]
        recs += [(None, r) for r in (doc.get("pending") or [])]
        by_iid = {_iid(i): i for i in instances}
        resolved: Dict[int, dict] = {}          # instance_id → override
        unresolved: List = []
        taken = set()

        def _anchors(rec):
            a = np.asarray(rec.get("anchors") or [], float).reshape(-1, 3)
            return a if len(a) else None

        # pass 1 — the id still names the object (verified by geometry when there is some)
        rest = []
        for key, rec in recs:
            inst = by_iid.get(int(key)) if key is not None else None
            if inst is None or sam3_label_of(inst) != rec.get("sam3_label") or _iid(inst) in taken:
                rest.append((key, rec))
                continue
            a = _anchors(rec)
            if points_of is not None and a is not None:
                same = [i for i in instances if sam3_label_of(i) == rec.get("sam3_label")
                        and _iid(i) not in taken]
                best, _d = _closest(a, same, points_of)
                if best is not None and best is not inst:
                    rest.append((key, rec))     # its number now names another object
                    continue
            resolved[_iid(inst)] = rec
            taken.add(_iid(inst))
        # pass 2 — found again by geometry
        for key, rec in rest:
            a = _anchors(rec)
            found = None
            if points_of is not None and a is not None:
                free = [i for i in instances if _iid(i) not in taken]
                pref = [i for i in free if sam3_label_of(i) == rec.get("sam3_label")]
                found, d = _closest(a, pref, points_of)
                if found is None or d > REIDENTIFY_MAX_M:
                    found, d = _closest(a, free, points_of)
                if found is not None and d > REIDENTIFY_MAX_M:
                    found = None
            if found is None:
                log(f"[labels] DECLARED: the user's name '{rec.get('label')}' (#{key}, segmented "
                    f"as '{rec.get('sam3_label')}') matches no object of this segmentation — "
                    f"kept, not applied")
                unresolved.append((key, rec))
                continue
            log(f"[labels] '{rec.get('label')}' found again by its geometry: #{key} → "
                f"#{_iid(found)}")
            rec = dict(rec, sam3_label=sam3_label_of(found))
            resolved[_iid(found)] = rec
            taken.add(_iid(found))

        # the names onto the instances; anchors refreshed where the geometry confirmed them
        for iid, rec in resolved.items():
            inst = by_iid[iid]
            if points_of is not None:
                fresh = anchors_from(points_of(inst))
                if fresh:
                    rec["anchors"] = fresh
            if not inst.get("label_sam3") and inst.get("label_origin") != HUMAN:
                inst["label_sam3"] = inst.get("label")
            if inst.get("label") != rec["label"] or inst.get("label_origin") != HUMAN:
                changed[iid] = rec["label"]
            inst["label"] = rec["label"]
            inst["label_origin"] = HUMAN
        # an instance the user had named whose override is gone gets its segmentation name back
        for inst in instances:
            iid = _iid(inst)
            if iid not in resolved and inst.get("label_origin") == HUMAN and inst.get("label_sam3"):
                inst["label"] = inst.pop("label_sam3")
                inst["label_origin"] = VLM
                changed[iid] = None

        new_doc = {"version": VERSION, "labels": {str(i): r for i, r in resolved.items()}}
        pending = []
        for key, rec in unresolved:
            if key is not None and key not in new_doc["labels"]:
                new_doc["labels"][key] = rec
            else:
                pending.append(rec)
        if pending:
            new_doc["pending"] = pending
        if new_doc != {"version": VERSION, **{k: v for k, v in doc.items() if k in ("labels", "pending")}}:
            _save(output_dir, new_doc)
    n_on = sum(1 for v in changed.values() if v is not None)
    n_back = sum(1 for v in changed.values() if v is None)
    if n_on or n_back:
        log(f"[labels] {n_on} user name(s) applied, {n_back} segmentation name(s) given back")
    return changed


def update_store(output_dir, changed: Dict[int, Optional[str]], *,
                 forget_dossier: bool = False, sam3_names: Optional[Dict[int, str]] = None) -> None:
    """The chat's store (scene_r.db) follows: a user name with the user's provenance, a name
    given back (``None`` → its segmentation name from ``sam3_names``) as the VLM's.
    ``forget_dossier`` drops each object's chat dossier (written under the old name — the chat
    regenerates it)."""
    p = Path(output_dir) / "scene_r.db"
    if not changed or not p.exists():
        return
    from phase_r.instance_store import InstanceStore
    st = InstanceStore(p)
    try:
        for iid, label in changed.items():
            if label is None:
                name = (sam3_names or {}).get(int(iid))
                if name is None:
                    continue
                st.conn.execute("UPDATE instances SET label=?, label_origin=? WHERE instance_id=?",
                                (str(name), VLM, int(iid)))
            else:
                st.conn.execute("UPDATE instances SET label=?, label_origin=? WHERE instance_id=?",
                                (str(label), HUMAN, int(iid)))
            if forget_dossier:
                st.conn.execute("DELETE FROM scene_meta WHERE key=?", (f"object_analysis_{int(iid)}",))
        st.conn.commit()
    finally:
        st.conn.close()


class Busy(RuntimeError):
    """A projection holds the session's matching lock: it applies the names itself."""


def apply_to_session(output_dir, *, log: Callable[[str], object] = print,
                     lock_timeout_s: Optional[float] = None) -> Dict[int, Optional[str]]:
    """The live session gets the user's names (and gives back the ones removed):
    segmentation_result.json (rewritten only when a name changes), the chat's store and the
    viewer's light copy — under the session's matching lock (``lock_timeout_s``: raise
    :class:`Busy` instead of waiting). The ids are verified with the chat store's points."""
    from atomic_io import atomic_write_json
    from segmentation.match_lock import MatchingBusy, matching_lock
    out = Path(output_dir)
    res_p = out / "segmentation_result.json"
    if not res_p.exists():
        return {}
    try:
        with matching_lock(out, timeout_s=lock_timeout_s, log=lambda m: log(f"[labels]{m}")):
            result = json.loads(res_p.read_text())
            insts = result.get("instances") or []
            if not load(out).get("labels") and not load(out).get("pending") \
                    and not any(i.get("label_origin") == HUMAN for i in insts):
                return {}
            sam3_before = {_iid(i): sam3_label_of(i) for i in insts}
            store = None
            if (out / "scene_r.db").exists():
                from phase_r.instance_store import InstanceStore
                store = InstanceStore(out / "scene_r.db")
            try:
                changed = apply(out, insts, log=log,
                                points_of=(lambda i: store.get_points(_iid(i))) if store else None)
            finally:
                if store is not None:
                    store.conn.close()
            if changed:
                atomic_write_json(res_p, result)
                try:
                    from segmentation.segments_view import segments_view
                    segments_view(res_p)                  # the viewer's light copy, refreshed now
                except Exception as e:  # noqa: BLE001 — a cache
                    log(f"[labels] light copy not refreshed ({e})")
            update_store(out, changed, sam3_names={k: v for k, v in sam3_before.items() if v})
            return changed
    except MatchingBusy as e:
        raise Busy(str(e)) from e


def _poisson_name(label: str, iid: int) -> str:
    """The Poisson export's own folder name (segmentation.poisson_object): another sanitiser,
    no length cut."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in (label or "segment")) \
        + f"_{int(iid)}_poisson"


def _folder_pairs(old_label: str, new_label: str, iid: int) -> List[tuple]:
    """(old folder name, new folder name, old file stem, new file stem) of every per-object
    folder a mesh export writes: tsdf/<safe>, tsdf/<safe>_pgsr, tsdf/<safe>_poisson and the
    Poisson export's own naming, shape/<safe>."""
    from segmentation.session_io import _safe_label
    o = _safe_label(old_label or "segment", int(iid))
    n = _safe_label(new_label or "segment", int(iid))
    pairs = [(o + sfx, n + sfx, o, n) for sfx in ("", "_pgsr", "_poisson")]
    po, pn = _poisson_name(old_label, iid), _poisson_name(new_label, iid)
    if po != o + "_poisson":
        pairs.append((po, pn, po, pn))
    return pairs


def rename_object_files(output_dir, old_label: str, new_label: str, instance_id: int,
                        log: Callable[[str], object] = print) -> List[str]:
    """The object's published files follow its name: its per-object folders under tsdf/ and
    shape/ (exactly the names the exports give THIS instance — never another instance's folder
    that merely starts the same), the files inside named after it, the label their meta carries,
    and the session's placed-object references to them (scene_objects.json)."""
    out = Path(output_dir)
    moved: List[str] = []
    url_moves: List[tuple] = []
    for sub in ("tsdf", "shape"):
        d = out / sub
        if not d.is_dir():
            continue
        for old_name, new_name, old_stem, new_stem in _folder_pairs(old_label, new_label,
                                                                      instance_id):
            f = d / old_name
            if old_name == new_name or not f.is_dir():
                continue
            dest = d / new_name
            if dest.exists():
                log(f"[labels] DECLARED: {sub}/{new_name} exists — {sub}/{old_name} not renamed")
                continue
            f.rename(dest)
            for g in sorted(dest.iterdir()):
                if g.is_file() and g.name.startswith(old_stem):
                    new_file = new_stem + g.name[len(old_stem):]
                    url_moves.append((f"/{old_name}/{g.name}", f"/{new_name}/{new_file}"))
                    g = g.rename(dest / new_file)
                if g.is_file() and g.suffix == ".json":
                    try:
                        meta = json.loads(g.read_text())
                    except (OSError, ValueError):
                        continue
                    if isinstance(meta, dict) and meta.get("label") == old_label:
                        meta["label"] = new_label
                        g.write_text(json.dumps(meta, indent=2))
            url_moves.append((f"/{old_name}/", f"/{new_name}/"))
            moved.append(f"{sub}/{old_name} → {sub}/{new_name}")
    so = out / "scene_objects.json"
    if url_moves and so.exists():
        text = so.read_text()
        new_text = text
        for a, b in url_moves:                    # file URLs first, then the folder prefix
            new_text = new_text.replace(a, b)
        if new_text != text:
            from atomic_io import atomic_write_text
            atomic_write_text(so, new_text)
    if moved:
        log(f"[labels] object files renamed: {'; '.join(moved)}")
    return moved


if __name__ == "__main__":
    import argparse
    import sys
    ap = argparse.ArgumentParser(description="Apply the user's segment names to a session.")
    ap.add_argument("--apply", required=True, help="the session's output directory")
    ap.add_argument("--lock-timeout", type=float, default=None,
                    help="seconds to wait for a running projection (exit 3 when it holds on)")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        got = apply_to_session(a.apply, lock_timeout_s=a.lock_timeout)
    except Busy as e:
        print(f"[labels] a projection is running ({e}) — it applies the names itself")
        sys.exit(3)
    print(json.dumps({"changed": {str(k): v for k, v in got.items()}}))
