"""The ONE translation between the mask store's frame keys and the frame
numbers everything else is keyed by.

Three index spaces live in one session and they were never written down:

  · KEYFRAME POSITION — 0, 1, 2 … N-1. What ``_prepare_valid_frames`` copies
    into ``frames_valid/`` and what SAM3 therefore keys its masks by. The npz
    key is ``f<position>_o<oid>``.
  · VIDEO FRAME NUMBER — 1, 60, 97 … 3026. What the camera poses, the
    intrinsics, the JPEG filenames and every point's ``frame_global`` are
    keyed by. ``output/camera_frames.txt`` line *i* is the video frame of
    keyframe *i*, and that file is the authority: the poses are read from it.
  · SAM3 SEQUENTIAL INDEX — the interactive session's own numbering, mapped
    back to filenames by ``kf_mapping``. It only exists inside main.py.

Nothing translated. Fourteen call sites took a mask key and handed it to
``cam.pose_map`` (or took a real frame and built a mask key from it), which on
pccr matched 13 of 216 keyframes — and those 13 took the mask of the WRONG
keyframe, because 60 is both a valid position and a valid video number. That
is measured: ``witness/mask_votes`` produced the point ``status`` the user
judges the cloud by out of 13 usable frames.

Two rules, and they are the whole module:

  1. The store DECLARES its space (``mask_frame_space`` in the npz). A store
     written before the declaration existed is MEASURED against the keyframe
     list — never assumed — exactly as ``_mask_frame_lookup`` did.
  2. A writer that holds frames in the other space is CONVERTED before the
     upsert, or the save FAILS. The batch pipeline wrote positions and the
     interactive/Resume paths wrote video numbers through the same upserting
     ``_save_masks``: one file, two spaces, and 60 meaning two different
     frames depending on who wrote it.

Hernán Barreto - Ingerop IN3 Session IV - STAC
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

SPACE_KEYFRAME = "keyframe_position"
SPACE_VIDEO = "video_frame"
SPACES = (SPACE_KEYFRAME, SPACE_VIDEO)

#: npz key holding the declaration. Deliberately NOT starting with "f": half
#: the readers iterate ``npz.files`` filtering on ``key.startswith("f")``.
NPZ_KEY = "mask_frame_space"

_MASK_KEY_RE = re.compile(r"^f(\d+)_o(\d+)$")


# ── the keyframe list (position → video frame number) ────────────────────

def keyframe_numbers(output_dir) -> Optional[List[int]]:
    """``camera_frames.txt`` (or ``frame_list.json``): line *i* = the video
    frame number of keyframe *i*. None when the session has no sidecar — then
    the backend's own frame numbering IS the keyframe ordinal and the two
    spaces coincide."""
    from segmentation.session_io import _load_frame_index_map
    try:
        nums = _load_frame_index_map(Path(output_dir))
    except Exception:  # noqa: BLE001 — a missing sidecar is not an error
        return None
    if not nums:
        return None
    return [int(x) for x in nums]


def mask_frames_of(masks) -> List[int]:
    """Every frame index present as an ``f<N>_o<OID>`` key."""
    files = getattr(masks, "files", None)
    keys = files if files is not None else list(masks)
    out = set()
    for k in keys:
        m = _MASK_KEY_RE.match(str(k))
        if m:
            out.add(int(m.group(1)))
    return sorted(out)


# ── the space itself ─────────────────────────────────────────────────────

class MaskSpace:
    """Translates between the mask store's frame keys and the cloud/pose frame
    numbers. Built by :func:`resolve`; never instantiate it by guessing."""

    def __init__(self, space: str, keyframes: Optional[Sequence[int]],
                 source: str = "measured",
                 mask_frames: Sequence[int] = (),
                 stray: Sequence[int] = (),
                 ambiguous: Sequence[int] = ()):
        if space not in SPACES:
            raise ValueError(f"unknown mask frame space {space!r}")
        self.space = space
        self.source = source
        self.keyframes: List[int] = [int(x) for x in (keyframes or [])]
        self._pos_of: Dict[int, int] = {f: i for i, f in enumerate(self.keyframes)}
        self.mask_frames: List[int] = [int(x) for x in mask_frames]
        #: frames the declared/measured space cannot explain but the OTHER one
        #: can — the fingerprint of a store written by two conventions
        self.stray: List[int] = [int(x) for x in stray]
        #: frames both spaces explain: a mixed store cannot be untangled there
        self.ambiguous: List[int] = [int(x) for x in ambiguous]

    # -- properties ------------------------------------------------------

    @property
    def is_identity(self) -> bool:
        """True when a mask key IS the cloud frame number (no translation)."""
        return self.space == SPACE_VIDEO

    @property
    def mixed(self) -> bool:
        return bool(self.stray)

    # -- translation -----------------------------------------------------

    def to_cloud(self, mask_frame) -> Optional[int]:
        """Mask key frame → the frame the poses / JPEGs / ``frame_global``
        use. None when the key has no keyframe (a stray from a mixed store)."""
        f = int(mask_frame)
        if self.space == SPACE_VIDEO:
            return f
        if 0 <= f < len(self.keyframes):
            return self.keyframes[f]
        return None

    def to_mask(self, cloud_frame) -> Optional[int]:
        """Cloud / pose frame number → the mask store's key frame. None when
        that frame is not a keyframe (the masks cannot testify about it)."""
        f = int(cloud_frame)
        if self.space == SPACE_VIDEO:
            return f
        return self._pos_of.get(f)

    def from_keyframe(self, kf_index) -> Optional[int]:
        """KEYFRAME POSITION (0, 1, 2 …) → the store's key frame. The UI and
        the interactive SAM3 session both count keyframes, so this is the
        third caller shape; it is a no-op on a positional store and a lookup
        on a video-keyed one."""
        i = int(kf_index)
        if self.space == SPACE_KEYFRAME:
            return i if (not self.keyframes or 0 <= i < len(self.keyframes)) else None
        if 0 <= i < len(self.keyframes):
            return self.keyframes[i]
        return None

    def key_of_keyframe(self, kf_index, oid) -> Optional[str]:
        """The npz key of an object at a KEYFRAME POSITION."""
        m = self.from_keyframe(kf_index)
        return None if m is None else f"f{int(m)}_o{int(oid)}"

    def key(self, cloud_frame, oid) -> Optional[str]:
        """The npz key for an object in a cloud frame, or None when that frame
        is not a keyframe. NEVER build ``f{frame}_o{oid}`` by hand."""
        m = self.to_mask(cloud_frame)
        return None if m is None else f"f{int(m)}_o{int(oid)}"

    def cloud_frames(self, mask_frames: Optional[Iterable[int]] = None) -> List[int]:
        """The store's frames, expressed in cloud space (strays dropped)."""
        src = self.mask_frames if mask_frames is None else mask_frames
        out = []
        for f in src:
            c = self.to_cloud(f)
            if c is not None:
                out.append(int(c))
        return sorted(set(out))

    def cloud_to_mask(self) -> Dict[int, int]:
        """The legacy ``_mask_frame_lookup`` contract: {cloud frame: mask
        frame}, EMPTY when the identity is right."""
        if self.space == SPACE_VIDEO:
            return {}
        return {f: i for i, f in enumerate(self.keyframes)}

    # -- reporting -------------------------------------------------------

    def describe(self) -> str:
        n = len(self.keyframes)
        if self.space == SPACE_VIDEO:
            what = "video frame number (identity — no translation)"
        else:
            what = f"keyframe POSITION over {n} keyframes"
        line = f"mask frame space: {what} [{self.source}]"
        if self.stray:
            line += (f" ⚠ MIXED STORE: {len(self.stray)} key(s) are in the "
                     f"other space (e.g. {self.stray[:5]}) and are unreadable")
        return line

    def __repr__(self) -> str:  # pragma: no cover — debugging aid
        return f"<MaskSpace {self.space} {self.source} kf={len(self.keyframes)}>"


# ── declaration + measurement ────────────────────────────────────────────

def declared_space(masks) -> Optional[str]:
    """The space the store declares, or None for a store written before the
    declaration existed."""
    try:
        files = getattr(masks, "files", None)
        if files is None:
            if NPZ_KEY not in masks:
                return None
            raw = masks[NPZ_KEY]
        else:
            if NPZ_KEY not in files:
                return None
            raw = masks[NPZ_KEY]
        val = str(np.asarray(raw).ravel()[0])
        return val if val in SPACES else None
    except Exception:  # noqa: BLE001 — a corrupt declaration is no declaration
        return None


def measure_space(mask_frames: Sequence[int], keyframes: Optional[Sequence[int]]
                  ) -> Tuple[str, int, int, List[int], List[int]]:
    """Which space explains the store's keys. Measured, never assumed.

    Both spaces are counted against the SAME keys and the larger count wins:
    a positional store has every key in ``[0, N)``, a video-keyed one has
    every key in the keyframe list. The overlap (a video number that is also a
    valid position, e.g. 60 of 216 keyframes) is reported as ambiguous — it is
    the only thing a mixed store makes unrecoverable.

    Returns (space, positional_hits, video_hits, stray, ambiguous).
    """
    frames = [int(f) for f in mask_frames]
    if not keyframes:
        return SPACE_VIDEO, 0, 0, [], []
    n = len(keyframes)
    kfset = {int(x) for x in keyframes}
    pos_ok = {f for f in frames if 0 <= f < n}
    vid_ok = {f for f in frames if f in kfset}
    if len(pos_ok) > len(vid_ok):
        space, winner, loser = SPACE_KEYFRAME, pos_ok, vid_ok
    else:
        space, winner, loser = SPACE_VIDEO, vid_ok, pos_ok
    stray = sorted(loser - winner)
    ambiguous = sorted(pos_ok & vid_ok)
    return space, len(pos_ok), len(vid_ok), stray, ambiguous


# ── resolution (cached per store) ────────────────────────────────────────

_CACHE: Dict[str, Tuple[tuple, MaskSpace]] = {}


def _stamp(p: Path) -> tuple:
    try:
        st = p.stat()
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return (0, 0)


def resolve(output_dir, masks=None, log=None) -> MaskSpace:
    """The session's mask frame space. Cached per store file (invalidated by
    its mtime), so every call site pays for the measurement once.

    ``masks`` is an already-open NpzFile when the caller has one; otherwise
    the store is opened here.
    """
    out = Path(output_dir)
    p = out / "seg_masks.npz"
    key = str(out.resolve()) if out.exists() else str(out)
    stamp = _stamp(p)
    hit = _CACHE.get(key)
    if hit is not None and hit[0] == stamp:
        return hit[1]

    kf = keyframe_numbers(out)
    opened = None
    if masks is None and p.exists():
        try:
            masks = opened = np.load(p, allow_pickle=True)
        except Exception:  # noqa: BLE001
            masks = None
    frames = mask_frames_of(masks) if masks is not None else []

    space = declared_space(masks) if masks is not None else None
    if space is not None:
        _sp, _pos, _vid, stray, amb = measure_space(frames, kf)
        # the declaration decides; the measurement still runs so a store the
        # declaration does not fit is reported instead of silently believed
        if _sp != space:
            stray = sorted(set(frames) - (
                {f for f in frames if 0 <= f < len(kf or [])} if space == SPACE_KEYFRAME
                else {f for f in frames if f in {int(x) for x in (kf or [])}}))
        ms = MaskSpace(space, kf, "declared", frames, stray, amb)
    else:
        sp, _pos, _vid, stray, amb = measure_space(frames, kf)
        src = "measured" if kf else "no camera_frames.txt — ordinal is the frame"
        ms = MaskSpace(sp, kf, src, frames, stray, amb)

    if opened is not None:
        try:
            opened.close()
        except Exception:  # noqa: BLE001
            pass
    _CACHE[key] = (stamp, ms)
    if log is not None:
        log(ms.describe())
    return ms


def invalidate(output_dir=None) -> None:
    """Drop the cache (a writer just changed the store)."""
    if output_dir is None:
        _CACHE.clear()
        return
    out = Path(output_dir)
    _CACHE.pop(str(out.resolve()) if out.exists() else str(out), None)


# ── the writer's side: never leave two spaces in one file ────────────────

def store_space(output_dir, incoming_space: str) -> str:
    """The space the store on disk already uses, or — for a store that does
    not exist yet — the incoming one. An existing store NEVER changes space:
    converting it would rewrite every key of a file the session depends on."""
    if incoming_space not in SPACES:
        raise ValueError(f"unknown mask frame space {incoming_space!r}")
    p = Path(output_dir) / "seg_masks.npz"
    if not p.exists():
        return incoming_space
    ms = resolve(output_dir)
    if not ms.mask_frames:
        return incoming_space
    return ms.space


def convert_frames(output_dir, frames: Iterable[int],
                   src_space: str, dst_space: str) -> Dict[int, int]:
    """{frame in ``src_space``: frame in ``dst_space``} for every frame.

    Raises when a frame cannot be translated — a save that would put two
    index spaces in one file must FAIL, not half-succeed. On pccr that silent
    half-success left oid 110 keyed 0,1,2,3… and oid 213 keyed 1,60,97,…
    """
    frames = [int(f) for f in frames]
    if src_space == dst_space:
        return {f: f for f in frames}
    kf = keyframe_numbers(output_dir)
    if not kf:
        raise RuntimeError(
            f"cannot convert mask frames from {src_space} to {dst_space}: "
            f"{Path(output_dir)}/camera_frames.txt (or frame_list.json) is "
            f"missing, so the keyframe↔video-frame map is unknown")
    out: Dict[int, int] = {}
    bad: List[int] = []
    if src_space == SPACE_VIDEO:          # video number → keyframe position
        pos = {int(f): i for i, f in enumerate(kf)}
        for f in frames:
            if f in pos:
                out[f] = pos[f]
            else:
                bad.append(f)
    else:                                  # keyframe position → video number
        for f in frames:
            if 0 <= f < len(kf):
                out[f] = int(kf[f])
            else:
                bad.append(f)
    if bad:
        raise RuntimeError(
            f"cannot write these masks: frame(s) {bad[:10]}"
            f"{' …' if len(bad) > 10 else ''} are not in the session's "
            f"keyframe list ({len(kf)} keyframes), so they cannot be "
            f"expressed in the store's {dst_space} space. The store holds "
            f"one space only — mixing them makes every mask key ambiguous.")
    return out


def normalize_masks(output_dir, masks_by_frame: Dict[int, dict],
                    incoming_space: str, log=print) -> Tuple[Dict[int, dict], str]:
    """Re-key a writer's ``{frame: {oid: mask}}`` into the store's space.

    Returns (masks_in_store_space, store_space). Raises through
    :func:`convert_frames` rather than writing a mixed store.
    """
    dst = store_space(output_dir, incoming_space)
    if dst == incoming_space:
        return masks_by_frame, dst
    conv = convert_frames(output_dir, masks_by_frame.keys(), incoming_space, dst)
    out = {conv[int(f)]: v for f, v in masks_by_frame.items()}
    if log is not None:
        sample = sorted(masks_by_frame)[:3]
        log(f"[MaskSpace] incoming masks are in {incoming_space}, the store is "
            f"in {dst} — translated {len(out)} frames "
            f"({sample} → {[conv[s] for s in sample]})")
    return out, dst


def declaration(space: str) -> np.ndarray:
    """The array to store under :data:`NPZ_KEY`."""
    if space not in SPACES:
        raise ValueError(f"unknown mask frame space {space!r}")
    return np.array(space)
