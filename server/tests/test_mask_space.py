"""The mask store has ONE frame space, it says which, and nobody guesses.

Three index spaces meet in a session (keyframe position, real video frame
number, SAM3's sequential index) and until 2026-09-18 nothing translated:
fourteen readers handed a mask key straight to ``cam.pose_map``, and three
writers upserted into the same npz with two different conventions. On pccr
that meant the mask witness — the point ``status`` painted in the kit — was
computed from 13 of 216 keyframes, and each of those 13 was matched to the
WRONG keyframe's masks (60 is both a valid position and a valid video
number).

These tests pin the two rules: the store declares its space, and a writer in
the other space is translated or refused.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmentation import mask_space  # noqa: E402

# a walk of 6 keyframes: video frames 1, 60, 97, 99, 116, 127 (pccr's real
# head). Positions 0..5 — and 1 is a valid number in BOTH spaces.
KEYFRAMES = [1, 60, 97, 99, 116, 127]


def _session(tmp_path: Path, mask_frames, declare=None, extra=None):
    out = tmp_path / "output"
    out.mkdir(parents=True, exist_ok=True)
    (out / "camera_frames.txt").write_text("\n".join(str(f) for f in KEYFRAMES) + "\n")
    data = {f"f{f}_o0": np.ones((4, 4), np.uint8) for f in mask_frames}
    data["frames"] = np.array(sorted(mask_frames), np.int32)
    data["obj_ids"] = np.array([0], np.int32)
    data["scaled_res"] = np.array([4, 4], np.int32)
    if declare is not None:
        data[mask_space.NPZ_KEY] = mask_space.declaration(declare)
    if extra:
        data.update(extra)
    np.savez_compressed(out / "seg_masks.npz", **data)
    mask_space.invalidate(out)
    return out


# ── measurement (legacy stores, written before the declaration existed) ──

def test_positional_store_is_measured_not_assumed(tmp_path):
    out = _session(tmp_path, range(len(KEYFRAMES)))
    ms = mask_space.resolve(out)
    assert ms.space == mask_space.SPACE_KEYFRAME
    assert ms.source == "measured"
    assert ms.to_cloud(0) == 1 and ms.to_cloud(5) == 127
    assert ms.to_mask(97) == 2
    assert ms.key(99, 7) == "f3_o7"


def test_video_keyed_store_is_measured_as_identity(tmp_path):
    out = _session(tmp_path, KEYFRAMES)
    ms = mask_space.resolve(out)
    assert ms.space == mask_space.SPACE_VIDEO
    assert ms.is_identity
    assert ms.to_cloud(97) == 97 and ms.to_mask(97) == 97
    assert ms.cloud_to_mask() == {}          # the legacy contract


def test_the_ambiguous_frames_are_the_ones_that_matched_by_accident(tmp_path):
    """Frame 1 is keyframe 0 AND video frame 1. Those coincidences are exactly
    what made the old comparison 'work' on a handful of frames."""
    out = _session(tmp_path, range(len(KEYFRAMES)))
    ms = mask_space.resolve(out)
    assert ms.ambiguous == [1]


def test_a_frame_that_is_not_a_keyframe_has_no_mask(tmp_path):
    out = _session(tmp_path, range(len(KEYFRAMES)))
    ms = mask_space.resolve(out)
    assert ms.to_mask(42) is None
    assert ms.key(42, 0) is None


def test_no_keyframe_sidecar_means_the_ordinal_is_the_frame(tmp_path):
    out = tmp_path / "output"
    out.mkdir(parents=True)
    np.savez_compressed(out / "seg_masks.npz",
                        **{"f0_o0": np.ones((2, 2), np.uint8)})
    mask_space.invalidate(out)
    ms = mask_space.resolve(out)
    assert ms.space == mask_space.SPACE_VIDEO
    assert ms.to_cloud(7) == 7 and ms.to_mask(7) == 7


# ── declaration ─────────────────────────────────────────────────────────

def test_the_declaration_wins_over_the_measurement(tmp_path):
    """Keys 0..5 measure as positions; a store that DECLARES video space is
    believed (and its unexplained keys are reported, not swallowed)."""
    out = _session(tmp_path, range(len(KEYFRAMES)), declare=mask_space.SPACE_VIDEO)
    ms = mask_space.resolve(out)
    assert ms.space == mask_space.SPACE_VIDEO
    assert ms.source == "declared"


def test_declaration_key_does_not_look_like_a_mask(tmp_path):
    """Half the readers iterate npz.files filtering on key.startswith('f') —
    the declaration must not land in that net."""
    assert not mask_space.NPZ_KEY.startswith("f")
    out = _session(tmp_path, range(3), declare=mask_space.SPACE_KEYFRAME)
    z = np.load(out / "seg_masks.npz")
    assert mask_space.mask_frames_of(z) == [0, 1, 2]


# ── the mixed store ─────────────────────────────────────────────────────

def test_a_mixed_store_is_detected_and_named(tmp_path):
    """pccr 2026-09-14: oid 110 keyed 0,1,2,3… and oid 213 keyed 1,60,97,…
    in one file. The majority space wins and the keys in the OTHER one are
    unreadable — they must be named, not silently taken for the winner's."""
    out = _session(tmp_path, [0, 2, 60, 97, 99, 116])
    ms = mask_space.resolve(out)
    assert ms.space == mask_space.SPACE_VIDEO          # 4 video numbers vs 2 positions
    assert ms.mixed
    assert ms.stray == [0, 2]                          # positions, not video frames
    assert "MIXED STORE" in ms.describe()


def test_a_clean_positional_store_is_not_called_mixed(tmp_path):
    out = _session(tmp_path, range(len(KEYFRAMES)))
    assert not mask_space.resolve(out).mixed


# ── the writer's side ───────────────────────────────────────────────────

def test_an_existing_store_never_changes_space(tmp_path):
    out = _session(tmp_path, range(len(KEYFRAMES)), declare=mask_space.SPACE_KEYFRAME)
    assert mask_space.store_space(out, mask_space.SPACE_VIDEO) == mask_space.SPACE_KEYFRAME
    assert mask_space.store_space(out, mask_space.SPACE_KEYFRAME) == mask_space.SPACE_KEYFRAME


def test_a_new_store_adopts_the_writer_space(tmp_path):
    out = tmp_path / "output"
    out.mkdir(parents=True)
    assert mask_space.store_space(out, mask_space.SPACE_VIDEO) == mask_space.SPACE_VIDEO


def test_video_masks_are_translated_into_a_positional_store(tmp_path):
    """The interactive and Resume paths hold REAL frame numbers; the batch
    pipeline's store is positional. Translate — do not append."""
    out = _session(tmp_path, range(len(KEYFRAMES)), declare=mask_space.SPACE_KEYFRAME)
    incoming = {97: {"a": 1}, 127: {"b": 2}}
    got, space = mask_space.normalize_masks(out, incoming, mask_space.SPACE_VIDEO,
                                            log=None)
    assert space == mask_space.SPACE_KEYFRAME
    assert sorted(got) == [2, 5]


def test_positional_masks_are_translated_into_a_video_store(tmp_path):
    out = _session(tmp_path, KEYFRAMES, declare=mask_space.SPACE_VIDEO)
    got, space = mask_space.normalize_masks(out, {0: {}, 3: {}},
                                            mask_space.SPACE_KEYFRAME, log=None)
    assert space == mask_space.SPACE_VIDEO
    assert sorted(got) == [1, 99]


def test_an_untranslatable_frame_fails_the_save(tmp_path):
    """A save that cannot be expressed in the store's space must FAIL. Half
    of it landing under the other convention is what corrupted pccr."""
    out = _session(tmp_path, range(len(KEYFRAMES)), declare=mask_space.SPACE_KEYFRAME)
    with pytest.raises(RuntimeError) as e:
        mask_space.normalize_masks(out, {5000: {}}, mask_space.SPACE_VIDEO, log=None)
    assert "5000" in str(e.value)


def test_conversion_without_a_keyframe_map_fails_loudly(tmp_path):
    out = tmp_path / "output"
    out.mkdir(parents=True)
    with pytest.raises(RuntimeError) as e:
        mask_space.convert_frames(out, [0], mask_space.SPACE_KEYFRAME,
                                  mask_space.SPACE_VIDEO)
    assert "camera_frames.txt" in str(e.value)


# ── the third caller shape: a keyframe position from the UI ─────────────

def test_keyframe_position_reaches_both_store_kinds(tmp_path):
    pos = _session(tmp_path / "a", range(len(KEYFRAMES)), declare=mask_space.SPACE_KEYFRAME)
    vid = _session(tmp_path / "b", KEYFRAMES, declare=mask_space.SPACE_VIDEO)
    assert mask_space.resolve(pos).key_of_keyframe(2, 0) == "f2_o0"
    assert mask_space.resolve(vid).key_of_keyframe(2, 0) == "f97_o0"


# ── the cache follows the file ──────────────────────────────────────────

def test_the_space_is_remeasured_after_the_store_changes(tmp_path):
    out = _session(tmp_path, range(len(KEYFRAMES)))
    assert mask_space.resolve(out).space == mask_space.SPACE_KEYFRAME
    _session(tmp_path, KEYFRAMES)          # rewritten in the other space
    assert mask_space.resolve(out).space == mask_space.SPACE_VIDEO


# ── the real writer: one file, one space, whoever calls ─────────────────

def _save(out, masks, space, labels=None):
    from segmentation.pipeline import _save_masks
    labels = labels or {o: "thing" for fm in masks.values() for o in fm}
    cfg = {"visualization": {"segment_colors": [[1, 2, 3], [4, 5, 6]]}}
    return _save_masks(out, masks, ["thing"], labels, cfg, frame_space=space)


def _store_keys(out):
    z = np.load(out / "seg_masks.npz")
    return sorted(k for k in z.files if k.startswith("f") and "_o" in k)


def test_the_batch_and_the_interactive_writer_share_one_space(tmp_path):
    """The exact sequence that corrupted pccr: the batch pipeline saves
    keyframe POSITIONS, then a Resume saves the same session in REAL frame
    numbers through the same upserting writer."""
    out = tmp_path / "output"
    out.mkdir(parents=True)
    (out / "camera_frames.txt").write_text("\n".join(str(f) for f in KEYFRAMES) + "\n")
    (tmp_path / "frames").mkdir()
    m = np.ones((4, 4), np.uint8)

    _save(out, {0: {0: m}, 1: {0: m}}, mask_space.SPACE_KEYFRAME)
    assert _store_keys(out) == ["f0_o0", "f1_o0"]

    # the interactive path holds video frame 97 = keyframe 2
    _save(out, {97: {1: m}}, mask_space.SPACE_VIDEO)
    assert _store_keys(out) == ["f0_o0", "f1_o0", "f2_o1"]

    ms = mask_space.resolve(out)
    assert ms.source == "declared" and ms.space == mask_space.SPACE_KEYFRAME
    assert not ms.mixed


def test_the_store_declares_itself_after_a_save(tmp_path):
    out = tmp_path / "output"
    out.mkdir(parents=True)
    (out / "camera_frames.txt").write_text("\n".join(str(f) for f in KEYFRAMES) + "\n")
    (tmp_path / "frames").mkdir()
    _save(out, {60: {0: np.ones((4, 4), np.uint8)}}, mask_space.SPACE_VIDEO)
    z = np.load(out / "seg_masks.npz")
    assert mask_space.NPZ_KEY in z.files
    assert mask_space.declared_space(z) == mask_space.SPACE_VIDEO


def test_the_writer_must_declare_its_space(tmp_path):
    from segmentation.pipeline import _save_masks
    with pytest.raises(TypeError):
        _save_masks(tmp_path, {}, [], {}, {})


# ── the site the user actually sees: the point status ───────────────────

def test_the_mask_witness_reads_every_keyframe_not_the_coincidences(tmp_path):
    """``label_maps`` must come out keyed by the CLOUD frame, because that is
    what ``compute_mask_votes`` compares against the pose/depth dict.

    Before the translation the two were compared raw: on pccr that left the
    mask witness — the green/yellow/red the kit paints — with 13 usable
    keyframes of 216, each carrying the WRONG keyframe's masks.
    """
    from reconstruction.witness.mask_votes import MaskStore, label_maps

    out = _session(tmp_path, range(len(KEYFRAMES)), declare=mask_space.SPACE_KEYFRAME)
    store = MaskStore(
        masks={f"f{i}_o0": np.ones((4, 4), np.uint8) for i in range(len(KEYFRAMES))},
        res=(4, 4), obj_of={1: 0}, space=mask_space.resolve(out))
    maps, present = label_maps(store, [1], 0)

    assert sorted(maps) == sorted(KEYFRAMES)      # not 0..5
    assert len(maps) == len(KEYFRAMES)            # every keyframe, not the 1 coincidence
    assert all(1 in s for s in present.values())


def test_the_mask_witness_drops_a_mixed_store_stray_instead_of_misreading_it(tmp_path):
    from reconstruction.witness.mask_votes import MaskStore, label_maps

    out = _session(tmp_path, range(len(KEYFRAMES)), declare=mask_space.SPACE_KEYFRAME)
    store = MaskStore(
        masks={"f0_o0": np.ones((4, 4), np.uint8),
               "f900_o0": np.ones((4, 4), np.uint8)},   # no such keyframe
        res=(4, 4), obj_of={1: 0}, space=mask_space.resolve(out))
    maps, _present = label_maps(store, [1], 0)
    assert sorted(maps) == [1]
