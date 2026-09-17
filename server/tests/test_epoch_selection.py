"""Every epoch lives; you only choose which one is shown (USER 2026-09-16).

*"el accept y undo no sirven para nada, porque en realidad deben quedar épocas
que deben ser seleccionables para verificación visual, nada más… todas viven,
solo se seleccionan y la que se selecciona se muestra"*.

What this replaced: Approve deleted every previous epoch and Undo deleted the
current one, so a session could only ever hold two states and choosing wrong
destroyed the other. On pccr 2026-09-15 two accidental Undos cost epochs 2 and
3 — *"no presioné dos veces, apareció por error, es muy confuso"*.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from correction.apply import (MANIFEST_NAME, PREV_PREFIX,  # noqa: E402
                              available_epochs, next_epoch, select_epoch,
                              session_artifacts)
from correction.epoch import epoch_lineage, epoch_path   # noqa: E402


def _epoch_dir(out: Path, e: int, arts, payload: str):
    d = out / f"{PREV_PREFIX}{e}"
    d.mkdir(parents=True)
    for rel in arts:
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text(payload if rel != "geometry_epoch.json"
                             else json.dumps({"epoch": e}))
    (d / MANIFEST_NAME).write_text(json.dumps(
        {"epoch_from": e - 1, "epoch_to": e,
         "artifacts": [{"rel": r, "existed_before": True} for r in arts]}))


@pytest.fixture()
def session(tmp_path):
    """A session live at epoch 3, with 0, 1 and 2 stored beside it."""
    out = tmp_path / "output"
    out.mkdir()
    arts = ["cleaned_cloud.ply", "camera_poses.txt", "geometry_epoch.json"]
    for rel in arts:
        (out / rel).write_text("live-3" if rel != "geometry_epoch.json"
                               else json.dumps({"epoch": 3}))
    for e in (0, 1, 2):
        _epoch_dir(out, e, arts, f"state-{e}")
    return out


def test_every_epoch_is_listed_and_one_is_live(session):
    got = available_epochs(session)
    assert [e["epoch"] for e in got] == [0, 1, 2, 3]
    assert [e["live"] for e in got] == [False, False, False, True]


def test_selecting_shows_that_epoch_and_keeps_the_others(session):
    res = select_epoch(session, 1, log=lambda m: None)
    assert res["changed"] and res["epoch"] == 1

    assert (session / "cleaned_cloud.ply").read_text() == "state-1"
    assert json.loads((session / "geometry_epoch.json").read_text())["epoch"] == 1
    # nothing was destroyed: the state we left is stored, the rest untouched
    assert sorted(e["epoch"] for e in available_epochs(session)) == [0, 1, 2, 3]
    assert (session / f"{PREV_PREFIX}3" / "cleaned_cloud.ply").read_text() == "live-3"
    assert (session / f"{PREV_PREFIX}0" / "cleaned_cloud.ply").read_text() == "state-0"
    assert (session / f"{PREV_PREFIX}2" / "cleaned_cloud.ply").read_text() == "state-2"


def test_you_can_go_back_up_again(session):
    """The move that Undo made impossible: down to 1, then back to 3."""
    select_epoch(session, 1, log=lambda m: None)
    select_epoch(session, 3, log=lambda m: None)
    assert (session / "cleaned_cloud.ply").read_text() == "live-3"
    assert sorted(e["epoch"] for e in available_epochs(session)) == [0, 1, 2, 3]


def test_every_epoch_survives_a_full_tour(session):
    for e in (0, 2, 1, 3, 0):
        select_epoch(session, e, log=lambda m: None)
        assert json.loads((session / "geometry_epoch.json").read_text())["epoch"] == e
        assert sorted(x["epoch"] for x in available_epochs(session)) == [0, 1, 2, 3]


def test_selecting_the_live_epoch_does_nothing(session):
    res = select_epoch(session, 3, log=lambda m: None)
    assert res["changed"] is False
    assert (session / "cleaned_cloud.ply").read_text() == "live-3"


def test_an_epoch_that_is_not_there_is_refused_by_name(session):
    with pytest.raises(RuntimeError) as e:
        select_epoch(session, 7, log=lambda m: None)
    assert "7" in str(e.value)
    assert (session / "cleaned_cloud.ply").read_text() == "live-3"


def test_stored_epochs_above_the_live_one_are_still_listed(session):
    """The bug a contiguous walk down from the current epoch would cause: after
    showing epoch 1, epochs 2 and 3 sit ABOVE it and must not disappear from
    the list."""
    select_epoch(session, 1, log=lambda m: None)
    got = [e["epoch"] for e in available_epochs(session)]
    assert 2 in got and 3 in got


def test_an_artifact_a_later_epoch_introduced_travels_too(tmp_path):
    """A file that only exists from epoch 2 on (the `depth_correction.json` of
    a depth correction, a floor transform) must LEAVE when epoch 1 is shown.

    Selecting used to move only the artifacts named in the chosen epoch's own
    manifest, so the sidecar of the newest epoch stayed live over older
    geometry: the cloud of epoch 1 with the depth of epoch 3."""
    out = tmp_path / "output"
    out.mkdir()
    base = ["cleaned_cloud.ply", "geometry_epoch.json"]
    # epoch 0 → 1 moved only the base artifacts
    _epoch_dir(out, 0, base, "state-0")
    # epoch 1 → 2 introduced the sidecar (epoch 1 never had it)
    _epoch_dir(out, 1, base, "state-1")
    # live: epoch 2, sidecar included
    for rel in base:
        (out / rel).write_text("live-2" if rel != "geometry_epoch.json"
                               else json.dumps({"epoch": 2}))
    (out / "depth_correction.json").write_text('{"k": {"7": 0.95}}')
    # the manifest of the epoch we are living in names the sidecar, because the
    # swap that produced epoch 2 staged it
    d = out / f"{PREV_PREFIX}1"
    man = json.loads((d / MANIFEST_NAME).read_text())
    man["artifacts"].append({"rel": "depth_correction.json",
                             "existed_before": False})
    (d / MANIFEST_NAME).write_text(json.dumps(man))

    assert "depth_correction.json" in [a["rel"] for a in session_artifacts(out)]
    select_epoch(out, 0, log=lambda m: None)
    assert (out / "cleaned_cloud.ply").read_text() == "state-0"
    assert not (out / "depth_correction.json").exists(), \
        "the depth of a later epoch stayed on the geometry of epoch 0"
    # and it comes back with its own epoch
    select_epoch(out, 2, log=lambda m: None)
    assert (out / "depth_correction.json").exists()
    assert (out / "cleaned_cloud.ply").read_text() == "live-2"


def test_the_transforms_never_travel(session):
    """`corrections/epoch_<N>.npz` is history, not state: the store follows the
    geometry with it and `correction.replay` reproduces any epoch, so it stays
    live whichever epoch is shown."""
    assert not any(a["rel"].startswith("corrections/")
                   for a in session_artifacts(session))


# ── branching: a correction runs on top of the epoch being SHOWN ────────────
# Selecting an older epoch and correcting from there is the normal flow now,
# so the history is a tree, not a line.

def _state(out: Path, epoch: int, parent, live: bool):
    """Write the geometry record of one epoch (live, or stored in its dir)."""
    rec = json.dumps({"epoch": epoch, "parent_epoch": parent})
    if live:
        (out / "geometry_epoch.json").write_text(rec)
    else:
        d = out / f"{PREV_PREFIX}{epoch}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "geometry_epoch.json").write_text(rec)
        (d / MANIFEST_NAME).write_text(json.dumps(
            {"epoch": epoch, "artifacts": [{"rel": "cleaned_cloud.ply"}]}))


@pytest.fixture()
def branched(tmp_path):
    """0 → 1 → 2 → 3, then epoch 1 was selected and corrected → 4 (live)."""
    out = tmp_path / "output"
    out.mkdir()
    for e, par in ((0, None), (1, 0), (2, 1), (3, 2)):
        _state(out, e, par, live=False)
    _state(out, 4, 1, live=True)
    return out


def test_a_new_epoch_never_reuses_a_number(branched):
    """`current + 1` would have called the new correction "epoch 2" while a
    different epoch 2 was on disk: same directory, same `epoch_2.npz`, and a
    ledger saying two things about one number."""
    assert next_epoch(branched) == 5
    (branched / "corrections").mkdir()
    (branched / "corrections" / "epoch_9.npz").write_bytes(b"")
    assert next_epoch(branched) == 10


def test_the_store_follows_the_real_line_not_the_arithmetic(branched):
    """From 4 (child of 1) to 3 (child of 2): undo 4, then apply 2 and 3.
    Arithmetic would have undone 4 — and nothing else."""
    assert epoch_lineage(branched, 4) == [0, 1, 4]
    assert epoch_lineage(branched, 3) == [0, 1, 2, 3]
    assert epoch_path(branched, 4, 3) == [(4, True), (2, False), (3, False)]
    assert epoch_path(branched, 3, 4) == [(3, True), (2, True), (4, False)]
    assert epoch_path(branched, 4, 1) == [(4, True)]


def test_a_session_that_never_branched_is_the_old_arithmetic(session):
    """No `parent_epoch` recorded anywhere (a session written earlier): the
    path must be exactly what the linear code did."""
    assert epoch_path(session, 3, 1) == [(3, True), (2, True)]
    assert epoch_path(session, 1, 3) == [(2, False), (3, False)]
    assert epoch_path(session, 3, 0) == [(3, True), (2, True), (1, True)]


def test_approve_and_undo_are_gone_from_the_whole_system():
    """Not only from the correction module: the certification kit approved and
    undid through the same function, and a leftover caller is a 500 waiting for
    the first user who presses the button."""
    root = Path(__file__).resolve().parents[1]
    src = (root / "correction" / "run.py").read_text()
    assert "def run_verdict(" not in src, "the approve/undo verdict is back"
    assert "def run_select(" in src
    apply_src = (root / "correction" / "apply.py").read_text()
    for gone in ("def approve_swap(", "def undo_swap(", "def pending_prev_dirs("):
        assert gone not in apply_src, f"{gone} is back"
    for api_rel in ("correction/api.py", "reconstruction/certify/api.py"):
        api = (root / api_rel).read_text()
        assert '/select' in api, api_rel
        assert '@router.post("/approve")' not in api, api_rel
        assert '@router.post("/undo")' not in api, api_rel
    # and nothing anywhere calls what no longer exists
    gone_calls = ("run_verdict", "approve_swap", "undo_swap", "pending_prev_dirs",
                  "ledger.pending_run")
    for py in root.glob("**/*.py"):
        if py.name == Path(__file__).name or "__pycache__" in py.parts:
            continue
        text = py.read_text(errors="ignore")
        for pat in gone_calls:
            assert pat not in text, f"{py.relative_to(root)} still calls {pat}"
