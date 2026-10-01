"""precision.product after the certification: the published cloud stays LIVE when the
live epoch descends from it through TRANSFORM epochs (the certification warps the
product per keyframe and publishes the result), with or without the stored epoch
directories (`certify.single_final_epoch` deletes them; the ledger then says what each
epoch was). A new-cloud epoch above the product, or a live epoch that does not descend
from it, is not the product."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from correction.epoch import EPOCH_DIR_PREFIX, EPOCH_FILE, LEDGER_FILE     # noqa: E402
from correction.ledger import save_epoch_npz                               # noqa: E402
from precision import product as PR                                        # noqa: E402

CLOUD = 7


def _rec(epoch, parent, kind):
    return json.dumps({"epoch": epoch, "parent_epoch": parent, "kind": kind, "correction_id": f"c{epoch}"})


def _ledger(out: Path, runs):
    out.joinpath(LEDGER_FILE).write_text("\n".join(json.dumps(
        {"type": "run", "correction_id": f"c{to}", "epoch_from": frm, "epoch_to": to, "kind": kind,
         "verdict": "applied"}) for frm, to, kind in runs) + "\n")


def _session(tmp_path: Path, live: int, stored: bool, runs, live_kind="transform"):
    out = tmp_path / "output"
    out.mkdir()
    (out / "cleaned_cloud.ply").write_bytes(b"ply\n")
    (out / "corrected_cloud.json").write_text(json.dumps({"epoch_to": CLOUD, "epoch_from": CLOUD - 1}))
    (out / EPOCH_FILE).write_text(_rec(live, live - 1, live_kind))
    if stored:
        for e in range(1, live):
            d = out / f"{EPOCH_DIR_PREFIX}{e}"
            d.mkdir()
            (d / EPOCH_FILE).write_text(_rec(e, e - 1, "new_cloud" if e == CLOUD else "transform"))
    _ledger(out, runs)
    for frm, to, kind in runs:
        if kind != "fuse":
            save_epoch_npz(out, to, np.tile(np.eye(3), (2, 1, 1)), np.zeros((2, 3)), np.ones(2), [0, 10])
    return out


RUNS = [(CLOUD - 1, CLOUD, "fuse"), (CLOUD, CLOUD + 1, "floor"), (CLOUD + 1, CLOUD + 2, "certify")]


def test_the_product_is_live_at_its_own_epoch(tmp_path):
    out = _session(tmp_path, live=CLOUD, stored=True, runs=RUNS[:1], live_kind="new_cloud")
    live, why = PR.product_is_live(out)
    assert live and why == f"corrected_cloud (epoch {CLOUD}) on disk"
    assert PR.transform_descent(out, CLOUD, CLOUD) == ([], "")


def test_the_product_stays_live_through_transform_epochs_with_the_stored_dirs(tmp_path):
    out = _session(tmp_path, live=CLOUD + 2, stored=True, runs=RUNS)
    live, why = PR.product_is_live(out)
    assert live, why
    assert f"corrected_cloud (epoch {CLOUD})" in why and f"[{CLOUD + 1}, {CLOUD + 2}]" in why
    assert f"live epoch {CLOUD + 2}" in why
    assert PR.transform_descent(out, CLOUD, CLOUD + 2) == ([CLOUD + 1, CLOUD + 2], "")


def test_the_product_stays_live_once_the_stored_dirs_are_gone(tmp_path):
    """certify.single_final_epoch deleted `_epoch_<N>/`: the kinds come from the ledger,
    the ancestry from the live record (linear below it)."""
    out = _session(tmp_path, live=CLOUD + 2, stored=False, runs=RUNS)
    assert not list(out.glob(f"{EPOCH_DIR_PREFIX}*"))
    live, why = PR.product_is_live(out)
    assert live, why
    assert f"[{CLOUD + 1}, {CLOUD + 2}]" in why
    assert PR.product_report(out)["epoch_to"] == CLOUD


def test_a_new_cloud_above_the_product_is_not_the_product(tmp_path):
    runs = RUNS[:2] + [(CLOUD + 1, CLOUD + 2, "fuse")]          # a second cloud, its report lost
    out = _session(tmp_path, live=CLOUD + 2, stored=False, runs=runs, live_kind="new_cloud")
    live, why = PR.product_is_live(out)
    assert not live and f"epoch {CLOUD + 2} is a NEW CLOUD" in why
    # the same session with the live record written before `kind` existed: the ledger says
    (out / EPOCH_FILE).write_text(json.dumps({"epoch": CLOUD + 2, "parent_epoch": CLOUD + 1}))
    live, why = PR.product_is_live(out)
    assert not live and f"epoch {CLOUD + 2} is a NEW CLOUD" in why


def test_a_live_epoch_below_the_product_does_not_descend_from_it(tmp_path):
    out = _session(tmp_path, live=CLOUD - 2, stored=False, runs=RUNS[:1])
    live, why = PR.product_is_live(out)
    assert not live and "does not descend" in why


def test_the_descent_still_needs_the_cloud_on_disk(tmp_path):
    out = _session(tmp_path, live=CLOUD + 2, stored=False, runs=RUNS)
    (out / "cleaned_cloud.ply").unlink()
    live, why = PR.product_is_live(out)
    assert not live and "cleaned_cloud.ply is not on disk" in why


def test_the_pipeline_probe_and_the_cloud_stage_read_this_one_answer():
    """The reconstruction stage's resume probe keyed on product_is_live: a
    certification used to turn it false and the resume re-ran the reconstruction."""
    src = (Path(__file__).resolve().parents[1] / "pipeline_manager.py").read_text()
    assert "return product_is_live(output_dir)" in src
    assert "from precision.product import product_is_live" in (
        Path(__file__).resolve().parents[1] / "workers" / "cloudcompy_worker.py").read_text()
