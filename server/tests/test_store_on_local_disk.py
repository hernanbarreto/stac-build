"""scene_r.db worked on the pod's local disk and put back in one copy (phase_r.instance_store.
store_on_local_disk, USER 2026-11-12: "bien con el arreglo"): the writes arrive, the target's stale journal
files go, a failure leaves the target as it was, an unclosed store is refused."""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase_r.instance_store import InstanceStore, store_on_local_disk  # noqa: E402


def _rows(db):
    con = sqlite3.connect(str(db))
    try:
        return sorted(r[0] for r in con.execute("SELECT instance_id FROM instances"))
    finally:
        con.close()


def test_a_fresh_store_is_built_locally_and_lands_on_the_target(tmp_path):
    target = tmp_path / "out" / "scene_r.db"
    with store_on_local_disk(target, keep_existing=False) as local:
        assert local.parent != target.parent
        st = InstanceStore(local)
        st.upsert_instance(7, "floor", source="sam3_concepts", status="proposed", n_views=0)
        st.set_points(7, np.zeros((5, 3)))
        st.close()
    assert _rows(target) == [7]
    assert not Path(str(target) + "-wal").exists() and not Path(str(target) + "-shm").exists()


def test_an_existing_store_is_updated_and_its_journal_folded_in(tmp_path):
    target = tmp_path / "scene_r.db"
    st = InstanceStore(target)
    st.upsert_instance(1, "wall", source="sam3_concepts", status="proposed", n_views=0)
    st.commit() if hasattr(st, "commit") else None
    keep_open = InstanceStore(target)                       # a reader keeps the journal alive
    with store_on_local_disk(target, keep_existing=True) as local:
        st2 = InstanceStore(local)
        assert [r.get("instance_id") for r in st2.list_instances()] == [1]
        st2.upsert_instance(2, "door", source="sam3_concepts", status="proposed", n_views=0)
        st2.close()
    keep_open.close()
    st.close()
    assert _rows(target) == [1, 2]


def test_a_failure_leaves_the_target_untouched(tmp_path):
    target = tmp_path / "scene_r.db"
    st = InstanceStore(target)
    st.upsert_instance(1, "wall", source="sam3_concepts", status="proposed", n_views=0)
    st.close()
    before = target.read_bytes()
    with pytest.raises(ValueError):
        with store_on_local_disk(target, keep_existing=True) as local:
            st2 = InstanceStore(local)
            st2.upsert_instance(9, "x", source="sam3_concepts", status="proposed", n_views=0)
            st2.close()
            raise ValueError("boom")
    assert target.read_bytes() == before and _rows(target) == [1]


def test_an_unclosed_store_is_refused(tmp_path):
    target = tmp_path / "scene_r.db"
    with pytest.raises(RuntimeError, match="was not closed"):
        with store_on_local_disk(target, keep_existing=False) as local:
            st = InstanceStore(local)
            st.upsert_instance(3, "pipe", source="sam3_concepts", status="proposed", n_views=0)
    assert not target.exists()
