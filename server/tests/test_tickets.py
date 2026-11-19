"""Tickets: a point of the model marked as a review item or a non-conformity.

USER 2026-11-09: "TICKETS — una herramienta simple para marcar tickets en cualquier parte del
proyecto, como un punto de revisión o una no conformidad: es para mostrar el concepto nomás".
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tickets as T  # noqa: E402


def test_a_ticket_is_marked_numbered_and_kept(tmp_path):
    a = T.create(tmp_path, kind="non_conformity", title="  The tray is not earthed ", position=[1.5, 0.2, -3.0], author="ana",
                 description="no bonding jumper at the joint", scan="2026-08-31/default", geometry_epoch=4,
                 instance_id=432, label="equipment_rack")
    b = T.create(tmp_path, kind="review", title="Check the clearance to the door", position=(0, 1, 2), author="luis")
    assert (a["number"], b["number"]) == (1, 2) and a["id"] != b["id"]
    assert a["title"] == "The tray is not earthed" and a["status"] == "open" and a["provenance"] == "human_stated"
    assert a["object"] == {"instance_id": 432, "label": "equipment_rack"} and b["object"] is None and b["scan"] is None
    assert a["position"] == [1.5, 0.2, -3.0] and a["created_by"] == "ana" and a["created_at"].endswith("Z")
    doc = json.loads((tmp_path / T.FILE).read_text())
    assert doc["version"] == 1 and doc["next_number"] == 3 and [t["number"] for t in doc["tickets"]] == [1, 2]
    assert not list(tmp_path.glob("*.part"))


def test_what_is_not_a_ticket_is_refused_and_nothing_is_written(tmp_path):
    ok = dict(kind="review", title="t", position=[0, 0, 0], author="ana")
    for change, why in (({"kind": "bug"}, "kind must be one of"), ({"title": "   "}, "needs a title"),
                        ({"title": "x" * 201}, "longer than 200"), ({"position": [0, 1]}, "three finite numbers"),
                        ({"position": [0, float("nan"), 1]}, "three finite numbers"), ({"position": [0, True, 1]}, "three finite numbers"),
                        ({"author": ""}, "needs a author"), ({"instance_id": 0}, "id of a segmented object"),
                        ({"description": 5}, "must be text")):
        with pytest.raises(T.TicketError, match=why):
            T.create(tmp_path, **{**ok, **change})
    assert not (tmp_path / T.FILE).exists()


def test_resolving_records_who_and_when_and_reopening_clears_it(tmp_path):
    t = T.create(tmp_path, kind="review", title="Look at this", position=[0, 0, 0], author="ana")
    r = T.update(tmp_path, t["id"], {"status": "resolved"}, "luis")
    assert r["status"] == "resolved" and r["resolved_by"] == "luis" and r["resolved_at"] and r["created_by"] == "ana"
    again = T.update(tmp_path, str(t["number"]), {"status": "resolved", "title": "Looked at"}, "eva")      # by its number too
    assert again["resolved_by"] == "luis" and again["title"] == "Looked at"   # (resolved already: who resolved it stays)
    o = T.update(tmp_path, t["id"], {"status": "open", "kind": "non_conformity", "description": "worse than it looked"}, "eva")
    assert o["status"] == "open" and o["resolved_by"] is None and o["resolved_at"] is None and o["kind"] == "non_conformity"
    with pytest.raises(T.TicketError, match="cannot be changed"):
        T.update(tmp_path, t["id"], {"position": [1, 1, 1]}, "eva")           # a ticket is where it was marked
    with pytest.raises(T.TicketError, match="status must be"):
        T.update(tmp_path, t["id"], {"status": "closed"}, "eva")
    with pytest.raises(KeyError):
        T.update(tmp_path, "nope", {"title": "x"}, "eva")


def test_a_removed_tickets_number_is_not_given_again(tmp_path):
    a = T.create(tmp_path, kind="review", title="a", position=[0, 0, 0], author="ana")
    T.create(tmp_path, kind="review", title="b", position=[0, 0, 0], author="ana")
    gone = T.delete(tmp_path, a["id"], "eva")
    assert gone["number"] == 1 and gone["status"] == "deleted" and gone["deleted_by"] == "eva"
    c = T.create(tmp_path, kind="review", title="c", position=[0, 0, 0], author="ana")
    # a deleted ticket STAYS, as history (USER 2026-11-19: "si se elimino deberia quedar como historico no?")
    kept = T.load(tmp_path)["tickets"]
    assert c["number"] == 3 and [(t["number"], t["status"]) for t in kept] == [(1, "deleted"), (2, "open"), (3, "open")]
    assert [h["action"] for h in kept[0]["history"]] == ["created", "deleted"] and kept[0]["history"][-1]["by"] == "eva"
    assert T.listing(tmp_path)["counts"]["open"] == 2 and T.listing(tmp_path)["counts"]["deleted"] == 1
    with pytest.raises(T.TicketError, match="deleted"):
        T.update(tmp_path, a["id"], {"status": "open"}, "eva")
    with pytest.raises(KeyError):
        T.delete(tmp_path, a["id"], "eva")


def test_the_listing_counts_and_tells_what_the_geometry_left_behind(tmp_path):
    assert T.listing(tmp_path) == {"tickets": [], "counts": {"open": 0, "resolved": 0, "deleted": 0, "review": 0, "non_conformity": 0}}
    a = T.create(tmp_path, kind="non_conformity", title="a", position=[0, 0, 0], author="ana", scan="2026-08-31/default", geometry_epoch=4)
    b = T.create(tmp_path, kind="review", title="b", position=[0, 0, 0], author="ana", scan="2026-08-24/default", geometry_epoch=2)
    c = T.create(tmp_path, kind="review", title="c", position=[0, 0, 0], author="ana")
    T.update(tmp_path, c["id"], {"status": "resolved"}, "ana")
    out = T.listing(tmp_path, {"2026-08-31/default": 5, "2026-08-24/default": 2})
    assert out["counts"] == {"open": 2, "resolved": 1, "deleted": 0, "review": 1, "non_conformity": 1}
    assert {t["number"]: t["stale"] for t in out["tickets"]} == {a["number"]: True, b["number"]: False, c["number"]: False}
    (tmp_path / T.FILE).write_text(json.dumps({"version": 99, "tickets": []}))
    with pytest.raises(T.TicketError, match="not a ticket document"):
        T.load(tmp_path)


# ── the routes ─────────────────────────────────────────────────────────────────────────────────
@pytest.fixture()
def client(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import tickets_api as R
    from correction.epoch import EPOCH_FILE
    project = tmp_path / "projects" / "site"
    outs = {}
    for scan in ("2026-08-24/default", "2026-08-31/default"):
        day, source = scan.split("/")
        outs[scan] = project / "scans" / day / f"src_{source}" / "output"
        outs[scan].mkdir(parents=True)
        (outs[scan] / EPOCH_FILE).write_text(json.dumps({"epoch": 4}))

    class Ctx:
        def __init__(self, scan):
            self.project_dir = project
            self.output_dir = outs[scan]
            self.session_dir = outs[scan].parent

    def resolve(session_id, scan=None):
        if session_id != "site":
            return type("Missing", (), {"session_dir": tmp_path / "nope", "output_dir": tmp_path / "nope" / "output"})()
        if scan is not None and scan not in outs:
            raise FileNotFoundError(f"no scan {scan}")
        return Ctx(scan or "2026-08-24/default")

    log = []
    R.configure(resolve, audit_log=lambda *a, **k: log.append((a, k)))
    app = FastAPI()
    app.include_router(R.router)
    who = {"user": {"username": "ana", "role": "viewer"}}
    app.dependency_overrides[R._user] = lambda: who["user"]
    return TestClient(app), who, outs, log, project


def test_tickets_through_the_routes(client):
    c, who, outs, log, project = client
    assert c.get("/api/tickets/site").json() == {"ok": True, "tickets": [], "counts": {"open": 0, "resolved": 0, "deleted": 0, "review": 0, "non_conformity": 0}}
    assert c.get("/api/tickets/elsewhere").status_code == 404
    r = c.post("/api/tickets/site", json={"kind": "non_conformity", "title": "Loose cover", "position": [1, 2, 3],
                                          "scan": "2026-08-31/default", "instance_id": 7, "label": "panel"})
    assert r.status_code == 200, r.text
    t = r.json()["ticket"]
    assert t["number"] == 1 and t["created_by"] == "ana" and t["geometry_epoch"] == 4 and t["stale"] is False
    assert r.json()["counts"]["non_conformity"] == 1 and (project / "tickets.json").exists()      # the PROJECT's
    assert c.post("/api/tickets/site", json={"kind": "review", "title": "", "position": [1, 2, 3]}).status_code == 400
    assert c.post("/api/tickets/site", json={"kind": "review", "title": "x", "position": [1, 2, 3], "scan": "2030-01-01/x"}).status_code == 404
    # the scan's geometry moves on: the ticket is told to be of the old one
    from correction.epoch import EPOCH_FILE
    (outs["2026-08-31/default"] / EPOCH_FILE).write_text(json.dumps({"epoch": 5}))
    assert c.get("/api/tickets/site").json()["tickets"][0]["stale"] is True
    # resolved by another user, by its number
    who["user"] = {"username": "luis", "role": "viewer"}
    r = c.patch("/api/tickets/site/1", json={"status": "resolved"})
    assert r.status_code == 200 and r.json()["ticket"]["resolved_by"] == "luis" and r.json()["counts"] == {"open": 0, "resolved": 1, "deleted": 0, "review": 0, "non_conformity": 0}
    assert c.patch("/api/tickets/site/1", json={"position": [0, 0, 0]}).status_code == 400
    assert c.patch("/api/tickets/site/99", json={"title": "x"}).status_code == 404
    # removed by its author, a manager or an administrator — not by another viewer
    assert c.delete(f"/api/tickets/site/{t['id']}").status_code == 403
    who["user"] = {"username": "eva", "role": "manager"}
    assert c.delete(f"/api/tickets/site/{t['id']}").json()["removed"] == t["id"]
    assert c.delete(f"/api/tickets/site/{t['id']}").status_code == 404
    assert [a[0][0] for a in log] == ["ticket.create", "ticket.update", "ticket.delete"]


def test_marking_and_changing_need_a_signed_in_user(client):
    c, who, outs, log, project = client
    who["user"] = None
    assert c.get("/api/tickets/site").status_code == 200                      # reading is anyone's who opens the session
    assert c.post("/api/tickets/site", json={"kind": "review", "title": "x", "position": [0, 0, 0]}).status_code == 401
    assert c.patch("/api/tickets/site/1", json={"status": "resolved"}).status_code == 401
    assert c.delete("/api/tickets/site/1").status_code == 401
    assert not (project / "tickets.json").exists()
