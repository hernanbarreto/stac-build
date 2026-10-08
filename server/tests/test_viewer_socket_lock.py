"""Every write to a viewer socket goes through the per-socket lock.

MEASURED 2026-09-23: the backend dropped the browser's websocket once per
session the user opened. `ERROR: keepalive ping failed` followed by websockets'
`_drain_helper` assertion — `assert waiter is None or waiter.cancelled()` — which
fires when TWO coroutines drain the same transport at once. `ViewerManager`
already held a lock per socket and used it in `broadcast_text`; the handler had
27 writes that went straight to the socket and raced those broadcasts.

The lock is not decoration, so this is a structural test: a direct
`websocket.send_text(...)` / `send_bytes(...)` inside a function that writes to a
VIEWER socket fails here, not once a user opens a big scene.

Parsed from source: importing main pulls the whole server.
"""
import ast
import pathlib

import pytest

MAIN = pathlib.Path(__file__).resolve().parent.parent / "main.py"

# Functions that hold or receive a viewer socket. `ws_team` and the log socket
# are OTHER endpoints with their own sockets and are deliberately not listed.
VIEWER_FUNCS = (
    "viewer_websocket",
    "_send_cleaned_cloud",
    "_send_sabana_cloud",
    # `_run_cloudcompy_postprocess_inner` (the on-load cloud rebuild) was REMOVED on
    # 2026-10-08 — docs/plan_determinismo.md points 111 / 158: opening a session never
    # builds; the cloud stage is ordered as a pipeline job instead
)


@pytest.fixture(scope="module")
def tree():
    return ast.parse(MAIN.read_text())


def _func(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {MAIN} — renamed? update this test")


@pytest.mark.parametrize("name", VIEWER_FUNCS)
def test_no_unlocked_write_to_a_viewer_socket(tree, name):
    fn = _func(tree, name)
    offenders = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if (isinstance(f, ast.Attribute)
                and f.attr in ("send_text", "send_bytes")
                and isinstance(f.value, ast.Name)
                and f.value.id == "websocket"):
            offenders.append(f"line {node.lineno}: websocket.{f.attr}(...)")
    assert not offenders, (
        f"{name} writes to the viewer socket without the lock:\n  "
        + "\n  ".join(offenders)
        + "\nUse viewer_manager.send_text(websocket, ...) / send_bytes(websocket, ...) — "
          "two concurrent drains kill the keepalive and the browser loses the socket.")


def test_the_manager_serialises_and_never_drops_silently(tree):
    """The lock must be taken for EVERY socket, registered or not.

    The first version returned early when the socket was not in `self.locks`,
    which dropped the message with no trace — a silent failure is the one kind
    this repo does not accept.
    """
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "ViewerManager")
    names = {n.name for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert "_lock_for" in names, "ViewerManager lost its on-demand lock helper"

    for meth in ("send_text", "send_bytes"):
        fn = next(n for n in cls.body
                  if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == meth)
        assert any(isinstance(n, ast.AsyncWith) for n in ast.walk(fn)), (
            f"ViewerManager.{meth} no longer writes under the lock")
        # No early return ANYWHERE: the old guard was `if websocket not in
        # self.locks: return`, nested in an `if`, so a check over fn.body alone
        # would have missed exactly the bug it is meant to catch.
        returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
        assert not returns, (
            f"ViewerManager.{meth} returns instead of writing (line "
            f"{returns[0].lineno}) — that is the silent drop this test exists "
            f"to prevent")
