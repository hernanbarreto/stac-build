"""The UI's polls never reach the log; a failing endpoint always does.

USER 2026-09-16: *"que no aparezcan estos INFO: 127.0.0.1:xxxxx - GET /api/"* —
neither on the console nor in the file (one stream, teed by scripts/start.sh).
"""

import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from access_log import (FALLBACK_QUIET, PollingNoiseFilter,  # noqa: E402
                        install, load_quiet_prefixes)


def _access(method: str, path: str, status: int) -> logging.LogRecord:
    """A record shaped exactly like uvicorn's access log."""
    return logging.LogRecord(
        name="uvicorn.access", level=logging.INFO, pathname=__file__, lineno=1,
        msg='%s - "%s %s HTTP/%s" %d',
        args=("127.0.0.1:42292", method, path, "1.1", status),
        exc_info=None)


@pytest.fixture()
def filt():
    return PollingNoiseFilter(("GET /health", "GET /api/correction/state/",
                               "GET /api/certify/state/", "GET /potree/"))


@pytest.mark.parametrize("path", [
    "/api/correction/state/pccr",
    "/api/certify/state/pccr",
    "/health",
    "/potree/pccr/metadata.json",
])
def test_the_polls_are_dropped(filt, path):
    assert filt.filter(_access("GET", path, 200)) is False


def test_everything_else_still_prints(filt):
    assert filt.filter(_access("GET", "/api/sessions", 200)) is True
    assert filt.filter(_access("POST", "/api/correction/select", 200)) is True
    # a POST to a quiet PREFIX is not a poll either
    assert filt.filter(_access("POST", "/api/certify/state/pccr", 200)) is True


@pytest.mark.parametrize("status", [400, 404, 409, 500, 503])
def test_a_failing_poll_is_never_silenced(filt, status):
    """The whole point of the exception: /api/certify/state polled every 5 s is
    noise at 200 and the only evidence you have at 500."""
    assert filt.filter(_access("GET", "/api/certify/state/pccr", status)) is True


def test_partial_content_spam_is_dropped(filt):
    """206 range requests: hundreds per cloud load (Potree nodes, video)."""
    assert filt.filter(_access("GET", "/sessions/pccr/video.mp4", 206)) is False


def test_a_non_access_line_passes_untouched(filt):
    rec = logging.LogRecord(name="uvicorn.error", level=logging.INFO,
                            pathname=__file__, lineno=1,
                            msg="Application startup complete.", args=None,
                            exc_info=None)
    assert filt.filter(rec) is True


def test_the_list_comes_from_config_yaml():
    quiet = load_quiet_prefixes()
    assert "GET /api/correction/state/" in quiet, quiet
    assert "GET /api/certify/state/" in quiet, quiet
    # the real config must cover what the user actually sees
    assert any(q.startswith("GET /health") for q in quiet)


def test_a_broken_config_does_not_stop_the_server(tmp_path):
    bad = tmp_path / "config.yaml"
    bad.write_text("server: [this is not a mapping\n")
    assert load_quiet_prefixes(bad) == FALLBACK_QUIET
    assert load_quiet_prefixes(tmp_path / "does_not_exist.yaml") == FALLBACK_QUIET


def test_install_is_idempotent():
    log = logging.getLogger("test.access.idempotent")
    log.filters.clear()
    assert install(log.name) is True
    assert install(log.name) is False
    assert sum(isinstance(f, PollingNoiseFilter) for f in log.filters) == 1
    log.filters.clear()
