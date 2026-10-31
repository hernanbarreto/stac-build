"""pytest bootstrap — make the server package importable from server/tests/."""
import sys
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parent.parent
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))


import pytest  # noqa: E402


@pytest.fixture(autouse=True, scope="session")
def _hole_audit_evidence_on_cpu():
    """The synthetic tests run the hole-audit evidence on the CPU — the same code path as the
    card (reconstruction.surface_fit.hole_audit.use_device); a GPU test picks cuda itself and
    restores this."""
    from reconstruction.surface_fit import fit_torch, hole_audit
    hole_audit.use_device("cpu")
    fit_torch.use_device("cpu")
    yield


@pytest.fixture(autouse=True, scope="session")
def _projects_dir_is_never_the_servers(tmp_path_factory):
    """No test writes into the server's projects directory: ``config.PROJECTS_DIR`` is a temporary
    directory for the whole session (2026-10-31: two eval projects had been left in server/projects by
    a test that published its variants there)."""
    import config
    tmp = tmp_path_factory.mktemp("projects")
    mp = pytest.MonkeyPatch()
    mp.setattr(config, "PROJECTS_DIR", tmp)
    yield
    mp.undo()
