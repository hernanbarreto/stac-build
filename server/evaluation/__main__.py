"""``python -m server.evaluation …`` (from the repo root) or ``python -m evaluation …`` (from server/)."""
import sys
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parent.parent
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from evaluation.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
