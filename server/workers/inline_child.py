"""Entry point of a worker hosted by another worker (workers.base.run_stage_inline).

    python -m workers.inline_child <module> <session_dir> <fd>

``fd`` is the child end of a multiprocessing Pipe the host created and passed
down (Popen pass_fds) — the same Connection protocol every pipeline stage speaks
(progress / log / done / error, cancel from the host). The config dict arrives as
the first message on that pipe (pickled, so YAML values survive as they are). The
host runs this as a plain subprocess because the pipeline's stage workers are
daemonic multiprocessing processes and Python forbids those to spawn children.
"""

from __future__ import annotations

import importlib
import os
import sys
from multiprocessing.connection import Connection


def main(argv) -> int:
    if len(argv) != 4:
        print("usage: python -m workers.inline_child <module> <session_dir> <fd>", file=sys.stderr)
        return 2
    module_name, session_dir, fd = argv[1], argv[2], int(argv[3])
    server_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)
    conn = Connection(fd)
    config = conn.recv()
    mod = importlib.import_module(module_name)
    mod.run(conn, session_dir, config)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
