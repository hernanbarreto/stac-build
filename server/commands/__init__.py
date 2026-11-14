"""THE DATABASE OF COMMANDS — everything STAC can be asked to do, one row per command.

USER 2026-11-14: "debes armar una base de datos con absolutamente todos los comandos a los que
luego se iran incorporando los nuevos comandos en la medida que stac build crezca en
funcionalidades" — and, on speaking to STAC: "debe poder discernir entre un comando y una consulta
de chat … manejar todos los comandos que tiene stac build y como se integraran los del futuro que
seran muchisimos".

    catalog/*.yaml  →  catalog.load_catalog()  →  db.build()  →  commands.db (sqlite + FTS5)

  - A command is DECLARED once, in its area's file under ``catalog/`` (text, in git: every command
    has a history and a review): what it is (action / query / talk), where it runs (the server's
    tool layer or the interface), its arguments, whether it asks for confirmation, and in the
    three languages of the system (es / en / fr) its name, how people ask for it and what STAC
    answers when it is done.
  - The DATABASE is what everything reads at run time: the chat and the voice (the tools the model
    is offered, chosen per sentence among all of them — :mod:`commands.router`), the interface
    (its executors are checked against the rows), the help. It is rebuilt whenever the
    declarations change (their sha256 is in the database).
  - What tells a COMMAND from a QUESTION is the row the model picks: ``kind`` = action (STAC
    operates the system), query (STAC measures / reads) or talk (conversation).

A new function of STAC = one more declaration (and its executor). Nothing else lists commands.
"""
from .catalog import AREAS_DIR, KINDS, LANGS, RUNS_ON, Catalog, CatalogError, load_catalog  # noqa: F401
from .db import CommandsDB, build, database  # noqa: F401
from .system_tools import SystemTools  # noqa: F401
