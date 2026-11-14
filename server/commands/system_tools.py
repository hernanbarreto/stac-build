"""The commands the SERVER runs with no scene (``runs_on: system``): the conversation and the
help. A handler takes the model's arguments and returns a result the answer is written from
(phase5_qa/answers.py), as the scene's tools do."""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from .db import CommandsDB, fold


class SystemTools:
    def __init__(self, db: CommandsDB, lang: str):
        self.db = db
        self.lang = lang

    def small_talk(self, reply: str = "") -> Dict[str, Any]:
        """Conversation: what the model wrote is the reply (answers.py refuses a figure in it)."""
        return {"ok": True, "reply": str(reply or ""),
                "note": "nothing was measured and nothing changed in the application"}

    def help_commands(self, area: Optional[str] = None) -> Dict[str, Any]:
        """The areas of commands, or the commands of one area — from the database, in the
        user's language. An area is named by its id or its title in any language."""
        areas = self.db.areas(self.lang)
        if not area:
            return {"areas": [{"id": a["id"], "title": a["title"], "commands": a["commands"]} for a in areas],
                    "total": self.db.count()}
        want = fold(area)
        hit = next((a for a in areas if fold(a["id"]) == want), None)
        if hit is None:
            for lang in ("es", "en", "fr"):
                hit = next((a for a, b in zip(areas, self.db.areas(lang)) if fold(b["title"]) == want), None)
                if hit is not None:
                    break
        if hit is None:
            hit = next((a for a in areas if want and (want in fold(a["title"]) or want in fold(a["id"]))), None)
        if hit is None:
            return {"insufficient_data": True,
                    "reason": f"no area '{area}' — the areas are: " + ", ".join(a["title"] for a in areas)}
        rows = self.db.list(self.lang, area=hit["id"], status=("available", "manual"))
        return {"area": hit["id"], "title": hit["title"],
                "commands": [r["title"] for r in rows if r["status"] == "available"],
                "manual": [r["title"] for r in rows if r["status"] == "manual"]}

    def impls(self) -> Dict[str, Callable[..., Any]]:
        """tool name → handler, for every system row of the database (a row with no handler
        here fails naming it: a command is never half built)."""
        out: Dict[str, Callable[..., Any]] = {}
        for row in self.db.list(runs_on="system"):
            fn = getattr(self, row["handler"], None)
            if not callable(fn):
                raise RuntimeError(f"commands: the system command {row['id']} has no handler "
                                   f"'{row['handler']}' in commands.system_tools.SystemTools")
            out[row["tool"]] = fn
        return out
