# STAC-BUILD: Activity Logger — Comprehensive File Persistence
# Logs to DB + personnel/{user}/activity.jsonl + teams/{id}/team.json
# Hernán Barreto — Ingerop IN3

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from db import async_session_factory
from db.team import ActivityLog

from config import DATA_DIR

PERSONNEL_DIR = DATA_DIR / "personnel"
TEAMS_DIR = DATA_DIR / "teams"


# ═══════════════════════════════════════════════════════════════
# 1.  Core activity logging (DB + user JSONL)
# ═══════════════════════════════════════════════════════════════

async def log_activity(
    user_id: int,
    username: str,
    action: str,
    *,
    session_id: str | None = None,
    team_id: int | None = None,
    detail: str = "",
):
    """
    Log an activity to both the database and the actor's personal JSONL file.
    Fire-and-forget safe — errors are caught and printed.
    """
    now = datetime.utcnow()

    # ── 1. Database write ──────────────────────────────────────
    try:
        async with async_session_factory() as session:
            entry = ActivityLog(
                user_id=user_id,
                username=username,
                team_id=team_id,
                session_id=session_id,
                action=action,
                detail=detail,
                timestamp=now,
            )
            session.add(entry)
            await session.commit()
    except Exception as e:
        print(f"[ActivityLogger] DB write failed: {e}")

    # ── 2. User's activity.jsonl ───────────────────────────────
    _append_user_log(username, action, detail=detail, session_id=session_id, team_id=team_id, ts=now)


def _append_user_log(
    username: str,
    action: str,
    *,
    detail: str = "",
    session_id: str | None = None,
    team_id: int | None = None,
    extra: dict | None = None,
    ts: datetime | None = None,
):
    """Append an entry to personnel/{username}/activity.jsonl (sync, non-throwing)."""
    try:
        now = ts or datetime.utcnow()
        user_dir = PERSONNEL_DIR / username
        user_dir.mkdir(parents=True, exist_ok=True)
        log_path = user_dir / "activity.jsonl"

        record: dict[str, Any] = {
            "ts": now.isoformat() + "Z",
            "action": action,
        }
        if session_id:
            record["session"] = session_id
        if team_id:
            record["team_id"] = team_id
        if detail:
            record["detail"] = detail
        if extra:
            record.update(extra)

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[ActivityLogger] File write failed for {username}: {e}")


# ═══════════════════════════════════════════════════════════════
# 2.  Personnel profile persistence
# ═══════════════════════════════════════════════════════════════

def save_user_profile(user_dict: dict, *, teams: list[int] | None = None):
    """
    Create or update personnel/{username}/profile.json.
    Call after user creation or any user update.
    """
    try:
        username = user_dict["username"]
        user_dir = PERSONNEL_DIR / username
        user_dir.mkdir(parents=True, exist_ok=True)
        profile_path = user_dir / "profile.json"

        profile: dict[str, Any] = {
            "id": user_dict.get("id"),
            "username": username,
            "full_name": user_dict.get("full_name"),
            "email": user_dict.get("email"),
            "role": user_dict.get("role"),
            "is_active": user_dict.get("is_active", True),
            "created_at": user_dict.get("created_at"),
            "last_login": user_dict.get("last_login"),
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        if teams is not None:
            profile["teams"] = teams

        # Merge with existing profile to preserve teams if not provided
        if profile_path.exists() and teams is None:
            try:
                existing = json.loads(profile_path.read_text(encoding="utf-8"))
                if "teams" in existing:
                    profile["teams"] = existing["teams"]
            except Exception:
                pass

        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)
            f.write("\n")

    except Exception as e:
        print(f"[ActivityLogger] Profile write failed: {e}")


def update_user_teams(username: str, team_ids: list[int]):
    """Update the 'teams' field in an existing profile.json."""
    try:
        profile_path = PERSONNEL_DIR / username / "profile.json"
        if profile_path.exists():
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            profile["teams"] = team_ids
            profile["updated_at"] = datetime.utcnow().isoformat() + "Z"
            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(profile, f, indent=2, ensure_ascii=False)
                f.write("\n")
    except Exception as e:
        print(f"[ActivityLogger] Profile teams update failed for {username}: {e}")


# ═══════════════════════════════════════════════════════════════
# 3.  Team file persistence
# ═══════════════════════════════════════════════════════════════

def save_team_file(team_data: dict):
    """
    Create or update teams/{team_id}_{sanitized_name}/team.json.
    Call after any team mutation.

    team_data should include:
      id, name, description, manager (dict with id/username),
      members (list), sessions (list), is_active
    """
    try:
        team_id = team_data["id"]
        name_safe = _sanitize(team_data["name"])
        team_dir = TEAMS_DIR / f"{team_id}_{name_safe}"
        team_dir.mkdir(parents=True, exist_ok=True)
        team_path = team_dir / "team.json"

        payload: dict[str, Any] = {
            "id": team_id,
            "name": team_data.get("name"),
            "description": team_data.get("description"),
            "is_active": team_data.get("is_active", True),
            "manager": team_data.get("manager"),
            "members": team_data.get("members", []),
            "sessions": team_data.get("sessions", []),
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }

        with open(team_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")

    except Exception as e:
        print(f"[ActivityLogger] Team file write failed: {e}")


def log_to_team(team_id: int, team_name: str, action: str, *, detail: str = "", actor: str = ""):
    """Append an entry to teams/{id}_{name}/activity.jsonl."""
    try:
        name_safe = _sanitize(team_name)
        team_dir = TEAMS_DIR / f"{team_id}_{name_safe}"
        team_dir.mkdir(parents=True, exist_ok=True)
        log_path = team_dir / "activity.jsonl"

        record: dict[str, Any] = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "action": action,
        }
        if actor:
            record["actor"] = actor
        if detail:
            record["detail"] = detail

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[ActivityLogger] Team log write failed: {e}")


def delete_team_folder(team_id: int, team_name: str):
    """Remove the team folder when a team is deleted."""
    import shutil
    try:
        name_safe = _sanitize(team_name)
        team_dir = TEAMS_DIR / f"{team_id}_{name_safe}"
        if team_dir.exists():
            shutil.rmtree(team_dir)
    except Exception as e:
        print(f"[ActivityLogger] Team folder delete failed: {e}")


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _sanitize(name: str) -> str:
    """Sanitize a name for use in a folder path."""
    return "".join(c if c.isalnum() or c in ("-", "_", " ") else "_" for c in name).strip().replace(" ", "_")
