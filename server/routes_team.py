# STAC-BUILD: Team Module — REST API Routes (with file persistence)
# Hernán Barreto — Ingerop IN3

from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from db import User, async_session_factory
from db_team import Team, TeamMember, SessionAssignment, ActivityLog, TeamMessage
from auth import get_current_user, require_role
from activity_logger import (
    log_activity, _append_user_log, save_user_profile,
    save_team_file, log_to_team, delete_team_folder, update_user_teams,
)

router = APIRouter(prefix="/api/teams", tags=["teams"])

# ─── Request Schemas ───────────────────────────────────────────

class CreateTeamRequest(BaseModel):
    name: str
    description: Optional[str] = None
    manager_id: int

class UpdateTeamRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    manager_id: Optional[int] = None
    is_active: Optional[bool] = None

class AddMemberRequest(BaseModel):
    user_id: int

class AssignSessionRequest(BaseModel):
    session_id: str

class SendMessageRequest(BaseModel):
    content: str


# ─── Helper: build full team snapshot for file persistence ─────

async def _build_team_snapshot(session, team: Team) -> dict:
    """Build a complete team dict for save_team_file()."""
    manager = await session.get(User, team.manager_id)
    manager_info = {"id": team.manager_id, "username": manager.username if manager else "unknown"}

    members_q = await session.execute(
        select(TeamMember, User).join(User, TeamMember.user_id == User.id).where(
            TeamMember.team_id == team.id
        )
    )
    members = []
    for tm, user in members_q.all():
        members.append({
            "user_id": user.id,
            "username": user.username,
            "full_name": user.full_name,
            "role": user.role,
            "added_at": tm.joined_at.isoformat() + "Z" if tm.joined_at else None,
        })

    sessions_q = await session.execute(
        select(SessionAssignment).where(SessionAssignment.team_id == team.id)
    )
    sessions = [sa.session_id for sa in sessions_q.scalars().all()]

    return {
        "id": team.id,
        "name": team.name,
        "description": team.description,
        "is_active": team.is_active,
        "manager": manager_info,
        "members": members,
        "sessions": sessions,
    }


async def _get_user_team_ids(session, user_id: int) -> list[int]:
    """Get a list of team IDs a user belongs to (as member or manager)."""
    member_q = await session.execute(
        select(TeamMember.team_id).where(TeamMember.user_id == user_id)
    )
    managed_q = await session.execute(
        select(Team.id).where(Team.manager_id == user_id)
    )
    ids = set(r[0] for r in member_q.all()) | set(r[0] for r in managed_q.all())
    return sorted(ids)


# ─── Team CRUD (Admin only) ───────────────────────────────────

@router.get("")
async def list_teams(current_user: dict = Depends(get_current_user)):
    """List teams. Admin sees all; others see only their teams."""
    async with async_session_factory() as session:
        if current_user["role"] == "admin":
            result = await session.execute(select(Team).order_by(Team.id))
            teams = result.scalars().all()
        else:
            member_team_ids = select(TeamMember.team_id).where(
                TeamMember.user_id == current_user["id"]
            )
            managed_team_ids = select(Team.id).where(
                Team.manager_id == current_user["id"]
            )
            result = await session.execute(
                select(Team).where(
                    Team.id.in_(member_team_ids) | Team.id.in_(managed_team_ids)
                ).order_by(Team.id)
            )
            teams = result.scalars().all()

        team_dicts = []
        for team in teams:
            td = team.to_dict()
            manager = await session.get(User, team.manager_id)
            td["manager_name"] = manager.full_name or manager.username if manager else "Unknown"

            members_q = await session.execute(
                select(TeamMember, User).join(User, TeamMember.user_id == User.id).where(
                    TeamMember.team_id == team.id
                )
            )
            td["members"] = []
            for tm, user in members_q.all():
                td["members"].append({
                    **tm.to_dict(),
                    "username": user.username,
                    "full_name": user.full_name,
                    "role": user.role,
                    "avatar_url": user.avatar_url,
                })

            sessions_q = await session.execute(
                select(SessionAssignment).where(SessionAssignment.team_id == team.id)
            )
            td["sessions"] = [sa.to_dict() for sa in sessions_q.scalars().all()]

            team_dicts.append(td)

        return {"teams": team_dicts}


@router.post("")
async def create_team(
    body: CreateTeamRequest,
    admin: dict = Depends(require_role("admin")),
):
    """Create a new team (admin only)."""
    async with async_session_factory() as session:
        manager = await session.get(User, body.manager_id)
        if not manager:
            raise HTTPException(status_code=404, detail="Manager user not found")

        team = Team(
            name=body.name,
            description=body.description,
            manager_id=body.manager_id,
        )
        session.add(team)
        await session.commit()
        await session.refresh(team)

        # Auto-add manager as member
        member = TeamMember(
            team_id=team.id,
            user_id=body.manager_id,
            added_by=admin["id"],
        )
        session.add(member)
        await session.commit()

        # Build snapshot and persist team file
        snapshot = await _build_team_snapshot(session, team)
        save_team_file(snapshot)

        # Update manager's profile with team list
        manager_teams = await _get_user_team_ids(session, body.manager_id)
        update_user_teams(manager.username, manager_teams)

    # ── Logs ──
    await log_activity(
        admin["id"], admin["username"], "team_created",
        team_id=team.id,
        detail=f"Created team '{body.name}' with manager {manager.username}",
    )
    log_to_team(team.id, body.name, "team_created",
        actor=admin["username"], detail=f"Team created with manager {manager.username}")
    _append_user_log(manager.username, "added_to_team",
        team_id=team.id, detail=f"Added as manager to new team '{body.name}'")

    return {"team": team.to_dict()}


@router.put("/{team_id}")
async def update_team(
    team_id: int,
    body: UpdateTeamRequest,
    admin: dict = Depends(require_role("admin")),
):
    """Update team (admin only)."""
    async with async_session_factory() as session:
        team = await session.get(Team, team_id)
        if not team:
            raise HTTPException(status_code=404, detail="Team not found")

        changes = []
        old_name = team.name

        if body.name is not None:
            changes.append(f"name: {team.name} → {body.name}")
            team.name = body.name
        if body.description is not None:
            changes.append(f"description updated")
            team.description = body.description
        if body.manager_id is not None:
            old_manager = await session.get(User, team.manager_id)
            new_manager = await session.get(User, body.manager_id)
            if not new_manager:
                raise HTTPException(status_code=404, detail="Manager user not found")
            changes.append(f"manager: {old_manager.username if old_manager else '?'} → {new_manager.username}")
            team.manager_id = body.manager_id
            # Log to both managers
            if old_manager:
                _append_user_log(old_manager.username, "removed_as_manager",
                    team_id=team_id, detail=f"No longer manager of '{team.name}'")
            _append_user_log(new_manager.username, "assigned_as_manager",
                team_id=team_id, detail=f"Now manager of '{team.name}'")
        if body.is_active is not None:
            changes.append(f"is_active: {team.is_active} → {body.is_active}")
            team.is_active = body.is_active

        await session.commit()
        await session.refresh(team)

        # Persist team file
        snapshot = await _build_team_snapshot(session, team)
        save_team_file(snapshot)

    change_detail = "; ".join(changes) if changes else "no changes"

    await log_activity(
        admin["id"], admin["username"], "team_updated",
        team_id=team.id, detail=f"Updated team '{team.name}': {change_detail}",
    )
    log_to_team(team.id, team.name, "team_updated",
        actor=admin["username"], detail=change_detail)

    return {"team": team.to_dict()}


@router.delete("/{team_id}")
async def delete_team(
    team_id: int,
    admin: dict = Depends(require_role("admin")),
):
    """Delete team (admin only)."""
    async with async_session_factory() as session:
        team = await session.get(Team, team_id)
        if not team:
            raise HTTPException(status_code=404, detail="Team not found")
        name = team.name

        # Get member list before deletion for logging
        members_q = await session.execute(
            select(TeamMember, User).join(User, TeamMember.user_id == User.id).where(
                TeamMember.team_id == team_id
            )
        )
        member_users = [(tm, user) for tm, user in members_q.all()]

        await session.delete(team)
        await session.commit()

        # Update profiles of all former members
        for _tm, user in member_users:
            user_teams = await _get_user_team_ids(session, user.id)
            update_user_teams(user.username, user_teams)
            _append_user_log(user.username, "team_deleted",
                team_id=team_id, detail=f"Team '{name}' was deleted")

    await log_activity(
        admin["id"], admin["username"], "team_deleted", detail=f"Deleted team '{name}'",
    )
    # Team folder is kept as archive (contains final activity log)
    return {"message": f"Team '{name}' deleted"}


# ─── Members (Manager of team or Admin) ───────────────────────

async def _require_team_manager(team_id: int, user: dict):
    """Check that user is admin or the manager of the given team."""
    if user["role"] == "admin":
        return
    async with async_session_factory() as session:
        team = await session.get(Team, team_id)
        if not team or team.manager_id != user["id"]:
            raise HTTPException(status_code=403, detail="Only the team manager can do this")


@router.post("/{team_id}/members")
async def add_member(
    team_id: int,
    body: AddMemberRequest,
    current_user: dict = Depends(get_current_user),
):
    """Add a member to a team (manager of the team or admin)."""
    await _require_team_manager(team_id, current_user)

    async with async_session_factory() as session:
        team = await session.get(Team, team_id)
        if not team:
            raise HTTPException(status_code=404, detail="Team not found")

        target_user = await session.get(User, body.user_id)
        if not target_user:
            raise HTTPException(status_code=404, detail="User not found")

        # Check duplicate
        existing = await session.execute(
            select(TeamMember).where(
                and_(TeamMember.team_id == team_id, TeamMember.user_id == body.user_id)
            )
        )
        if existing.scalar_one_or_none():
            raise HTTPException(status_code=409, detail="User already in team")

        member = TeamMember(
            team_id=team_id,
            user_id=body.user_id,
            added_by=current_user["id"],
        )
        session.add(member)
        await session.commit()

        # Persist team file
        snapshot = await _build_team_snapshot(session, team)
        save_team_file(snapshot)

        # Update user's profile with team list
        user_teams = await _get_user_team_ids(session, body.user_id)
        update_user_teams(target_user.username, user_teams)

    # ── Logs to actor, member, and team ──
    await log_activity(
        current_user["id"], current_user["username"], "member_added",
        team_id=team_id,
        detail=f"Added {target_user.username} to team '{team.name}'",
    )
    _append_user_log(target_user.username, "added_to_team",
        team_id=team_id, detail=f"Added to team '{team.name}' by {current_user['username']}")
    log_to_team(team_id, team.name, "member_added",
        actor=current_user["username"], detail=f"Added {target_user.username}")

    return {"message": f"Added {target_user.username} to team"}


@router.delete("/{team_id}/members/{user_id}")
async def remove_member(
    team_id: int,
    user_id: int,
    current_user: dict = Depends(get_current_user),
):
    """Remove a member from a team (manager or admin)."""
    await _require_team_manager(team_id, current_user)

    async with async_session_factory() as session:
        team = await session.get(Team, team_id)
        result = await session.execute(
            select(TeamMember).where(
                and_(TeamMember.team_id == team_id, TeamMember.user_id == user_id)
            )
        )
        member = result.scalar_one_or_none()
        if not member:
            raise HTTPException(status_code=404, detail="Member not found in team")

        target_user = await session.get(User, user_id)
        await session.delete(member)
        await session.commit()

        # Persist team file
        snapshot = await _build_team_snapshot(session, team)
        save_team_file(snapshot)

        # Update user's profile
        if target_user:
            user_teams = await _get_user_team_ids(session, user_id)
            update_user_teams(target_user.username, user_teams)

    username = target_user.username if target_user else str(user_id)
    team_name = team.name if team else str(team_id)

    await log_activity(
        current_user["id"], current_user["username"], "member_removed",
        team_id=team_id, detail=f"Removed {username} from team '{team_name}'",
    )
    _append_user_log(username, "removed_from_team",
        team_id=team_id, detail=f"Removed from team '{team_name}' by {current_user['username']}")
    log_to_team(team_id, team_name, "member_removed",
        actor=current_user["username"], detail=f"Removed {username}")

    return {"message": f"Removed {username} from team"}


# ─── Session Assignments (Admin only) ─────────────────────────

@router.post("/{team_id}/sessions")
async def assign_session(
    team_id: int,
    body: AssignSessionRequest,
    admin: dict = Depends(require_role("admin")),
):
    """Assign a scan session to a team (admin only)."""
    async with async_session_factory() as session:
        team = await session.get(Team, team_id)
        if not team:
            raise HTTPException(status_code=404, detail="Team not found")

        existing = await session.execute(
            select(SessionAssignment).where(
                and_(
                    SessionAssignment.team_id == team_id,
                    SessionAssignment.session_id == body.session_id,
                )
            )
        )
        if existing.scalar_one_or_none():
            raise HTTPException(status_code=409, detail="Session already assigned to team")

        assignment = SessionAssignment(
            team_id=team_id,
            session_id=body.session_id,
            assigned_by=admin["id"],
        )
        session.add(assignment)
        await session.commit()

        # Persist team file
        snapshot = await _build_team_snapshot(session, team)
        save_team_file(snapshot)

    await log_activity(
        admin["id"], admin["username"], "session_assigned",
        team_id=team_id, session_id=body.session_id,
        detail=f"Assigned session '{body.session_id}' to team '{team.name}'",
    )
    log_to_team(team_id, team.name, "session_assigned",
        actor=admin["username"], detail=f"Session '{body.session_id}' assigned")

    return {"message": f"Session '{body.session_id}' assigned to team '{team.name}'"}


@router.delete("/{team_id}/sessions/{session_id}")
async def unassign_session(
    team_id: int,
    session_id: str,
    admin: dict = Depends(require_role("admin")),
):
    """Unassign a session from a team (admin only)."""
    async with async_session_factory() as session:
        team = await session.get(Team, team_id)
        result = await session.execute(
            select(SessionAssignment).where(
                and_(
                    SessionAssignment.team_id == team_id,
                    SessionAssignment.session_id == session_id,
                )
            )
        )
        assignment = result.scalar_one_or_none()
        if not assignment:
            raise HTTPException(status_code=404, detail="Assignment not found")

        await session.delete(assignment)
        await session.commit()

        # Persist team file
        if team:
            snapshot = await _build_team_snapshot(session, team)
            save_team_file(snapshot)

    team_name = team.name if team else str(team_id)
    await log_activity(
        admin["id"], admin["username"], "session_unassigned",
        team_id=team_id, session_id=session_id,
        detail=f"Unassigned session '{session_id}' from team '{team_name}'",
    )
    log_to_team(team_id, team_name, "session_unassigned",
        actor=admin["username"], detail=f"Session '{session_id}' unassigned")

    return {"message": f"Session '{session_id}' unassigned"}


# ─── Activity Log ─────────────────────────────────────────────

@router.get("/{team_id}/activity")
async def get_activity(
    team_id: int,
    limit: int = 50,
    current_user: dict = Depends(get_current_user),
):
    """Get activity log for a team."""
    if current_user["role"] != "admin":
        async with async_session_factory() as session:
            is_member = await session.execute(
                select(TeamMember).where(
                    and_(TeamMember.team_id == team_id, TeamMember.user_id == current_user["id"])
                )
            )
            team = await session.get(Team, team_id)
            if not is_member.scalar_one_or_none() and (not team or team.manager_id != current_user["id"]):
                raise HTTPException(status_code=403, detail="Not a member of this team")

    async with async_session_factory() as session:
        result = await session.execute(
            select(ActivityLog)
            .where(ActivityLog.team_id == team_id)
            .order_by(ActivityLog.timestamp.desc())
            .limit(limit)
        )
        entries = result.scalars().all()
        return {"activity": [e.to_dict() for e in entries]}


# ─── Team Messages ─────────────────────────────────────────────

@router.get("/{team_id}/messages")
async def get_messages(
    team_id: int,
    limit: int = 100,
    current_user: dict = Depends(get_current_user),
):
    """Get recent messages for a team."""
    async with async_session_factory() as session:
        result = await session.execute(
            select(TeamMessage)
            .where(TeamMessage.team_id == team_id)
            .order_by(TeamMessage.timestamp.desc())
            .limit(limit)
        )
        messages = result.scalars().all()
        return {"messages": [m.to_dict() for m in reversed(messages)]}


@router.post("/{team_id}/messages")
async def send_message(
    team_id: int,
    body: SendMessageRequest,
    current_user: dict = Depends(get_current_user),
):
    """Send a message to a team chat."""
    async with async_session_factory() as session:
        team = await session.get(Team, team_id)
        msg = TeamMessage(
            team_id=team_id,
            user_id=current_user["id"],
            username=current_user["username"],
            content=body.content,
        )
        session.add(msg)
        await session.commit()
        await session.refresh(msg)

    # Log message to user and team
    await log_activity(
        current_user["id"], current_user["username"], "message_sent",
        team_id=team_id, detail=f"Sent message to team chat",
    )
    if team:
        log_to_team(team_id, team.name, "message_sent",
            actor=current_user["username"], detail=f"Message: {body.content[:80]}")

    return {"message": msg.to_dict()}


# ─── Users list (for member management) ───────────────────────

@router.get("/available-users")
async def list_available_users(current_user: dict = Depends(get_current_user)):
    """List all active users (for adding to teams)."""
    async with async_session_factory() as session:
        result = await session.execute(
            select(User).where(User.is_active == True).order_by(User.username)
        )
        users = result.scalars().all()
        return {"users": [
            {"id": u.id, "username": u.username, "full_name": u.full_name, "role": u.role, "avatar_url": u.avatar_url}
            for u in users
        ]}
