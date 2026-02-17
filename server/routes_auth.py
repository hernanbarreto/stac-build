# STAC-BUILD: Auth REST API Routes
# Hernán Barreto — Ingerop IN3

from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db import User, async_session_factory
from auth import (
    hash_password, verify_password, create_access_token,
    get_current_user, require_role,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])

# ─── Request / Response Schemas ────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    password: str
    email: Optional[str] = None
    role: str = "viewer"
    full_name: Optional[str] = None

class UpdateUserRequest(BaseModel):
    email: Optional[str] = None
    role: Optional[str] = None
    full_name: Optional[str] = None
    is_active: Optional[bool] = None
    password: Optional[str] = None

class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str

# ─── Endpoints ─────────────────────────────────────────────────

@router.post("/login")
async def login(body: LoginRequest):
    """Authenticate user and return JWT token."""
    async with async_session_factory() as session:
        result = await session.execute(
            select(User).where(User.username == body.username)
        )
        user = result.scalar_one_or_none()

    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account disabled")

    # Update last_login
    async with async_session_factory() as session:
        user_to_update = await session.get(User, user.id)
        user_to_update.last_login = datetime.utcnow()
        await session.commit()

    token = create_access_token({
        "sub": str(user.id),
        "username": user.username,
        "role": user.role,
    })

    return {"token": token, "user": user.to_dict()}


@router.get("/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    """Get current authenticated user info."""
    async with async_session_factory() as session:
        user = await session.get(User, current_user["id"])
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return {"user": user.to_dict()}


@router.post("/change-password")
async def change_password(
    body: ChangePasswordRequest,
    current_user: dict = Depends(get_current_user),
):
    """Change own password."""
    async with async_session_factory() as session:
        user = await session.get(User, current_user["id"])
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        if not verify_password(body.current_password, user.password_hash):
            raise HTTPException(status_code=400, detail="Current password is incorrect")
        user.password_hash = hash_password(body.new_password)
        await session.commit()
    return {"message": "Password changed successfully"}


# ─── Admin-only User Management ───────────────────────────────

@router.get("/users")
async def list_users(admin: dict = Depends(require_role("admin"))):
    """List all users (admin only)."""
    async with async_session_factory() as session:
        result = await session.execute(select(User).order_by(User.id))
        users = result.scalars().all()
        return {"users": [u.to_dict() for u in users]}


@router.post("/users")
async def create_user(
    body: RegisterRequest,
    admin: dict = Depends(require_role("admin")),
):
    """Create a new user (admin only)."""
    if body.role not in User.ROLES:
        raise HTTPException(status_code=400, detail=f"Invalid role. Must be one of: {User.ROLES}")

    async with async_session_factory() as session:
        # Check duplicate username
        existing = await session.execute(
            select(User).where(User.username == body.username)
        )
        if existing.scalar_one_or_none():
            raise HTTPException(status_code=409, detail="Username already exists")

        user = User(
            username=body.username,
            email=body.email,
            password_hash=hash_password(body.password),
            role=body.role,
            full_name=body.full_name,
            is_active=True,
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return {"user": user.to_dict()}


@router.put("/users/{user_id}")
async def update_user(
    user_id: int,
    body: UpdateUserRequest,
    admin: dict = Depends(require_role("admin")),
):
    """Update user details (admin only)."""
    async with async_session_factory() as session:
        user = await session.get(User, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        if body.email is not None:
            user.email = body.email
        if body.role is not None:
            if body.role not in User.ROLES:
                raise HTTPException(status_code=400, detail=f"Invalid role")
            user.role = body.role
        if body.full_name is not None:
            user.full_name = body.full_name
        if body.is_active is not None:
            user.is_active = body.is_active
        if body.password is not None:
            user.password_hash = hash_password(body.password)

        await session.commit()
        await session.refresh(user)
        return {"user": user.to_dict()}


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    admin: dict = Depends(require_role("admin")),
):
    """Delete a user (admin only). Cannot delete yourself."""
    if user_id == admin["id"]:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")

    async with async_session_factory() as session:
        user = await session.get(User, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        await session.delete(user)
        await session.commit()
    return {"message": "User deleted"}
