# STAC-BUILD: Database Models & Init
# Hernán Barreto — Ingerop IN3

import asyncio
from datetime import datetime
from pathlib import Path
from sqlalchemy import (
    Column, Integer, String, Boolean, DateTime, Text,
    create_engine
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker as async_sessionmaker

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "stac.db"
DATABASE_URL = f"sqlite+aiosqlite:///{DB_PATH}"

Base = declarative_base()

# ─── Models ────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(120), unique=True, nullable=True)
    password_hash = Column(String(128), nullable=False)
    role = Column(String(20), nullable=False, default="viewer")
    full_name = Column(String(100), nullable=True)
    avatar_url = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

    # Valid roles
    ROLES = ("admin", "manager", "editor", "viewer", "recorder")

    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "role": self.role,
            "full_name": self.full_name,
            "avatar_url": self.avatar_url,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
        }

# ─── Engine & Session Factory ──────────────────────────────────

engine = create_async_engine(DATABASE_URL, echo=False)
async_session_factory = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

async def get_session() -> AsyncSession:
    async with async_session_factory() as session:
        yield session

# ─── Init / Bootstrap ──────────────────────────────────────────

async def init_db():
    """Create tables and bootstrap admin user if not exists."""
    from auth import hash_password  # local import to avoid circular
    import db_team  # noqa: F401 — register Team/TeamMember/etc. models
    import db_project  # noqa: F401 — register Project/ScanDay/Source/IFC models

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Bootstrap: create admin user if no users exist
    async with async_session_factory() as session:
        from sqlalchemy import select, func
        result = await session.execute(select(func.count(User.id)))
        count = result.scalar()
        if count == 0:
            admin = User(
                username="admin",
                email="admin@stac.local",
                password_hash=hash_password("admin"),
                role="admin",
                full_name="Administrator",
                is_active=True,
            )
            manager = User(
                username="manager",
                email="manager@stac.local",
                password_hash=hash_password("manager"),
                role="manager",
                full_name="Project Manager",
                is_active=True,
            )
            session.add_all([admin, manager])
            await session.commit()
            print("[DB] ✅ Created test users: admin/admin, manager/manager")
        else:
            print(f"[DB] ✅ Database ready ({count} users)")
